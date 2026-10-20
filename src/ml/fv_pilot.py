"""Matched, per-instance SWE fields and dimensionless FV/IC/wall/depth losses.

A conservative FV operator inside a soft loss does not enforce conservation
of the predicted trajectory. The HBPP branch is constant for each case.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .hbpp import HBPPOp
from .hbpp_layers import FourierFeatures, MLP
from swe.grid import pad_scalar
from swe.physics import compute_dt
from swe.solver import Config, step


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hbpp_options(settings):
    width = settings.get("width", 32)
    return dict(branch_dim=1, coord_dim=3, out_dim=3, d=width,
        n_basis=settings.get("n_basis", 8), branch_hidden=[width],
        trunk_hidden=[width, width], scaling_hidden=[width],
        value_hidden=[width, width], basis_hidden=[8, 8],
        n_fourier=settings.get("n_fourier", 16), fourier_scale=1.0,
        scale_axes=[0, 1], activation="tanh", seed=0)


class PilotField(nn.Module):
    """Common depth/velocity output transform for MLP and HBPP representations."""
    def __init__(self, bi, architecture, settings):
        super().__init__()
        self.architecture = architecture
        self.H = float(settings.get("depth_scale_m", 10.0))
        self.V = math.sqrt(bi.g * self.H)
        self.L = bi.grid.nx * bi.grid.dx
        self.T = self.L / self.V
        self.velocity_bound = float(settings.get("velocity_bound", 4.0))
        # Optional hard initial condition: raw outputs are multiplied by t/t_end,
        # so depth equals h0 and momentum is zero at t=0 by construction.
        self.hard_ic = bool(settings.get("hard_ic", False))
        self.register_buffer("state_scale", torch.tensor([self.H, self.H*self.V, self.H*self.V], dtype=torch.float64))
        self.register_buffer("lo", torch.tensor([bi.grid.x0, bi.grid.y0, 0.0]))
        self.register_buffer("hi", torch.tensor([bi.grid.x0+self.L, bi.grid.y0+bi.grid.ny*bi.grid.dy, bi.t_end]))
        self.register_buffer("background", bi.U0[0].float() / self.H)
        self.register_buffer("branch_input", torch.ones(1, 1))
        options = hbpp_options(settings)
        if architecture in ("hbpp", "hbpp_no_warp"):
            self.net = HBPPOp(**options, use_scaling=architecture != "hbpp_no_warp")
        elif architecture == "mlp":
            # Approximate parameter-count control; same feature map and activation.
            target = sum(p.numel() for p in HBPPOp(**options).parameters())
            nfreq = options["n_fourier"]
            layers = settings.get("mlp_layers", 4)
            def count(w):
                return (2*nfreq+1)*w+(layers-1)*(w*w+w)+(w+1)*3
            width = min(range(4, 257), key=lambda w: abs(count(w)-target))
            self.embed = FourierFeatures(3, nfreq, 1.0, seed=0)
            self.net = MLP(2*nfreq, [width]*layers, 3, "tanh")
        else:
            raise ValueError(architecture)

    def raw(self, xyt):
        z = 2*(xyt.to(self.lo)-self.lo)/(self.hi-self.lo)-1
        if self.architecture.startswith("hbpp"):
            out = self.net(self.branch_input, z.unsqueeze(0))[0]
        else:
            out = self.net(self.embed(z))
        if self.hard_ic:
            out = out * ((z[..., 2:3] + 1) / 2)
        return out

    def decode(self, raw, background):
        # Inverse-softplus anchor: raw depth zero recovers h0 above the dry floor.
        base = background.to(raw).clamp_min(1e-7)
        offset = base + torch.log(-torch.expm1(-base))
        h = self.H*torch.nn.functional.softplus(raw[..., 0]+offset)
        u = self.velocity_bound*self.V*torch.tanh(raw[..., 1])
        v = self.velocity_bound*self.V*torch.tanh(raw[..., 2])
        return torch.stack([h, h*u, h*v], dim=-1)

    def points(self, xyt, background):
        return self.decode(self.raw(xyt), background)

    def predict_grid(self, t, grid):
        X,Y = grid.centers()
        xyt = torch.stack([X.flatten(),Y.flatten(),torch.full_like(X.flatten(), float(t))], dim=1)
        return self.points(xyt, self.background.flatten()).T.reshape(3,grid.ny,grid.nx).double()


def advance_cfl(U, z_pad, duration, solver, max_steps=128):
    """Differentiable updates with detached adaptive CFL decisions at every step.

    No unsafe substep cap: exceeding the work limit fails the run explicitly.
    """
    remaining=float(duration)
    steps=0
    while remaining > 1e-12:
        if steps >= max_steps:
            raise RuntimeError("FV work limit exceeded; increase intervals/max_substeps")
        safe=float(compute_dt(U.detach(),solver.grid,solver.g,cfl=solver.cfl,h_eps=solver.h_eps))
        dt=min(remaining,safe)
        if not math.isfinite(dt) or dt<=0:
            raise FloatingPointError("Invalid CFL time step")
        U=step(U,z_pad,dt,solver)
        if not torch.isfinite(U).all():
            raise FloatingPointError("Nonfinite FV state")
        remaining-=dt;steps+=1
    return U,steps


class PilotObjective:
    def __init__(self,bi,times,gauges,settings):
        self.bi,self.times,self.gauges=bi,times,gauges
        self.settings=settings
        self.solver=Config(grid=bi.grid,bc=bi.bc,scheme="hllc",order=2,limiter="van_leer",g=bi.g,manning_n=bi.manning_n,cfl=0.4)
        z=bi.z if bi.z is not None else torch.zeros_like(bi.U0[0])
        self.z_pad=pad_scalar(z.double(),bi.bc,bi.grid.ng)

    def boundary_loss(self,model,t):
        grid=self.bi.grid
        x=grid.xc.to(model.lo);y=grid.yc.to(model.lo)
        walls=[(torch.full_like(y,grid.x0),y,model.background[:,0],1),
               (torch.full_like(y,grid.x0+grid.nx*grid.dx),y,model.background[:,-1],1),
               (x,torch.full_like(x,grid.y0),model.background[0,:],2),
               (x,torch.full_like(x,grid.y0+grid.ny*grid.dy),model.background[-1,:],2)]
        terms=[]
        for xx,yy,base,axis in walls:
            points=torch.stack([xx,yy,torch.full_like(xx,float(t))],dim=1)
            U=model.points(points,base)
            terms.append(((U[:,axis]/U[:,0].clamp_min(1e-7))/model.V).square().mean())
        return torch.stack(terms).mean()

    def __call__(self,model,intervals,use_data,fv_weight=1.0):
        scale=model.state_scale[:,None,None]
        losses=[];steps=[];walls=[]
        for k in intervals:
            t0,t1=self.times[k:k+2]
            current=model.predict_grid(t0,self.bi.grid)
            advanced,n=advance_cfl(current,self.z_pad,t1-t0,self.solver,self.settings.get("max_substeps",128))
            residual=(model.predict_grid(t1,self.bi.grid)-advanced)/scale/((t1-t0)/model.T)
            losses.append(residual.square().mean());steps.append(n)
            walls.append(self.boundary_loss(model,t0))
        fv=torch.stack(losses).mean()
        ic=((model.predict_grid(0,self.bi.grid)-self.bi.U0)/scale).square().mean()
        bc=torch.stack(walls).mean()
        data=fv.new_zeros(())
        if use_data:
            pred=model.points(self.gauges["xyt"],self.gauges["background"])
            data=((pred[:,0]-self.gauges["depth"])/model.H).square().mean()
        weights=self.settings.get("weights",{"fv":1.,"ic":10.,"bc":1.,"data":5.})
        total=weights["fv"]*fv_weight*fv+weights["ic"]*ic+weights["bc"]*bc+weights["data"]*data
        return total,{"fv":float(fv.detach()),"ic":float(ic.detach()),"bc":float(bc.detach()),"data":float(data.detach()),"max_substeps":max(steps)}
