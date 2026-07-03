# Literature review: neural solvers vs. classical schemes for 2D dam-break SWEs

*Compiled 2026-07-03. All references verified against publisher pages / arXiv / Crossref;
BibTeX in `paper/references.bib`. Citation keys below match the bib file.*

## 1. Why this review

The paper compares three classical schemes (LW, HLL, MUSCL–Rusanov) and three neural
formulations (primitive PINN, conservative PINN, FVM-informed PINN) across six dam-break
initial-condition geometries over a Gaussian-hump bed. Several recent papers occupy nearby
territory, so the related-work section must position precisely: what has been compared
before, what was found, and what remains open.

## 2. Closest prior work (threats to novelty — read these first)

| Paper | What they did | What they found | How we differ |
|---|---|---|---|
| **Mumtaz et al. 2025, PLOS ONE** (`mumtaz2025dambreak`) | PINNs vs Lax–Wendroff, 1D+2D dam breaks, varying dam geometries and initial water-height profiles | PINNs reproduce baselines with limited accuracy; smoother/more diffused fronts; improve when numerical solutions are injected into training | No topography source term treatment like ours; single scheme baseline (LW only); no conservative-form PINN; no discrete FVM residual. We cross 3 scheme families × 6 ICs × 3 neural formulations over non-flat bed. |
| **Tian et al. 2025, WRR 61:e2025WR040052** (`tian2025pinnswe`) | PINNs for 2D SWEs with topography + rainfall source terms vs HLL, incl. circular dam break | Errors concentrate at shock/rarefaction fronts; conservative-form PINN beats first-order HLL on coarse grids; their entropy-stability-conditioned variant **failed** on the dam-break case | They vary source terms, not IC geometry; single-scheme baseline. Their conservative-form result motivates our primitive-vs-conservative axis; their entropy failure is a useful contrast for our FVM residual. |
| **Qi et al. 2024, J. Hydrology 636:131263** (`qi2024pinnswe`) | Data-free FCNN and CNN PINNs vs FV solver, incl. a real flood event | CNN-PINNs train faster/more accurately than FCNN; PINNs show better speed–accuracy trade-off than FV for depth on idealized cases | No systematic IC-geometry sweep; no FV-informed loss. |
| **Dazzi 2024, WRR 60:e2023WR036589** (`dazzi2024augmented`) | Augmented-system PINN: bed elevation as extra conserved-like variable, 1D dam break + non-flat terrain | Feasible topography handling inside a strong-form PINN | 1D; the augmented-system trick is an alternative to our source-term handling — cite as the reference approach. |
| **Liu 2026, arXiv:2605.11001** (`liu2026fvmpinn`) | "Data-Guided FVM-PINN": differentiable well-balanced Roe FV loss on unstructured meshes | **Physics-only training collapses to a trivial low-momentum state** (loss only ~7× above truth); sparse data guidance (e.g. 200 velocity points) creates ~310× separation and ~22× error reduction | This is the method claim closest to our FVM–PINN. Ours differs: HLLC/SSP-RK discrete-consistency residual through a full solver *step* (not a Roe flux-balance residual), softplus positivity reparameterization around the IC depth, stochastic single-interval sampling, structured grid. We must (a) cite prominently, (b) test whether the zero-momentum collapse reproduces in our setting → the data-guidance ablation is now a required experiment. |

**Bottom line on novelty:** the naive framing ("compare PINNs with classical schemes on
dam breaks") is taken by Mumtaz 2025. The defensible framing is the *systematic crossing*:
IC geometry (6) × classical scheme family (3) × neural formulation (primitive/conservative
strong-form + discrete FVM residual), all with a topography source term, with conservation
and front-position metrics — no prior paper does that matrix.

## 3. The method taxonomy (for Related Work structure)

**Strong-form PINNs** (`raissi2019physics`): pointwise residual via autograd. Known
failure modes on hyperbolic systems: residual ill-defined at shocks (`mao2020highspeed`),
no discrete conservation (mass leakage), spectral bias smearing high-frequency fronts
(`rahaman2019spectral`), rugged multi-term loss landscapes. Loss balancing for SWEs
specifically addressed by NDAWL-PINN (`qi2025ndawl`).

**Weak-form / entropy PINNs** (`oubarka2026wepinn`, arXiv:2603.24819): integral
weak-form residual + entropy admissibility constraints; well-defined across shocks;
selects the physical weak solution. Single-instance, implementation-heavy.

**FV-informed / hybrid neural solvers** — the family our FVM–PINN belongs to:
- `liu2026fvmpinn` — FV loss with differentiable Roe solver (see table above).
- `wei2025ffvpinn` — FFV-PINN, CMAME 444:118139 (2025): simplified FV discretization of
  convective terms + residual correction; order-of-magnitude training speedup.
- `patsatzis2025gorinns` — GoRINNs, JCP 534:114002 (2025): shallow NNs learn flux
  functions inside 2nd-order Godunov schemes; conservation by construction; SWE among
  test cases; mainly inverse/closure problems.
- `lichtle2025unfv` — (U)NFV, arXiv:2505.23702 (Berkeley): learns FV update rules over
  extended stencils, supervised or unsupervised (weak-form); up to 10× lower error than
  Godunov on first-order conservation laws. **Preprint; official title is "Supervised and
  Unsupervised…", not "(Un)Supervised".**
- `wu2025rimnet` — RimNet, SSRN preprint 5433524: neural Riemann-flux surrogate inside a
  Godunov FV scheme; conserves by construction; generalizes across mesh resolutions and
  unseen dam-break ICs; up to 29.66% runtime reduction. **Preprint only — no journal
  version found.**

**Neural operators** (learn IC→solution maps; generalize without retraining):
DeepONet (`lu2021deeponet`), PI-DeepONet (`wang2021pideeponet`), FNO (`li2021fourier`),
the general framework (`kovachki2023neuraloperator`), PINO (`li2024pino`). SWE/flood
applications: DeepONet hydraulics comparing data-driven vs physics-informed training
(`liu2026hydraulics`, arXiv:2601.08086 — PI variants generalize better out-of-distribution),
PINO 2D SWE flood surrogate incl. radial dam break with ~4 orders of magnitude speedup
(`keppler2025pinoswe`, ESS Open Archive preprint), FNO debris-flow surrogate trained on
FV data (`secchi2026debrisflow`, Land 2026). Known limitation: Fourier mode truncation
acts as a low-pass filter that smears shock fronts over long rollouts; remedies are
local–global operators (`wang2026lgno`, arXiv:2606.18221 — beats WENO-Z-at-higher-resolution
on dissipation in long rollouts), dual-channel wavelet operators (`lu2026dcmawno`,
IJNMHFF 36(5), 2026), and latent-space operator learning (`kontolati2024latent`,
Nat. Commun. 15:5101). We cite this family as future work, not competitors — our study is
single-instance.

## 4. Classical numerics and benchmarks (settled references)

- Schemes: Godunov (`godunov1959difference`), Roe (`roe1981approximate`), HLL
  (`harten1983upstream`), HLLC (`toro1994hllc`), Rusanov (`rusanov1961calculation` — note
  the correct pagination: Russian original 1(2):267–279 (1961), English translation
  1(2):304–320 (1962)), MUSCL (`vanleer1979towards`), Lax–Wendroff (`lax1960systems`),
  SSP-RK (`gottlieb2001strong`), WENO-Z (`borges2008improved`). Textbooks: Toro
  (`toro2001shockcapturing`, `toro2009riemann`); modern FV-SWE survey: Kurganov
  (`kurganov2018finite`).
- Source terms / well-balancing: C-property (`bermudez1994upwind`), hydrostatic
  reconstruction (`audusse2004fast`), positivity-preserving wet/dry (`xing2010positivity`).
- Benchmarks: Stoker analytic dam break (`stoker1957water`), SWASHES analytic library
  (`delestre2013swashes`), Fennema–Chaudhry partial breach (`fennema1990explicit`),
  three humps (`kawahara1986finite`), isolated-obstacle experiment
  (`soaresfrazao2007experimental`).

## 5. Experiment implications from the review

1. **Data-guidance ablation is mandatory** — Liu 2026 makes physics-only FV-loss collapse a
   documented pathology; we must show whether it occurs with our step-consistency residual
   and softplus reparameterization (which may itself prevent the collapse — that would be a
   genuine finding worth headline billing).
2. **Conservative vs primitive PINN comparison has a hook** — Tian 2025 found the
   conservative form beats coarse HLL; our six-geometry sweep tests whether that holds
   across IC complexity.
3. **Report error localization** (shock/rarefaction front maps), mass drift, and front
   position — the metrics prior work used, so results are directly comparable.
4. **Cite preprints as preprints** — RimNet (SSRN), (U)NFV, WE-PINN, Liu 2026, LGNO,
   keppler2025pinoswe are all unrefereed as of 2026-07-03; flagged in the bib.
