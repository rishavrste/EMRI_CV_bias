# EMRI Cutler–Vallisneri bias studies

This folder holds the EMRI runs that fit an **imperfect 0PA (adiabatic) template** — optionally
with a flux **deviation** — to a **1PA injected signal**, and locate the resulting maximum-likelihood
point (0PA "MAP"). The offset of that point from the injection is the systematic (Cutler–Vallisneri)
bias; a nonzero best-fit deviation says the extra freedom absorbed some of the 0PA↔1PA mismatch.

## Method (all scripts share it)

**LM-damped Cutler–Vallisneri iteration** — the CV bias formula `Δθ = Γ⁻¹⟨∂h|s−h⟩` is one
Gauss–Newton step; we iterate it with Levenberg–Marquardt damping (Nielsen gain-ratio λ update),
re-computing the StableEMRIFisher derivatives at each point, until it converges on the 0PA MAP.
Because it is a *local* climb, each run is `CV → (if it stalls below the overlap target) one
1000-step Nelder–Mead escape → CV again`. Common controls: `OVERLAP_TARGET=0.9999999999`,
`NM_MAXITER=1000`, `F_MIN=1e-5`, high-pass clip before the PSD, `deriv_type="stable"`.

## Conventions

- **Signal** = 1PA injection (`evolve_1PA=True`, deviation off). **Template** = 0PA (`evolve_1PA=False`),
  with the deviation on for the deviation models. Overlap is the noise-weighted normalized match;
  `chi2 = ⟨s−h|s−h⟩` (my-pipeline value, `≈ 2·SNR²·(1−overlap)`).
- **Params inferred**: `[m1,m2,a,p0,e0,qS,phiS,Phi_phi0,Phi_r0]` (9, for 0PA) plus
  `[dev_1,dev_2]` (11, for a deviation). Secondary spin `chi2` is a 1PA effect, not inferred.
- **Deviation wiring depends on the SuperKludge_r branch** (verify before running):
  - **hybrid**: `additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p(5), C_e(6), del_0_p(7), del_0_e(8)]`
    - **PN**: `C_p`=dev_1, `C_e`=dev_2 — additive 2.5PN `pdot/edot`.
    - **simple**: `del_0_p`=dev_1, `del_0_e`=dev_2 — multiplicative `Edot/Ldot` flux rescale.
  - **dev_a_pe**: `additional_args = [..., deviation_included, del_0_p(5), del_0_e(6)]`
    - **simple_pe**: `del_0_p`=dev_1, `del_0_e`=dev_2 — multiplicative `pdot/edot` rescale (≡ hybrid-simple via the Jacobian).
- **Start points** (the CV seed) used across scripts:
  - `from_injection` — the 1PA injected parameters. *Usually stalls* (0PA template is dephased there).
  - `from_MAP` — the 0PA-vs-2PA recovered best fit (from the critical-SNR grid). Re-phasing baked in → reliable seed.
  - `from_0PA` / `from_0PAcv` — our own CV-optimized 0PA best fit (+dev=0). Cleanest deviation seed
    (starts *at* the 0PA optimum, so the result is guaranteed ≥ 0PA).
  - `from_NMdev` — an earlier Nelder-Mead / differential-evolution deviation best fit.
- **Output**: each script writes `results_*.json` (per case: `ov_start`, `ov_final`, `chi2`, optimized
  `params`, and the start) and a matching `*.log`. Runs on GPU via `../batch.sh` (Apptainer + venv).

---

## Scripts

### Diverse grid points (spin & eccentricity spread) — **hybrid branch**
| file | points (a, e0) | models | starts | output |
|------|----------------|--------|--------|--------|
| `gauss_cv_emri_grid_diverse.py` | idx0 (−0.9, 0.1), idx6 (−0.5, 0.2), idx12 (0.0, 0.3), idx18 (0.5, 0.4) | 0PA, PN, simple | from_injection **and** from_MAP | `results_emri_grid_diverse.json` |
| `gauss_cv_emri_grid_diverse_dev_from_0pa.py` | same 4 points | PN, simple | from the 0PA best fit (from_MAP 0PA) | `results_emri_grid_diverse_dev_from_0pa.json` |
| `gauss_cv_emri_grid_diverse_simple_pe.py` (**dev_a_pe branch**) | same 4 points | simple_pe | from_MAP **and** from_0PAcv | `results_emri_grid_diverse_simple_pe.json` |
| `gauss_cv_emri_grid_diverse2.py` | idx3 (0.5,0.1), idx5 (−0.9,0.2), idx11 (−0.5,0.3), idx17 (0.0,0.4), idx20 (−0.9,0.5), idx24 (0.9,0.5) | 0PA, PN, simple | from_injection **and** from_MAP | `results_emri_grid_diverse2.json` |
| `gauss_cv_emri_grid_diverse2_dev_from_0pa.py` | same 6 points | PN, simple | from the 0PA best fit (from_MAP 0PA) | `results_emri_grid_diverse2_dev_from_0pa.json` |

`gauss_cv_emri_grid_diverse.py`: 24 CV climbs (4 points × 3 models × 2 starts) — the broad spin/ecc
sweep. Result: injection-start stalls everywhere; from_MAP gives the 0PA MAP; deviations help only
idx18 (highest e0), where **PN → 0.99999**. `..._dev_from_0pa.py` re-runs the two deviations from
the 0PA best fit (guaranteed ≥ 0PA) for clean, artifact-free deviation numbers on all 4 points.

### Least-biased grid points idx4 (a=+0.9, e0=0.1) & idx14 (a=+0.9, e0=0.3) — **hybrid branch**
| file | model | start | output |
|------|-------|-------|--------|
| `gauss_cv_emri_grid_0pa.py` | 0PA (9-param) | from_MAP | `results_emri_grid_0pa.json` |
| `gauss_cv_emri_grid_pn_dev.py` | 0PA + PN | from_MAP | `results_emri_grid_PN.json` |
| `gauss_cv_emri_grid_simple_dev.py` | 0PA + simple | from_MAP | `results_emri_grid_simple.json` |
| `gauss_cv_emri_grid_dev_from_0pa.py` | idx4→PN, idx14→simple | from 0PA best fit (idx14 currently high-SNR/`dist=1.0` from an earlier `x_bf`) | `results_emri_grid_dev_from_0pa.json` |

`gauss_cv_emri_grid_dev_from_0pa.py` exists to remove the "deviation worse than 0PA" numerical
artifact by seeding at the 0PA optimum (guarantees ≥ 0PA). Result: idx4 PN reaches 0.999975
(`dev_1≈−36.8`); idx14 simple confirmed = 0PA (no benefit).

### idx4 & idx14 pe-deviation — **dev_a_pe branch**
| file | model | starts | output |
|------|-------|--------|--------|
| `gauss_cv_emri_grid_simple_pe.py` | 0PA + simple_pe | from_MAP **and** from_0PAcv | `results_emri_grid_simple_pe.json` |

Confirmed `simple_pe ≡ hybrid-simple` (same reachable overlap, Jacobian-related coefficients);
no benefit for idx4/idx14 (pe is degenerate with the intrinsic params there).

### Earlier EMRI test set (idx0 / idx9 / idx13; see each file's `CASES` for exact a, e0, dt, T)
| file | model | starts | branch | output |
|------|-------|--------|--------|--------|
| `gauss_cv_emri_pn_dev.py` | 0PA + PN | from_0PA **and** from_NMdev | hybrid | `results_emri_PN.json` |
| `gauss_cv_emri_simple_dev.py` | 0PA + simple | from_0PA **and** from_NMdev | hybrid | `results_emri_simple.json` |
| `gauss_cv_pn_dev.py` | 0PA + PN | single (0PA best fit) → NM(500) → CV | hybrid | — |
| `gauss_cv_simple_dev.py` | 0PA + simple | single (0PA best fit) → NM(500) → CV | hybrid | — |
| `gauss_cv_pe_dev.py` | 0PA + simple_pe | single (0PA best fit) → NM(500) → CV | dev_a_pe | — |

The `gauss_cv_pn_dev.py` / `gauss_cv_simple_dev.py` / `gauss_cv_pe_dev.py` trio are the earlier
single-start versions, superseded by the two-start `gauss_cv_emri_{pn,simple}_dev.py`.

---

## Running

Runs go through `../batch.sh` (PBS → Apptainer → venv). Point its `python …` line at the desired
script(s) and `qsub batch.sh`. **Set the SuperKludge_r branch first** to match the model
(`hybrid` for PN/simple, `dev_a_pe` for simple_pe) — the deviation slot indices differ between them.

## Key findings so far

- **a=+0.9 EMRIs (idx4, idx9, idx14)**: 0PA already fits well; **PN is the only helpful deviation**
  (`dev_1 ≈ −36.7…−36.9` in all three — a real signature). simple / simple_pe collapse to 0PA.
- **Seeding matters**: seed deviation runs from the 0PA best fit (guaranteed ≥ 0PA); the 1PA
  injection is a poor seed (dephased) and the 2PA MAP is a good one.
- Diverse points (idx0/6/12/18) are more biased (retrograde / eccentric), so a deviation is more
  likely to matter there — that's what `gauss_cv_emri_grid_diverse.py` probes.
