# IMRI Cutler–Vallisneri bias studies

Same method as the EMRI folder (see `../EMRI/README.md` for the full method write-up), applied
to IMRI systems. A 1PA signal is fitted with an imperfect 0PA template (optionally + a flux
deviation); LM-damped Cutler–Vallisneri iteration (`CV → if it stalls, 1000-step Nelder–Mead →
CV`) finds the 0PA maximum-likelihood point. Common controls: `OVERLAP_TARGET=0.9999999999`,
`NM_MAXITER=1000`, `F_MIN=1e-5`, high-pass clip before the PSD, `deriv_type="stable"`, 3 TDI
channels, no noise.

## IMRI-specific conventions
- **Mass ratio** ~1e-3 (m1 = 1e6, m2 = 1e3 for the grid; the tails use m2 = 1e4).
- **Params inferred**: `[m1,m2,a,p0,e0,qS,phiS,Phi_phi0,Phi_r0]` (9, 0PA) plus `[dev_1,dev_2]`
  (11, deviation). Secondary spin `chi2 = 0.95` throughout (not inferred).
- **Deviation wiring depends on the SuperKludge_r branch** — verify before running:
  - **hybrid**: PN = `C_p`(5)/`C_e`(6) (additive 2.5PN pdot/edot); simple = `del_0_p`(7)/`del_0_e`(8)
    (multiplicative Edot/Ldot).
  - **dev_a_pe**: simple_pe = `del_0_p`(5)/`del_0_e`(6) (multiplicative pdot/edot; ≡ hybrid-simple).
- **Start points**: `from_injection` (1PA injected params, usually stalls — 0PA dephased);
  `from_MAP` (0PA-vs-2PA recovered best fit); `from_0PA`/`from_0PAcv` (our CV 0PA best fit);
  `from_NMdev` (an NM/DE deviation best fit).
- **Output**: `results_imri_*.json` + `imri_*.log`; run on GPU via `../batch.sh`.

---

## Scripts

### dt=5 re-run (aliasing fix) — **hybrid branch**
At dt=10 the Nyquist frequency is 0.05 Hz; prograde IMRIs carry power above it, which aliases
into the null T channel. `snr_channels_imri.py` (dt=10) found T carrying **29–97%** of the SNR at
every grid point; `snr_overlap_dt5_imri.py` re-ran the same setup at dt=5 and T% collapsed to
**~0.00 everywhere**, with all grid SNRs falling back to 20.5–32.8. Aliasing confirmed — so the
dt=10 best fits were optimised against a partly-fake signal.

| file | purpose | dt | output |
|------|---------|----|--------|
| `snr_channels_imri.py` | per-channel SNR (A/E/T) for all 28 cases — the diagnostic | 10 | `snr_channels_imri.json` |
| `snr_overlap_dt5_imri.py` | same, plus overlap/chi2 of the dt=10 best fits re-evaluated at dt=5 | 5 | `snr_overlap_dt5_imri.json` |
| `gauss_cv_imri_dt5_rerun.py` | **full CV re-climb at dt=5**, seeded from the dt=10 best fits | 5 | `results_imri_dt5_rerun.json` |

`gauss_cv_imri_dt5_rerun.py` covers the 12 grid points that have dt=10 best fits, ordered by
dt=10 T-contamination (worst first, so a walltime cut still returns the cases that matter):
idx14, 4, 24, 18, 8, 12, 6, 16, 22, 10, 0, 20. pt4 / pt20 / adhoc_A are excluded (T% already 0.00).

Seeds — 0PA gets one climb, each deviation gets **two** and the better result is kept:
| model | seeds climbed |
|-------|---------------|
| 0PA | `from_dt10_0PA` |
| PN | `from_dt10_PN` **and** `from_0PA_dt5` (this run's dt=5 0PA + dev=0) |
| simple | `from_dt10_simple` **and** `from_0PA_dt5` |

Both are run because the better seed is point-dependent (established on the EMRI grid), and the
`from_0PA_dt5` seed guarantees a deviation can never end below 0PA — the trap that put idx14's
dt=10 PN under its own 0PA. => 12 × (1 + 2 + 2) = **60 climbs**.

60 dt=5 climbs is ~90 h, so split it across two 72 h jobs with `DT5_POINTS` / `DT5_TAG`:
`../batch_dt5_a.sh` (idx14, 4, 24, 18, 8, 12 → `results_imri_dt5_rerun_a.json`) and
`../batch_dt5_b.sh` (idx6, 16, 22, 10, 0, 20 → `..._b.json`), ~45 h each. Running
`../batch.sh` instead does all 12 in one job (worst-contaminated first). Results save
incrementally, and each point's JSON carries a `best` block naming the winning seed per model.

### a × e0 grid (fresh — m1=1e6, m2=1e3, dt=10, T=1.0) — **hybrid branch**
| file | points (a, e0) | models | starts | output |
|------|----------------|--------|--------|--------|
| `gauss_cv_imri_grid_diverse.py` | idx0 (−0.9, 0.1), idx6 (−0.5, 0.2), idx12 (0.0, 0.3), idx18 (0.5, 0.4), idx20 (−0.9, 0.5), idx24 (0.9, 0.5) | 0PA, PN, simple | from_injection **and** from_MAP | `results_imri_grid_diverse.json` |

| `gauss_cv_imri_grid_diverse_dev_from_0pa.py` | same 6 points | PN, simple | from the 0PA best fit (from_MAP 0PA) | `results_imri_grid_diverse_dev_from_0pa.json` |
| `gauss_cv_imri_grid_diverse2.py` | idx4 (0.9,0.1), idx8 (0.5,0.2), idx10 (−0.9,0.3), idx14 (0.9,0.3), idx16 (−0.5,0.4), idx22 (0.0,0.5) | 0PA, PN, simple | **from_MAP only** | `results_imri_grid_diverse2.json` |

`grid_diverse`: 36 CV climbs (6 points × 3 models × 2 starts) spanning all 5 spins and all 5
eccentricities — the IMRI analogue of `../EMRI/gauss_cv_emri_grid_diverse.py`. Result: PN helps
at **every** point (0PA 0.9977–0.9992 → PN 0.9984–0.99997). `..._dev_from_0pa.py` re-runs the two
deviations from the 0PA best fit (guaranteed ≥ 0PA) for clean best-of-both numbers.

### Ad-hoc IMRI set idx0 / idx1 / idx16 (m2 = 5e3 / 1e3 / 1e3, dt=5, T=1.0)
| file | model | starts | branch | output |
|------|-------|--------|--------|--------|
| `gauss_cv_imri_0pa.py` | 0PA | from DE/NM 0PA best fit → NM → CV | (branch-agnostic) | `results_imri_0pa.json`* |
| `gauss_cv_imri_pn_dev.py` | 0PA + PN | from_0PA **and** from_NMdev | hybrid | `results_imri_PN.json` |
| `gauss_cv_imri_simple_dev.py` | 0PA + simple | from_0PA **and** from_NMdev | hybrid | `results_imri_simple.json` |
| `gauss_cv_imri_simple_pe.py` | 0PA + simple_pe | from the 0PA best fit | dev_a_pe | `results_imri_simple_pe.json` |

### IMRI-tails pt4 (a=+0.9, e0=0.1) & pt20 (a=−0.9, e0=0.5) (m2 = 1e4, dt=10, T=0.25)
| file | model | starts | branch | output |
|------|-------|--------|--------|--------|
| `gauss_cv_imri_tails_0pa.py` | 0PA | from DE/NM 0PA best fit | (branch-agnostic) | `results_imri_tails_0pa.json` |
| `gauss_cv_imri_tails_pn_dev.py` | 0PA + PN | from_0PA **and** from_NMdev | hybrid | `results_imri_tails_PN.json` |
| `gauss_cv_imri_tails_simple_dev.py` | 0PA + simple | from_0PA **and** from_NMdev | hybrid | `results_imri_tails_simple.json` |
| `gauss_cv_imri_tails_simple_pe.py` | 0PA + simple_pe | from_MAP **and** from_0PAcv | dev_a_pe | `results_imri_tails_simple_pe.json` |

\* the `gauss_cv_imri_0pa.py` log is `imri0pa.log`; results are printed there.

---

## Running
Point `../batch.sh`'s `python …` line at the desired script(s), set the SuperKludge_r branch to
match the model (`hybrid` for PN/simple, `dev_a_pe` for simple_pe), and `qsub batch.sh`.

## Lesson carried over from EMRI
For deviation runs, **seeding is point-dependent**: neither the 0PA-vs-2PA MAP nor the CV 0PA
best-fit universally finds the deviation optimum — run both starts and take the best. The
`grid_diverse` script runs `from_injection` + `from_MAP`; a follow-up `..._dev_from_0pa`-style
run (as in EMRI) can add the 0PA-best-fit seed for the deviations if needed.
