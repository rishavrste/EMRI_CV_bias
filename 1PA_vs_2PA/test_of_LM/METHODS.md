# test_of_LM: LM climbs from the injection (IMRI_TAIL, 1PA vs 2PA)

These are the methods tried here for fitting a 1PA template to a 2PA signal on the IMRI_TAIL grid, and what each one gave.
Every run starts at the injected parameters.

## Setup

| item | value |
|---|---|
| system | IMRI_TAIL: m1 1e6, m2 1e4, T 0.25 yr, dt 10, SNR 20 at the stored distance |
| signal | 2PA, flags (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, True), chi2 = 0.95 |
| template | 1PA, (True, False, False); `--template 0pa` is available but not used in this study |
| held | distance, sky angles (qS, phiS, qK, phiK), Phi_theta0, always at the injected values |
| evaluation | SK_files `overlap.py` `doc` setup (`FIT_GEN`: err 1e-11, mode threshold 1e-5, AE), so overlaps compare directly with SK_files |
| DC bin | every result below was made with the DC bin included; `config.DROP_DC` has been `True` since 2026-09-30 (never fit on DC), so a re-run now drops it |
| resid | <s-h\|s-h>, quoted at the stored distance (<s\|s> = 400) |

The stored mle and overlap rows have the same held columns as the injection (max difference 0).
So a fitted vector can be dropped onto the injected row without shifting anything.

## Files

| file | role |
|---|---|
| `config.py` | case, flags, evaluation stack, LM controls, output folder names |
| `model.py` | signal, template, inner product and SEF derivatives at one grid point |
| `lm.py` | the damped LM climb (plain and `--regularise`) |
| `run_lm_from_inj.py` | method 1: LM from the injection; `run_point` is also used by method 3 |
| `run_lm_short_T.py` | method 3: two-stage shorter-T seed |
| `run_lm_T_ladder.py` | method 4: observation-time ladder, any grid and template |
| `batch_lm_from_inj.sh`, `batch_lm_short_T.sh`, `batch_lm_T_ladder.sh` | PBS wrappers for methods 1, 3 and 4 |
| `results/{case}/lm_idx{i}.json` | one climb per point, with its full history |
| `logs/lm_{case}_idx{i}.log` | the matching stdout of each climb |

`__pycache__/` and the PBS `.out`/`.err` files (resource summaries only) were removed.
The runs those PBS files flagged as killed (exit 271) are the ones marked "unfinished" or "killed" in the table below.

## The optimiser (lm.py)

- The optimiser is damped Levenberg–Marquardt, i.e. iterated Cutler–Vallisneri.
- Each step solves (G + lam diag G) delta = g, with G_ij = <d_i h|d_j h> and g_i = <d_i h|s-h>.
- lam is updated with Nielsen's gain-ratio rule.
- Derivatives come from SEF: 6th-order stencil, NDELTA 12, stable deltas re-scanned every 3 iterations.
- A proposed step is pulled back into the physical box: a within ±0.999, e0 in [0, 0.99], chi2 in [0, 0.999].

**Stop conditions:**

| stop_reason | meaning |
|---|---|
| overlap_target | overlap ≥ 0.9999999999 |
| rel_tol | relative drop in resid < 1e-9 |
| lambda_exhausted | no accepted step within 30 inner tries, or lam > 1e8 |
| max_iters | 1000 iterations |

The result JSON is rewritten after every accepted step, so a killed job still leaves its latest point.

## Methods

### 1. Direct LM from the injection (run_lm_from_inj.py)

```
python run_lm_from_inj.py --idx 6 9 [--fix-chi2] [--fix-phases] [--dist-div D] [--regularise]
```

**Options:**
- `--fix-chi2` holds chi2 at 0.95, leaving 7 free parameters.
- `--fix-phases` also holds Phi_phi0 and Phi_r0, leaving 5.
- `--dist-div D` divides the distance by D.
  - This multiplies the SNR by D and the resid by D².
  - It leaves the overlap unchanged, and so leaves the LM path unchanged apart from the tolerances.

**Output:** `results/IMRI_TAIL_1pa[_fixchi2][_fixphases][_distdiv{D}][_reg]/lm_idx{i}.json`.

### 2. Regularised inverse (--regularise)

- The damped matrix A = G + lam diag G is inverted explicitly.
- The step is taken through Sigma_reg = A^-1 + eps diag(A^-1), with eps = \|\|A A^-1 - 1\|\|_F.
- **Verdict: not useful.** Tried on idx8, it follows the same path as plain LM with smaller steps (see results).

### 3. Two-stage shorter-T seed (run_lm_short_T.py) — the method that works

```
python run_lm_short_T.py --idx 9 --fix-chi2 --dist-div 10 [--t-short 0.2]
```

**Stages:**
1. The same source is observed for only 0.20 yr, and LM runs from the injection.
   - chi2 is held at 0.95 and dist_div is 10, so this stage runs at SNR ~200.
   - Output: `lm_T0.2_idx{i}.json`.
2. The full 0.25 yr, with LM seeded at the stage-1 final point.
   - Output: `lm_idx{i}.json`, in the same layout as method 1, so SK_files `make_best_fit.py` can merge it.

**Why it works:**
- The shorter signal accumulates less dephasing, so the injection is already inside the right basin: stage-1 start overlaps are 0.86–0.99.
- The stage-1 point is then close enough to the 0.25 yr optimum that stage 2 starts at O ≈ 0.9995.
- The direct 0.25 yr climb from the injection instead falls onto a degeneracy branch at O ≈ 0.5–0.85.

**Output:** `results/IMRI_TAIL_1pa_fixchi2_distdiv10_Tseed0.2/`.

**Batch script:** `batch_lm_short_T.sh` (method 1 uses `batch_lm_from_inj.sh`). Logs go to `logs/`.

### 4. Observation-time ladder (run_lm_T_ladder.py)

```
python run_lm_T_ladder.py --grid IMRI --template 0pa --idx 3 --start best --t-steps 0.75 1
python run_lm_T_ladder.py --grid IMRI --template 0pa --idx 3 --start inj --t-steps 0.2 0.4 0.6 0.8 1
```

**How it works:** method 3 generalised to any number of rungs.
- The same source is observed for each T in `--t-steps` in turn.
- Rung 1 starts at `--start`: the injection (`inj`) or the current SK_files best fit, `LM_iterations/best_fit_{GRID}_{pa}.json` (`best`).
- Every later rung starts at the final point of the rung before.
- `--grid` and `--template` select any SK_files grid (IMRI, IMRI_TAIL, EMRI) and 0PA or 1PA.

**Output:** `results/{GRID}_{pa}[_fixchi2][_distdiv{D}]_Tladder_{start}_{T1-T2-...}/`.
- Each rung writes `lm_T{T}_idx{i}.json`.
- The last rung, at the stored T, writes `lm_idx{i}.json`, the layout `make_best_fit.py` merges.

**Batch script:** `batch_lm_T_ladder.sh`. T_STEPS is passed hyphen-separated, because `qsub -v` splits on commas and spaces, e.g. `-v GRID=IMRI,TEMPLATE=0pa,IDX=3,START=best,T_STEPS=0.75-1`.

**Status here:** no finished run in this folder. The one test job (IMRI 0PA idx3, `--start best`) was killed after 1.5 min.
The production ladders are run from the extended copy in `SK_files/LM_iterations/code/t_ladder/`.
That copy adds `--start mle|overlap`, the `rishav` setup and drop_dc, and writes to `LM_iterations/climbs/`.

## Results

O_start is the overlap at the injection (for stage 2: at the stage-1 final). resid is at SNR 20.

| folder | idx | free | O_start | O_final | resid | stop | iters |
|---|---|---|---|---|---|---|---|
| IMRI_TAIL_1pa | 6 | 8 | -0.045534 | 0.718180 | — | unfinished | — |
| IMRI_TAIL_1pa | 8 | 8 | 0.117267 | 0.757628 | — | unfinished | — |
| IMRI_TAIL_1pa | 9 | 8 | -0.415400 | 0.314368 | — | unfinished | — |
| _fixchi2 | 6 | 7 | -0.045534 | **0.999999948336** | 4.66e-05 | rel_tol | 385 |
| _fixchi2 | 8 | 7 | 0.117267 | 0.845722602582 | 123.45 | rel_tol | 98 |
| _fixchi2 | 9 | 7 | -0.415400 | 0.489292 | 349.34 | lambda_exhausted | 165 |
| _fixchi2 | 14 | 7 | -0.266206 | 0.519492 | 359.40 | max_iters | 1000 |
| _fixchi2_distdiv0.5 | 9 | 7 | -0.415400 | 0.475695 | — | unfinished | — |
| _fixchi2_distdiv0.5 | 14 | 7 | -0.266206 | 0.499988 | — | unfinished | — |
| _fixchi2_distdiv1000 | 8 | 7 | 0.117267 | 0.845722617456 | 123.45 | rel_tol | 96 |
| _fixchi2_distdiv1000 | 9 | 7 | -0.415400 | 0.490594 | 348.60 | lambda_exhausted | 215 |
| _fixchi2_fixphases_distdiv1000 | 8 | 5 | 0.117267 | 0.829099 | 136.27 | lambda_exhausted | 92 |
| _fixchi2_reg | 8 | 7 | 0.117267 | 0.845566054010 | — | killed (SIGTERM, 1 h 14 m, it 359) | — |
| _Tseed0.2, stage 1 (T 0.20) | 3 | 7 | 0.858700 | 0.999999999893 | — | rel_tol | 23 |
| _Tseed0.2, stage 1 (T 0.20) | 9 | 7 | 0.985021 | 0.999999999938 | — | overlap_target | 17 |
| _Tseed0.2, stage 1 (T 0.20) | 14 | 7 | 0.980481 | 0.999999999913 | — | overlap_target | 18 |
| _Tseed0.2, stage 2 (T 0.25) | 3 | 7 | 0.999513 | **0.999999986894** | 1.08e-05 | rel_tol | 23 |
| _Tseed0.2, stage 2 (T 0.25) | 9 | 7 | 0.999553 | **0.999999991297** | 7.08e-06 | rel_tol | 20 |
| _Tseed0.2, stage 2 (T 0.25) | 14 | 7 | 0.999532 | **0.999999971680** | 2.31e-05 | rel_tol | 22 |

## What was learned

- **Direct LM from the injection works only at idx6.** Elsewhere the injection starts at O between -0.4 and 0.1, and the climb stalls at O between 0.49 and 0.85.
- **Distance does not change the outcome.** dist_div 1000 lands where dist_div 1 did: idx8 at 0.8457226 both times (a difference of 1.5e-8), and idx9 at 0.4906 vs 0.4893. dist_div 0.5 stalls at 0.48–0.50 as well. The stall is a basin problem, not a noise-scale or tolerance problem.
- **Freeing chi2 is slower and no better.** The 8-parameter direct runs never finished and sat below their chi2-fixed counterparts.
- **Holding the phases does not help.** It makes idx8 worse (0.829 vs 0.846).
- **The regularised inverse does not help.** It gives smaller steps along the same path: idx8 reached 0.8456 after 359 iterations, against 0.8457 in 98 for plain LM.
- **The two-stage shorter-T seed fixes every point it was tried on.** idx3, 9 and 14 reached O > 0.99999998 in about 5 minutes each.
  - The stage-1 segment carries only about 1% of the SNR², but that is enough to place the seed in the right basin.

## Merged into SK_files

`SK_files/LM_iterations/code/make_best_fit.py`'s `MERGE_TEST_OF_LM` takes the following points into `best_fit_IMRI_TAIL_1pa.json`:

| idx | source file |
|---|---|
| 6 | `_fixchi2/lm_idx6.json` |
| 3, 9, 14 | `_fixchi2_distdiv10_Tseed0.2/lm_idx{3,9,14}.json` |

Held parameters are filled in at their injected values.
The resid is rescaled to the stored distance, which is exact because the resid goes as 1/distance².
After these merges, every IMRI_TAIL 1PA point is above O = 0.99999.

## Follow-up (in SK_files, not here)

This follow-up does not run in test_of_LM:
- It starts from each IMRI_TAIL 1PA best-fit point, with all 8 parameters free.
- It allows at most 100 iterations, with chi2 in [0, 0.999].
- It runs as `lm_climb.py --start frombest --chi2-min 0`, and writes to `SK_files/LM_iterations/climbs/IMRI_TAIL_1pa/frombest/`.
