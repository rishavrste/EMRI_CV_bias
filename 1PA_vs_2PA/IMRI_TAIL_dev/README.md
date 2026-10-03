# IMRI_TAIL_dev: 2PA signal vs 1PA + deviation template, T-ladder LM

The aim is to see how much of the 1PA↔2PA mismatch on the IMRI_TAIL grid a free flux deviation absorbs, and how that changes the bias.

## Setup

| item | value |
|---|---|
| system | IMRI_TAIL: m1 1e6, m2 1e4, T 0.25 yr, dt 10, SK_files `signal_parameter_array_IMRI_TAIL.npy` |
| signal | 2PA, flags (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, True), chi2 = 0.95, no deviation |
| template | 1PA + deviation: (True, False, False), deviation_included = True |
| free | m1, m2, a, p0, e0, Phi_phi0, Phi_r0, C_p, C_e (9); chi2 held at 0.95 (`--fix-chi2`) |
| held | distance, sky angles, Phi_theta0 at the injected values |
| evaluation | SK_files `overlap.py` `rishav` setup (signal SNR exactly 20), DC bin dropped |
| fit | LM from the injection (C_p = C_e = 0): T = 0.2 yr, then 0.25 yr seeded at the 0.2 yr final; dist_div 10 (SNR 200) |

## The deviation (SuperKludge_r @ 2PA_PN_deviation, commit 407d3cc)

```
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e]
pdot += q^2 C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
edot += q^2 e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5        q = m1 m2 / (m1 + m2)^2
```

- The terms are added only inside the `evolve_1PA` block, so they act only with a 1PA template.
- The shape is Peters–Mathews × p^(-1/2), i.e. 0.5PN beyond the leading radiation reaction, at 2PA order (q²).
  The code comment says "0PN-2PA type"; the shape is used as coded (confirmed 2026-10-02).
- `run_lm_T_ladder.py` refuses to run unless SuperKludge_r is on `2PA_PN_deviation`.

**Wiring check (2026-10-02).** Trajectory at idx9 over 0.25 yr, CPU, final p:

| case | final p |
|---|---|
| 1PA, no deviation args | 3.3894299085 |
| 1PA, deviation on, C = 0 | 3.3894299085 (identical) |
| C_p = 10 | 3.8318002014 |
| C_e = 10 | 3.4559903568 |
| deviation off, C_p = 10 | 3.3894299085 (ignored) |
| 2PA signal | 3.3256120949 |
| 2PA signal with explicit deviation-off args | 3.3256120949 (identical) |

## Files

| file | role |
|---|---|
| `config.py` | case, flags, deviation slots, evaluation stack, LM controls |
| `model.py` | signal, template, inner product, SEF derivatives (C_p, C_e differentiated as `add_param_args`) |
| `lm.py` | damped LM climb (copied unchanged from SK_files t_ladder) |
| `run_lm_T_ladder.py` | the ladder; writes `results/{case}/lm_T0.2_idx{i}.json` and `lm_idx{i}.json`. `--seed` picks the start (`config.SEEDS`); `--then-free` adds a chi2-free climb |
| `batch_lm_T_ladder.sh` | PBS wrapper; logs in `logs/` |

```
qsub -N dev_Tl_idx9 -v IDX=9,T_STEPS=0.2-0.25,FIX_CHI2=1,DIST_DIV=10 \
     -o logs/pbs_idx9.out -e logs/pbs_idx9.err batch_lm_T_ladder.sh
```

## Runs

| job | points | status |
|---|---|---|
| 642606 | idx9 (a 0.9, e0 0.2) | test point: done |
| 642610 | idx0, 1, 2, 3, 4 | rest of the grid (2026-10-02) |
| 642611 | idx5, 6, 7, 8, 10 | rest of the grid |
| 642612 | idx11, 12, 13, 14, 15 | rest of the grid |
| 642613 | idx16, 17, 18, 19, 20 | rest of the grid |
| 642614 | idx21, 22, 23, 24 | rest of the grid |
| 643197 | idx0, 1, 2, 3, 4 | chi2 free, T 0.25, seeded at the fixed-chi2 result (`--seed devinj`, 2026-10-02) |
| 643198 | idx5, 6, 7, 8, 10 | same |
| 643199 | idx11, 12, 13, 14, 16 | same |
| 643200 | idx17, 18, 19 | same |
| 643201 | idx20, 21, 22, 23, 24 | same; held until 642613 and 642614 finish |
| 643202 | idx15 | fixed chi2 from the 1PA-only best fit, C = 0 (`--seed 1pabest`), then chi2 free (`--then-free`); hung at start-up (10 s CPU in 50 min), deleted |
| 643428 | idx2, 4, 7 | fixed chi2, from the injection straight at T 0.25, no ladder (`inj_0.25`); cancelled: idx2 0.1109, idx4 0.999999996306, idx7 ~0.111 |
| 643429 | idx12, 17, 24 | same; cancelled while idx12 ran away (C_p ~ -2.4e3) |
| 643430 | idx15, 20, 21 | same; cancelled before it started |
| 643492 | idx2, 4, 7 | chi2 free, from the 1PA-only best fit, C_p = C_e = 0, T 0.25 (`1pabest_0.25`) |
| 643493 | idx12, 17, 24 | same |
| 643494 | idx15, 20, 21 | same |

Result of 643492-643494: no point drops below its 1PA-only fit, but C_p and C_e stay at ~0 everywhere
except idx2 (gain 1.18, C_e = -1.04): the climb sits on the flat dev = 0 ridge. idx15, 20, 21 (a < 0)
gain nothing, although their row-mates gain 4-60x, so for them this is a stall, not an absence of signal.

| 643614 | idx2, 4, 7 | fixed chi2, from the 1PA-only best fit, C_p = C_e = 0, ladder T 0.242 -> 0.25 (`1pabest_0.242-0.25`) |
| 643615 | idx12, 17, 24 | same |
| 643616 | idx15, 20, 21 | same |

Result of 643614-643616: only idx15 works (gain 15.7, C_p = -16.8, in line with its a = -0.9 row).
idx20 leaves the ridge (C_p = -11.2) but gains 1.01x. idx2, 4, 7, 12, 24 end below 1PA-only; idx17 ties.
idx21 runs away (C_p ~ -1e5, overlap 0.51). With chi2 held at 0.95, idx4, 21, 24 start at overlap < 0 at
T 0.242 (their 1PA fits have chi2 0.943, 0.970, 0.943), so chi2 has to be free at those points.

| 644299 | idx2, 4, 7 | chi2 free, from the 1PA-only best fit, C_p = C_e = 0, ladder T 0.242 -> 0.25 (`1pabest_0.242-0.25`) |
| 644300 | idx12, 17, 24 | same |
| 644301 | idx20, 21 | same |
| 644342 | idx20, 21 | chi2 held at the 1PA-only best-fit value (`--chi2-at seed`), from the 1PA-only best fit, C = 0, ladder T 0.242 -> 0.25 (`fixchi2seed_..._1pabest_0.242-0.25`) |
| 644376 / 644377 | idx2, 4, 7 / idx12, 17 | same, ladder T 0.248 -> 0.25 (`fixchi2seed_..._1pabest_0.248-0.25`) |
| 644380 | idx21 | chi2 held at the 1PA value, from the 1PA-only best fit but C_p = -8, C_e = 0 (`--dev-start`), ladder T 0.242 -> 0.25 (`fixchi2seed_..._1pabest_Cp-8_Ce0_0.242-0.25`) |
| 644498 | idx20 | chi2 free, from the end of the chi2-held-at-1PA ladder 644342 (`--seed devchi2seed`), single rung T 0.25 (`..._devchi2seed_0.25`) |
| 644540 | all | Fisher at both best fits (1PA-only from SK_files, 1PA + dev from `best_fit_dev.py` -> `results/best_fit_IMRI_TAIL_1pa_dev.json`), SNR 20, T 0.25 (`fisher_at_best.py`, `results/fisher_at_best/{1pa,1pa_dev}/`) |
| 644820 | idx4 | 1PA + dev Fisher redone with SEF's at-zero step grid for C (its C ~ 4e-16 had given steps ~ 1e-17 and a noise C block); failed (list in place of array for `delta_range`) |
| 644822 | idx4 | the same, fixed; replaces the 644540 idx4 1pa_dev Fisher (C block 1.4e3 / 3.6, corr 0.991; D^2 2115 -> 0.0023) |

The fixed-chi2 climb of idx15 from the injection (642612) ran away (C_p ~ -1.9e5, overlap 0.64)
and crashed at the spin bound. idx2, 4, 7, 12 and 17 ended below their 1PA-only overlaps, which
the nested template rules out at a true optimum: these are stalls.

Chi2 free (643197-643201) changed almost nothing: chi2 stayed at 0.950 and the overlap moved only at
idx18. idx20 and idx21 ran away in the fixed-chi2 ladder like idx15. The nine points below their
1PA-only overlap or run away (idx2, 4, 7, 12, 15, 17, 20, 21, 24) are re-climbed at full T without
the ladder (643428-643430). That failed (idx2, 7 stuck at overlap 0.11; idx4 still below 1PA-only), so
they restart from the 1PA-only best fit with chi2 free (643492-643494): the start is then exactly the
1PA-only fit, which the nesting makes a floor.

Chi2 held at the 1PA value, ladder 0.248 -> 0.25 (644376/644377) improved nothing: gains at T = 0.25
are idx2 0.989, idx4 1.000 (stayed exactly at C = 0, rel_tol stop), idx7 0.871, idx12 1.008,
idx17 1.095. idx21 started at C_p = -8 (644380) went back to C_p = +1.32, gain 1.425, the same basin
as the C = 0 start (1.43).

idx20 with chi2 free from the chi2-held point (644498) took no step (1 iteration, none accepted,
lambda_exhausted): that point (chi2 0.950, C_p -12.58, overlap 0.999999707815) is a local maximum
with chi2 free too, distinct from the chi2-free ladder's (chi2 0.998, C_p -2.93, 0.999999719502).

Chosen best fit per point: the maximum T = 0.25 overlap over all runs, except idx4 (the chi2-free
run of the tied pair, 1pabest_0.25) and idx20 (point A above, chi2 held at the 1PA value, gain 3.81,
chosen over the 1.2e-8 higher chi2 = 0.998 fit). idx4 and idx7 have C = 0: they are the 1PA-only fit.

Bias (644540 Fishers, `compare_bias.py` -> `results/bias_comparison_snr20.txt`; plots and copy-ready
tables from `results/plot_codes/` -> `results/plot/`): 1D bias is below 0.2 sigma everywhere at SNR 20
(below 1.6 at SNR 200). The deviation inflates sigma(m2) 15-30x (C_p-m2 degeneracy), so its own-sigma
biases shrink, while its m2 bias in 1PA sigma grows (to -3.3 sigma at SNR 200, a < 0; -6.2 at idx22).
sigma(C_p) ~ 1e3 at SNR 20: the spurious C_p is invisible. The 1PA-only nD bias is large (D ~ 46 at SNR 20,
nD critical SNR O(1)): the IMRI_TAIL Fisher widths are very narrow even at mismatch ~1e-7. idx4's 644540 1PA + dev Fisher was wrong:
its C (~4e-16, round-off, not 0) made SEF's value-relative step search pick C steps ~1e-17, below
round-off, so the C block was noise (~1e12, rank 1) and marginalising C did nothing. `fisher_at_best.py`
now gives |C| < 1e-10 SEF's own at-zero grid (1e-4 .. 1e-9, as idx7 at C = 0 exactly), theta unchanged;
redone in 644822. Marginalising C means Gaussian marginalisation: invert the full 10 x 10 Fisher and
keep the physical block of the covariance (Schur complement); the bias vector is not marginalised.

Plots (`results/plot_codes/`, regenerate any one by running it): `results/plot/` holds fit/ (fit
table, r-factor maps, trend check), bias_1D/snr{20,200}/ (split violins of x_best - x_inj, left 1PA, right
1PA + dev, from the Fisher; 1D table; per_param/ one figure per parameter), bias_nD/{with,no}_phases/, critical_snr/{with,no}_phases/ (90 and 99% CL, convention of
Fisher_results/plot_bias.py) and tables/ (copy-ready markdown).
