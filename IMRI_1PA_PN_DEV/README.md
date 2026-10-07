# IMRI_1PA_PN_DEV: 1PA signal vs 0PA and 0PA + PN deviation, 2nd-generation TDI

This folder redoes the IMRI 0PA-vs-1PA fits of `src/IMRI` (1st-generation TDI, A, E, T, dt 5) in the
1PA-vs-2PA setup: 2nd-generation TDI, A and E only, the SK_files `overlap.py` `rishav` setup, with the
DC bin dropped. Only the PN deviation is fitted. The code is EMRI_1PA_PN_DEV's, adapted.

## Setup

| item | value |
|---|---|
| system | SK_files IMRI grid: m1 1e6, m2 1e3, T 1 yr, dt 10, chi2 0.95, signal SNR 20 (the old fits injected the same rows; `make_seeds.py` checks it) |
| dt | 10 s, as in SK_files (user, 2026-10-05). The grid aliases at dt 10 (AGENTS.md section 3); dt 5 puts the 2nd-generation TDI PSD null at ~0.06 Hz in band |
| signal | 1PA, flags (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation |
| templates | `0pa`: (False, False, False); `pn`: the same plus deviation_included, C_p and C_e free |
| free | m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0 (+ C_p, C_e), as in the old fits |
| held | distance, qK, phiK, Phi_theta0, chi2 at the injected values |
| SuperKludge_r | `hybrid`: additional_args = [chi2, 1PA, primary, 2PA, deviation_included, C_p, C_e, del_0_p, del_0_e] |
| seeds | `old`: the old best fit of the same template (`seeds_old_1stgen.json`). `sk`: the SK_files 0PA LM best fit (`LM_iterations/best_fit_IMRI_0pa.json`, fitted to a 2PA signal) with qS, phiS at the injection; for pn also C_p = C_e = 0 (`seeds_sk.json`) |
| fit | T-ladder LM at 0.9, 0.95, 1 yr, each rung from the previous rung's final point; every point and template from both seeds |

## Files

| file | role |
|---|---|
| `config.py` | case, flags, deviation slots, seeds, evaluation stack, LM controls |
| `model.py`, `lm.py` | signal, template, inner product, SEF derivatives; damped LM (unchanged from EMRI_1PA_PN_DEV) |
| `make_seeds.py` | both seed files; run outside the container, which only binds this folder |
| `score_seeds.py` | overlap of a seed at full T, no fitting -> `results/seed_overlap/{seed}_idx{i}.json`, `{seed}.md` |
| `run_lm_T_ladder.py`, `batch_lm_T_ladder.sh` | scores the seeds, then the ladder -> `results/{case}/lm_T{T}_idx{i}.json`, `lm_idx{i}.json` |
| `make_results_table.py` | copy-ready table of the full-T fits, the best seed per template -> `results/lm_results.md` |
| `fisher/best_fits.py` | the full-T best fit per template and point: the highest overlap over every run in `results/` (all ladders, both seeds, the direct T = 1 climb; user, 2026-10-07) -> `results/best_fits.json` |
| `fisher/fisher_at_best.py`, `fisher/batch_fisher_at_best.sh` | Fisher at both best fits, SNR 20, T 1 yr, at-zero spin steps at a = 0 (as EMRI_1PA_PN_DEV) -> `results/fisher_at_best/{0pa,pn}/fisher_idx{i}.json` |
| `fisher/compare_bias.py` | 1D and nD bias tables -> `results/fisher_at_best/bias_comparison_snr{S}.txt` |
| `plot_codes/` | the EMRI_1PA_PN_DEV plots without the flag marks (user, 2026-10-07): fit table, r-factor, trend check, 1D/nD bias tables, violins, critical SNR -> `plots/` |

## Runs

| job | what | status |
|---|---|---|
| 648760-648764 | seed old, rows idx0-4, 5-9, 10-14, 15-19, 20-24, 0PA then PN | done 2026-10-06, exit 0, 4h05m-6h03m. The full-T rung ends below the seed's own full-T overlap at idx9, 14, 19, 24 (0PA) and idx14, 19 (PN): the T 0.95 fit drops to O 0.35-0.55 at T 1 |
| 648765-648769 | seed sk, the same rows | done 2026-10-06, exit 0. Final overlaps agree with seed old to <3e-8 wherever both fits converged |
| 649428, 649429 | seed old, sk: the a=+0.9 cells idx4, 9, 14, 19, 24 on the finer ladder 0.95-0.975-1, 0PA then PN -> `results/IMRI_{template}_rishav_{seed}_Tfrac0.95-0.975-1/` | done 2026-10-06, exit 0, 4h58m / 4h30m. Fixes 0PA idx9 (0.99915) and idx14 (0.99861), and PN idx14 (0.99961) and idx19 (0.99891), all above their seeds' overlap at T=1. idx4 and idx9 PN reproduce the 0.9-0.95-1 fits to 1e-9. 0PA idx19 (0.789) and idx24 (0.746) still fail: rho_s doubles from ~10 to ~20 between T=0.975 and T=1 (the plunge), so that rung is not a small step, and LM settles with rho_h ~18 |
| 651070 | seed old, 0PA idx19, 24: direct LM climb at T = 1 yr (T_FRACS=1) -> `results/IMRI_0pa_rishav_old_Tfrac1/` | done 2026-10-06, exit 0, 19m. Fixes both: idx19 0.9968471044, idx24 0.9967932236 (seed 0.99654, 0.99066 at T=1) |
| 651071 | seed old, 0PA idx19, 24: ladder inside the plunge stretch 0.975-0.99-1 -> `results/IMRI_0pa_rishav_old_Tfrac0.975-0.99-1/` | done 2026-10-07, exit 0, 38m: idx19 0.9968471121, idx24 0.9967933096, the same fits as 651070 (1e-8, 1e-7) |
| 651272 | Fisher at both best fits (`results/best_fits.json`), all 25 points | done 2026-10-07: exit 0, 42 min, 50 Fishers; every overlap reproduced exactly, rho_s 19.80-20.62 |

Wiring check (CPU, idx4, T 0.02 yr): the 0PA waveform with the deviation on and C = 0 is identical to the
deviation-off waveform, and C_p = 8.4 changes it.
