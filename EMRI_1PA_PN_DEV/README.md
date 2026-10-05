# EMRI_1PA_PN_DEV: 1PA signal vs 0PA and 0PA + PN deviation, 2nd-generation TDI

This folder redoes the EMRI 0PA-vs-1PA fits of `src/EMRI` (1st-generation TDI, A, E, T) in the 1PA-vs-2PA setup:
2nd-generation TDI, A and E only, the SK_files `overlap.py` `rishav` setup, with the DC bin dropped. Only
the PN deviation is fitted; the simple deviation is dropped.

## Setup

| item | value |
|---|---|
| system | SK_files EMRI grid: m1 1e6, m2 10, T 2.5 yr, dt 10, chi2 0.95, signal SNR 20; plus `adhoc_A` from the old runs (a 0.9, e0 0.5, p0 7.5, T 1 yr, dt 5, chi2 0, dist 5, SNR ~44.7; `config.ADHOC`) |
| signal | 1PA, flags (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation |
| templates | `0pa`: (False, False, False); `pn`: the same plus deviation_included, C_p and C_e free |
| free | m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0 (+ C_p, C_e), as in the old fits |
| held | distance, qK, phiK, Phi_theta0, chi2 at the injected values |
| SuperKludge_r | `hybrid`: additional_args = [chi2, 1PA, primary, 2PA, deviation_included, C_p, C_e, del_0_p, del_0_e] |
| seeds | the old best fit of the same template (`make_seeds.py` -> `seeds_old_1stgen.json`) |
| fit | T-ladder LM at 0.9 then 1.0 of each point's T (grid 2.25 -> 2.5 yr, adhoc_A 0.9 -> 1 yr), the second rung from the first's final point |

## Files

| file | role |
|---|---|
| `config.py` | case, flags, deviation slots, evaluation stack, LM controls |
| `model.py` | signal, template, inner product, SEF derivatives |
| `lm.py` | damped LM climb (unchanged from IMRI_TAIL_dev) |
| `make_seeds.py` | old best fits via `Fisher_results/parse_results.py`; run outside the container, which only binds this folder |
| `overlap_old_fits.py` | overlap of the old fits in this setup, no fitting -> `results/overlap_old_fits.{json,md}` |
| `make_seeds_row.py` | row seeds for stalled PN fits -> `seeds_row.json` |
| `make_results_table.py` | copy-ready table of the full-T fits, grid points, best PN fit over seeds old/row/ramp -> `results/lm_results.md` |
| `run_cp_ramp.py`, `batch_cp_ramp.sh` | PN re-climb by a C_p ramp at the first rung (PHYS re-fitted at fixed C_p = +-2.5 ... +-40, C_e = 0), then the ladder -> `results/EMRI_pn_rishav_ramp_Tfrac0.9-1/ramp_T2.25_{label}.json` (profile), `lm_*` |
| `run_lm_T_ladder.py`, `batch_lm_T_ladder.sh` | the ladder -> `results/{case}/lm_T{T}_{label}.json`, `lm_{label}.json` (label `idx4`, `adhoc_A`) |
| `fisher/best_fits.py` | the full-T best fit per template (0PA seed old; PN the best over old/row/ramp/rampfine) -> `results/best_fits.json` |
| `fisher/fisher_at_best.py`, `fisher/batch_fisher_at_best.sh` | Fisher at both best fits, SNR 20, T 2.5 yr -> `results/fisher_at_best/{0pa,pn}/fisher_{label}.json` |
| `fisher/compare_bias.py` | 1D and nD bias tables -> `results/fisher_at_best/bias_comparison_snr{S}.txt` |
| `fisher/check_fisher_wiring.py` | CPU check: AET[:2] = AE response, signal/template flags, deviation on at C = 0 = 0PA, SEF derivatives vs central finite differences over a range of steps (`--deltas` reuses SEF steps from an earlier run) |
| `plot_codes/` | plots as in IMRI_TAIL_dev (fit table, r-factor, trend check, 1D/nD bias tables, violins, critical SNR) -> `plots/` |

## Runs

| job | what | status |
|---|---|---|
| 644900 | `overlap_old_fits.py`, all 25 grid points, both templates | done (2026-10-03) |
| 644902-644906 | T ladder 0.9 -> 1 of T, 0PA then PN, grid rows idx0-4, 5-9, 10-14, 15-19, 20-24 | done; `make_results_table.py` -> `results/lm_results.md` |
| 644907 | adhoc_A: old fits scored, then the same ladder | done, discarded (below) |
| 645637 | PN re-climb off the C = 0 ridge, idx7, 12, 17, 20, `--seed row` (`make_seeds_row.py`: own 0PA fit + row-median C_p, C_e) -> `results/EMRI_pn_rishav_row_Tfrac0.9-1/` | done: idx20 found the ridge-free fit; idx7, 12, 17 failed (below) |
| 646058-646060 | C_p ramp re-climb, idx7, 12, 17 (`run_cp_ramp.py`) | 646058, 646059 done (idx7 81x at C_p +24.4; idx12 3.4x, full-T rung re-dephased: O 0.99999978 at 2.25 yr -> 0.99998961); 646060 done (idx17 stalled at full T, see rampfine) |
| 646388, 646389 | last try for idx12, 17: T ladder 0.95, 1 from the ramp fit at 2.25 yr (`run_lm_T_ladder.py --seed rampfine`) | done: idx17 69x (O 0.9999994560, C_p +28.6) -- used; idx12 stalled again at full T (O 0.9999895729), the ramp fit (3.4x) is kept as final |
| 647353 | Fisher at both best fits, all 25 points (`fisher/fisher_at_best.py`); 647351 cancelled when the code moved to `fisher/` | done 2026-10-05: 50 of 50, overlaps reproduced exactly, rho_s 20.000; bias in `results/fisher_at_best/bias_comparison_snr{20,200}.txt`, plots in `plots/` |
| 647447 | step test at a = 0: idx2, 7, 12, 17, 22, both templates, SEF's stored steps of each of the five points (`--steps-from`) -> `results/fisher_at_best/{0pa,pn}_steps/` | done: the a = 0 Fishers change with the steps (0PA bias/sigma in a 0.6-103, idx17's own steps the worst); SEF's spin step there was ~3e-9 (relative to a best fit ~1e-4) |
| 647537 | a = 0 Fishers redone with the at-zero step grid for a (`NEAR_ZERO` in `fisher_at_best.py`), idx2, 7, 12, 17, 22, both templates; the 647353 ones kept in `results/fisher_at_best/{0pa,pn}_sefdefault/`, its log in `logs/fisher_at_best_0pa-pn_647353.log` | done 2026-10-05: spin steps 4e-6 to 3.5e-5, sigma_a smooth in e0 (0PA 2.0-3.4e-4, PN 2.1-3.4e-3), idx17 0PA bias/sigma in a 16.6 -> 0.49; bias tables and plots regenerated |
| 647745-647749 | PN re-climb at the a = 0 points idx2, 7, 12, 17, 22 with the at-zero spin steps (now in `model.derivs`, `config.NEAR_ZERO`), one rung at full T from the PN best fit (`--seed best`) -> `results/EMRI_pn_rishav_best_Tfrac1/` | done 2026-10-05: no improvement -- lambda_exhausted after 1-6 iterations at all five (idx2, 17, 22 no accepted step; idx7 +1.5e-11, idx12 +5e-10 in overlap). The a = 0 PN fits are local optima for LM; best_fits.json left unchanged |

Wiring check (CPU, idx4, T 0.05 yr): the 0PA waveform with the deviation on and C = 0 is identical to the
deviation-off waveform, and C_p = -36.8 changes it.

The old fits, rescored in 2nd-generation AE: rho_s is 20.000 everywhere. The 0PA mismatch rises by
1.0-4.7x (largest at idx17, then 16 and 22). The PN mismatch rises by up to ~2800x: the retrograde fits
at ~1e-8 drop to ~1e-5. At idx0, 3, 7, 10 and 20 the old PN fit now scores below the old 0PA fit; the
nested template rules that out at an optimum, so every PN fit needs re-climbing.

Ladder results (2026-10-04): the 0PA fit improved on the rescored old fit at all 25 points, and the PN fit at
every point. PN ends >= 0PA everywhere. PN stalls at C_p ~ 0 (1.0-1.1x) at idx7, 12, 17 (a = 0) and idx20
(a = -0.9, whose row-mates reach C_p ~ +20 at >800x): the dev = 0 ridge, not yet re-climbed.

adhoc_A is dropped (user, 2026-10-04): at dt 5 the band reaches the 2nd-generation TDI PSD null at 0.05996 Hz,
where one bin carries SNR^2 ~ 2.4e11 (rho_s 490826; below 0.05 Hz rho_s = 44.58). The grid (dt 10, Nyquist
0.05 Hz) does not reach it.

Row re-climb (645637): idx20 reached O = 0.9999999915 (C_p +21.15, 625x), replacing the stalled old-seed fit.
At a = 0 the row seed (own 0PA fit + C_p +29.4) starts at O ~ 0.05 and LM ends on spurious maxima (O 0.50-0.72,
C_p from -483 to +3546, qS off by up to 0.8 rad); the old-seed fits of idx7, 12, 17 stand.
