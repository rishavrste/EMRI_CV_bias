# run_2 -- fresh circular EMRI population, 2nd-generation TDI

As run_1 (cv_population.py: env Fisher, 0PA and 0PA+PN CV climbs LM -> NM -> LM), except:
2nd-gen TDI (A2/E2TDISens), seed 44, t_plunge ~ U(3, 4) yr (run_1: U(2, 4)), m1 ~ U(1e6, 1e7),
m2 = 50, SNR 200, dt = 10. The dt = 5 aliasing check is band-limited to f < 0.05 Hz because the
2nd-gen PSD has a null at ~0.06 Hz. SuperKludge_r on `environ_shubham` (a71f184).

| job | what | status |
|---|---|---|
| 653026 (`env_pop2`) | sample_population.py -> population.json | submitted 2026-10-07 |
| 653027 (`env_cv2`) | cv_population.py -> results_env_cv.json, held until 653026 exits 0 | submitted 2026-10-07 |
| 653027 | outcome | killed 2026-10-08 after 14 of 25 sources (e0 free). Converged: 1, 4-9, 13 (p0 <= 7). Failed: 0, 2, 3, 10, 11, 12 (p0 >= 8.6). Saved as `results_env_cv_e0free_partial.json` |

## e0 fixed at 0 (`lm_e0fixed.py`, 2026-10-08)

e0 removed from the fit (0PA 7 params, PN 8), LM_MAX_ITERS 300, no Nelder-Mead.
Converged in 653027: LM from its fit with e0 dropped. All others: 0PA T-ladder from the truth,
0.5 ... 1 in steps of 0.05 (idx11, p0 = 11.1: 0.1 ... 0.8 then 0.05 steps); PN from two seeds,
the higher overlap kept -- its own T-ladder from the truth (C_p = 0) and a full-T climb from the
0PA T=1 fit with C_p = 0.

| job | sources | output |
|---|---|---|
| 654139 (`e2lm`) | 1, 4, 5, 6, 7, 8, 9, 13 (LM) | `results_lm_e0fixed_lm.json` |
| 654140 (`e2la`) | 14, 16, 18, 19, 20, 21, 23, 24 | `results_lm_e0fixed_ladder_a.json` |
| 654141 (`e2lb`) | 0, 11 | `results_lm_e0fixed_ladder_b.json` |
| 654142 (`e2lc`) | 3, 10 | `results_lm_e0fixed_ladder_c.json` |
| 654143 (`e2ld`) | 2, 12 | `results_lm_e0fixed_ladder_d.json` |
| 654144 (`e2le`) | 15, 17, 22 | `results_lm_e0fixed_ladder_e.json` |
