# run_1_second_gen -- run_1 population refit with 2nd-generation TDI, LM only

Same 23 sources as run_1, A and E, dt = 10, dist rescaled to SNR 200 under the 2nd-gen PSD.
Converged run_1 sources: one full-T LM per model from the run_1 best fit. idx 8, 13, 14, 16:
T-ladder 0.5, 0.75, 0.9, 0.95, 1 from the truth. Details in the `lm_second_gen.py` docstring.
SuperKludge_r on `environ_shubham` (a71f184).

| job | what | status |
|---|---|---|
| 653024 (`lm2g_a`) | 19 converged sources -> `results_lm_second_gen_converged.json` | submitted 2026-10-07 |
| 653025 (`lm2g_b`) | T-ladder idx 8, 13, 14, 16 -> `results_lm_second_gen_ladder.json` | submitted 2026-10-07 |
| 653024 / 653025 | outcome | exit 0. Converged 19: overlaps match run_1 to ~1e-8. Ladder: idx8 0PA 0.99993, idx13/14 0.998-0.999 still climbing at 150 its, idx16 failed; PN ended below 0PA at 8, 13, 14 |

## e0 fixed at 0 (`lm_e0fixed.py`, 2026-10-08)

e0 removed from the fit (0PA 7 params, PN 8), LM_MAX_ITERS 300, no Nelder-Mead. Converged
sources: LM from the e0-free 2nd-gen fit, e0 dropped. idx 8, 13, 14, 16: 0PA ladder from the
truth in steps of 0.05 from T = 0.5 (idx16: 0.1 ... 0.8, then 0.05 steps). PN from two seeds,
the higher overlap kept: its own T-ladder from the truth (C_p = 0), and a full-T climb from the
0PA T=1 fit with C_p = 0.

| job | what | status |
|---|---|---|
| 654131 (`e0c`) | 19 converged -> `results_lm_e0fixed_converged.json` | submitted 2026-10-08 |
| 654137 (`e0la`) | idx 8, 14 -> `results_lm_e0fixed_ladder_a.json` | submitted 2026-10-08 (654132 cancelled before start: PN two-seed added) |
| 654138 (`e0lb`) | idx 13, 16 -> `results_lm_e0fixed_ladder_b.json` | submitted 2026-10-08 (654133 cancelled likewise) |
