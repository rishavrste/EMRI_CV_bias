# EMRI_1PA_PN_POWER_DEV: 1PA signal vs 0PA + power-law PN deviation

Same setup as `EMRI_1PA_PN_DEV` (2nd-generation TDI, A and E, `rishav`, DC bin dropped, T 2.5 yr, dt 10, SNR 20),
with the PN term multiplied by a power of p:

    pdot += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5 (p/10)^n_p
    edot += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5 (p/10)^n_e

| item | value |
|---|---|
| SuperKludge_r | `PN_power_deviation`: additional_args = [chi2, 1PA, primary, 2PA, deviation_included, C_p, C_e, n_p, n_e] |
| free | m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, C_p, C_e, n_p, n_e |
| points | idx0, 4, 8, 12, 19 |
| pivot | p = 10 |
| LM start | `LAMBDA0` = 1e-6, not 1e-2: at 1e-2 idx4 stopped at iteration 0 with no accepted step (`diagnose_step.py`) |
| SEF steps of n | always absolute, `config.N_STEPS` = geomspace(1e-1, 1e-5, 12) |

## Arms

| arm | seed | rungs [yr] |
|---|---|---|
| `best_full` | PN best fit of `EMRI_1PA_PN_DEV`, n = 0 | 2.5 |
| `best_ladder` | same | 2.2, 2.3, 2.4, 2.45, 2.47, 2.5 |
| `inj_ladder` | injection, C_p = C_e = 1, n = 0 | 2.2, 2.3, 2.4, 2.45, 2.47, 2.5 |

## Files

| file | role |
|---|---|
| `config.py` | case, arms, deviation slots, evaluation stack, LM controls |
| `model.py` | signal, template, inner product, SEF derivatives (from `EMRI_1PA_PN_DEV`; the n step grids are new) |
| `lm.py` | damped LM climb, unchanged |
| `run_lm_arms.py`, `batch_lm_arms.sh` | the arms -> `results/{arm}/lm_T{T}_{label}.json`, `lm_{label}.json` |
| `make_results_table.py` | arm comparison -> `results/lm_results.md` |
| `fisher_at_best.py`, `batch_fisher_at_best.sh` | 13-parameter Fisher at the fits, repeated with steps doubled -> `results/fisher_at_best/` |
| `critical_snr_intrinsic.py` | nD critical SNR over m1, m2, a, p0, e0 -> `results/critical_snr_intrinsic.md` |
| `diagnose_step.py`, `batch_diagnose.sh` | SEF steps, Fisher spectrum, gradient, trial steps and n profile at one seed |

Consistency checks of the model are in `../PN_power_dev/`.
