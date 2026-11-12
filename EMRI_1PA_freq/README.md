# EMRI_1PA_freq: 1PA signal vs 0PA + PN deviation with a C_p-driven frequency term

Same setup as `EMRI_1PA_PN_ADD_DEV` (2nd-generation TDI, A and E, `rishav`, DC bin dropped, T 2.5 yr, dt 10,
SNR 20). The template carries the 2.5PN-type C_p, C_e rate terms, and C_p also shifts both frequencies, as a
guess at the 1PA frequency correction:

    pdot      += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot      += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5
    Omega_phi += -(15/56) q C_p (1-e^2)^1.5 / p^2,     Omega_r += the same

| item | value |
|---|---|
| SuperKludge_r | `PN_power_deviation`, flux.py edited in the working tree (not committed); power-law slots removed: additional_args = [chi2, 1PA, primary, 2PA, deviation_included, C_p, C_e] |
| free | m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, C_p, C_e |
| points | idx0, idx4 |
| seed | the injection, with C_p, C_e from the exact 1PA pdot, edot at the injected (a, p0, e0, chi2) (`physical_c.py`): idx0 C_p = -17.36, C_e = -0.986; idx4 C_p = +0.692, C_e = +0.0209 |
| strategy | `full`: one LM climb at T = 2.5. If it ends below the 0PA best-fit overlap of `EMRI_1PA_PN_DEV` (C = 0 is nested, so that is a failed climb), `ladder`: T = 2.2, 2.3, 2.4, 2.45, 2.47, 2.5 from the same seed |
| LM | `lm.py` unchanged, `LAMBDA0` = 1e-6, up to 1000 iterations per rung |

## Files

| file | role |
|---|---|
| `check_freq_deviation.py` | CPU checks of the flux term -> `logs/check_freq_deviation.log` (12 of 12 pass) |
| `physical_c.py` | C_p, C_e implied by the exact 1PA rates at the injection (the seed) |
| `config.py` | case, arms, deviation slots, evaluation stack, LM controls |
| `model.py`, `lm.py` | from `EMRI_1PA_PN_ADD_DEV` |
| `run_lm_arms.py`, `batch_lm.sh` | full, then ladder if needed -> `results/{full,ladder}/lm_idx{p}.json` (ladder rungs `lm_T{T}_idx{p}.json`); log `logs/lm_idx{p}.log` |
| `fisher_at_best.py`, `batch_fisher.sh` | 11-parameter Fisher at the higher-overlap arm, repeated with steps doubled -> `results/fisher_at_best/fisher_idx{p}.json` |
| `make_summary.py` | fits, conditioning, intrinsic critical SNR vs 0PA, PN, PN_power, PN+env -> `results/summary.md` |
