# EMRI_1PA_PN_ADD_DEV: 1PA signal vs 0PA + PN deviation + environmental power law

Same setup as `EMRI_1PA_PN_POWER_DEV` (2nd-generation TDI, A and E, `rishav`, DC bin dropped, T 2.5 yr, dt 10,
SNR 20). The template carries the 2.5PN-type C_p, C_e terms and a power-law factor on the vacuum rates, meant
to absorb environmental perturbations:

    pdot = pdot_GR (1 + A_p (p/10)^n_p) + q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot = edot_GR (1 + A_e (p/10)^n_e) + q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5

The environmental factor multiplies the vacuum 0PA rate only, so it carries q, edot keeps its factor of e, and
A = 0 is GR. All six at zero is GR; A = 0 is the earlier PN model.

| item | value |
|---|---|
| SuperKludge_r | `PN_power_deviation`, flux.py edited in the working tree (not committed): additional_args = [chi2, 1PA, primary, 2PA, deviation_included, C_p, C_e, A_p, A_e, n_p, n_e] |
| free | m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, C_p, C_e, A_p, A_e, n_p, n_e |
| points | idx0, idx4 |
| pivot | p = 10 |
| seed | PN best fit of `EMRI_1PA_PN_DEV`, A_p = A_e = 1e-7, n_p = n_e = 0 (A = 0 would make d/dn vanish) |
| ladder | T = 2.2, 2.3, 2.4, 2.45, 2.47, 2.5 yr |
| LM | `lm.py` unchanged, `LAMBDA0` = 1e-6 |
| SEF steps | A: absolute geomspace(1e-5, 1e-10, 12); n: absolute geomspace(1, 1e-4, 12) |

Expected degeneracies: at n = 0 with A_p = A_e the factor is a constant rescaling of the 0PA rates, i.e. a
change of m2; A and n are correlated because ln(p/10) keeps one sign over most of the inspiral.
`make_summary.py` reports both.

## Files

| file | role |
|---|---|
| `check_add_deviation.py` | CPU checks of the flux term and trajectories -> `logs/check_add_deviation.log` (31 of 31 pass) |
| `config.py` | case, arm, deviation slots, evaluation stack, LM controls |
| `model.py` | signal, template, inner product, SEF derivatives (from `EMRI_1PA_PN_POWER_DEV`) |
| `lm.py` | damped LM climb, unchanged |
| `run_lm_arms.py`, `batch_lm.sh` | the ladder -> `results/pn_ladder/lm_T{T}_idx{p}.json`, `lm_idx{p}.json`; log `logs/lm_idx{p}.log` |
| `fisher_at_best.py`, `batch_fisher.sh` | 15-parameter Fisher at the fits, repeated with steps doubled -> `results/fisher_at_best/` |
| `make_summary.py` | fits, Fisher conditioning, intrinsic critical SNR vs 0PA, PN, PN_power -> `results/summary.md` |
