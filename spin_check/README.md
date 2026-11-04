# spin_check: idx22 at small spins, 0PA vs 0PA + PN, critical SNR

The EMRI idx22 injection (a = 0, e0 = 0.5, T 2.5 yr, dt 10) repeated at
a = -0.1, -0.05, -0.001, 0, +0.001, +0.05, +0.1. Everything else is EMRI_1PA_PN_DEV's setup
(1PA signal, rishav, 2nd-generation TDI A + E, DC bin dropped, SuperKludge_r `hybrid`).

| item | value |
|---|---|
| p0 | solved per spin: the 1PA trajectory plunges at 2.5 yr (`get_p_at_t`). At a = 0 this gives 10.99510, not idx22's stored 11.08166, so a = 0 is re-run, not reused |
| dist | rescaled per spin so the 1PA signal has SNR 20 |
| seed | new injection + (idx22 best fit - idx22 injection), same template (`results/best_fits.json`) |
| fit | T ladder 0.9 -> 1 of T (2.25, 2.5 yr), LM as in EMRI_1PA_PN_DEV |
| spin steps | the absolute at-zero grid 1e-4 .. 1e-9 for a at every spin (`spin_config.py`), in the LM and the Fisher |
| critical SNR | 1D and nD at 90 / 99 % CL, with and without the phases, as `plot_critical_snr.py` |

## Files

| file | role |
|---|---|
| `spin_config.py` | spins, rungs, paths, the spin step override |
| `run_spin_fits.py`, `batch_spin_fits.sh` | injection -> `results/a{spin}/injection.json`; rungs -> `{template}_lm_T{T}.json`; Fisher -> `{template}_fisher.json` |
| `critical_snr.py` | CPU, after the jobs -> `results/critical_snr.md` |

## Runs

| job | spins | status |
|---|---|---|
| 695625 | -0.1, +0.1 | done; 0PA at +0.1 stuck at overlap 0.565 (secondary maximum), needs a re-seed |
| 695626 | -0.05, +0.05 | done |
| 695627 | -0.001, 0, +0.001 | done |
| 697189 | +0.1, 0PA only, from the injection, ladder 2.25 / 2.5 yr (`--seed injection --tag inj`) | done; stuck at overlap 0.326 |
| 697192 | +0.1, 0PA only, from the injection, ladder 0.4 .. 1.0 of T in 0.1 (`--tag inj_fine`) | done; stuck at overlap 0.865 with Phi_phi0 off by about pi |

`critical_snr.py` uses, per spin and template, the highest-overlap Fisher among the default and tagged runs
(the "runs used" column).
