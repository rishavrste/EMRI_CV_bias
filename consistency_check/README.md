# consistency_check

Why the PN + frequency deviation does not lower the absolute bias at idx0 (a = -0.9, e0 = 0.1, chi2 = 0.95).
CPU only; SuperKludge_r @ PN_power_deviation, working tree.

| script | output |
|---|---|
| `check_lo_terms.py` | `logs/check_lo_terms.log`: is the deviation the leading 1PA term of `SKequatorialfluxes.py`? |
| `trajectories_idx0.py` | `results/trajectories_idx0.npz`: p, e, phases, pdot, edot, Omega_phi, Omega_r, dOmega/dt for 7 models |
| `plot_trajectories_idx0.py` | `plots/*.png`, plus a table in `logs/plot_trajectories_idx0.log` |

Run all three from `EMRI_1PA_freq/` with `/home/svu/e1583490/bias_inference_emri/.venv/bin/python`.

## 1. Leading-order check (40 of 40 pass)

Leading (lowest v = p^-1/2) terms of the SK 1PA functions, s = 5a - 3chi2:

    pdot1PA                 -> (28/15) s (1-e^2)^1.5 (8 + 7e^2) / p^3.5
    edot1PA                 -> (4/45)  s e (1-e^2)^1.5 (304 + 121e^2) / p^4.5
    OmegaPhi1PA, Omegar1PA  -> -(1/2)  s (1-e^2)^1.5 / p^2          (the same for both)

The deviation is exactly this term with C_p = (28/15) s and C_e = (4/45) s (so C_e = C_p / 21). The coded
frequency term -(15/56) C_p is then -(1/2) s, so it is consistent, and using one shift for Omega_phi and Omega_r is
correct at this order. The averaged form -(15/112) C_p - (45/16) C_e gives the same -(1/2) s.

At idx0: C_LO = (-13.72, -0.6533). The full 1PA rate projected at p0 gives (-17.36, -0.9862). Over the inspiral,
the leading term carries 61-79% of pdot1PA, 48-66% of edot1PA and 77-86% of OmegaPhi1PA, and overshoots Omegar1PA
by 16-17%. At idx4 (s = +1.65), it does not even have the right sign over most of the inspiral.

## 2. Models (idx0, T = 2.5 yr, each on its own trajectory)

1PA (signal); 0PA at the injection; 0PA fit and 0PA+PN fit (EMRI_1PA_PN_DEV, the PN fit without the frequency
term); 0PA+PN_freq fit (`results/full/lm_idx0.json`); 0PA+PN_freq at the injection with C_LO and with C_proj.

The unit factors are checked in the log: dPhi/dt gives 1/(M MTSUN), and dp/dt gives q/(M MTSUN) to 1e-8.

## 3. Result

    model                    | dPhi_phi(T) | dPhi_r(T) | dp(T)     | pdot frac | edot frac | Om_phi frac | Om_r frac
    0PA injection            |     +6.8824 |   +3.3239 | -1.41e-04 |    1.0000 |    1.0000 |    1.0000 |    1.0000
    0PA fit                  |     +0.0083 |   +0.0063 | +3.95e-04 |   18.6529 |   11.3881 |    4.7535 |    1.1278
    0PA+PN fit               |     +0.0001 |   -0.0010 | +1.58e-04 |    7.6902 |    1.6277 |    1.9398 |    0.4652
    0PA+PN_freq fit          |     -0.0001 |   +0.0001 | +2.41e-04 |   12.0319 |    6.4290 |    3.0176 |    0.7201
    0PA+PN_freq inj, C_LO    |     +1.2600 |   -0.6566 | -3.51e-05 |    0.2557 |    0.3792 |    0.1596 |    0.1644
    0PA+PN_freq inj, C_proj  |     -0.2333 |   -1.7141 | -7.17e-06 |    0.0705 |    0.0936 |    0.0686 |    0.4737

"frac" = rms over p of (1PA - model) / (1PA - 0PA injection), compared at the same p.

- At the injection, the deviation with the physical C cuts the rate errors to 7-38% of the 0PA ones, but it still
  leaves 0.2-1.7 rad of dephasing over 2.5 yr. That is far more than an overlap of 1 - 1e-8 allows.
- The fit removes that dephasing (|dPhi| < 1e-3 rad) by moving m1, p0, ... and C. It does not do it by
  reproducing the 1PA rates: at the same p, the fitted rates are 3-12 times further from 1PA than 0PA at the
  injection is. The fit buys phase coherence with parameter shifts, which shows up as the bias.
- The fitted C are not the physical C: C_p = -6.87 (vs -13.7 / -17.4) and C_e = +6.79 (vs -0.65 / -0.99), with
  the wrong sign. The PN fit is similar: C_p = +19.45, also the wrong sign.
- Two knobs cannot match four rates: one C_p drives pdot and both frequencies, while C_e drives edot only. With
  C_proj, Omega_r keeps a 47% error (Omega_phi only 7%), so Phi_r runs off by 1.7 rad, and the remaining
  freedom (m1, p0, a) is used to cancel it.
