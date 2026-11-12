# Can a flux/frequency deviation absorb 1PA without biasing the physical parameters?

Status: 2026-11-11. Covers EMRI_1PA_freq (the C_p-driven frequency term), consistency_check (why it does not lower the
absolute bias), and check_models (the search for a better deviation). Section 4b's idx7 / IMRI idx4 fits are still running.

## 0. The question

Signal: 1PA SuperKludge waveform. Template: 0PA plus a few deviation coefficients. The fit always gets better
(higher overlap) when the template has more freedom. The question is whether the **absolute bias** of the physical
parameters (m1, m2, a, p0, e0, phases) also goes down, i.e. whether the deviation absorbs the 1PA physics instead of
trading against the masses.

Working criterion used throughout: a deviation can lower the bias only if, **at the injected physical parameters**,
some choice of its coefficients reproduces the 1PA phases to well below ~1/SNR = 0.05 rad. Otherwise the remaining
dephasing is cancelled by moving m1, p0, a, ... and that shift is the bias.

## 1. The C_p-driven frequency term (EMRI_1PA_freq)

SuperKludge_r @ PN_power_deviation (now committed as ae0adc8):

    pdot      += q C_p (1-e^2)^1.5 (8 + 7e^2) / p^3.5
    edot      += q C_e e (1-e^2)^1.5 (304 + 121e^2) / p^4.5
    Omega_phi += -(15/56) q C_p (1-e^2)^1.5 / p^2,  Omega_r gets the same term          q = m1 m2/(m1+m2)^2

Versions tried for the frequency term: the C_p + C_e average with +45/16 C_e (a sign error; it cancelled the C_p
term), the average with -45/16 C_e, and C_p only. C_p only was kept (most accurate at small p; the average weights the
unphysical C_e 21 times more).

LM (lm.py, LAMBDA0 = 1e-6) from the "physical" seed: injected parameters with C_p, C_e = projection of the exact
pdot1PA, edot1PA onto the shapes at p0 (physical_c.py). Full climb at T = 2.5 yr first, the T ladder
(2.2, 2.3, 2.4, 2.45, 2.47, 2.5) if it ends below the 0PA overlap.

| point | seed | arm | overlap | note |
|---|---|---|---|---|
| idx0 (a=-0.9) | physical | full | 0.999999990866 | best of all templates: 1.88x PN, 1.71x PN_power, 1.87x PN+env |
| idx4 (a=+0.9) | physical | full | 0.739024 | C_e ran to -696 |
| idx4 | physical | ladder | 0.505566 | every rung lost overlap |
| idx4 | PN best fit + physical C | full | 0.081 at it 399 | stalled; stopped by hand (job 711754) |
| idx4 | 0PA best fit + physical C | full | 0.783118 | below the floor |
| idx4 | 0PA best fit + physical C | ladder | **0.999986860** | worked: 2.0x better than PN (0.999974087), below PN_power (0.999996433) and PN+env (0.999992603) |

idx4 0pa_fit ladder best fit:

    m1=1000040.0338072256, m2=9.999674448605294, a=0.9000033288626035, p0=9.086623172841826, e0=0.10000240930186352,
    qS=1.0473510210208163, phiS=0.7853301319270013, Phi_phi0=0.09619915513935225, Phi_r0=0.340995192916717,
    C_p=-15.098211099090863, C_e=-0.2936573255471579

Absolute bias, idx0 (best fit - injection):

| param | 0PA | PN | PN_power | PN+env | PN+freq |
|---|---|---|---|---|---|
| m1 | +43.97 | +7.968 | +7.986 | +7.959 | +25.94 |
| m2 | +5.18e-5 | +8.84e-5 | | | -1.42e-4 |
| a | +1.38e-4 | +6.29e-5 | | | +8.11e-5 |
| p0 | -4.70e-4 | -1.95e-4 | | | -3.05e-4 |
| e0 | -1.72e-5 | +9.94e-6 | | | +7.48e-6 |

idx4 (new 0pa_fit ladder): m1 +40.0, m2 -3.3e-4, a +3.3e-6, p0 -1.75e-4, e0 +2.4e-6, against PN m1 -6.50,
m2 -1.36e-3, a +9.0e-6, p0 -6.75e-5, e0 +6.54e-5.

**Result: the overlap improves but the m1 bias gets worse at both points.** idx0 critical SNR (intrinsic nD, 90%):
0PA 0.77, PN 10.3, PN_power 16.9, PN+env 17.0, PN+freq 15.0. The 11-parameter Fisher at idx0 is not positive
semi-definite (cond 4.8e10; corr C_p-m1 -0.96, C_p-p0 +0.95). The idx4 Fisher of the new fit has not been run
(fisher_at_best.py checks for branch PN_power_deviation; SuperKludge_r is on PN_freq_evol now).

## 2. Is the frequency term coded right? (consistency_check/check_lo_terms.py, 40/40 pass)

The lowest power of v = p^-1/2 in each SKequatorialfluxes 1PA function, s = 5a - 3chi2:

    pdot1PA -> (28/15) s (1-e^2)^1.5 (8+7e^2)/p^3.5      edot1PA -> (4/45) s e (1-e^2)^1.5 (304+121e^2)/p^4.5
    OmegaPhi1PA, Omegar1PA -> -(1/2) s (1-e^2)^1.5/p^2   (identical at this order)

So the deviation is exactly the leading 1PA term with C_p = (28/15)s, C_e = (4/45)s = C_p/21, and -(15/56)C_p = -s/2.
The code is consistent. But the leading term carries only 61-79% of pdot1PA, 48-66% of edot1PA, 77-86% of
OmegaPhi1PA along idx0, overshoots Omegar1PA by ~16%, and has the wrong sign over most of idx4.

## 3. Why the bias does not go down (consistency_check, idx0)

Trajectories of 1PA, 0PA (injection and fit), 0PA+PN fit, 0PA+PN_freq fit, and PN_freq at the injection with
C_LO = (-13.72, -0.653) and C_proj = (-17.36, -0.986). Rates from the ODE right-hand side; units checked
(dPhi/dt gives 1/(M MTSUN), dp/dt gives q/(M MTSUN) to 1e-8).

| model | end dPhi_phi | end dPhi_r | rate error at fixed p vs 0PA-injection (pdot, edot, Om_phi, Om_r) |
|---|---|---|---|
| 0PA injection | +6.88 | +3.32 | 1, 1, 1, 1 |
| 0PA fit | +0.008 | +0.006 | 18.7, 11.4, 4.75, 1.13 |
| 0PA+PN fit | +0.0001 | -0.001 | 7.69, 1.63, 1.94, 0.47 |
| 0PA+PN_freq fit | -0.0001 | +0.0001 | 12.0, 6.43, 3.02, 0.72 |
| PN_freq inj, C_LO | +1.26 | -0.66 | 0.26, 0.38, 0.16, 0.16 |
| PN_freq inj, C_proj | -0.23 | -1.71 | 0.07, 0.09, 0.07, 0.47 |

- At the injection the deviation leaves 0.2-1.7 rad, far above what an overlap of 1 - 1e-8 allows.
- The fits remove that by shifting parameters: their rates at fixed p are further from 1PA than plain 0PA is,
  because m1 moved by +2.6e-5 (relative), twice the whole 1PA effect on Omega_phi (1.2e-5). The data constrain
  Phi(t), not the rates at fixed p.
- The fitted C are not physical (PN_freq: C_p = -6.87, C_e = +6.79; PN: C_p = +19.45).
- One C_p drives pdot and both frequencies; Omega_r keeps a 47% error with C_proj and Phi_r runs off by 1.7 rad.

Plots: consistency_check/plots/ (diff_vs_t, diff_vs_p, phase_diff_vs_t; *_inj = 1PA vs 0PA and PN_freq at the
injection only; (1PA - model)/1PA, one column per model; dOmega_r/dt normalised by max|1PA| since it changes sign).

## 4. Candidate deviation families (check_models)

**2026-11-12: flux.py on PN_freq_evol now carries only the SVD_4 deviation (section 5d); the model switch below is
removed, so families.py / expected_values.py / phase_fit.py / check_wiring.py no longer run on that branch.**

SuperKludge_r @ **PN_freq_evol**, one flux.py with a model switch (check_wiring.py: 18/18 pass):

    additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, model, c1, ..., c8]

| model | name | parameters | terms |
|---|---|---|---|
| 0 | PN_freq | C_p, C_e | as section 1 |
| 1 | A | C_p, C_e, C_Om | Omega_phi, Omega_r += q C_Om (1-e^2)^1.5/p^2 |
| 2 | B | C_p, C_e, C_Omphi, C_Omr | separate Omega_phi, Omega_r amplitudes |
| 3 | power law | B + n_p, n_e, n_Omphi, n_Omr | each term times (10/p)^n |

Each contains the one before (n = 0 gives B; C_Omphi = C_Omr gives A; C_Om = -(15/56)C_p gives PN_freq).

### 4a. Expected coefficients and the dephasing they leave (expected_values.py)

Coefficients by least squares of each family's terms to the exact 1PA rates along the injected 1PA trajectory
(uniform in t; linear except the power law), then one trajectory at the injection. End dephasing (Phi_phi, Phi_r), rad:

| family | EMRI idx0 (a=-0.9) | EMRI idx4 (a=+0.9) | EMRI idx7 (a=0, e0=0.2) | IMRI idx4 (a=+0.9) |
|---|---|---|---|---|
| 0PA | +6.86, +3.32 | +4.15, +2.97 | +8.67, +4.18 | -178, -86 |
| PN_freq | -0.87, -2.16 | -11.6, -3.27 | +1.03, -0.16 | -639, -294 |
| A | +0.60, -0.70 | -13.4, -5.14 | +0.40, -0.79 | -635, -291 |
| B | -0.11, +0.01 | -13.9, -4.64 | -0.33, -0.06 | -638, -289 |
| power law | +0.002, 0.000 | -2.71, -0.82 | +0.011, +0.002 | -192, -94 |

- Retrograde and a = 0: B is close (0.1-0.3 rad), the power law nearly exact.
- a = +0.9 (EMRI and IMRI): the 1PA rates are not dominated by their leading term (the leading spin term
  s = 5a - 3chi2 = 1.65 is small); one amplitude per channel leaves ~50% (EMRI) to ~80% (IMRI) of the rates and
  dephases *more* than 0PA. Only the power law helps at EMRI idx4, with steep indices (n ~ +2.1 to +2.75).

### 4b. Phase fits by re-integration (phase_fit.py)

From the expected values: (i) coefficients only at the injection; (ii) coefficients + m1, m2, a, p0, e0, Phi_phi0,
Phi_r0 free. The shifts in (ii) are a CPU proxy for the absolute bias (phases only, no noise weighting: a ranking).
PBS job 712351 (EMRI idx1, 2, 3, 10, 19; exit 0). End (Phi_phi, Phi_r) residual in rad; m1 shift for M = 1e6:

| point (a, e0) | family | (i) at injection: end residual | (ii) free: end residual | (ii) m1 shift | (ii) a shift | (ii) coefficients |
|---|---|---|---|---|---|---|
| idx1 (-0.5, 0.1) | A | -0.4453, -0.1093 | +0.00508, +0.002652 | +77.5 | +0.0001461 | C_p=-10.8603, C_e=-7.63543, C_Om=+2.81761 |
| idx1 (-0.5, 0.1) | B | +0.00151, +0.02346 | +0.004153, +0.002597 | +18.43 | +3.974e-05 | C_p=-17.023, C_e=-3.23813, C_Omphi=+3.34775, C_Omr=+2.21007 |
| idx2 (0, 0.1) | A | -0.2741, -0.1733 | +0.005021, +0.005732 | +52.75 | +8.254e-05 | C_p=-1.07137, C_e=-4.22758, C_Om=+1.83893 |
| idx2 (0, 0.1) | B | +0.006725, +0.073 | +0.007911, +0.004332 | +11.06 | +2.726e-05 | C_p=-8.58825, C_e=-7.50149, C_Omphi=+2.29338, C_Omr=+1.02235 |
| idx3 (+0.5, 0.1) | A | -0.02945, +0.1079 | +0.01517, +0.005819 | +84.7 | +7.82e-05 | C_p=-0.695828, C_e=-22.0396, C_Om=+2.13233 |
| idx3 (+0.5, 0.1) | B | -0.03598, +0.2684 | +0.01464, +0.01932 | +20.64 | +2.695e-05 | C_p=-2.37939, C_e=-6.81124, C_Omphi=+2.04252, C_Omr=+1.12539 |
| idx10 (-0.9, 0.3) | A | -0.4564, -0.0808 | +0.0009822, +0.001564 | +59.87 | +0.0001327 | C_p=-23.1143, C_e=-0.561608, C_Om=+3.54963 |
| idx10 (-0.9, 0.3) | B | -0.0052, +0.003144 | +0.001329, +0.00141 | +5.177 | +1.49e-05 | C_p=-23.7801, C_e=-0.913096, C_Omphi=+4.11752, C_Omr=+2.72006 |
| idx19 (+0.9, 0.4) | A | -0.1226, +0.7307 | +0.09481, +0.39 | +0.08323 | -3.995e-06 | C_p=-0.00563412, C_e=+0.980242, C_Om=+1.78363 |
| idx19 (+0.9, 0.4) | B | -0.08772, +0.7235 | +0.1831, +0.4192 | -0.182 | +8.702e-06 | C_p=-2.25308, C_e=+0.20116, C_Omphi=+0.314455, C_Omr=-0.0330945 |

- Retrograde to a = +0.5: B reproduces the 1PA phases at the injection to 0.002-0.07 rad with its expected
  coefficients alone (A: 0.1-0.45 rad). With the physical parameters free, both reach ~1e-3 rad, but **A does it by
  moving m1 by +53 to +85 and B by +5 to +21**. B keeps the optimum 3-10x closer to the injection in m1 (and a).
- a = +0.9 (idx19): neither family gets below ~0.4 rad in Phi_r even with everything free; the fit barely moves
  (m1 shift ~0) because there is no direction that helps. Same failure as the expected values at idx4.
- Not yet comparable to 0PA on this proxy: the 0PA free fit was not part of job 712351. idx7 and IMRI idx4 (0PA, A,
  B) died with the previous session and were relaunched on the login node (2026-11-12).
- Caveat on IMRI end values: the IMRI tracks run to p ~ 3.3, so the "end" phase difference is taken where it
  changes fastest and depends on the time grid (IMRI idx4 0PA: -178 rad here at N_T = 2000, -299 rad in
  pn_orders/fit_forms at N_T = 1000). Use the rms values for IMRI (18 vs 22 rad).

## 5. Which PN orders matter, and better fitting forms

Every SK 1PA rate divided by its leading shape is exactly a cubic in v (sk_orders.py splits each into its four
order terms, checked to add up to the full function).

### 5a. PN-order importance at all 50 grid points (pn_orders.py, job 712462, exit 0)

rms(order-i term)/rms(1PA) along the 1PA track, and end Phi_phi dephasing (rad) against 0PA. The pattern depends on
spin only (e0 changes the numbers by a few %), so one e0 row per spin is shown (e0 = 0.1, idx0-4):

| grid | a | pdot fractions o0, o1, o2, o3 | Omega_phi fractions o0..o3 | Phi_phi: o0 alone | o1 alone | o2 alone | o3 alone | full 1PA |
|---|---|---|---|---|---|---|---|---|
| EMRI | -0.9 | 0.72 0.10 0.32 0.70 | 0.84 0.04 0.02 0.22 | +5.62 | -0.39 | -0.69 | +2.34 | +6.88 |
| EMRI | -0.5 | 0.70 0.04 0.41 0.67 | 0.69 0.15 0.10 0.26 | +5.49 | +0.90 | -1.51 | +2.89 | +7.77 |
| EMRI | 0 | 0.60 0.26 0.49 0.63 | 0.45 0.37 0.16 0.35 | +4.73 | +2.92 | -2.43 | +3.72 | +8.93 |
| EMRI | +0.5 | 0.15 0.57 0.46 0.75 | 0.07 0.53 0.26 0.66 | +1.18 | +5.36 | -3.13 | +5.81 | +9.22 |
| EMRI | +0.9 | 0.84 0.24 0.10 1.46 | 0.46 0.47 0.66 1.61 | -15.37 | +5.44 | -2.55 | +16.73 | +4.24 |
| IMRI | -0.9 | 0.79 0.10 0.30 0.63 | 0.90 0.04 0.02 0.16 | +262 | -22 | -46 | +82 | +275 |
| IMRI | -0.5 | 0.75 0.04 0.39 0.61 | 0.76 0.14 0.08 0.19 | +249 | +15 | -59 | +75 | +280 |
| IMRI | 0 | 0.63 0.26 0.47 0.60 | 0.52 0.36 0.14 0.28 | +197 | +63 | -63 | +60 | +260 |
| IMRI | +0.5 | 0.15 0.56 0.46 0.77 | 0.09 0.56 0.25 0.62 | +41 | +103 | -46 | +52 | +152 |
| IMRI | +0.9 | 0.62 0.19 0.08 1.32 | 0.43 0.41 0.59 1.51 | -465 | +54 | -2 | +79 | -299 |

Plots of all 50 points, all four rates and both phases: plots/pn_orders_EMRI.png, plots/pn_orders_IMRI.png
(plot_pn_orders.py). edot is the exception to the table: its order-3 term is 0.7-1.6 of the total at every spin.

- a < 0: the leading term carries most of the 1PA (80-95% of the phase). This is why B works there.
- a = +0.5: the leading term is almost absent (s = 5a - 3chi2 = -0.35); the 1PA comes from orders 1 and 3.
- a = +0.9: orders 0 and 3 are both larger than the total and cancel (fractions > 1). The leading term alone dephases
  with the **wrong sign** (EMRI -15 rad against +4 for the full 1PA), and orders 0..2 together are still wrong.
  No deviation with the leading-order p-dependence can absorb this; it needs the v^3 (order 3) structure.
- Orders 1 and 2 are never negligible anywhere; the truncated series 0..i does not converge monotonically.

### 5b. Fitting forms (fit_forms.py, job 712471, exit 0)

| form | parameters | R_X (deviation = q shape_X R_X) |
|---|---|---|
| B | 4 | C_X |
| PL | 8 | C_X (10/p)^n_X |
| PL6 | 6 | n shared by the fluxes and by the frequencies |
| LIN | 8 | C_X (1 + k_X v), next order as a free slope |
| J5 | 5 | C_X (1 + k v), one slope for all |
| PADE | 8 | C_X / (1 - k_X v) |
| SVD_K | K = 1..6 | sum_j C_j b_jX(v): the top K singular vectors of the 1PA ratios over a in [-0.99, 0.99], chi2 in [0, 1], e in [0.05, 0.5], p in [5, 31]. Each b_jX is a cubic in v with fixed coefficients, so the basis is analytic. One C_j moves all four channels together. |

SVD singular values (normalised): 1, 0.19, 0.093, 0.011, 0.009, 0.001; cumulative energy 0.957, 0.992, 0.9998,
0.99992, 0.999999. The 1PA family is effectively 3-4 dimensional over the whole box.

The analytic basis (cubic coefficients of v^0, v^1, v^2, v^3, per channel; deviation_X = q shape_X sum_j C_j b_jX(v)):

    b1: pdot +0.37632 +0.081864 -2.1385 +15.567 | edot +0.01792 +0.001718 -0.11166 +1.0619 | Om_phi -0.1008 -0.12129 +0.079166 -1.8224 | Om_r -0.1008 -0.12129 +0.38157 -1.2003
    b2: pdot +1.0068 -1.3032 -3.8859 +14.716 | edot +0.047944 -0.062627 -0.17513 +0.70948 | Om_phi -0.26969 +0.73205 -0.5469 +5.4386 | Om_r -0.26969 +0.73205 +0.26215 +3.4601
    b3: pdot +0.94918 +1.703 -4.367 -33.039 | edot +0.045199 +0.10239 -0.22295 -1.6137 | Om_phi -0.25425 -0.24301 -0.14125 +5.9525 | Om_r -0.25425 -0.24301 +0.62149 +4.3438
    b4: pdot -1.2065 +1.7298 +12.221 +0.23448 | edot -0.057452 +0.12368 +1.0052 -3.7428 | Om_phi +0.32317 -1.1662 -5.1156 +17.942 | Om_r +0.32317 -1.1662 -6.0851 +15.803

Rate level, max over channels of the relative rms remainder (all 50 points in logs/fit_forms_rates.log). Worst
case per spin, EMRI / IMRI:

| a | B (4) | J5 (5) | PL6 (6) | PL (8) | PADE (8) | SVD_3 (3) | SVD_4 (4) | SVD_5 (5) |
|---|---|---|---|---|---|---|---|---|
| -0.9 | 0.067 / 0.17 | 0.038 / 0.11 | 0.013 / 0.034 | 0.002 / 0.023 | 0.001 / 0.009 | 0.011 / 0.010 | 0.005 / 0.005 | 0.005 / 0.005 |
| +0.9 | 0.51 / 0.76 | 0.40 / 0.51 | 0.068 / 0.12 | 0.068 / 0.12 | 0.17 / 0.25 | 0.13 / 0.15 | 0.050 / 0.037 | 0.034 / 0.020 |

Phase level: coefficients from the rate fit, one trajectory at the injection, end (Phi_phi, Phi_r) of 1PA - model, rad:

| form (n) | EMRI idx0 (a=-0.9) | EMRI idx7 (a=0) | EMRI idx4 (a=+0.9) | EMRI idx19 (a=+0.9) | IMRI idx0 (a=-0.9) | IMRI idx4 (a=+0.9) |
|---|---|---|---|---|---|---|
| 0PA (0) | +6.88, +3.32 | +8.71, +4.18 | +4.24, +3.01 | +4.99, +3.21 | +275, +93 | -299, -85 |
| B (4) | -0.110, +0.010 | -0.329, -0.060 | -14.3, -4.65 | -13.4, -4.71 | -48.3, -7.3 | -495, -273 |
| J5 (5) | -0.017, +0.028 | -0.005, +0.033 | -11.6, -3.86 | -10.7, -3.79 | -5.5, +2.0 | -117, -36 |
| PL6 (6) | +0.007, +0.006 | +0.064, +0.026 | -2.93, -0.83 | -2.30, -0.71 | +17.8, +4.4 | -143, -80 |
| SVD_2 (2) | +0.007, -0.038 | +0.383, +0.037 | -6.83, -2.29 | -5.82, -2.06 | -6.2, -2.3 | -157, -91 |
| SVD_3 (3) | +0.031, -0.027 | +0.044, -0.050 | -2.57, -1.07 | -1.93, -0.86 | +0.46, -0.24 | -21.8, -8.6 |
| **SVD_4 (4)** | **+0.004, -0.002** | **+0.056, -0.026** | **+0.23, -0.02** | **+0.40, +0.04** | **-0.047, -0.014** | **-9.9, -3.5** |
| PL (8) | +0.002, 0.000 | +0.012, +0.002 | -2.79, -0.82 | -2.25, -0.71 | +8.5, +1.7 | -142, -80 |
| PADE (8) | +0.000, 0.000 | +0.003, +0.001 | -6.10, -1.95 | -5.35, -1.83 | +3.1, +0.6 | crashed (pole: integration did not converge) |

- **SVD_4 is the best form with fewer than 5 parameters, and the only one that works at a = +0.9**: 15-20x below
  0PA at EMRI idx4/idx19, 30x at IMRI idx4, and 0.002-0.05 rad at retrograde/a = 0. Every form built on the leading
  shape (B, J5, PL6, PL, PADE) fails at a = +0.9, as 5a predicts; PL and PADE (8 parameters) win only where B is
  already good.
- SVD_4 still misses the 0.05 rad criterion at a = +0.9 (0.2-0.4 rad EMRI, ~10 rad IMRI). SVD_5 halves the
  rate remainder there and is the natural next test.
- These are phases at the injection with rate-fitted coefficients, not a free fit: whether SVD_4 also keeps the
  physical parameters near the injection (the actual goal) needs the phase_fit (ii) test or a GPU LM run.

### 5d. SVD_4 in flux.py (PN_freq_evol)

    additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_1, C_2, C_3, C_4]
    X += eta shape_X(p, e) sum_j C_j b_jX(v),  X = pdot, edot, Omega_phi, Omega_r   (SuperKludgeFlux.SVD_BASIS)

check_svd_wiring.py: 4/4 pass (basis identical to fit_forms; rates to 5.5e-16; C = 0 gives GR exactly; the EMRI idx0
trajectory equals fit_forms' to 0 rad, and the deviation moves Phi_phi by +6.878 rad against the 1PA's +6.88).

### 5c. Other ideas

Ideas considered and not (yet) built: scaling each PN order of the 1PA (ppE/TIGER-like, lambda_k; exact at
lambda = 1 but uses the 1PA structure and higher orders); lambda x 1PA(chi2_eff) (an upper bound, essentially the
signal model); shifting the 0PA model's own arguments (q d_a dOmega/da + q d_p dOmega/dp, separately for fluxes and
frequencies; degenerate with a and p0, and its p-dependence, p^-3 and p^-2.5, does not match the 1PA p^-2); priors on
C in the LM (only acts inside a GPU fit).

## 6. Files

| path | what |
|---|---|
| EMRI_1PA_freq/ | LM, Fisher, critical SNR of the PN_freq template (run_lm_arms.py --seed physical / pn_fit / 0pa_fit) |
| consistency_check/check_lo_terms.py | section 2 |
| consistency_check/trajectories_idx0.py, plot_trajectories_idx0.py | section 3 (`--only-injection` for the 2-model plots) |
| check_models/check_wiring.py | flux.py model switch, 18 checks |
| check_models/families.py | families, points ("I4" = IMRI idx4), trajectories |
| check_models/expected_values.py | 4a |
| check_models/phase_fit.py, batch_cpu.sh | 4b |
| check_models/sk_orders.py, pn_orders.py, batch_pn_orders.sh | section 5, PN orders |
| check_models/fit_forms.py, batch_fit_forms.sh | section 5, fitting forms and SVD basis |

Python: /home/svu/e1583490/bias_inference_emri/.venv/bin/python. SuperKludge_r: branch PN_freq_evol, flux.py edited
in the working tree (uncommitted). EMRI_1PA_freq scripts require PN_power_deviation and will refuse to run until the
branch is switched back.
