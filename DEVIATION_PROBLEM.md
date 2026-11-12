# Absorbing 1PA physics with a few-parameter deviation: problem statement and everything tried

Status: 2026-11-12 (job 713215 results included). This file is a self-contained briefing. It states the problem, the equations, every model tried,
the numbers obtained, and what is still open. Detailed logs are in `check_models/README.md`, `EMRI_v_3/`,
`consistency_check/` and `EMRI_1PA_freq/`.

Equations are written in plain text: `^` is a power, `v = p^(-1/2)`, `eta = m1 m2 / (m1 + m2)^2` (symmetric mass
ratio; called `massratio` in the code), `e` is eccentricity, `p` semi-latus rectum (in units of M), `a` the primary spin,
`chi2` the secondary spin.

---

## 1. The problem in one paragraph

We study systematic (Cutler-Vallisneri) bias in LISA parameter estimation of EMRIs and IMRIs. The injected signal is
a **1PA** (first post-adiabatic) SuperKludge waveform. The template is the **0PA** (adiabatic) waveform, which
misses the 1PA terms, plus a small number of **deviation parameters** added to the orbital evolution equations. A best
fit with more freedom always reaches a higher overlap. **The goal is different: find a deviation, ideally with fewer
than 5 extra parameters, whose best fit leaves the physical parameters (m1, m2, a, p0, e0, phases) as close to the
injection as possible, i.e. the deviation must absorb the 1PA physics rather than trade off against the masses.** A
model that raises the overlap but moves m1 further from the truth is a failure for this purpose.

---

## 2. Setup

### 2.1 Waveform and evolution (FEW fork: SuperKludge_r)

The trajectory is an ODE in slow time for (p, e) and the phases (Phi_phi, Phi_r), equatorial orbits (x = 1):

```
dp/dt       = pdot_0PA(p, e)      + eta * pdot1PA(a, p, e, chi2)      [+ deviation]
de/dt       = edot_0PA(p, e)      + eta * edot1PA(a, p, e, chi2)      [+ deviation]
dPhi_phi/dt = Omega_phi_0PA(p, e) + eta * OmegaPhi1PA(a, p, e, chi2)  [+ deviation]
dPhi_r/dt   = Omega_r_0PA(p, e)   + eta * Omegar1PA(a, p, e, chi2)    [+ deviation]
```

- Code: `packages_to_install/SuperKludge_r/src/few/trajectory/ode/flux.py`, class `SuperKludgeFlux`, method
  `modify_rhs` (slots ydot[0], ydot[1], ydot[3], ydot[5]). The 1PA functions are in `SKequatorialfluxes.py`.
- Units inside `modify_rhs`: pdot, edot are in slow time (overall factor eta/(M MTSUN) applied outside), the
  frequencies in 1/(M MTSUN). The 1PA terms and every deviation carry exactly one explicit factor `eta`.
- `additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, <deviation slots...>]`.
  Signal: evolve_1PA = 1, deviation_included = 0. Template: evolve_1PA = 0, deviation_included = 1.
- `few` is an editable install: **the git branch of SuperKludge_r decides which deviation runs.** Always check
  `git -C packages_to_install/SuperKludge_r branch --show-current` (see section 7).

### 2.2 Grids

| grid | m1 | m2 | T | dt | spins a | e0 | other |
|---|---|---|---|---|---|---|---|
| EMRI | 1e6 | 10 | 2.5 yr | 10 s | -0.9, -0.5, 0, 0.5, 0.9 | 0.1 ... 0.5 | chi2 = 0.95, Phi_phi0 = 0.1, Phi_r0 = 0.3, SNR about 20 |
| IMRI | 1e6 | 1e3 | 1 yr | 5 s (dt = 10 aliases) | same | same | same |

Grid index: `idx = e0_index*5 + a_index`. So idx0 = (a -0.9, e0 0.1), idx4 = (a +0.9, e0 0.1), idx7 = (a 0, e0 0.2),
idx19 = (a +0.9, e0 0.4). p0 is chosen so the 1PA system plunges at T.

### 2.3 Fitting and bias

```
<a|b>      = 4 Re sum_f a(f) b*(f) / S_n(f)      (TDI A, E [, T]; DC bin always excluded)
overlap    = <s|h> / sqrt(<s|s><h|h>)
chi2       = <s-h|s-h>  ~  2 SNR^2 (1 - overlap)
bias       = theta_bestfit - theta_injection
LM step    : delta = (Gamma + lambda diag Gamma)^(-1) g,   g_i = <d_i h | s - h>,   Gamma_ij = <d_i h | d_j h>
```

Derivatives come from StableEMRIFisher. The CV bias formula is one Gauss-Newton step; we iterate it with
Levenberg-Marquardt to the true best fit. The 1D max bias reported is max |bias_i| / sigma_i over the **physical**
parameters only (deviation coefficients excluded).

### 2.4 Working criterion (cheap CPU proxy)

A deviation can only lower the bias if, **at the injected physical parameters**, some choice of its coefficients
reproduces the 1PA phases Phi_phi(t), Phi_r(t) to well below 1/SNR, about 0.05 rad. Otherwise the remaining
dephasing is removed by moving m1, p0, a, ..., and that shift is the bias. So every candidate is screened by:

1. **Expected coefficients**: linear least squares of the deviation to the exact 1PA rates along the injected 1PA
   track (no waveform needed).
2. **Dephasing at the injection**: one trajectory with those coefficients; end and rms of
   Phi_1PA(t) - Phi_model(t).
3. **Free phase fit** (optional): least squares on the phases with the physical parameters also free; the shift of
   m1, a, ... is a noise-free ranking proxy for the bias.

Only after that, a GPU LM fit on the full waveform.

---

## 3. Structure of the SuperKludge 1PA (the key fact)

Every SK 1PA function is exactly

```
1PA_X(a, p, e, chi2) = v^k0 * (A0 + A1 v + A2 v^2 + A3 v^3),     A_i = A_i(a, e, chi2),   v = p^(-1/2)
k0 = 7 (pdot), 9 (edot), 4 (Omega_phi), 4 (Omega_r)
```

Leading term (lowest power of v), with s = 5a - 3 chi2 (checked numerically, 40/40 pass):

```
pdot1PA     -> (28/15) s (1-e^2)^1.5 (8 + 7e^2) / p^3.5
edot1PA     -> (4/45)  s e (1-e^2)^1.5 (304 + 121e^2) / p^4.5
OmegaPhi1PA -> -(1/2)  s (1-e^2)^1.5 / p^2          (Omegar1PA identical at this order)
```

Define the **leading shapes**

```
shape_pdot = (1-e^2)^1.5 (8 + 7e^2) / p^3.5
shape_edot = e (1-e^2)^1.5 (304 + 121e^2) / p^4.5
shape_Om   = (1-e^2)^1.5 / p^2                (Omega_phi and Omega_r)
```

Then `r_X = 1PA_X / shape_X` is exactly a cubic in v for each rate. `check_models/sk_orders.py` splits each function
into its four order terms `A_i v^(k0+i)` exactly (4x4 Vandermonde solve).

**Importance of each order over all 50 grid points** (`check_models/pn_orders.py`, plots
`check_models/plots/pn_orders_{EMRI,IMRI}.png`). rms(order-i term)/rms(full 1PA) along the track, e0 = 0.1 rows
(e0 changes these by a few %); last columns: end Phi_phi dephasing of 0PA + that order alone vs 0PA (rad):

| grid | a | pdot o0 o1 o2 o3 | Omega_phi o0 o1 o2 o3 | Phi_phi: o0 | o1 | o2 | o3 | full 1PA |
|---|---|---|---|---|---|---|---|---|
| EMRI | -0.9 | 0.72 0.10 0.32 0.70 | 0.84 0.04 0.02 0.22 | +5.62 | -0.39 | -0.69 | +2.34 | +6.88 |
| EMRI | -0.5 | 0.70 0.04 0.41 0.67 | 0.69 0.15 0.10 0.26 | +5.49 | +0.90 | -1.51 | +2.89 | +7.77 |
| EMRI | 0 | 0.60 0.26 0.49 0.63 | 0.45 0.37 0.16 0.35 | +4.73 | +2.92 | -2.43 | +3.72 | +8.93 |
| EMRI | +0.5 | 0.15 0.57 0.46 0.75 | 0.07 0.53 0.26 0.66 | +1.18 | +5.36 | -3.13 | +5.81 | +9.22 |
| EMRI | +0.9 | 0.84 0.24 0.10 1.46 | 0.46 0.47 0.66 1.61 | -15.37 | +5.44 | -2.55 | +16.73 | +4.24 |
| IMRI | -0.9 | 0.79 0.10 0.30 0.63 | 0.90 0.04 0.02 0.16 | +262 | -22 | -46 | +82 | +275 |
| IMRI | +0.9 | 0.62 0.19 0.08 1.32 | 0.43 0.41 0.59 1.51 | -465 | +54 | -2 | +79 | -299 |

- a < 0: the leading term carries 80-95% of the phase. Leading-shape deviations work here.
- a = +0.5: the leading term almost vanishes (s = -0.35); the 1PA comes from orders 1 and 3.
- a = +0.9: orders 0 and 3 are each larger than the total and cancel; the leading term alone gives the **wrong sign**
  (-15 rad vs +4.2 rad). No deviation with only the leading p-dependence can absorb this.
- edot: its order-3 term is 0.7-1.6 of the total at every spin.

---

## 4. Everything tried, in order

### 4.1 Earlier deviations (before this study; from AGENTS.md and src/)

- **PN** (branch `hybrid` / `PN_power_deviation`): `pdot += eta C_p shape_pdot`, `edot += eta C_e shape_edot`
  (2 parameters). Big overlap gains for a < 0 (IMRI 18-28x mismatch reduction), little for a >= 0.
- **simple / simple_pe**: multiplicative Edot/Ldot or pdot/edot changes. Collapse to about 0PA everywhere.
- **PN_power** (`pdot_GR (1 + A_p (p/10)^n_p)` environment-like terms) and **PN+env**: better overlaps, same bias problem.

### 4.2 PN_freq: leading 1PA term including the frequencies (EMRI_1PA_freq/)

```
pdot      += eta C_p shape_pdot
edot      += eta C_e shape_edot
Omega_phi += eta * (-(15/56) C_p) * shape_Om        (Omega_r the same)
```

It is exactly the leading 1PA term with C_p = (28/15)s, C_e = C_p/21 (verified). Variants tried for the frequency
term: average with +45/16 C_e (sign error, cancelled), average with -45/16 C_e, C_p only (kept).

GPU LM results (T = 2.5 yr, T-ladder 2.2 -> 2.5 when the full climb failed):

| point | seed | overlap | m1 bias | compare |
|---|---|---|---|---|
| EMRI idx0 (a=-0.9) | physical C | 0.999999990866 (best of all models) | +25.94 | 0PA +43.97, PN +7.97 |
| EMRI idx4 (a=+0.9) | physical C, full | 0.739 (C_e ran to -696) | - | |
| EMRI idx4 | 0PA best fit + physical C, ladder | 0.999986860 (2x PN) | +40.0 | PN -6.50, 0PA -8.71 |

idx4 best fit: `m1=1000040.0338072256, m2=9.999674448605294, a=0.9000033288626035, p0=9.086623172841826,
e0=0.10000240930186352, qS=1.0473510210208163, phiS=0.7853301319270013, Phi_phi0=0.09619915513935225,
Phi_r0=0.340995192916717, C_p=-15.098211099090863, C_e=-0.2936573255471579`.

**Result: overlap improves, m1 bias gets worse.** idx0 critical SNR: 0PA 0.77, PN 10.3, PN_power 16.9, PN+env 17.0,
PN_freq 15.0. The idx0 Fisher with C_p, C_e is near-singular (corr C_p-m1 -0.96, C_p-p0 +0.95).

### 4.3 Why the bias does not go down (consistency_check/, EMRI idx0)

| model | end dPhi_phi | end dPhi_r |
|---|---|---|
| 0PA at injection | +6.88 | +3.32 |
| 0PA best fit | +0.008 | +0.006 |
| PN_freq at injection, C from leading order | +1.26 | -0.66 |
| PN_freq at injection, C projected on the exact rates | -0.23 | -1.71 |

At the injection the deviation leaves 0.2-1.7 rad, so the fit removes it by moving m1 (+2.6e-5 relative, which is
**twice the whole 1PA effect on Omega_phi**). The data constrain Phi(t), not the rates at fixed p; the fitted C are
unphysical (PN_freq C_p = -6.87, C_e = +6.79). One C_p drives pdot and both frequencies, so Omega_r keeps a 47% error.

### 4.4 Leading-shape families with more freedom (check_models/, branch PN_freq_evol, now replaced)

```
A          : C_p, C_e, C_Om              Omega_phi and Omega_r += eta C_Om shape_Om
B          : C_p, C_e, C_Omphi, C_Omr    separate frequency amplitudes            (4 parameters)
power law  : B with each term times (10/p)^n_X                                    (8 parameters)
```

End dephasing at the injection with expected coefficients (rad, Phi_phi, Phi_r):

| family | EMRI idx0 | EMRI idx4 | EMRI idx7 | IMRI idx4 |
|---|---|---|---|---|
| 0PA | +6.86, +3.32 | +4.15, +2.97 | +8.67, +4.18 | -178, -86 |
| PN_freq | -0.87, -2.16 | -11.6, -3.27 | +1.03, -0.16 | -639, -294 |
| A | +0.60, -0.70 | -13.4, -5.14 | +0.40, -0.79 | -635, -291 |
| B | -0.11, +0.01 | -13.9, -4.64 | -0.33, -0.06 | -638, -289 |
| power law | +0.002, 0.000 | -2.71, -0.82 | +0.011, +0.002 | -192, -94 |

(IMRI end values depend on the time grid near plunge; use rms for IMRI.)

Free phase fits (coefficients plus m1, m2, a, p0, e0, phases free; CPU, phases only). All reach about 1e-3 rad
except at a = +0.9; the m1 shift (bias proxy, M = 1e6):

| point | A | B |
|---|---|---|
| idx1 (a=-0.5) | +77.5 | +18.4 |
| idx2 (a=0) | +52.8 | +11.1 |
| idx3 (a=+0.5) | +84.7 | +20.6 |
| idx10 (a=-0.9) | +59.9 | +5.2 |
| idx19 (a=+0.9) | +0.08 (fails: Phi_r 0.39 rad left) | -0.18 (fails: 0.42 rad) |

B keeps m1 3-10x closer than A where it works; nothing with leading shapes works at a = +0.9. A 0PA free fit on the
same proxy was not run, so these are not yet compared with 0PA.

### 4.5 Other fitting forms (check_models/fit_forms.py)

All of the form `X += eta * shape_X * R_X(v)`:

| form | params | R_X |
|---|---|---|
| B | 4 | C_X |
| J5 | 5 | C_X (1 + k v) |
| PL6 | 6 | C_X (10/p)^n, n shared by the fluxes and by the frequencies |
| PL | 8 | C_X (10/p)^n_X |
| LIN | 8 | C_X (1 + k_X v) |
| PADE | 8 | C_X / (1 - k_X v) |
| SVD_K | K | sum_j C_j b_jX(v) (section 4.6) |

### 4.6 SVD basis (idea: learn the 1PA shapes)

Construction (`check_models/fit_forms.py`, `build_basis`):

1. Training box: a in 23 values [-0.99, 0.99], chi2 in 5 values [0, 1], e in 6 values [0.05, 0.5], v in 60 values
   for p from 31 down to 5.
2. One row per (a, chi2, e): `[r_pdot(v_1..v_60), r_edot(...), r_Omphi(...), r_Omr(...)]` (240 columns,
   690 rows), with `r_X = 1PA_X / shape_X`.
3. Divide each channel block by its rms; SVD `M = U S V^T`.
4. Each right singular vector is, per channel, an exact cubic in v; refit it as one and put the channel scale back:
   `b_jX(v) = beta[j,X,0] + beta[j,X,1] v + beta[j,X,2] v^2 + beta[j,X,3] v^3`.
5. Deviation (SVD_K): `X += eta * shape_X(p, e) * sum_{j=1..K} C_j b_jX(v)`. Free parameters: C_1..C_K only; the
   table beta (K x 4 channels x 4 powers) is fixed.

Normalised singular values 1, 0.19, 0.093, 0.011, 0.009, 0.001; cumulative energy 0.957, 0.992, 0.9998,
0.99992, 0.999999.

The basis (SVD_4), coefficients of v^0, v^1, v^2, v^3:

```
b1: pdot  +0.37632  +0.081864 -2.1385   +15.567  | edot +0.01792  +0.001718 -0.11166  +1.0619  | Om_phi -0.1008  -0.12129 +0.079166 -1.8224 | Om_r -0.1008  -0.12129 +0.38157  -1.2003
b2: pdot  +1.0068   -1.3032   -3.8859   +14.716  | edot +0.047944 -0.062627 -0.17513  +0.70948 | Om_phi -0.26969 +0.73205 -0.5469   +5.4386 | Om_r -0.26969 +0.73205 +0.26215  +3.4601
b3: pdot  +0.94918  +1.703    -4.367    -33.039  | edot +0.045199 +0.10239  -0.22295  -1.6137  | Om_phi -0.25425 -0.24301 -0.14125  +5.9525 | Om_r -0.25425 -0.24301 +0.62149  +4.3438
b4: pdot  -1.2065   +1.7298   +12.221   +0.23448 | edot -0.057452 +0.12368  +1.0052   -3.7428  | Om_phi +0.32317 -1.1662  -5.1156  +17.942 | Om_r +0.32317 -1.1662  -6.0851  +15.803
```

Expected coefficients depend almost only on spin (EMRI idx0 and IMRI idx0, q differing by 100:
(-26.76, -4.04, 0.90, 0.43) vs (-26.72, -3.94, 0.92, 0.44); EMRI idx4: (-0.94, 2.25, 1.56, 0.22)).

### 4.7 Dephasing at the injection for all forms (expected coefficients; rad, end Phi_phi, Phi_r)

| form (n) | EMRI idx0 a=-0.9 | EMRI idx7 a=0 | EMRI idx4 a=+0.9 | EMRI idx19 a=+0.9 | IMRI idx0 a=-0.9 | IMRI idx4 a=+0.9 |
|---|---|---|---|---|---|---|
| 0PA (0) | +6.88, +3.32 | +8.71, +4.18 | +4.24, +3.01 | +4.99, +3.21 | +275, +93 | -299, -85 |
| B (4) | -0.110, +0.010 | -0.329, -0.060 | -14.3, -4.65 | -13.4, -4.71 | -48.3, -7.3 | -495, -273 |
| J5 (5) | -0.017, +0.028 | -0.005, +0.033 | -11.6, -3.86 | -10.7, -3.79 | -5.5, +2.0 | -117, -36 |
| PL6 (6) | +0.007, +0.006 | +0.064, +0.026 | -2.93, -0.83 | -2.30, -0.71 | +17.8, +4.4 | -143, -80 |
| SVD_2 (2) | +0.007, -0.038 | +0.383, +0.037 | -6.83, -2.29 | -5.82, -2.06 | -6.2, -2.3 | -157, -91 |
| SVD_3 (3) | +0.031, -0.027 | +0.044, -0.050 | -2.57, -1.07 | -1.93, -0.86 | +0.46, -0.24 | -21.8, -8.6 |
| **SVD_4 (4)** | **+0.004, -0.002** | **+0.056, -0.026** | **+0.23, -0.02** | **+0.40, +0.04** | **-0.047, -0.014** | **-9.9, -3.5** |
| PL (8) | +0.002, 0.000 | +0.012, +0.002 | -2.79, -0.82 | -2.25, -0.71 | +8.5, +1.7 | -142, -80 |
| PADE (8) | +0.000, 0.000 | +0.003, +0.001 | -6.10, -1.95 | -5.35, -1.83 | +3.1, +0.6 | crashed (pole) |

SVD_4 is the best below 5 parameters and the only form that works at a = +0.9, but still above 0.05 rad there.
SVD_2 (99.2% energy) is not enough: energy is variance (0.8% left = about 9% rms in amplitude), weighted towards
large retrograde ratios, and phase errors accumulate over the inspiral.

### 4.8 The objection, and the generalisation now being tested (EMRI_v_3/, branch PN_v_3)

**Objection:** the SVD basis is learned from the SK 1PA itself, so the template is given the 1PA shapes in advance;
it may not absorb a post-adiabatic effect that differs from the SK 1PA. Options under test (job 713215, results in
`EMRI_v_3/results/`, logs in `EMRI_v_3/logs/`):

```
Option 2 (in flux.py on PN_v_3), 5 parameters, only PN structure assumed:
    X += eta * shape_X(p, e) * (C_X + K3 v^3)
    additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, C_Omphi, C_Omr, K3]

Option 1 (forms_v3.py only), shape = the template's own 0PA rate, no SK numbers:
    REL5_0 : X += eta * X_0PA(p, e) * (c_X + K v^3)         (as first proposed)
    REL5_1 : X += eta * X_0PA(p, e) * (c_X v + K v^4)       (PN-consistent: 1PA/0PA starts at relative v^1)
    REL8_1 : X += eta * X_0PA(p, e) * (c_X v + k_X v^4)

Option 3 (forms_v3.py only), a basis that knows less:
    SVD basis rebuilt from SK 1PA cubics whose four order terms are each multiplied by an independent U(1-f, 1+f),
    4 random draws per (a, chi2, e); f = 0, 0.2, 0.5, 1.0; K = 3..6 directions.
```

Smoke test (rate level, max over the four rates of the relative rms remainder):

| form | EMRI idx4 (a=+0.9) | IMRI idx0 (a=-0.9) |
|---|---|---|
| B | 0.510 | 0.172 |
| V3 (option 2) | 0.563 | 0.111 |
| REL5_0 / REL5_1 / REL8_1 (option 1) | 0.284 / 0.232 / 0.110 | 0.311 / 0.287 / 0.238 |
| SVD f=0, K=4 / 5 | 0.026 / 0.006 | 0.001 / 0.000 |
| SVD f=0.5, K=4 / 5 | 0.254 / 0.036 | 0.019 / 0.008 |
| SVD f=1.0, K=4 / 5 | 0.311 / 0.051 | 0.046 / 0.011 |

Singular values of the perturbed bases (normalised): f=0.2: 1, 0.196, 0.099, 0.062, 0.052, 0.019;
f=0.5: 1, 0.226, 0.151, 0.128, 0.117, 0.041; f=1.0: 1, 0.319, 0.287, 0.250, 0.139, 0.074.
**Results of job 713215 (exit 0).** Worst rate remainder per spin over all 50 tracks (max over the four rates):

| grid a | B | V3 | REL5_1 | REL8_1 | SVD f=0 K=4 | f=0 K=5 | f=0.5 K=5 | f=1.0 K=5 |
|---|---|---|---|---|---|---|---|---|
| EMRI -0.9 | 0.067 | 0.048 | 0.178 | 0.108 | 0.005 | 0.005 | 0.003 | 0.003 |
| EMRI 0 | 0.106 | 0.082 | 0.176 | 0.033 | 0.011 | 0.008 | 0.014 | 0.018 |
| EMRI +0.5 | 0.202 | 0.195 | 0.197 | 0.043 | 0.010 | 0.009 | 0.023 | 0.038 |
| EMRI +0.9 | 0.510 | 0.563 | 0.232 | 0.110 | 0.042 | 0.030 | 0.054 | 0.065 |
| IMRI -0.9 | 0.172 | 0.111 | 0.287 | 0.238 | 0.005 | 0.005 | 0.009 | 0.011 |
| IMRI +0.9 | 0.761 | 0.807 | 0.305 | 0.202 | 0.029 | 0.021 | 0.042 | 0.053 |

End dephasing at the injection (rad, Phi_phi, Phi_r), expected coefficients from the rate fit:

| form (n) | EMRI idx0 a=-0.9 | EMRI idx7 a=0 | EMRI idx4 a=+0.9 | EMRI idx19 a=+0.9 | IMRI idx0 a=-0.9 | IMRI idx4 a=+0.9 |
|---|---|---|---|---|---|---|
| 0PA (0) | +6.88, +3.32 | +8.71, +4.18 | +4.24, +3.01 | +4.99, +3.21 | +275, +93 | -299, -85 |
| B (4) | -0.110, +0.010 | -0.329, -0.060 | -14.3, -4.65 | -13.4, -4.71 | -48.3, -7.3 | -495, -273 |
| V3, option 2 (5) | -0.114, -0.001 | -0.320, -0.070 | -13.8, -4.63 | -12.4, -4.41 | -45.6, -7.5 | -484, -268 |
| REL5_0, option 1 (5) | +0.318, -0.040 | +0.573, +0.013 | +1.26, +0.12 | +1.00, -0.01 | +68.6, +10.5 | +71.7, +24.8 |
| REL5_1, option 1 (5) | +0.307, -0.057 | +0.541, -0.018 | +0.388, -0.264 | +0.165, -0.381 | +72.4, +10.8 | -39.7, -19.6 |
| REL8_1, option 1 (8) | -0.017, +0.003 | -0.059, -0.004 | +0.033, +0.085 | +0.196, +0.149 | -24.4, -4.1 | +14.3, +8.3 |
| SVD f=0 K=4 | +0.004, -0.002 | +0.056, -0.025 | +0.056, -0.075 | +0.075, -0.063 | -0.101, -0.024 | -5.04, -1.70 |
| SVD f=0 K=5 | +0.001, -0.000 | +0.014, -0.003 | +0.069, +0.039 | +0.103, +0.077 | -0.078, -0.004 | +2.59, +1.15 |
| SVD f=0.2 K=5 | +0.001, +0.000 | +0.018, -0.001 | +0.475, +0.190 | +0.370, +0.169 | +0.97, +0.29 | +29.6, +12.1 |
| SVD f=0.5 K=5 | -0.002, +0.002 | +0.038, -0.006 | +0.329, +0.188 | +0.250, +0.155 | -0.84, +0.11 | +21.3, +9.5 |
| SVD f=1.0 K=5 | -0.004, +0.003 | +0.056, -0.008 | +0.461, +0.245 | +0.364, +0.194 | -1.84, -0.00 | +28.4, +12.6 |

(SVD f=0 K=4 at EMRI idx4 differs from section 4.7, +0.23/-0.02, because forms_v3 weights each rate by 1/rms of the
rate while fit_forms weighted the shape-normalised ratio: the rate-fitted coefficients are not phase-optimal, so
0.1 rad level differences reflect the fitting objective, not the capacity of the form. K = 6 is often worse than
K = 5 in phase for the same reason.)

Conclusions:
- **Option 2 (V3) does not help**: within a few % of B everywhere; one shared v^3 slope cannot fix a = +0.9.
- **Option 1 (0PA-relative)**: the 5-parameter versions are worse than B for a <= 0 and only partly fix a = +0.9
  (EMRI 0.2-0.4 rad); IMRI fails (tens of rad). REL8_1 (8 parameters, no SK numbers at all) is the best generic form
  for EMRI (<= 0.2 rad at all four points) but still 4-24 rad for IMRI.
- **Option 3**: with K = 5 a basis that assumes only "shape x cubic in v with SK-sized, randomly perturbed order
  terms" keeps a < +0.9 at about 0.002-0.06 rad for EMRI, and costs about 5x at a = +0.9 (0.25-0.47 rad vs 0.07-0.10
  for the exact basis). f = 0.5 and f = 1.0 behave alike, i.e. the extra direction already captures the freedom.
  IMRI a = +0.9 stays at 20-30 rad (exact basis: 2.6-5 rad).
- None of the generalisations reaches the 0.05 rad criterion at a = +0.9; the exact SK basis (f = 0, K = 4-5) comes
  closest. The rate fit is a limitation: a phase fit of the coefficients at the injection would show each form's
  real capacity.

### 4.9 Bias proxy for REL5_1 (EMRI_v_3/phase_fit_v3.py, jobs 713431/713432, exit 0)

Free phase fit (deviation coefficients + m1, m2, a, p0, e0, Phi_phi0, Phi_r0, phases only, no noise weighting),
compared up to 0.99 of the 1PA signal's end. Shift = best fit - injection:

| point | form | m1 | m2 | a | p0 | rms residual after fit (Phi_phi, Phi_r) |
|---|---|---|---|---|---|---|
| EMRI idx0 (a=-0.9) | 0PA | +57.7 | +3.8e-5 | +1.7e-4 | -5.9e-4 | 0.0026, 0.0018 |
| | B | +23.9 | +1.8e-5 | +5.6e-5 | -2.0e-4 | 0.0005, 0.0004 |
| | REL5_1 | -26.9 | +2.8e-4 | -4.2e-5 | +2.2e-4 | 0.0009, 0.0011 |
| EMRI idx7 (a=0) | 0PA | +46.3 | +6.2e-5 | +9.3e-5 | -3.9e-4 | 0.0034, 0.0018 |
| | B | +37.0 | +1.5e-5 | +5.9e-5 | -2.7e-4 | 0.0010, 0.0009 |
| | REL5_1 | -3.5 | +7.7e-5 | +2.3e-6 | +1.7e-5 | 0.0013, 0.0015 |
| EMRI idx4 (a=+0.9) | 0PA | +33.5 | +1.8e-4 | +2.9e-5 | -1.7e-4 | 0.041, 0.026 |
| | B | -34.0 | +2.4e-4 | -7.7e-7 | +2.5e-4 | 0.0087, 0.015 |
| | REL5_1 | -3.8 | -1.7e-6 | -6.4e-6 | +3.8e-5 | 0.0002, 0.0001 |
| EMRI idx19 (a=+0.9) | 0PA | +43.7 | +2.8e-4 | +3.5e-5 | -2.1e-4 | 0.038, 0.022 |
| | B | -36.3 | +3.5e-4 | -3.9e-6 | +2.8e-4 | 0.0071, 0.012 |
| | REL5_1 | -2.4 | -3.5e-6 | -4.8e-6 | +3.2e-5 | 0.0006, 0.0004 |
| IMRI idx0 (a=-0.9) | 0PA | +1709 | -0.51 | +7.8e-3 | -0.041 | 0.015, 0.012 |
| | B | +260 | +2.49 | +8.7e-3 | -0.024 | 0.0029, 0.0041 |
| | REL5_1 | +23.3 | +0.27 | +3.3e-4 | +1.6e-3 | 0.0004, 0.0006 |
| IMRI idx4 (a=+0.9) | 0PA | +352 | -0.21 | +3.7e-4 | -4.0e-3 | 0.016, 0.018 |
| | B | +2356 | -2.76 | +1.4e-3 | -0.037 | 0.0029, 0.0045 |
| | REL5_1 | -195 | -0.12 | -4.6e-4 | +3.0e-3 | 0.0002, 0.0003 |

REL5_1 free-fit coefficients (c_p, c_e, c_Omphi, c_Omr, K):

    EMRI idx0  +0.330, +11.82, +1.939, +1.020, +62.84
    EMRI idx7  -0.600, +3.733, +0.355, +0.273, +68.87
    EMRI idx4  -3.196, -0.086, -1.511, -1.458, +67.05
    EMRI idx19 -2.347, -0.009, -1.214, -1.219, +63.11
    IMRI idx0  +6.960, +5.531, +3.872, +3.823, +67.13
    IMRI idx4  -1.512, -6.911, -0.821, -0.817, +55.73

- REL5_1 has the smallest m1 shift at every point: 2x (EMRI idx0), 9-18x (EMRI idx4, 7, 19), 73x (IMRI idx0) and
  1.8x (IMRI idx4) smaller than 0PA, and a, p0 shifts smaller as well except at IMRI idx4 (about equal). It also
  fits the phases best (rms 1e-4 to 1e-3 rad). B helps at a <= 0 EMRI, flips the sign at a = +0.9, and is 7x worse
  than 0PA at IMRI idx4.
- The shared v^4 coefficient K comes out at +56 to +69 at all six points (EMRI and IMRI, all spins), so it may be
  close to universal; fixing it would leave 4 parameters. c_Omphi and c_Omr are nearly equal at a = +0.9.
- Caveat: this proxy has no noise weighting, amplitude or sky parameters. Its 0PA m1 shift at EMRI idx4 (+33.5)
  has the opposite sign to the real 0PA LM bias (-8.71), so it ranks forms; the GPU LM fit decides.

---

## 5. Ideas considered and not built

- Scaling each PN order of the 1PA (ppE/TIGER-like lambda_k): exact at lambda = 1 but uses the full 1PA structure.
- lambda x 1PA(chi2_eff): essentially the signal model, an upper bound only.
- Shifting the 0PA model's own arguments (eta d_a dOmega/da + eta d_p dOmega/dp): degenerate with a and p0, and the
  p-dependence does not match the 1PA.
- Priors on the deviation coefficients inside the LM.
- Separate SVD bases per spin region (retrograde / prograde), or a basis weighted by dephasing instead of rate size.
- Hold-out test: train the basis without a > 0.5 and test at a = +0.9; test against a non-SK 1PA model.

---

## 6. Open questions / next steps

1. GPU LM fit with REL5_1 (or REL4 with K fixed near +65), seeded from the free-fit coefficients above; then Fisher
   and critical SNR.
1b. Phase-optimal coefficients for SVD f=0/0.5 K=4-5 and REL8_1.
2. Free phase fit (bias proxy) for SVD_4 and the winners, alongside a 0PA free fit on the same proxy.
3. GPU LM fit with the winning deviation, seeded from the expected coefficients, then Fisher and critical SNR.
4. How general must the deviation be to be defensible (vs absorbing only SK-like 1PA)?
5. idx4 Fisher for the PN_freq 0pa_fit result (needs branch PN_power_deviation).

---

## 7. Branch map of SuperKludge_r (deviation slots after deviation_included at index 4)

| branch | deviation |
|---|---|
| hybrid | C_p, C_e (2.5PN pdot/edot) + del_0_p, del_0_e (multiplicative) |
| dev_a_pe | del_0_p, del_0_e (multiplicative pdot/edot) |
| PN_power_deviation | C_p, C_e, A_p, A_e, n_p, n_e |
| PN_freq_evol | committed by the user; last state: SVD_4, C_1..C_4 (earlier: model switch 0-3, PN_freq / A / B / power law) |
| PN_v_3 | option 2: C_p, C_e, C_Omphi, C_Omr, K3 (uncommitted) |

Python: `/home/svu/e1583490/bias_inference_emri/.venv/bin/python`. PBS project `CFP05-CF-131`.

## 8. File map

| path | content |
|---|---|
| EMRI_1PA_freq/ | GPU LM runs of PN_freq (lm.py, run_lm_arms.py, batch_lm.sh, fisher_at_best.py) |
| consistency_check/ | leading-order check, idx0 trajectories and rate plots |
| check_models/README.md | detailed log of sections 4.4-4.7 |
| check_models/sk_orders.py | exact split of each 1PA function into PN orders |
| check_models/pn_orders.py, plot_pn_orders.py | order importance over the grids, plots/ |
| check_models/fit_forms.py | all fitting forms and the SVD basis |
| check_models/expected_values.py, phase_fit.py, families.py | families A, B, power law (need the old model switch) |
| EMRI_v_3/forms_v3.py, check_v3_wiring.py, batch_v3.sh | options 1, 2, 3 (5/5 wiring checks pass) |
