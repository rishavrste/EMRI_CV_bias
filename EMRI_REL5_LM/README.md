# EMRI_REL5_LM — 1PA signal vs 0PA + REL5_1 deviation, LM best fits, Fisher, critical SNR

Goal: a deviation with ≤ 5 free coefficients that absorbs the 1PA physics, so that the best fit of the
0PA template lands as close to the injection as possible (smallest absolute bias).

## 1. Setup

| item | value |
|---|---|
| signal | 1PA, `(evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False)`, no deviation |
| template | 0PA `(False, False, False)` + REL5_1 deviation, 5 coefficients free |
| deviation (SuperKludge_r @ **PN_v_3**, `few/trajectory/ode/flux.py`) | `X += eta · X_0PA(p, e) · (c_X v + K v^4)`, `v = p^-1/2`, `eta = m1 m2/(m1+m2)^2`, X = pdot, edot, Omega_phi, Omega_r |
| `additional_args` | `[chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, c_p, c_e, c_Omphi, c_Omr, K]` |
| fitted (14) | m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0 + c_p, c_e, c_Omphi, c_Omr, K |
| held at injection | distance, spin orientation, Phi_theta0, chi2 (secondary spin) |
| data | EMRI grid (25 points, `idx = e0_index·5 + a_index`), T = 2.5 yr, dt = 10 s, 2nd-gen TDI A+E, DC bin dropped, "rishav" setup, signal SNR 20 |
| LM | `lm.py`, LAMBDA0 1e-6, SEF stable derivatives (order 6, 12 deltas, re-scanned every 3 steps), absolute step grid near zero for the coefficients and for a |
| 0PA / PN references | `EMRI_1PA_PN_DEV/results/best_fits.json` and its `results/fisher_at_best/{0pa,pn}` (same TDI setup) |

**Do not touch `flux.py` on PN_v_3** while any REL5_1 GPU job runs — the template is the
editable `few` install. `run_lm.py` and `fisher_at_best.py` stop if SuperKludge_r is not on PN_v_3.

## 2. Files

| file | what it does |
|---|---|
| `config.py` | controls, paths, parameter lists, arms (`full`, `ladder`, `reseed`) |
| `model.py`, `lm.py` | waveform, inner product, SEF derivatives; damped LM climb |
| `check_model.py`, `batch_check.sh` | GPU wiring check: GR limit (all coefficients 0 = plain 0PA), response, every slot reaches flux.py |
| `make_seeds.py` | CPU seeds: injected physical parameters + coefficients from a phase fit to the 1PA phases (`EMRI_v_3`) -> `results/seeds.json` |
| `run_lm.py`, `batch_lm.sh` | LM per point: `full` at T = 2.5 yr; `ladder` only if `full` ends below the 0PA overlap -> `results/full/lm_idx{p}.json` |
| `run_lm.py --coeff-seed-from J`, `batch_reseed.sh` | `reseed` arm: injected physical parameters + point J's fitted coefficients -> `results/reseed/` |
| `fisher_at_best.py`, `batch_fisher.sh` | 14-parameter Fisher at each best fit, repeated with SEF steps doubled -> `results/fisher_at_best/fisher_idx{p}.json` |
| `make_summary.py` | fits, conditioning, critical SNR vs 0PA and PN -> `results/summary.md` |
| `plot_codes/` | all plots and tables below -> `plots/` (each script regenerates its own outputs) |

## 3. Runs

| job | what | result |
|---|---|---|
| rel5_check | `check_model.py` | passed: GR limit exact, all five slots active |
| 713537–713541 (rel5_a … rel5_e) | LM, all 25 points, `full` arm | done, exit 0; **all 25 beat the 0PA overlap at full T, the ladder was never needed** |
| 713791 (rel5_rs3) | idx3 reseeded from idx8's coefficients | **failed**: O −0.048 → 0.245 after 398 iterations (c_e ≈ 8e4, qS moved by 1.0); the `full` idx3 fit (O 0.999999976) stands |
| 714132 (rel5_fisher) | Fisher at all 25 best fits + summary | done in 30 min, exit 0 |

## 4. Results

### 4.1 Overlap
Every REL5_1 fit has a higher overlap than 0PA. Gain `(1 − O_0PA)/(1 − O_REL5)`:

```
            e0=0.1     e0=0.2     e0=0.3     e0=0.4     e0=0.5
a=-0.9      1120×      1050×      1700×      3900×      1450×
a=-0.5       999×      1020×      1420×      12.8×     11500×
a= 0.0      1570×      1970×      52.7×      3160×      87.5×
a=+0.5      4930×      4440×     18100×       164×     12600×
a=+0.9      8120×     32700×     44100×     11900×     91400×
```
Against PN, REL5_1 is lower at 5 points, all a ≤ −0.5: idx16 (0.0036×), idx6 (0.50×), idx1 (0.52×),
idx11 (0.64×), idx21 (0.98×). REL5_1 does not nest PN, so this is allowed. idx16, though, is almost surely
an unfinished climb (stopped on rel_tol after 12 iterations at 1 − O = 1.8e-6). idx12 (6.8e-7) and
idx18 (1.4e-6) also stopped early. A tighter re-climb from their current fits has been proposed, not run.

### 4.2 Fitted coefficients (spin-structured, like C_p in the PN model)
c_p ≈ +15…+29 for a ≤ +0.5 and ≈ −1 at a = +0.9; K ≈ −45…−64 at a = −0.9 and ≈ +67…+70 at a = +0.9;
c_e is poorly determined at e0 = 0.1. idx3 and idx13 (a = +0.5, e0 0.1 / 0.3) sit on a different
solution (c_Omphi ≈ c_Omr ≈ 21, K ≈ 33–37) with m1 bias +28 / +32. Map: `plots/fit/coefficients.png`.

### 4.3 Absolute bias, per parameter
`r = |θ_0PA − θ_inj| / |θ_REL5 − θ_inj|`, r > 1: REL5_1 closer to the injection (phases wrapped).
Full point × parameter table: `plots/tables/r_sigma.md`, `plots/fit/r_table.png`.

| | m1 | m2 | a | p0 | e0 | qS | phiS | Phi_phi0 | Phi_r0 |
|---|---|---|---|---|---|---|---|---|---|
| median r | 9.61 | 0.277 | 7.11 | 12.7 | 1.26 | 87.5 | 71.5 | 14.2 | 38.4 |
| r > 1 (of 25) | 24 | 10 | 21 | 23 | 18 | 25 | 24 | 25 | 25 |

- m1 bias goes from −110…+92 (0PA) to about −6…+9 (idx3, 13: +28, +32).
- **m2 is the one parameter that gets worse** (typically −5e-4 vs 0PA's +1e-4): c_p trades against the
  mass ratio. Where c_p stays small (idx12, 16, 18, 22, a = +0.9 row) m2 is as good as 0PA or better.
- All 5 intrinsic parameters improved at 7 points; m2 alone worse at 9; idx3 is the only point where
  m1, a and p0 all get worse.
- vs PN: REL5_1 has the smaller |bias| in m1 23/25, m2 12, a 24, p0 25, e0 16, qS 18, phiS 21,
  Phi_phi0 12, Phi_r0 18.

### 4.4 Fisher and the cost of the extra coefficients
- All 25 Fishers (and the doubled-step ones) are positive definite, but barely: Jacobi-rescaled
  eigenvalues 1e-12…11, condition number 1e10–1e13 (0PA 1e7–1e10, PN 1e8–1e11). The smallest eigenvalue
  moves up to 5× when the steps are doubled; the critical SNR moves < 10 % (worst idx22, 31 %).
- Degeneracies: corr(c_p, m1) = +0.84…+0.999; corr(c_Omphi, c_Omr) = +0.92…+1.000 except idx12, 18, 22;
  corr(c_p, m2) and corr(c_p, p0) reach −0.99 at idx3, 12, 18, 19, 22.
- Marginal σ, `σ_REL5 / σ_0PA` (SNR 20, own Fisher, coefficients marginalised) — median over the grid:

| m1 | m2 | a | p0 | e0 | qS | phiS | Phi_phi0 | Phi_r0 |
|---|---|---|---|---|---|---|---|---|
| 86 | 505 | 59 | 102 | 42 | 1.28 | 1.19 | 1.52 | 1.66 |

  σ grows at every point for every parameter; the intrinsic ones by 4.6–1480×, the sky and phases by
  only 1.1–2×. Largest at a = +0.5. PN (+2 coefficients) by comparison: σ_m1 6.5×, σ_m2 16×, σ_a 5.8×,
  σ_p0 5.7×, σ_e0 9.7×. Per point: `plots/tables/r_sigma.md`, `plots/extra_dof/`.

### 4.5 Critical SNR
`ρ_crit = 20 D_CL / D`, `D² = bᵀ Σ_S⁻¹ b` over the intrinsic (m1, m2, a, p0, e0), every other fitted
parameter marginalised, `D_CL² = χ²₅` quantile. (EMRI_1PA_PN_DEV's own maps used all nine physical
parameters, so its 0PA / PN numbers differ from these.)

ρ_crit at 90 % CL, 0PA / PN / REL5_1:
```
            e0=0.1             e0=0.2             e0=0.3             e0=0.4             e0=0.5
a=-0.9   0.77/10.3/1220     0.69/8.69/1260     0.62/4.2/1480      0.55/6.24/1880     0.50/5.63/2470
a=-0.5   0.71/10.8/1070     0.65/9.26/1130     0.58/7.86/1080     0.52/6.83/1.5e5 *  0.47/6.11/1780
a= 0.0   0.67/12.5/598      0.62/10.6/816      0.55/9.08/1020     0.50/7.84/1190     0.45/6.93/1950
a=+0.5   0.75/18.4/509      0.69/14.1/457      0.63/11.7/496      0.57/10/664        0.52/8.8/811
a=+0.9   2.1/39.7/88.7      1.96/22.8/77.5     1.63/15.8/67.8     1.56/13.4/76       1.33/10.9/71.6
```
\* idx16: unfinished fit sitting almost exactly on the injection; not meaningful.
Worst 1D |b|/σ at SNR 20: 0PA 0.013–1.6, PN 0.005–1.16, REL5_1 0.0012–0.076.
99 % CL and the per-point table: `results/summary.md`, `plots/tables/comparison.md`.

**Reading:** ρ_crit rises ~100–2000× over PN, but mostly because σ grows (median 60–500×) rather than
because the bias falls (median 7–13× for m1, a, p0; m2 worse). REL5_1 fits much better and moves most
parameters toward the injection, at the price of near-degenerate coefficients that make the masses and
p0 far harder to measure.

## 5. Side studies

### 5.1 idx3 reseed (713791) — negative
idx8's coefficients at idx3's injection give O = −0.048; LM ran off to |coefficients| ~ 1e3–1e4. Seeding
across points does not move idx3 off its solution.

### 5.2 Shifting the bias to the coefficients: REL5_1S (CPU phase-fit proxy only) — negative
Hypothesis: c_p v is nearly constant along a track (v ≈ 0.30–0.33), so it is degenerate with the mass
ratio. Test form `c_X (v − v0) + K v^4`, v0 = p0^-1/2 of the template, added to `EMRI_v_3/forms_v3.py`
and `phase_fit_v3.py` only (flux.py untouched; backups in the session scratchpad). Free-fit m1 shift
(phase proxy, ranking only):

| point | 0PA | REL5_1 | REL5_1S |
|---|---|---|---|
| idx0 | +57.7 | −26.9 | +82.8 |
| idx3 | +42.4 | −10.2 | −11.8 |
| idx7 | +46.3 | −3.5 | −12.4 |
| idx8 | +44.3 | −0.65 | −12.8 |
| idx16 | +92.6 | −6.2 | +37.2 |

REL5_1S matches the phases 3–9× better but the physical parameters drift more; idx4 and idx19 did not
converge. REL5_1S2 (K term shifted too) is ill-conditioned (rate-fit coefficients ~600, −5800), dropped.
REL5_1 stays the best tested form.

### 5.3 Prior on the coefficients — not done
Discussed; a Fisher + prior scan `Δθ = (Γ + P)⁻¹[⟨∂h|r⟩ + P(μ − θ)]` was proposed as a cheap first look.
Not requested.

## 6. Plots (`plots/`, scripts in `plot_codes/`)

| file | content | script |
|---|---|---|
| `fit/mismatch.png` | 1 − O of all three templates; REL5_1 gain over 0PA and PN | `plot_fit_maps.py` |
| `fit/coefficients.png` | the five fitted coefficients on the grid | `plot_fit_maps.py` |
| `fit/r_factor.png` | worst / best per-parameter r and count of improved parameters, vs 0PA and PN (r defined as REL5/ref here) | `plot_fit_maps.py` |
| `fit/r_table.png`, `tables/r_sigma.md` | **r = \|b_0PA\|/\|b_REL5\| per point and parameter**, and σ_REL5/σ_0PA | `r_sigma_table.py` |
| `extra_dof/sigma_table.png` | σ_REL5/σ_0PA, every point × physical parameter | `r_sigma_table.py` |
| `fit/overall_r.png`, `tables/overall_r.md` | one overall r per point (geometric mean and 0PA-Fisher metric; inverse convention, REL5/0PA) | `overall_r.py` |
| `critical_snr/snr_crit_{nD,1D}_{90,99}CL.png` | ρ_crit of the three templates and the REL5_1 ratios | `plot_critical_snr.py` |
| `extra_dof/sigma_bias.png`, `summary.png`, `conditioning.png` | σ growth vs bias reduction for PN (+2) and REL5_1 (+5); condition numbers and correlations | `plot_extra_dof.py` |
| `bias_1D/violin_grid.png`, `per_param/`, `zoom/` | half-violins, 0PA left, REL5_1 right; `zoom/` keeps the bias bars visible under the wide REL5_1 σ | `plot_violins.py` |
| `tables/comparison.md` | per point: 1 − O, ρ_crit 90/99 %, worst 1D b/σ, intrinsic biases, all three templates | `make_table.py` |

Python for the plots: `/home/svu/e1583490/bias_inference_emri/.venv/bin/python` (needs scipy).

## 7. Open

1. Re-climb idx16 (and idx12, 18) from their current fits with a tighter rel_tol.
2. idx3 / idx13 alternative solution: still the only cells where REL5_1 is worse than 0PA in m1.
3. Reduce the degeneracy (c_Omphi ≈ c_Omr; c_p ↔ mass ratio) — e.g. tie c_Omphi = c_Omr, or a prior.
4. The `fit/overall_r` products use the inverse r convention; the agreed r is the one in `r_table`.
