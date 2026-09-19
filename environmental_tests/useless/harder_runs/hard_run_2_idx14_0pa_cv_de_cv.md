# run_2 / idx 14 — 0PA, CV → DE → CV

`hard_run_2_idx14_0pa_cv_de_cv.json` (results) · `hard_run_2_idx14_0pa_cv_de_cv.log` (full run log)

The vacuum-recovery answer for the one source the population run could not climb. Produced by
`harder_environmental/cv_de_cv.py` (PBS job 616035, 9.57 h on one GPU), copied here unchanged.

## The source

| | |
|---|---|
| run / idx | `run_2` / 14 |
| m1 | 1 046 286.8476 M☉ |
| m2 | 50 M☉ |
| a | 0.9 |
| p0 | 14.7724747 |
| e0 | 0.2 |
| T, dt | 3.646218 yr, 10 s |
| distance | 5.188972 Gpc |
| SNR | 100.00 |
| injected deviation | A_PM = 2.0088707e-05, n_PM = 8, A_GC = 1e-12, n_GC = 4 |
| channels | A, E |

The environment is in the **injection only**; every template is vacuum. Recovery model here is
**0PA**, the 9 vacuum parameters, with no `C_p`/`C_e`.

## Why this run exists

The population run climbs from the injected point with Levenberg–Marquardt alone. On this source
that stalls at **ov = 0.7534** — m1, m2, a, p0, e0, qS and phiS are recovered to seven digits and
the two orbital phases sit in the wrong basin, which a damped linear step cannot cross. That is an
optimiser failure, not a statement about the model, so the fix is a better optimiser on the same
likelihood.

Three stages:

1. **The complete CV step** from the injected point — `dθ = Γ⁻¹⟨∂h|s−h⟩`, Jacobi-preconditioned,
   taken unconditionally and undamped. This is what the CV formalism predicts, and it is *not*
   what an LM iteration gives; `lm_climb` shrinks the step by `λ·diag(Γ)` and accepts it only on a
   positive Nielsen gain ratio. (Setting `λ = 0` there does nothing useful: a rejected step does
   `λ ← λ·LAMBDA_UP`, which leaves zero at zero, so the climb exhausts without moving. Hence the
   standalone `cv_step_exact`.) χ² is allowed to rise here — the point of the stage is the CV
   prediction and the box it defines, not a descent.
2. **Differential evolution** in a box of ±15σ about that point, σ from the Fisher evaluated
   *there*. `--full-phase-bounds`, so qS, phiS, Phi_phi0 and Phi_r0 span their full physical
   ranges instead of a Fisher width. 1100 generations, popsize 15, sobol init, chi2 objective,
   no polish.
3. **A CV climb** from the DE winner — the full LM → Nelder-Mead → LM arrangement, 150-iteration
   cap, which takes the answer from "in the right basin" to "at the maximum".

## Result

| stage | overlap | χ² | cost |
|---|---|---|---|
| injected point | −0.0628243337 | — | — |
| 1 · CV step (exact) | 0.2222834503 | 1.3269e+04 | 41 s |
| 2 · DE (1100 gens, 281 856 evals) | 0.9090470066 | 1.7364e+03 | 8.96 h |
| 3 · CV climb (174 iters, `rel_tol`) | **0.9483672239** | **1.0281e+03** | 32 min |

Against the population run's **0.7533716417** (0PA) and 0.7929170355 (0PA+PN) on the same source.
The recovered template has SNR 99.535 against the injected 100.

## Parameters

| param | injected | CV step | DE | **CV climb** | σ | b/σ |
|---|---|---|---|---|---|---|
| m1 | 1046286.848 | 1046090.001 | 1046106.823 | **1046508.177** | 8.5426e+00 | +25.909 |
| m2 | 50 | 50.00081568 | 49.99980317 | **50.00277752** | 8.4324e-05 | +32.939 |
| a | 0.9 | 0.8998291145 | 0.8998322969 | **0.9001798588** | 7.7899e-06 | +23.089 |
| p0 | 14.7724747 | 14.77417431 | 14.77399537 | **14.77092736** | 6.8565e-05 | −22.568 |
| e0 | 0.2 | 0.2000452557 | 0.1999858638 | **0.1997660850** | 3.7421e-06 | −62.510 |
| qS | 0.2 | 0.04828571262 | 0.1588427902 | **0.1832523488** | 3.6790e-03 | −4.552 |
| phiS | 0.2 | 0.1250197188 | 6.042074343 | **6.6965802431** | 2.0156e-02 | +10.587 |
| Phi_phi0 | 0.3 | 1.080169532 | 6.282989204 | **2.7467642108** | 6.0714e-02 | +40.300 |
| Phi_r0 | 0.5 | −0.466766728 | 0.4717638243 | **−0.8882363247** | 7.3694e-02 | −18.838 |

σ is the Fisher error at the final point. Biases are angle-unwrapped: phiS, Phi_phi0 and Phi_r0 are
folded into (−π, π] before dividing by σ, because a fit a whole turn away is the same physical
answer. qS is polar and bounded to [0, π], so it is never unwrapped.

## Reading it

**This is a genuine recovery of the intrinsics.** 0.7534 → 0.9484 on a point the population run
could not move, and the sky and intrinsic parameters now sit tens of σ from the injected values
rather than the hundreds the stalled climb reported. At SNR 100 with a σ this tight, tens of σ is
still a large systematic bias — which is the CV result being measured, not a defect.

**The remaining 5% of overlap is in the phases.** Phi_phi0 is +40σ and Phi_r0 −19σ off; the
intrinsics are the parameters that have actually converged.

**The DE box was still not centred on the answer.** Stage 3 walked outside the box it started
from — +2.95 half-widths on m1, +2.68 on a, −2.83 on p0, −4.70 on e0. DE went to the box edge and
the climb carried it the rest of the way. Widening to 15σ (from 10σ) helped but did not fix it: a
Fisher σ evaluated at an ov ≈ 0.22 point is not the right yardstick for a distance of this size,
however the step to that point is taken. A box built from the converged Fisher, or a fixed
fractional box on the intrinsics, is the outstanding idea.

**Do not read σ at face value for a "measurement" claim.** The Fisher condition number at the final
point is 5.911e+17. The inverse is Jacobi-preconditioned (`cov_precond`), which is what makes it
usable at all with m1 ~ 1e6 and e0 ~ 0.2 in one matrix, but the error bars are still stiff-direction
numbers.

## The 0PA+PN answer for this source

`hard_run_2_idx14_pn_from_0pa.json` · `hard_run_2_idx14_pn_from_0pa.log`

The first attempt, `harder_environmental/results_cv_de_cv_pn.json`, ran the same three stages with
the 11-parameter model and a 10σ box and reached **ov = 0.8907** — *below* the 0PA answer above.
That ordering is impossible for a converged pair: 0PA is a strict subspace of 0PA+PN, so setting
`C_p = C_e = 0` reproduces the 0.9484 point exactly. The 0PA answer is even *inside* that run's DE
box (worst intrinsic offset 1.55 half-widths), so the box was not at fault; DE lost the basin in
the two extra dimensions, and the two models ended up on opposite sky hemispheres (qS 0.183 against
2.973). Its `C_p = −30.1`, `C_e = −72.1` are what the wrong basin needed, not a measurement of the
deviation. **That file should not be used.**

It was redone by `harder_environmental/pn_from_0pa.py` (PBS job 617186, 12.4 min), which drops DE
entirely and seeds a single LM → Nelder-Mead → LM climb from the 0PA fit above with
`C_p = C_e = 0` — the dual-seeding `cv_population.py` already applies across the population. That
makes `ov_PN ≥ ov_0PA` true by construction. The seed check confirms the two models agree about the
point: read as a PN template it gives ov = 0.9483672239, identical to the 0PA run to all ten digits.

| | overlap | χ² |
|---|---|---|
| 0PA (CV → DE → CV) | 0.9483672239 | 1.028073e+03 |
| 0PA+PN (CV → DE → CV) — *discarded* | 0.8906779624 | 2.098886e+03 |
| **0PA+PN (from the 0PA answer)** | **0.9483681337** | **1.028069e+03** |

| param | injected | seed (0PA) | 0PA+PN fit | σ | b/σ |
|---|---|---|---|---|---|
| m1 | 1046286.848 | 1046508.177 | 1046508.177 | 1.5826e+01 | +13.985 |
| m2 | 50 | 50.00277752 | 50.0027775 | 3.9786e-04 | +6.981 |
| a | 0.9 | 0.9001798588 | 0.9001798605 | 1.2556e-05 | +14.325 |
| p0 | 14.7724747 | 14.77092736 | 14.77092736 | 1.0431e-04 | −14.833 |
| e0 | 0.2 | 0.199766085 | 0.1997660798 | 9.4543e-06 | −24.742 |
| qS | 0.2 | 0.1832523488 | 0.1832577311 | 3.9329e-03 | −4.257 |
| phiS | 0.2 | 0.4133949359 | 0.4136374688 | 2.0823e-02 | +10.260 |
| Phi_phi0 | 0.3 | 2.746764211 | 2.746186419 | 6.2394e-02 | +39.205 |
| Phi_r0 | 0.5 | 5.394948982 | 5.393859749 | 8.8591e-02 | −15.682 |
| C_p | 0 | 0 | −6.617769e-05 | 6.0164e-02 | −0.0011 |
| C_e | 0 | 0 | +3.849988e-04 | 5.5223e-01 | +0.0007 |

phiS and Phi_r0 are folded into [0, 2π) here and appear unwrapped in the 0PA table above; 0.4134 is
6.6966 − 2π and 5.3949 is −0.8882 + 2π. Same point.

### Reading it

**The climb did not move.** 12 LM iterations, 10 accepted, Nelder-Mead rejected, stopped on
`lambda_exhausted`; the gain over the seed is +9.1e-07 in overlap and −0.004 in χ². Every parameter
is unchanged to eight or more digits.

**`C_p` and `C_e` came back at zero** — 0.0011σ and 0.0007σ from their injected values. Given a
correctly seeded start, the deviation parameters have nothing to absorb on this source: the
environmental signature is not something the vacuum residual at ov 0.948 trades against. This is
the same behaviour as run_3's idx 2, 3, 4, 6, 7, 9 and 14, which is why those need `FORCE_KEEP`
past the PN-gain test in `collect_sensible.py` — a vanishing gain there is the measurement, not a
stalled optimiser.

**The b/σ column fell without the bias falling.** m1 reads +14.0σ here against +25.9σ in the 0PA
table, e0 −24.7σ against −62.5σ — but the fit is the *same point*, so the physical bias is
identical. All nine σ inflated (m1 8.54 → 15.83, e0 3.74e-6 → 9.45e-6) because two more parameters
now share the Fisher. Error-bar inflation, not bias removal.

**The deviation error bars are soft.** The delta calibration warned `minimum relative error is
greater than 1% for C_p` and the same for `C_e`, and the final Fisher condition number is 2.033e+18.
σ(C_p) = 0.060 and σ(C_e) = 0.552 should not be quoted hard. The central values sitting at zero is
the robust part — that is the climb declining to move, not a Fisher artifact.
