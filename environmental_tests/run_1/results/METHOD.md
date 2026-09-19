# Systematic bias from an environmental flux deviation: what was done, and what was assumed

This is the method behind `README.md` in this folder.  `README.md` is generated from the
per-source records and carries the numbers; this page carries the reasoning, the conventions
and — in one place, deliberately — every assumption the numbers rest on.  Nothing here is
auto-generated, so if the two disagree the numbers win and this page is stale.

Written for someone who knows LISA EMRI parameter estimation but was not present for the
runs.  Section 9 is the part to read if you are checking whether you believe the result.

---

## 1. The question

An EMRI embedded in an astrophysical environment radiates slightly differently from one in
vacuum.  If we then analyse it with a vacuum template, the fit absorbs the difference by
shifting the parameters, and the answer we report is wrong by a *systematic* amount that has
nothing to do with noise.  This is the Cutler–Vallisneri problem.

The specific question here: **if the template is given one extra degree of freedom — a
post-Newtonian deviation to the flux — does it absorb the environmental effect?**  And if it
does, does that *reduce the bias on the astrophysical parameters*, or does it only improve
the fit while leaving the parameters just as wrong?

Those are different questions and the study is built to keep them apart.  A template can
match a signal beautifully and still return the wrong masses.

## 2. The injected signal

Waveforms come from `SuperKludgeWaveform` (FEW / SuperKludge_r, branch `environ_shubham`),
propagated through `fastlisaresponse.ResponseWrapper` with `EqualArmlengthOrbits`.

The environment modifies the radial flux multiplicatively, in
`SuperKludge_r/src/few/trajectory/ode/flux.py`:

```
env  = 1 + A_PM * (p / 10)^n_PM + A_GC * p^n_GC
pdot = env * pdot
```

with, across the population,

| term | value | what it is |
|---|---|---|
| `n_PM` | 8 | steep power law — a migration-type torque, strongest at large `p` |
| `A_PM` | `1.92e-5 * (m1 / 1e6)` | amplitude, scaled with the primary mass |
| `n_GC` | 4 | shallow term |
| `A_GC` | `1e-12` | numerically dead at these `p` (fractional flux change ~1e-8) |

**Why the deviation is applied to `pdot` and nothing else.** The physical deviation is a
torque on `Ldot`.  On a circular orbit `L = L_circ(p)`, so `Ldot = (dL/dp) * pdot`, and the
consistent energy loss is `Edot = Omega_phi * Ldot`.  The two together are *exactly*
`pdot -> env * pdot`.  This is a circular-orbit identity and it is the reason the population
is circular.  Scaling `Ldot` alone would be a torque that does no work, and at `e = 0` it
carries the opposite sign.  **At finite eccentricity this shortcut is invalid** — the
deviation would have to be applied to `Edot` and `Ldot` before the Jacobian, with
`flux_output_convention = "ELQ"`.

`n_PM = 8` is what makes this study interesting and also what makes it hard: the effect is
negligible at small `p0` and violent at large `p0`, so the population spans sources where the
deviation is invisible and sources where a vacuum template cannot fit at all.

## 3. The two recovery models

Both are **vacuum** templates — the environment is off in the template, always.  That is the
point: we are measuring what happens when the environment is in the data and not in the model.

| model | free parameters | n |
|---|---|---|
| `0PA` | `m1, m2, a, p0, e0, qS, phiS, Phi_phi0` | 8 |
| `0PA+PN` | the same eight, plus `C_p` | 9 |

`C_p` is the amplitude of an additive 2.5PN deviation on the radial flux:

```
pdot += massratio * C_p * (1 - e^2)^1.5 * (8 + 7 e^2) / p^3.5
```

**`0PA` is a strict subspace of `0PA+PN`.** A PN template at `C_p = 0` is the 0PA template
exactly, bit for bit.  Two consequences used throughout:

1. The PN maximum can never lie below the 0PA maximum.  `PN >= 0PA` on overlap is a theorem
   here, not a result.  A PN overlap below the 0PA overlap is always an optimiser failure and
   is treated as one.
2. It also means the comparison is honest in the other direction: any PN improvement is the
   `C_p` direction doing work, because every other direction was already available to 0PA.

**Two parameters are deliberately not free**, because a circular orbit kills them:

* `Phi_r0` — the phase of a radial oscillation that does not exist.  Shifting it by 2 radians
  leaves the waveform bit-for-bit identical; its Fisher row is exactly zero and `Gamma` is
  singular if it is kept.
* `C_e` — the eccentric deviation amplitude, which enters as `edot += massratio * e * C_e * ...`
  and so carries an explicit factor of `e`.  With `e` pinned at 0 for the whole inspiral the
  direction is dead (measured mismatch 2.22e-16 on every source).  **Restore it the moment the
  population becomes eccentric.**

`e0` *is* free, and is the awkward one — see §9.

## 4. The population

Generated once (`population.json`, seed 43), 25 draws kept 23 sources:

| | |
|---|---|
| `m1` | log-uniform in `[1e6, 1e7]` Msun |
| `p0` | set so time-to-plunge lands in `[2, 4]` yr |
| SNR | every source rescaled in distance to **SNR = 200** |
| `T` | `0.99 * t_plunge` (`T_safety`), so no source plunges inside the window |
| `dt` | 10 s |
| channels | **A and E only** |
| band | `[1e-5, 1e-1]` Hz, DC bin dropped |

Everything not drawn is held fixed across the population: `m2 = 50`, `a = 0.9`, `e0 = 0`,
`xI0 = 1`, `qS = 0.2`, `phiS = 0.2`, `qK = 0.8`, `phiK = 0.8`, `Phi_phi0 = 0.3`,
`Phi_theta0 = 0.5`, `Phi_r0 = 0.5`.  So the population varies in **`m1`, `p0`, `T` and
distance only** — it is a one-parameter family dressed up as 23 sources, and `p0` is the
variable everything is organised by.

Two sources were removed by hand at generation time: **idx 1** (`m1 = 1.18e6`, dephasing
`+160.6` rad) and **idx 12** (`m1 = 9.98e6`, dephasing `+0.008` rad) — the two extremes of
the environmental deviation.  A third, **idx 16**, was dropped later (§10).

**`T` is dropped as a channel.** The `dt = 10` check in `population.json` shows T carrying
11–16% of the SNR² at `dt = 10` and essentially none at `dt = 5` — i.e. the T content at
`dt = 10` is aliasing, not signal.  Dropping T is the right call; keeping `dt = 10` is a
speed compromise and is listed as a caveat in §11.

## 5. Finding each best fit

The bias is `best_fit - injected`, so the whole study rests on the best fit actually being
the best fit.  Most of the effort went here.

### The CV climb

A Levenberg–Marquardt-damped Cutler–Vallisneri iteration: the Gauss–Newton step is
`dtheta = Gamma^-1 <dh | s - h>`, damped with Nielsen gain-ratio `lambda` updates, arranged
as **LM → Nelder–Mead → LM**.  The Fisher and the derivatives are recomputed at every LM
iteration.  Controls: `lambda0 = 1e-2`, `lambda` clamped to `[1e-12, 1e8]`, at most 150 LM
iterations, `rel_tol = 1e-9`, overlap target `0.9999999999`.

Every proposed point is pushed back inside a physical box (`project_physical`) before the
waveform sees it.  This is not cosmetic: the waveform generator raises outright on `e0 < 0`,
and with `e0` injected at exactly zero roughly half the CV steps want to cross that boundary,
so without the clip a source simply dies mid-climb.

Every climb records **why it stopped** — `stop_reason` is one of `overlap_target`, `rel_tol`,
`lambda_exhausted` or `max_iters` — per stage and for the climb as a whole.  This matters: in
the first version of this run *every* climb terminated on a `lambda` blowup while printing a
converged-looking `rel`, and the JSON could not tell you so.

### Two seeds for the PN model, always

The PN climb is run from **both** the injected truth and the 0PA best fit with `C_p = 0`, and
both are kept.  The tempting alternative — substituting the 0PA point whenever the PN climb
scores lower — manufactures an exact tie and hides the failure.  The spread between the two
seeds is the honest per-source statement of how converged the PN answer is.  The
higher-overlap climb is the one that enters the tables.

### The follow-up searches (`harder_environmental/`)

At large `p0` the deviation is strong enough that a climb seeded at the truth walks away and
never comes back.  Four sources needed a wider search: a three-stage
**CV step → differential evolution → CV climb** pipeline, with the DE box specified in units
of the Fisher `sigma` and centred either on the 0PA fit or (`--box-at-injection`) on the
injected point.

Two rules were adopted after early failures and are enforced:

* **The CV step must improve the overlap**, and the DE box must contain its own starting
  point.  An early version produced a stage-1 CV jump that moved *away* from the maximum and
  a box that excluded the seed, so DE was searching a region known not to contain the answer.
* **On the hard points `e0` is held at its injected value during the search**, then freed in a
  final seeded LM climb (the `_ecc` runs).  `e0`'s boundary Fisher direction was crashing the
  hard sources outright.  Freeing it at the end restores the same 8-parameter list the
  population run uses, so the follow-ups drop into the population's tables with no special
  casing.

**Each model arm is taken from whichever run reached the higher overlap for that model.**  A
source's two arms need not come from the same run — source 14's do not (§10).

## 6. The Fisher matrix

`sigma` comes from `StableEMRIFisher` (SEF), and is taken **at each model's own best fit**,
not at the injected truth — 8×8 at the 0PA fit, 9×9 at the PN fit.  That is what the CV
formalism assumes, and it is why the two models have different error bars on the same source.

For these results the stencil was refined from the population run's `der_order = 6`,
`Ndelta = 12` to **`der_order = 8`, `Ndelta = 24`**, so that `b/sigma` no longer depends
meaningfully on the choice of finite-difference step.  Cost scales roughly as
`der_order x Ndelta x n_params`.

**`C_p`'s finite-difference step is measured per source, not assumed.** SEF gives a parameter
sitting at zero a fixed absolute ladder calibrated to a parameter of order unity.  `C_p`
enters `pdot` multiplied by the mass ratio and is integrated over ~1e5 radians, so its usable
step is set by the source.  In an early run this produced `sigma(C_p)` of 7e-3, 2.7e0 and
1.5e6 on three *similar* sources, and one indefinite PN covariance.  `calibrate_dev_deltas`
now sweeps delta over `geomspace(1e-20, 1e2, 45)`, measures the mismatch each step induces,
and keeps the window where that mismatch lands in `[1e-9, 1e-5]` — well above waveform
evaluation noise (~1e-5 fractional in chi2) and still in the linear regime.  The ladder length
is `Ndelta`, so this had to be recalibrated when `Ndelta` changed from 12 to 24.

## 7. Conditioning: rescale, analyse, map back

The raw Fisher is catastrophically ill-conditioned — `cond(Gamma)` between **1e20 and 1e25**.
Almost all of that is units, not physics: the matrix holds `m1 ~ 1e6` next to `C_p ~ 1e-4`
next to angles of order 1.  At `cond ~ 1e22` in float64 a computed eigenvalue near zero is
indistinguishable from rounding noise, so *every* derived quantity has to be formed in a
better-scaled basis.

The transform is Jacobi (diagonal) rescaling.  With `D = diag(sqrt(|M_ii|))`:

```
M_hat = D^-1 M D^-1        (unit diagonal)
```

Three properties are used, and they are why this is safe:

1. **Quadratic forms are exactly invariant.**  `b^T M b = (D b)^T M_hat (D b)` and
   `b^T M^-1 b = (D^-1 b)^T M_hat^-1 (D^-1 b)`.  Every bias number below is computed in the
   scaled basis and is the same number the unscaled algebra defines — only formed without
   catastrophic cancellation.
2. **Inertia is preserved.**  The rescaling is a *congruence*, not a similarity, so it does
   not preserve eigenvalues — but by **Sylvester's law of inertia** it preserves their signs.
   A positive-definiteness test in the scaled basis therefore answers the question asked of
   the unscaled matrix, and answers it some ten orders of magnitude further from the noise
   floor.
3. **The inverse maps back exactly.**  `C = C_hat / outer(d, d)`.

**What this changed.** The population run tested definiteness on the raw covariance and
reported most PN covariances as "not positive semi-definite", with minimum eigenvalues around
`-5e-17`.  Re-tested in the scaled basis, `cond` drops from ~1e22 to **1e9–1e11** and every
minimum eigenvalue is a healthy **1e-11 to 1e-8** positive.  Those matrices were positive
definite all along; the negative eigenvalues were float noise against the largest eigenvalue.
Each record stores `cond_raw`, `cond_scaled`, `cov_min_eig_scaled` and `cov_pos_def`, the last
two from the scaled basis.

**How we check that it worked.**  A condition number says how much error an inversion
*could* amplify.  It does not say how much it *did*.  So each record also stores the direct
test:

* `inv_residual` = `max|G_hat C_hat - I|`, and `inv_residual_ratio`, the same number
  divided by `cond_scaled * 2.2e-16`.  **The ratio is the diagnostic, not the residual.**
  That product is the accuracy double precision can deliver at this conditioning, so a
  residual of 1e-6 at `cond = 1e11` is an excellent inversion while the same residual at
  `cond = 1e3` would be a broken one.  An absolute threshold gets this exactly backwards and
  flags the arithmetic floor as if it were a failure — which is what a first pass here did.
  A ratio above 10 is flagged in the log and named in `README.md`.
* `numerical_rank` = the number of eigenvalues above `max(eig) * eps`.  A deficit here is a
  genuinely dead parameter direction, not a scaling problem, and no rescaling will fix it.

Both are tabulated per source and model in `README.md`, and plotted in `conditioning.png`
beside the raw and rescaled condition numbers.  The verdict paragraph there is generated from
the records, so it cannot drift from what was actually computed.

This is a numerical fix, not a physical one.  It does not make the Fisher a good error
estimate where the likelihood is genuinely non-Gaussian — see §11.  A perfectly conditioned
Fisher at a fit that only reaches overlap 0.77 is still the wrong error model.

## 8. What "bias" means here — four quantities

Write `b` for the bias vector, `b_i = fit_i - truth_i`.

**Angles are unwrapped first.**  For `qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0` the
difference is mapped into `(-pi, pi]` via `(d + pi) mod 2pi - pi`, so a fit on the far side of
`2 pi` is not counted as a huge error.  Same convention everywhere, defined once in
`results_inputs.signed_bias_of`.  The per-parameter panels take `|b|`; the n-d quantities need
the sign, because a quadratic form has cross terms.

| quantity | definition | what it answers |
|---|---|---|
| **absolute bias** | `\|b_i\|` | how wrong is this parameter, in its own units.  Needs no Fisher, so it is the solid quantity. |
| **normalized bias** | `\|b_i\| / sigma_i` | is it wrong by more than we claim to know it |
| **max 1-d bias** | `max_i \|b_i\| / sigma_i` | is *any* parameter past its error bar — one is enough to make the answer wrong |
| **n-d bias** | `sqrt(b^T M b)` | the length of the whole bias vector in units of the error ellipsoid — the quantity the CV formalism bounds |

The n-d bias is computed **two ways**, differing only in what is done with the parameters held
out of the block:

* **marginalized** — `M = (C_vv)^-1`, the inverse of the kept block of the *covariance*.  The
  held-out parameters have been integrated over, so the freedom they carry is still paid for.
  This is the SK/CV convention.
* **conditional** — `M = G_vv`, the kept block of the *Fisher*, which is the inverse covariance
  with the held-out parameters frozen at their fitted values.

`G_vv >= (C_vv)^-1` in the Loewner order, so **the conditional value is never smaller than the
marginalized one**, and the gap between them is exactly what the extra freedom costs.  For
`0PA+PN` that gap is the price of letting `C_p` float.

**The conditional comparison between the two models is asymmetric, and must not be read as a
like-for-like contest.**  The two models do not freeze the same things: `0PA` freezes only
`e0`, while `0PA+PN` freezes `e0` *and* `C_p`.  Freezing an extra parameter tightens the error
ellipsoid, so the PN conditional number is computed against a stricter yardstick than the 0PA
one.  This is why PN comes out "worse" conditionally on most sources while coming out better
marginalized — the two statements are not in conflict, they are answers to different
questions.  **The marginalized panel is the like-for-like comparison** and is the SK/CV
convention; the conditional panel is there to show the size of the freedom `C_p` buys, by
comparing each model against *itself*.

This is the single most important thing to keep in view when reading the results: **a PN model
can show a smaller `b/sigma` with an absolute bias that has not moved at all**, purely because
`C_p` correlates with the vacuum parameters and inflates their `sigma`.  A shrinking normalized
bias is not by itself evidence that the template fixed anything.  That is why the absolute bias
is always plotted next to it.

**Four parameter blocks**, and every summary figure is drawn over all of them.  Two
independent choices, crossed:

*Is `e0` in?*  It sits on its physical boundary, so its Fisher direction is degenerate and
`sigma(e0)` is not a confidence interval.  Held out, a figure shows the parameters whose
`sigma` means what it says; kept in, it shows what that boundary direction does to the totals.

*All parameters, or intrinsic only?*  `m1, m2, a, p0` are the astrophysics.  A template that
absorbs an environmental deviation by shifting the sky position has not damaged the
measurement the way one absorbing it into the masses has, so the intrinsic block asks the
narrower and harder question.

| block | parameters | k | where the figures live |
|---|---|---|---|
| base | `m1, m2, a, p0, qS, phiS, Phi_phi0` | 7 | `results/` |
| with `e0` | the same plus `e0` | 8 | `results/`, `_with_e0` |
| intrinsic | `m1, m2, a, p0` | 4 | `results/intrinsic/` |
| intrinsic with `e0` | `m1, m2, a, p0, e0` | 5 | `results/intrinsic/`, `_with_e0` |

The intrinsic figures use the same filenames as the top-level ones, so the two folders can be
opened side by side.

**The n-d credible threshold moves with `k`** and is recomputed per block — `sqrt(chi2_ppf(p, k))`
is 3.64 sigma at `k = 4` and 4.30 at `k = 7` for `p = 0.99`.  A smaller space needs a smaller
radius to enclose the same probability, so the blocks are not comparable against a shared
line and none is drawn.

`C_p` (and `C_e`, when it returns) is held out of all of them: it is injected at zero and is
the thing being absorbed, not a parameter whose bias is being measured — and `0PA` has no such
parameter to compare against.

## 9. Assumptions, in one place

Ordered roughly by how much damage each would do if wrong.

1. **The best fit found is the global maximum.**  This is the load-bearing assumption of the
   whole study and it is *known false for at least one source* (idx 14, §10).  Where a search
   settled in a secondary maximum, the "bias" is the distance to that basin, not to the truth.
   Overlap and `stop_reason` are reported per source precisely so this can be checked rather
   than trusted.
2. **The Fisher is an adequate error estimate at the best fit.**  Linear-signal, Gaussian
   likelihood.  At SNR 200 this is usually defensible; at a fit that only reaches overlap
   0.77 it is not, because the likelihood there is nowhere near Gaussian.
3. **The environmental deviation is captured by `pdot -> env * pdot`.**  Exact on a circular
   orbit (§2), invalid at finite `e`.
4. **`n_PM = 8` and `A_PM = 1.92e-5 (m1/1e6)` describe the environment.**  These are a chosen
   benchmark, not a measured astrophysical prediction.  Every quantitative conclusion is
   conditional on them, and `n_PM = 8` in particular makes the whole effect a steep function
   of `p0`.
5. **A and E only, with T dropped.**  Justified by the `dt` check (§4), but it does discard
   real T-channel information at any `dt` where T is not aliased.
6. **`dt = 10` is adequate.**  A speed compromise.  The `dt = 5` comparison shows the A/E SNRs
   are stable, so this is believed safe for A and E, but it has not been re-run end to end.
7. **`e0` is a meaningful free parameter at `e0 = 0`.**  It sits on its physical boundary, so
   the likelihood is one-sided and the Fisher direction is degenerate.  `sigma(e0)` ranges over
   *nine orders of magnitude* across the population (2e-10 to 0.53) and is **not** a confidence
   interval.  The absolute bias on `e0` is meaningful; `b/sigma(e0)` is not, and no conclusion
   in `README.md` rests on it.  This is why every summary figure comes in a with-`e0` and a
   without-`e0` version rather than one being chosen.
8. **`C_e` and `Phi_r0` are dead directions and may be dropped.**  Exactly true on a circular
   population (measured mismatch 2.22e-16 and bit-identical waveforms respectively).  Both must
   come back if the population becomes eccentric.
9. **`A_GC` is negligible.**  A fractional flux change of ~1e-8 at these `p`.  It is in the
   injection but is not freed in any Fisher; freeing it makes the matrix singular.
10. **The Jacobi rescaling is exact.**  It is — see §7 — but it is worth stating that all
    reported `sigma` and n-d values are computed in the scaled basis and mapped back, not
    computed on the raw matrices.  The inversion residual is reported per arm so this is
    checkable rather than assumed.
11. **A population median is meaningful over 21–22 sources** whose only real variation is `m1`,
    `p0`, `T` and distance.  This is a narrow family, not a realistic astrophysical population;
    it answers "how does the comparison behave as a function of `p0`", not "what fraction of
    LISA EMRIs are biased".
12. **The `flux.py` patch is correct.**  It is uncommitted work on branch `environ_shubham`
    (§11).

## 10. What converged, and what did not

`README.md` has the per-source table.  The cases that need judgement:

* **idx 8** (`p0 = 7.76`).  0PA stops at overlap **0.9278** and this is a *converged* answer,
  not a failure: a vacuum template genuinely cannot do better against that environmental term.
  PN reaches 0.9999992.  This is the one source where PN unambiguously rescues the parameters —
  and it is that source precisely because 0PA failed there outright.
* **idx 13** (`p0 = 10.31`).  Both converge; PN wins on overlap and chi2 but is *worse* on
  several individual parameters.  A good example of §8's warning.
* **idx 14** (`p0 = 9.45`).  **The searches did not find the true basin.**  Its two arms come
  from different runs, each being that model's best: 0PA from the `harder_environmental`
  follow-up at overlap **0.7742**, PN from the population run at **0.9206**.  Both arms settle
  at `qS ~ 1.5` against an injected `0.2` and `phiS ~ 3.9-4.0` against `0.2` — a secondary sky
  maximum.  The PN follow-up (DE in a ±5σ box centred on the injected point, 2000 iterations)
  *also* found that basin and scored 0.7795, below the population run's 0.9206, so the box was
  not the problem.  **Its bias numbers measure the distance to the wrong maximum.**  It is
  plotted solid and included, but read it as a statement about where the searches break down.
* **idx 16** (`p0 = 11.75`) — **dropped entirely**.  Four PN searches, three of which collapsed
  to overlap ~1e-3.  Its numbers would report where the optimiser stopped, not what the
  template absorbs.
* **idx 1, idx 12** — removed at population generation as the two extremes (§4).

**Convergence here is not an overlap threshold.**  Source 8 at 0.9278 is converged; source 14
at 0.9206 is not.  The distinction is whether the climb reached a maximum, which is what
`stop_reason` and the seed spread are recorded for.

## 11. Limitations

* **The Fisher is a Fisher.**  No MCMC anywhere in this study.  Where a fit is poor the
  Gaussian approximation is not just imprecise, it is the wrong object.
* **`p0` is the only real independent variable** (§4, assumption 11).
* **`dt = 10` aliasing** is an accepted fidelity compromise.
* **Not reproducible from a clean checkout.**  The study depends on an uncommitted patch to
  `flux.py` on branch `environ_shubham` of `SuperKludge_r`.  Until that is committed, the
  results cannot be regenerated by anyone else.  **This is the most serious practical
  limitation.**
* **`sigma(C_p)` depends on a calibration** (§6).  It is measured rather than assumed, which is
  a large improvement, but it is still a choice of mismatch band.
* **No noise realisation.**  This is a pure systematics calculation: `s = h_env`, no noise is
  added.  The bias reported is the systematic offset alone, which is the CV question, but it
  means nothing here speaks to how the systematic compares to a statistical fluctuation in a
  given realisation beyond the `b/sigma` ratio.

## 12. Files

Numbers and figures: see `README.md`, generated by `make_results_md.py`.

| file | what it is |
|---|---|
| `idx*.json` | one record per source: both fits, both Fishers, covariances, scale, conditioning, bias, `b/sigma` |
| `bias_panels.png` / `_with_e0.png` | per parameter, `\|b\|` and `\|b\|/sigma` |
| `overlap.png` | `1 - overlap` against `p0`, both models |
| `max_bias*.png` | per source, worst single-parameter `\|b\|/sigma` |
| `nd_bias*.png` | per source, n-d bias, marginalized beside conditional |
| `improvement*.png` | per source, the factor by which PN beats 0PA on all three quantities |
| `*_with_e0`, `*_p90` | the e0 and credible-level variants above |
| `intrinsic/` | the same figures over `m1, m2, a, p0` only, under the same filenames, with their own `README.md` |
| `conditioning.png` | raw vs rescaled condition number, and the inversion residual |
| `../results_inputs.py` | which sources enter and where each model arm is read from |
| `../refit_fisher.py` | the order-8 Fisher job, including the rescaling |
| `../plot_results.py` | all figures |
| `../make_results_md.py` | `README.md` |
| `../cv_population.py` | the population run: waveforms, CV climbs, SEF machinery |
| `../../harder_environmental/cv_de_cv.py` | the CV → DE → CV follow-up pipeline |

To regenerate after a change to the fits:

```
qsub batch_refit_fisher_a.sh          # sources 0-11
qsub batch_refit_fisher_b.sh          # sources 13-24
python plot_results.py                # all 24 figures
python make_results_md.py             # README.md
```
