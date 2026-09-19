"""CV step -> differential evolution in the Fisher box -> CV climb, on one hard source.

The population run climbs from the injected point with Levenberg-Marquardt alone, and on the
hard sources that fails.  On the circular population, source 8 (m1 = 3.78e6, p0 = 7.76) ends
its LM climb at ov = 3.6e-4 with chi2 = 7.1e4 -- not a near miss, an entirely wrong basin.
The failures are the large-p0 sources, which is where the (p/10)^8 environmental deviation is
biggest, so the injected signal is many radians from any vacuum template LM starts near.  That
is an optimiser failure, not a statement about the model, so the fix is a better optimiser on
the same likelihood.

Three stages, in this order:

1. **One CV step** from the injected point.  The Gauss-Newton step the CV formalism predicts,
   `dtheta = Gamma^-1 <dh|s-h>`, taken once with LM damping.  It is also where the Fisher --
   and therefore sigma -- is evaluated, which is what stage 2 needs.

2. **Differential evolution** inside a box of +-`prior_sigma_range` sigma about that point,
   as in bias_inference_emri/src/inference.py: the same scipy call, the same sobol
   initialisation, the stage-1 point seeded into the population, bounds clipped to the
   physical ranges.  DE is a population search, so it can cross the ridge LM cannot.  It is
   run without polish -- stage 3 is the polish, and a CV climb is a better one than the
   L-BFGS-B scipy would apply.

3. **CV again** from the DE winner, the full LM -> Nelder-Mead -> LM arrangement of the
   population run, to take the answer from "in the right basin" to "at the maximum".

Nothing about the signal model changes: the injection carries the environment, every template
has it off, A and E channels, dt and T from the frozen point.

Settings live in config.py.  This file reads no constants of its own.

Run:  python cv_de_cv.py
      python cv_de_cv.py --model 0PA+PN --tag pn
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import differential_evolution

import config as C
from config import Config, apply_cv_controls

sys.path.insert(0, C.ENV_TESTS)
import cv_population as cvp          # noqa: E402  (path must be set first)


# ---------------------------------------------------------------- the point

def point_from_run(cfg):
    """Pull one source straight out of a population run's results file.

    The frozen hard_point.json is a convenience, not a dependency: this rebuilds the same
    structure from cfg.point_results, so the run is reproducible from the population results
    alone.  The column names are read off the file rather than assumed, so an archived
    eccentric run and the circular one both load.
    """
    path, idx = cfg.point_results, cfg.point_idx
    with open(path) as f:
        data = json.load(f)
    src = next((r for r in data["sources"] if r["idx"] == idx), None)
    if src is None:
        raise SystemExit(f"[point] no source {idx} in {path}")
    return {"from_run": cfg.point_run or "live", "from_results": path,
            "params_0pa": data["params_0pa"], "params_pn": data["params_pn"],
            "params_env": data["params_env"], "source": src}


def load_point(cfg):
    """The source record for cfg.point_run/cfg.point_idx.

    Prefers the frozen hard_point.json when it is present and holds the right source, and
    otherwise -- or on a mismatch -- reads the population results directly.  Either way the
    config, not the file on disk, decides which point is run.
    """
    want = (cfg.point_run or "live", cfg.point_idx)
    if os.path.exists(cfg.point_json):
        with open(cfg.point_json) as f:
            point = json.load(f)
        got = (point["from_run"], point["source"]["idx"])
        if got == want:
            print(f"[point] frozen {os.path.basename(cfg.point_json)}", flush=True)
            return point
        print(f"[point] {os.path.basename(cfg.point_json)} holds {got[0]}/{got[1]}, config "
              f"wants {want[0]}/{want[1]}: reading the run results instead", flush=True)
    else:
        print(f"[point] no {os.path.basename(cfg.point_json)}: reading the run results",
              flush=True)
    return point_from_run(cfg)


def truth_vector(point, names):
    """The injected values of the parameters being optimised."""
    truth = dict(zip(point["params_pn"], point["source"]["truth_pn"]))
    return np.array([truth[n] for n in names], dtype=float)


def describe_point(point):
    s = point["source"]
    print(f"[point] {point['from_run']}/{s['idx']}: m1 = {s['m1']:.6e}, p0 = {s['p0']:.6f}, "
          f"T = {s['T']:.6f} yr, dt = {s['dt']:.0f}, d = {s['dist']:.6f} Gpc, "
          f"SNR = {s['snr']:.4f}", flush=True)
    print(f"[point] injected A_PM = {s['A_PM']:.6e}, n_PM = {s['n_PM']:.0f}, "
          f"A_GC = {s['source_params']['A_GC']:.1e}, n_GC = {s['source_params']['n_GC']:.1f}",
          flush=True)
    print(f"[point] population run reached ov = {s['fit_0pa']['ov_final']:.8f} (0PA), "
          f"{s['fit_pn']['ov_final']:.8f} (0PA+PN)", flush=True)


# ---------------------------------------------------------------- context

# cv_population.build_source reads the environmental constants off the top level of a
# *population* entry; a CV *results* entry carries A_PM and n_PM there but keeps A_GC and
# n_GC only inside source_params.  This fills the gap from source_params rather than
# reaching for a population.json that may belong to a different run.
ENV_CONSTANTS = ("A_PM", "n_PM", "A_GC", "n_GC")


def as_population_source(src):
    """A results-file source in the shape build_source expects, without mutating the input."""
    out = dict(src)
    for name in ENV_CONSTANTS:
        if name not in out:
            out[name] = float(src["source_params"][name])
    return out


def build_context(cfg, point):
    """(S, ctx): the source-level objects and the recovery-model context for cfg.model."""
    S = cvp.build_source(as_population_source(point["source"]))
    print(f"[build] SNR check: {S['snr']:.6f} (A = {S['snr_channels'][0]:.4f}, "
          f"E = {S['snr_channels'][1]:.4f})", flush=True)
    ctx = cvp.build_ctx(S, cfg.param_names)
    return S, ctx


def calibrate_if_pn(cfg, ctx, theta0):
    """Measure the C_p finite-difference step, as the population run does.

    SEF hands a parameter sitting at zero a fixed absolute ladder calibrated to a parameter of
    order unity; C_p enters pdot times the mass ratio, so its usable step is set by the source
    and has to be measured.  The 0PA model has no such parameter and skips this.
    """
    if not cfg.dev_params:
        return None
    return cvp.calibrate_dev_deltas(ctx, theta0, tag=f" {cfg.model}")


# ---------------------------------------------------------------- angles

def wrap_angles(cfg, theta):
    """Fold the periodic parameters into [0, 2pi), which is the range the DE box lives in."""
    out = np.array(theta, dtype=float)
    for i, n in enumerate(cfg.param_names):
        if n in C.ANGLE_PARAMS:
            out[i] = out[i] % (2 * np.pi)
    return out


def unwrap_bias(cfg, theta, truth):
    """theta - truth with every angular component folded into (-pi, pi].

    A fit a whole turn from the injected value is the same physical answer; taken raw the
    difference would read as a bias of hundreds of sigma.
    """
    bias = np.array(theta, dtype=float) - np.asarray(truth, dtype=float)
    for i, n in enumerate(cfg.param_names):
        if n in C.ANGLE_PARAMS:
            bias[i] = (bias[i] + np.pi) % (2 * np.pi) - np.pi
    return bias


# ---------------------------------------------------------------- stage 1, 3

class lm_iters:
    """Run a climb with a different LM iteration cap, then put the module back as it was."""

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.saved = cvp.LM_MAX_ITERS
        cvp.LM_MAX_ITERS = self.n

    def __exit__(self, *exc):
        cvp.LM_MAX_ITERS = self.saved


def project_and_report(names, theta, tag):
    """Pull a proposed point back inside the physical box, naming what moved.

    `lm_climb` projects every proposal it makes.  The exact CV step did not, because the
    eccentric point this file was written for had e0 = 0.2 sitting well inside its range.  The
    circular population injects e0 = 0 exactly, so a step with any negative e0 component leaves
    the domain and the waveform generator refuses the point outright.

    Clipping is what a bounded optimiser does, but a clipped component is no longer the CV step
    the formalism predicts, so it is printed and recorded rather than applied silently.
    """
    out = np.array(theta, dtype=float)
    clipped = []
    for i, n in enumerate(names):
        lo, hi = C.PHYS_LO.get(n, -np.inf), C.PHYS_HI.get(n, np.inf)
        new = min(max(out[i], lo), hi)
        if new != out[i]:
            clipped.append(dict(param=n, proposed=float(out[i]), used=float(new)))
            out[i] = new
    for c in clipped:
        print(f"    [clip]{tag} {c['param']}: {c['proposed']:+.6e} -> {c['used']:+.6e} "
              f"(outside the physical box; this component is no longer the full CV step)",
              flush=True)
    return out, clipped


def backtrack_step(ctx, theta, delta, ov_start, tag, max_halvings):
    """The largest fraction of the CV step that still improves the overlap.

    The undamped step is what the formalism predicts, but on a hard source it can overshoot
    so far that stage 1 ends *below* where it started -- and since the stage-1 point is what
    centres the DE box, a bad step does not merely waste an iteration, it puts the truth
    outside the box for every later stage.  So the step is tried at 1, 1/2, 1/4, ... and the
    first scale that beats `ov_start` is taken.  Scale 1 is the ordinary outcome; anything
    smaller is reported, because it means the quadratic model did not hold at full length.

    If no scale improves the overlap the point is left where it was (scale 0) rather than
    moved to a worse one.
    """
    scale = 1.0
    for k in range(max_halvings + 1):
        proposed = theta + scale * delta
        out, clipped = project_and_report(ctx["params"], proposed, f"{tag}@{scale:g}")
        try:
            ov = float(ctx["ov"](out))
        except Exception as exc:
            # A proposal outside the waveform generator's domain of validity is just a step
            # that is too long -- it is the backtracking loop's job to shorten it, not a
            # reason to lose the run.  Source 13 died here on e0 = 0.99 before this existed.
            print(f"    CV{tag} scale {scale:g} rejected by the generator ({exc}); halving",
                  flush=True)
            scale *= 0.5
            continue
        if np.isfinite(ov) and ov > ov_start:
            if k:
                print(f"    CV{tag} backtracked to scale {scale:g} after {k} halving(s)",
                      flush=True)
            return scale, out, proposed, clipped, ov
        print(f"    CV{tag} scale {scale:g} gives ov {ov:.10f} <= start {ov_start:.10f}; "
              f"halving", flush=True)
        scale *= 0.5
    print(f"    CV{tag} no scale improved the overlap; staying put", flush=True)
    return 0.0, np.array(theta, dtype=float), np.array(theta, dtype=float), [], ov_start


def cv_direction(ctx, theta):
    """The raw CV step at `theta`, with the Fisher and its sigma at the same point.

    Split out of cv_step_exact so the iterated stage 1 can re-linearise without duplicating
    the solve.  Jacobi-preconditioned for the same reason cov_precond is: m1 ~ 1e6 and
    e0 ~ 0.2 sit in one matrix.
    """
    theta = np.asarray(theta, dtype=float)
    G, dH, _ = ctx["fisher_derivs"](theta, None)
    G = 0.5 * (G + G.T)
    r = ctx["s"] - ctx["make0"](theta)
    g = np.array([ctx["ip"](dH[j], r) for j in range(len(theta))])
    d = np.sqrt(np.abs(np.diag(G))) + 1e-300
    delta = np.linalg.solve(G / np.outer(d, d), g / d) / d
    sigma = np.sqrt(np.abs(np.diag(cvp.cov_precond(G))))
    return delta, G, sigma


def cap_to_sigma(delta, sigma, cap):
    """Shrink `delta` until no component exceeds `cap` sigma.  Returns (delta, scale).

    The whole vector is scaled by one factor rather than clipped per component, so the step
    keeps the direction the Fisher solve chose -- clipping would bend it, and on a matrix this
    correlated the direction is the part worth preserving.
    """
    reach = np.max(np.abs(np.asarray(delta) / (np.asarray(sigma) + 1e-300)))
    if not np.isfinite(reach) or reach <= cap:
        return np.asarray(delta, dtype=float), 1.0
    scale = float(cap / reach)
    return np.asarray(delta, dtype=float) * scale, scale


def cv_step_exact(ctx, theta, tag, backtrack=True, max_halvings=12,
                  cap_sigma=None, cap_sigma_units=None):
    """The complete Cutler-Vallisneri step: `dtheta = Gamma^-1 <dh|s-h>`.

    This is the step the CV formalism actually predicts.  `lm_climb` cannot produce it: its
    Levenberg-Marquardt damping solves `(Gamma + lam*diag(Gamma)) dtheta = g` and only accepts
    the result if the Nielsen gain ratio is positive, so the step is both shrunk and
    conditional.  Setting lambda to zero there does not help -- a rejected step multiplies
    lambda by LAMBDA_UP, which leaves zero at zero, so the climb exhausts without moving.

    The solve is Jacobi-preconditioned for the same reason cov_precond is: m1 ~ 1e6 and e0 ~ 0.2
    sit in one matrix.  The step is taken unconditionally; chi2 is allowed to rise, because the
    point of this stage is the CV prediction and the DE box it defines, not a descent.

    Returns the record shape cv_stage returns, so stage 1 and stage 3 stay comparable.
    """
    t0 = time.time()
    theta = np.asarray(theta, dtype=float)
    ov_start = float(ctx["ov"](theta))
    print(f"    start overlap = {ov_start:.10f}", flush=True)

    delta, G, sigma_here = cv_direction(ctx, theta)
    if cap_sigma:
        delta, cap_scale = cap_to_sigma(delta, cap_sigma_units if cap_sigma_units is not None
                                        else sigma_here, cap_sigma)
        if cap_scale < 1.0:
            print(f"    CV{tag} step capped to {cap_sigma:g} sigma "
                  f"(scaled by {cap_scale:.4g})", flush=True)
    if backtrack:
        scale, out, proposed, clipped, ov_final = backtrack_step(
            ctx, theta, delta, ov_start, tag, max_halvings)
    else:
        scale = 1.0
        proposed = theta + delta
        out, clipped = project_and_report(ctx["params"], proposed, tag)
        ov_final = float(ctx["ov"](out))
    chi2 = float(ctx["chi2r"](out))
    sigma = sigma_here
    print(f"    CV{tag} exact step  scale {scale:g}  ov {ov_start:.10f} -> {ov_final:.10f}  "
          f"chi2 {float(ctx['chi2r'](theta)):.3e} -> {chi2:.3e}", flush=True)

    info = dict(n_iter=1, n_accept=1, lam_final=0.0,
                stop_reason="exact_cv_step" if scale == 1.0 else f"cv_step_scaled_{scale:g}")
    fit = dict(ov_start=ov_start, ov_final=ov_final, chi2=chi2, params=out.copy(),
               params_unclipped=proposed.copy(), clipped=clipped, step_scale=scale,
               delta_full=delta.copy(),
               start=theta.copy(), sigma=sigma.copy(),
               stages=[dict(stage="CV_exact", ov=ov_final, **info)],
               nm_accepted=False, seconds=time.time() - t0, **info)
    return fit, out


def remaining_budget(theta0, theta, sigma_units, total_cap):
    """How much of the cumulative trust region is left, as a cap for the next step.

    The walk is measured from the injected point in sigma at that point, which is the unit the
    DE box is eventually built in.  Returns 0.0 when the budget is spent.
    """
    used = np.max(np.abs((np.asarray(theta) - np.asarray(theta0)) / (sigma_units + 1e-300)))
    return float(max(total_cap - used, 0.0)), float(used)


def cv_iterate_stage1(cfg, ctx, theta0, tag="1"):
    """Stage 1 as a trust-region climb: capped CV steps, each required to raise the overlap.

    The single undamped step is the literal CV prediction, but on a hard source it is a leap of
    tens of sigma, and since the stage-1 point centres the DE box the leap leaves the box unable
    to contain its own starting point.  Here the same step direction is re-derived at each
    iterate and taken at a capped length, so the walk stays inside a trust region the box can
    cover, and the overlap that the shorter steps give up is recovered by taking several.

    Sigma at the injected point is the metric throughout -- fixed, so the budget means the same
    thing at every iterate.  Returns the stage-1 record shape with the per-iteration log
    attached, so downstream reporting is unchanged.
    """
    t0 = time.time()
    theta0 = np.asarray(theta0, dtype=float)
    ov0 = float(ctx["ov"](theta0))
    _, _, sigma0 = cv_direction(ctx, theta0)
    print(f"[stage1] {cfg.cv1_steps} capped CV steps, {cfg.cv1_step_cap_sigma:g} sigma each, "
          f"{cfg.cv1_total_cap_sigma:g} sigma total; start ov = {ov0:.10f}", flush=True)

    theta, ov, iters, stop = theta0.copy(), ov0, [], "steps_exhausted"
    for k in range(cfg.cv1_steps):
        budget, used = remaining_budget(theta0, theta, sigma0, cfg.cv1_total_cap_sigma)
        if budget <= 0.0:
            stop = "trust_region_exhausted"
            print(f"[stage1] budget spent after {k} step(s) ({used:.2f} sigma)", flush=True)
            break
        cap = min(cfg.cv1_step_cap_sigma, budget)
        print(f"  -- stage-1 step {k + 1}/{cfg.cv1_steps}  (walked {used:.2f} sigma, "
              f"cap {cap:.2f})", flush=True)
        fit, theta_new = cv_step_exact(ctx, theta, f"{tag}.{k + 1}",
                                       backtrack=cfg.cv1_backtrack,
                                       max_halvings=cfg.cv1_max_halvings,
                                       cap_sigma=cap, cap_sigma_units=sigma0)
        iters.append(dict(iter=k + 1, ov_start=fit["ov_start"], ov_final=fit["ov_final"],
                          chi2=fit["chi2"], step_scale=fit["step_scale"],
                          clipped=fit["clipped"], params=theta_new.copy(),
                          walked_sigma=used, seconds=fit["seconds"]))
        if fit["step_scale"] == 0.0:
            stop = "no_scale_improved_overlap"
            print(f"[stage1] step {k + 1} could not improve the overlap; stopping", flush=True)
            break
        theta, ov = theta_new, fit["ov_final"]

    _, used = remaining_budget(theta0, theta, sigma0, cfg.cv1_total_cap_sigma)
    print(f"[stage1] {len(iters)} step(s), ov {ov0:.10f} -> {ov:.10f}, "
          f"walked {used:.2f} sigma, stop = {stop}", flush=True)

    sigma = stage1_sigma(ctx, theta)
    info = dict(n_iter=len(iters), n_accept=len(iters), lam_final=0.0, stop_reason=stop)
    fit = dict(ov_start=ov0, ov_final=float(ov), chi2=float(ctx["chi2r"](theta)),
               params=theta.copy(), params_unclipped=theta.copy(),
               clipped=[c for it in iters for c in it["clipped"]],
               start=theta0.copy(), sigma=sigma.copy(), sigma_injected=sigma0.copy(),
               walked_sigma=used, iterations=iters,
               stages=[dict(stage="CV_capped_iter", ov=float(ov), **info)],
               nm_accepted=False, seconds=time.time() - t0, **info)
    return fit, theta


def dev_decoherence_bound(cfg, dev_deltas, name="C_p"):
    """The |C_p| at which the template decoheres from the data, times a margin.

    calibrate_dev_deltas keeps the steps whose mismatch lands in
    [DEV_MISMATCH_LO, DEV_MISMATCH_HI]; mismatch is quadratic in C_p near zero, so the largest
    kept step d_hi gives mismatch DEV_MISMATCH_HI and mismatch reaches 1 near
    d_hi / sqrt(DEV_MISMATCH_HI).  Past that the likelihood is flat noise, which is where the
    source-8 fit found C_p = 2088 with a straight face.
    """
    steps = dev_deltas.get(name) if dev_deltas else None
    if steps is None or not len(steps):
        print(f"[bounds] no calibrated step for {name}; falling back to the Fisher box",
              flush=True)
        return None
    d_hi = float(np.max(np.asarray(steps, dtype=float)))
    decoh = d_hi / np.sqrt(cvp.DEV_MISMATCH_HI)
    half = cfg.dev_decoh_margin * decoh
    print(f"[bounds] {name}: calibrated d_hi = {d_hi:.4e} at mismatch "
          f"{cvp.DEV_MISMATCH_HI:.1e} -> decoherence at |{name}| ~ {decoh:.4e}; "
          f"box = +-{half:.4e} ({cfg.dev_decoh_margin:g}x)", flush=True)
    return (-half, half)


def cv_stage(ctx, theta, iters, use_nm, tag):
    """One CV stage: `iters` LM iterations, optionally the LM -> NM -> LM arrangement.

    Returns the same record shape in both cases, so stage 1 and stage 3 are directly
    comparable in the output file.
    """
    t0 = time.time()
    with lm_iters(iters):
        if use_nm:
            fit = cvp.cv_from(ctx, tag, theta)
            return fit, np.asarray(fit["params"], dtype=float)
        ov_start = float(ctx["ov"](theta))
        print(f"    start overlap = {ov_start:.10f}", flush=True)
        out, ov1, sigma, info = cvp.lm_climb(ctx, theta, tag=f" {tag}")
        fit = dict(ov_start=ov_start, ov_final=float(ov1),
                   chi2=float(ctx["chi2r"](out)), params=out.copy(),
                   start=np.array(theta, dtype=float), sigma=sigma.copy(),
                   stages=[dict(stage="LM", ov=float(ov1), **info)],
                   nm_accepted=False, seconds=time.time() - t0, **info)
    return fit, out


def no_stage1(ctx, theta):
    """Stage 1 as a no-op: the DE box is centred on the injected point itself.

    The capped CV walk exists to move the box centre onto a better point than injection, but
    that also moves the prior off the truth.  With this the prior is literally +-N sigma about
    the injected parameters, so the question DE answers is whether a PN template within N sigma
    of the truth beats the 0PA answer -- not whether one within N sigma of a CV-stepped point
    does.  Returns the ordinary stage-1 record shape so nothing downstream changes.
    """
    theta = np.asarray(theta, dtype=float)
    ov = float(ctx["ov"](theta))
    print("[stage1] skipped: the DE box is centred on the injected point", flush=True)
    print(f"[stage1] overlap at injection = {ov:.10f}", flush=True)
    return dict(ov_start=ov, ov_final=ov, chi2=float(ctx["chi2r"](theta)),
                params=theta.copy(), start=theta.copy(), skipped=True,
                seconds=0.0), theta


def stage1_sigma(ctx, theta):
    """The Fisher sigma at the stage-1 point, which is the half-width unit of the DE box.

    lm_climb already returns it, but only for the point it *started* from; the box has to be
    centred on where stage 1 ended, so it is recomputed there.
    """
    G, _, _ = ctx["fisher_derivs"](np.asarray(theta, dtype=float), None)
    G = 0.5 * (G + G.T)
    try:
        cov = cvp.cov_precond(G)
    except ValueError as exc:
        print(f"[WARN] cov_precond failed at the stage-1 point ({exc}); using fishinv")
        cov = cvp.fishinv(theta[0], G, index_of_M=0)
    return np.sqrt(np.abs(np.diag(cov)))


# ------------------------------------------------------- seeding from the 0PA fit

def load_seed(cfg, path, truth):
    """The start vector for this run, taken from a finished run's best fit.

    The seed is matched by parameter *name*, so one loader covers every case: a 0PA file
    seeding a PN run (the deviations are absent from the seed and enter at zero), a PN file
    reseeding a PN run, and either of them seeding a run that frees a parameter the seed run
    held fixed -- that parameter enters at its injected value, which is exactly where the
    seed run pinned it, so the seed point is the seed run's own answer and its overlap is the
    overlap the climb starts from.

    A name in the seed that this run does not fit is an error rather than a silent drop: it
    would mean quietly discarding a fitted value, which is how a seed lands in the wrong slot.
    """
    with open(path) as fh:
        src = json.load(fh)
    if src["point"]["idx"] != cfg.point_idx:
        raise ValueError(f"{path} is source {src['point']['idx']}, this run is {cfg.point_idx}")
    extra = [n for n in src["params"] if n not in cfg.param_names]
    if extra:
        raise ValueError(f"{path} fits {extra}, which this run does not: it fits "
                         f"{cfg.param_names}")

    # Anything the seed does not carry falls back to its injected value.  For a deviation
    # that is zero, which is the 0PA template; for a parameter the seed run held fixed it is
    # the value that run pinned it at.  Either way the fallback is where the seed run sat.
    fitted = dict(zip(src["params"], src["stage3_cv"]["params"]))
    theta = np.array([fitted.get(n, inj) for n, inj in zip(cfg.param_names, truth)],
                     dtype=float)

    added = [n for n in cfg.param_names if n not in fitted]
    print(f"[seed] fit from {os.path.basename(path)}: ov = "
          f"{src['stage3_cv']['ov_final']:.10f}, chi2 = {src['stage3_cv']['chi2']:.4f}",
          flush=True)
    if added:
        at = ", ".join(f"{n} = {theta[cfg.param_names.index(n)]:.6g}" for n in added)
        print(f"[seed] not fitted there, entering at its seed-run value: {at}", flush=True)
    print(f"[seed] {cfg.model} starts at that point, so its overlap starts at the seed "
          f"value and the LM climb can only raise it", flush=True)
    return theta


def climb_from_seed(cfg, ctx, theta_seed):
    """The LM climb, straight from the seed point, with no DE and no capped stage first.

    Nothing here needs a trust region.  The capped stage exists because a CV step off the
    injected point is a leap of tens of sigma that has to be reined in; the 0PA fit is already
    at the overlap the run is trying to beat, so the climb starts where the capped walk was
    only ever trying to get to.  Its budget is measured in the Fisher sigma at the seed, which
    is singular at both 0PA points, so imposing it would cap the climb in units that mean
    nothing.

    The LM climb accepts only proposals that raise the overlap, so the result is >= the 0PA
    overlap whatever the deviation direction does.
    """
    print("\n=== LM climb from the seed point ===", flush=True)
    fit, theta = cv_stage(ctx, theta_seed, cfg.cv2_iters, cfg.cv2_nm, "CV2")
    print(stage_line("CV climb", fit), flush=True)
    return fit, theta


# ---------------------------------------------------------------- stage 2

def sigma_scale(cfg, name):
    """The box half-width for one parameter, in sigma."""
    if name in cfg.per_param_sigma:
        return float(cfg.per_param_sigma[name])
    if name in cfg.dev_params:
        return float(cfg.dev_prior_sigma_range)
    return float(cfg.prior_sigma_range)


def safe_sigma(cfg, sigma, theta):
    """Replace a dead or non-finite sigma with 10% of the value, as inference.py does.

    A zero sigma means the Fisher column is dead and the box would collapse to a point,
    which silently removes that parameter from the search.
    """
    out = np.array(sigma, dtype=float)
    for i, n in enumerate(cfg.param_names):
        if not np.isfinite(out[i]) or out[i] == 0.0:
            fallback = max(abs(theta[i]) * 0.1, 1e-4)
            print(f"[WARN] invalid sigma for {n} (was {sigma[i]:.3e}), using {fallback:.3e}",
                  flush=True)
            out[i] = fallback
    return out


def fisher_bounds(cfg, theta, sigma):
    """The DE search box: theta +- scale*sigma per parameter, clipped to physical ranges."""
    bounds = []
    for i, n in enumerate(cfg.param_names):
        if n in cfg.dev_params and cfg.dev_hard_bounds is not None:
            lo, hi = float(cfg.dev_hard_bounds[0]), float(cfg.dev_hard_bounds[1])
        elif cfg.full_phase_bounds and n in ("qS",) + C.ANGLE_PARAMS:
            lo, hi = C.PHYS_LO[n], C.PHYS_HI[n]
        else:
            half = sigma[i] * sigma_scale(cfg, n)
            lo = max(theta[i] - half, C.PHYS_LO.get(n, -np.inf))
            hi = min(theta[i] + half, C.PHYS_HI.get(n, np.inf))
        bounds.append((float(lo), float(hi)))
    return bounds


def print_bounds(cfg, bounds, theta, truth):
    """The box, with the injected value marked in or out of it -- the diagnostic that says
    whether DE could even reach the right answer."""
    print("[DE] Fisher-based bounds:", flush=True)
    for i, n in enumerate(cfg.param_names):
        lo, hi = bounds[i]
        inside = "in " if lo <= truth[i] <= hi else "OUT"
        print(f"  {n:>9}: [{lo:+.8e}, {hi:+.8e}]  centre {theta[i]:+.8e}  "
              f"injected {truth[i]:+.8e} {inside}", flush=True)


def de_objective(cfg, ctx):
    """What DE minimises.  chi2 is the raw residual; overlap is the normalised version."""
    if cfg.de_objective == "chi2":
        def score(x):
            return float(ctx["chi2r"](np.asarray(x, dtype=float)))
    elif cfg.de_objective == "overlap":
        def score(x):
            try:
                return -float(ctx["ov"](np.asarray(x, dtype=float)))
            except Exception:
                return 1e7
    else:
        raise ValueError(f"unknown de_objective: {cfg.de_objective}")
    return counted(cfg, score)


def counted(cfg, score):
    """Wrap the objective in an evaluation counter that prints progress.

    scipy's differential_evolution is silent for as long as it runs, and one evaluation here
    is a full waveform plus the LISA response -- so without this the log shows nothing between
    the bounds and the answer, and there is no way to tell a slow run from a hung one or to
    cost the next one.  The per-evaluation time it prints is the number the DE budget should
    be set from.
    """
    state = dict(n=0, best=np.inf, t0=time.time())

    def wrapped(x):
        val = score(x)
        state["n"] += 1
        if val < state["best"]:
            state["best"] = val
        if state["n"] % cfg.de_report_every == 0:
            dt = time.time() - state["t0"]
            print(f"    [DE] {state['n']:>7} evals  {dt / 3600:6.2f} h  "
                  f"{dt / state['n']:6.2f} s/eval  best = {state['best']:.6e}", flush=True)
        return val

    return wrapped


def run_de(cfg, ctx, theta0, bounds):
    """Differential evolution inside the Fisher box, seeded at the stage-1 point."""
    func = de_objective(cfg, ctx)
    x0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds]) \
        if cfg.de_x0_from_cv else None
    print(f"[DE] ndim = {len(bounds)}, maxiter = {cfg.de_maxiter}, popsize = {cfg.de_popsize}, "
          f"objective = {cfg.de_objective}, init = {cfg.de_init}", flush=True)
    t0 = time.time()
    res = differential_evolution(
        func=func, bounds=bounds, x0=x0,
        maxiter=cfg.de_maxiter, tol=cfg.de_tol, atol=cfg.de_atol,
        popsize=cfg.de_popsize, init=cfg.de_init, mutation=cfg.de_mutation,
        recombination=cfg.de_recombination, polish=cfg.de_polish, seed=cfg.de_seed)
    theta = np.asarray(res.x, dtype=float)
    out = dict(params=theta, fun=float(res.fun), nit=int(res.nit), nfev=int(res.nfev),
               success=bool(res.success), message=str(res.message),
               ov_final=float(ctx["ov"](theta)), chi2=float(ctx["chi2r"](theta)),
               seconds=time.time() - t0)
    print(f"[DE] {out['nit']} iters, {out['nfev']} evals, {out['seconds'] / 3600:.2f} h: "
          f"ov = {out['ov_final']:.10f}, chi2 = {out['chi2']:.6e} ({out['message']})", flush=True)
    return out


# ---------------------------------------------------------------- reporting

def stage_line(name, fit):
    return (f"[stage] {name:<10} ov = {fit['ov_final']:.10f}  chi2 = {fit['chi2']:.6e}  "
            f"{fit.get('seconds', 0.0) / 60:.1f} min")


def print_parameter_table(cfg, truth, stages, sigma):
    """Injected against every stage, with the final bias in sigma -- copy-ready."""
    head = f"{'param':>9} {'injected':>18}" + "".join(f" {n:>18}" for n, _ in stages)
    print("\n" + head + f" {'sigma':>12} {'b/sigma':>10}", flush=True)
    final = stages[-1][1]
    bias = unwrap_bias(cfg, final, truth)
    for i, n in enumerate(cfg.param_names):
        row = f"{n:>9} {truth[i]:>18.10g}" + "".join(f" {th[i]:>18.10g}" for _, th in stages)
        print(row + f" {sigma[i]:>12.4e} {bias[i] / sigma[i]:>+10.4f}", flush=True)


def save(cfg, point, record):
    with open(cfg.out_path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[saved] {cfg.out_path}", flush=True)


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ---------------------------------------------------------------- driver

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="'0PA' or '0PA+PN'")
    ap.add_argument("--point-idx", type=int, default=None,
                    help="which source of the population run to fit")
    ap.add_argument("--point-run", default=None,
                    help="archived run holding the population results; '' is the live run")
    ap.add_argument("--tag", default=None, help="suffix on the output filename")
    ap.add_argument("--de-maxiter", type=int, default=None)
    ap.add_argument("--sigma-range", type=float, default=None,
                    help="DE box half-width in sigma, applied to every parameter including C_p")
    ap.add_argument("--fix-params", default=None,
                    help="comma-separated parameters held at their injected value "
                         "('none' fits everything); default fixes e0")
    ap.add_argument("--cv1-steps", type=int, default=None,
                    help="number of capped CV steps in stage 1; 1 is the old single leap")
    ap.add_argument("--cv1-step-cap", type=float, default=None,
                    help="per-step cap in sigma at the injected point")
    ap.add_argument("--cv1-total-cap", type=float, default=None,
                    help="cap on the whole stage-1 walk, same units")
    ap.add_argument("--cp-margin", type=float, default=None,
                    help="C_p box in multiples of its decoherence scale")
    ap.add_argument("--no-cv1-backtrack", action="store_true",
                    help="take the full undamped CV step even if it lowers the overlap")
    ap.add_argument("--cv1-iters", type=int, default=None,
                    help="LM iterations in stage 1 when --damped-cv1 is used")
    ap.add_argument("--damped-cv1", action="store_true",
                    help="stage 1 uses LM-damped iterations instead of the exact CV step")
    ap.add_argument("--seed-from", dest="seed_from", default=None,
                    help="any results_cv_de_cv_*.json to start from; skips DE and climbs "
                         "from that fit, matching by parameter name")
    ap.add_argument("--seed-from-0pa", dest="seed_from", default=None,
                    help="the former name of --seed-from, kept so the earlier batch scripts "
                         "still run")
    ap.add_argument("--box-at-injection", action="store_true",
                    help="skip stage 1 and centre the DE box on the injected point itself")
    ap.add_argument("--full-phase-bounds", action="store_true",
                    help="let qS, phiS, Phi_phi0 span their full physical range")
    return ap.parse_args()


def config_from_args(args):
    over = {}
    if args.model is not None:
        over["model"] = args.model
    if args.point_idx is not None:
        over["point_idx"] = args.point_idx
    if args.point_run is not None:
        over["point_run"] = args.point_run
    if args.tag is not None:
        over["tag"] = args.tag
    if args.de_maxiter is not None:
        over["de_maxiter"] = args.de_maxiter
    if args.sigma_range is not None:
        # both, so that 0PA and 0PA+PN are the same box on the shared parameters
        over["prior_sigma_range"] = args.sigma_range
        over["dev_prior_sigma_range"] = args.sigma_range
    if args.full_phase_bounds:
        over["full_phase_bounds"] = True
    if args.box_at_injection:
        over["box_at_injection"] = True
    if args.damped_cv1:
        over["cv1_exact"] = False
    if args.no_cv1_backtrack:
        over["cv1_backtrack"] = False
    if args.fix_params is not None:
        over["fixed_params"] = () if args.fix_params.strip().lower() == "none" else \
            tuple(n.strip() for n in args.fix_params.split(",") if n.strip())
    if args.cv1_steps is not None:
        over["cv1_steps"] = args.cv1_steps
    if args.cv1_iters is not None:
        over["cv1_iters"] = args.cv1_iters
    if args.cv1_step_cap is not None:
        over["cv1_step_cap_sigma"] = args.cv1_step_cap
    if args.cv1_total_cap is not None:
        over["cv1_total_cap_sigma"] = args.cv1_total_cap
    if args.cp_margin is not None:
        over["dev_decoh_margin"] = args.cp_margin
    if args.seed_from is not None:
        over["seed_fit"] = args.seed_from
    return Config(**over)


def main():
    cfg = config_from_args(parse_args())
    apply_cv_controls(cfg, cvp)
    point = load_point(cfg)
    describe_point(point)
    print(f"[cfg] model = {cfg.model}, parameters = {', '.join(cfg.param_names)}", flush=True)

    t_all = time.time()
    S, ctx = build_context(cfg, point)
    truth = truth_vector(point, cfg.param_names)
    theta_start = wrap_angles(cfg, truth)
    dev_deltas = calibrate_if_pn(cfg, ctx, theta_start)

    # C_p is bounded by where it stops meaning anything, not by 15 * a sigma that has come out
    # anywhere between 7e-3 and 1.5e6 on comparable sources.  Set before stage 1 so the value
    # used is on the record even if the run dies in DE.
    if cfg.dev_params and cfg.dev_hard_bounds is None and cfg.dev_bounds_from_decoherence:
        cfg.dev_hard_bounds = dev_decoherence_bound(cfg, dev_deltas)

    if cfg.seed_fit:
        theta_seed = load_seed(cfg, cfg.seed_fit, truth)
        fit3, theta3 = climb_from_seed(cfg, ctx, theta_seed)
        fit1, theta1 = fit3, theta3          # no separate stage 1; the record keeps one climb
        # the seeded branch has no stage-1 Fisher box, so there is no stage-1 sigma to record
        sigma1, bounds, de, theta2 = None, None, None, theta3
        stages = [("seed", theta_seed), ("CV climb", theta3)]
    else:
        print("\n=== stage 1: CV step from the injected point ===", flush=True)
        if cfg.box_at_injection:
            fit1, theta1 = no_stage1(ctx, theta_start)
        elif cfg.cv1_exact and cfg.cv1_steps > 1:
            fit1, theta1 = cv_iterate_stage1(cfg, ctx, theta_start, "1")
        elif cfg.cv1_exact:
            fit1, theta1 = cv_step_exact(ctx, theta_start, "1",
                                         backtrack=cfg.cv1_backtrack,
                                         max_halvings=cfg.cv1_max_halvings)
        else:
            fit1, theta1 = cv_stage(ctx, theta_start, cfg.cv1_iters, cfg.cv1_nm, "CV1")
        if not cfg.box_at_injection:
            print(stage_line("CV step", fit1), flush=True)

        print("\n=== stage 2: differential evolution in the Fisher box ===", flush=True)
        sigma1 = safe_sigma(cfg, stage1_sigma(ctx, theta1), theta1)
        bounds = fisher_bounds(cfg, wrap_angles(cfg, theta1), sigma1)
        print_bounds(cfg, bounds, wrap_angles(cfg, theta1), truth)
        de = run_de(cfg, ctx, wrap_angles(cfg, theta1), bounds)
        theta2 = de["params"]

        print("\n=== stage 3: CV climb from the DE point ===", flush=True)
        fit3, theta3 = cv_stage(ctx, theta2, cfg.cv2_iters, cfg.cv2_nm, "CV2")
        print(stage_line("CV climb", fit3), flush=True)
        centre = "injected" if cfg.box_at_injection else "CV step"
        stages = [(centre, theta1), ("DE", theta2), ("CV climb", theta3)]

    cov = cvp.cov_at(ctx, theta3, f"{cfg.model} final")
    sigma3 = np.sqrt(np.abs(np.diag(cov)))
    bias = unwrap_bias(cfg, theta3, truth)

    print_parameter_table(cfg, truth, stages, sigma3)
    if cfg.seed_fit:
        print(f"\n[done] ov: seed {fit3['ov_start']:.10f} -> LM climb "
              f"{fit3['ov_final']:.10f}", flush=True)
    elif cfg.box_at_injection:
        print(f"\n[done] ov: injected {fit1['ov_start']:.10f} -> DE "
              f"{de['ov_final']:.10f} -> CV climb {fit3['ov_final']:.10f}", flush=True)
    else:
        print(f"\n[done] ov: injected {fit1['ov_start']:.10f} -> CV step "
              f"{fit1['ov_final']:.10f} -> DE {de['ov_final']:.10f} -> CV climb "
              f"{fit3['ov_final']:.10f}", flush=True)
    print(f"[done] population run reached {point['source']['fit_0pa']['ov_final']:.10f} (0PA), "
          f"{point['source']['fit_pn']['ov_final']:.10f} (0PA+PN)", flush=True)
    print(f"[done] total {(time.time() - t_all) / 3600:.2f} h", flush=True)

    save(cfg, point, jsonable(dict(
        study=("harder environmental points: seeded LM climb" if cfg.seed_fit
               else "harder environmental points: CV -> DE -> CV"),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=cfg.as_dict(),
        point=dict(run=point["from_run"], idx=point["source"]["idx"],
                   m1=point["source"]["m1"], T=point["source"]["T"],
                   dt=point["source"]["dt"], dist=point["source"]["dist"],
                   snr=point["source"]["snr"], A_PM=point["source"]["A_PM"],
                   n_PM=point["source"]["n_PM"]),
        params=cfg.param_names, truth=truth,
        population_run=dict(ov_0pa=point["source"]["fit_0pa"]["ov_final"],
                            ov_pn=point["source"]["fit_pn"]["ov_final"],
                            chi2_0pa=point["source"]["fit_0pa"]["chi2"],
                            chi2_pn=point["source"]["fit_pn"]["chi2"]),
        dev_delta_range=dev_deltas,
        stage1_cv=fit1, stage1_sigma=sigma1, de_bounds=bounds, stage2_de=de,
        stage3_cv=fit3, sigma=sigma3, bias=bias, bias_over_sigma=bias / sigma3,
        seconds=time.time() - t_all)))


if __name__ == "__main__":
    main()
