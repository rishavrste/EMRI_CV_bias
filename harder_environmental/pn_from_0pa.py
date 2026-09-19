"""One 0PA+PN CV climb seeded from the converged 0PA answer, with C_p = C_e = 0.

The CV -> DE -> CV run gave 0PA ov = 0.9484 and 0PA+PN ov = 0.8907 on run_2/14.  That
ordering is impossible for a converged pair: 0PA is a strict subspace of 0PA+PN -- set the
two deviation parameters to zero and the models are identical -- so the PN likelihood has a
point worth 0.9484 that its search failed to find.  The 0PA answer is even inside the PN DE
box, within 1.55 half-widths on every intrinsic, so the box was not the problem; DE simply
got lost in the two extra dimensions.

The fix is the dual-seeding cv_population already uses on the population run, applied to this
one point: start the PN climb from the 0PA answer instead of from a DE winner.  By
construction it begins at the 0PA overlap, so ov_PN >= ov_0PA holds whatever the climb does,
and whatever C_p and C_e drift to is measured in the basin 0PA found rather than in whatever
basin DE wandered into.

There is no DE here.  One LM -> Nelder-Mead -> LM climb, which is the same stage 3
cv_de_cv.py runs, at a cost of minutes rather than the nine hours the DE stage took.

`--seed-stage` picks which point of the 0PA run to start from, and the choice is the whole
experiment -- see `load_0pa_answer`.  `cv3`, the converged 0PA answer, guarantees the ordering
but starts at a maximum where the vacuum gradient is already zero, so the deviation parameters
never get to trade against the intrinsics on the way up.  `de`, the DE winner, sits in the same
basin a little below the top and leaves the 11-dimensional climb room to find its own maximum.

Run:  python pn_from_0pa.py                               # seed cv3, the 0PA answer
      python pn_from_0pa.py --seed-stage de --out results_pn_from_0pa_de.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

import config as C
from config import Config, apply_cv_controls

sys.path.insert(0, C.ENV_TESTS)
import cv_population as cvp          # noqa: E402  (path must be set first)

from cv_de_cv import (build_context, calibrate_if_pn, cv_stage, describe_point,  # noqa: E402
                      jsonable, load_point, print_parameter_table, stage_line,
                      truth_vector, unwrap_bias, wrap_angles)


# ---------------------------------------------------------------- the seed

# Which point of the 0PA run to start the PN climb from.  `cv3` is the converged answer, `de`
# the differential-evolution winner it climbed from, `cv1` the single CV step off the injected
# point.  Each maps to (record key, the field holding the parameters).
SEED_STAGES = {"cv1": ("stage1_cv", "params"),
               "de": ("stage2_de", "params"),
               "cv3": ("stage3_cv", "params")}


def load_0pa_answer(path, stage):
    """One point out of the 0PA run, as the seed for this climb.

    `cv3`, the converged 0PA answer, is the seed that makes ov_PN >= ov_0PA true by
    construction -- but it is also a maximum, so the gradient in all nine vacuum directions is
    already zero and LM has nothing to push against.  C_p and C_e can then only move if they
    improve the fit on their own; the coupled motion, where the deviation parameters trade
    against m1, p0 and e0 on the way up, never happens.  A run from there answers "do the
    deviation parameters help at the 0PA maximum", which is a narrower question than the one
    worth asking.

    `de` is the honest seed for that wider question: the same basin -- 0PA climbed from it to
    its answer -- but short of the top, so the full 11-dimensional climb has room to find its
    own maximum, which need not sit above the 0PA one in the vacuum coordinates.  It gives up
    the guarantee; compare against the 0PA overlap in the output and keep the better of the
    two seeds, exactly as cv_population does across the population.
    """
    with open(path) as f:
        d = json.load(f)
    if d["params"] != C.PARAMS_9:
        raise SystemExit(f"[seed] {path} is not a 0PA run: parameters are {d['params']}")
    key, field = SEED_STAGES[stage]
    rec = d[key]
    return dict(path=path, stage=stage, point=d["point"],
                theta=np.array(rec[field], float),
                ov=float(rec["ov_final"]), chi2=float(rec["chi2"]),
                ov_0pa_best=float(d["stage3_cv"]["ov_final"]),
                chi2_0pa_best=float(d["stage3_cv"]["chi2"]),
                theta_0pa_best=np.array(d["stage3_cv"]["params"], float))


def check_same_point(cfg, seed, point):
    """Refuse a seed from a different source -- the climb would start nowhere near the signal."""
    want = (cfg.point_run, cfg.point_idx)
    got = (seed["point"]["run"], seed["point"]["idx"])
    if got != want:
        raise SystemExit(f"[seed] {os.path.basename(seed['path'])} is {got[0]}/{got[1]}, "
                         f"config wants {want[0]}/{want[1]}")
    if not np.isclose(seed["point"]["m1"], point["source"]["m1"], rtol=1e-12):
        raise SystemExit("[seed] m1 disagrees with the point being run")


def pad_with_zero_deviations(theta9):
    """The 9-parameter 0PA answer as an 11-parameter PN point: C_p = C_e = 0.

    Zero is the injected value of both, so this is the 0PA fit read as a PN template, and its
    overlap is identical to the 0PA overlap up to the finite-difference noise in nothing at
    all -- the waveform call is the same one.
    """
    return np.concatenate([np.asarray(theta9, float), np.zeros(2)])


# ---------------------------------------------------------------- reporting

def report_ordering(seed, fit, pn_de):
    """The one thing this run exists to settle: does PN beat the 0PA answer?

    The comparison is always against the converged 0PA overlap, never against the seed --
    a `de`-seeded climb starts below it on purpose, so beating its own start proves nothing.
    """
    print(f"\n[order] 0PA (CV->DE->CV)      ov = {seed['ov_0pa_best']:.10f}  "
          f"chi2 = {seed['chi2_0pa_best']:.6e}", flush=True)
    if pn_de is not None:
        print(f"[order] 0PA+PN (CV->DE->CV)   ov = {pn_de['ov']:.10f}  chi2 = {pn_de['chi2']:.6e}",
              flush=True)
    print(f"[order] seed ({seed['stage']})            ov = {seed['ov']:.10f}  "
          f"chi2 = {seed['chi2']:.6e}", flush=True)
    print(f"[order] 0PA+PN (from {seed['stage']})      ov = {fit['ov_final']:.10f}  "
          f"chi2 = {fit['chi2']:.6e}", flush=True)
    gain = fit["ov_final"] - seed["ov_0pa_best"]
    verdict = "PN >= 0PA" if gain >= -1e-12 else "BELOW 0PA -- this seed lost it, keep the other"
    print(f"[order] ov_PN - ov_0PA = {gain:+.3e}   {verdict}", flush=True)
    print(f"[order] the climb moved {fit['ov_final'] - seed['ov']:+.3e} in overlap from its seed",
          flush=True)


def save(path, record):
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[saved] {path}", flush=True)


def build_record(cfg, point, seed, pn_de, fit, theta, sigma, truth, dev_deltas, seconds):
    bias = unwrap_bias(cfg, theta, truth)
    return jsonable(dict(
        study="harder environmental points: 0PA+PN CV climb seeded from the 0PA answer",
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=cfg.as_dict(),
        point=dict(run=point["from_run"], idx=point["source"]["idx"],
                   m1=point["source"]["m1"], T=point["source"]["T"], dt=point["source"]["dt"],
                   dist=point["source"]["dist"], snr=point["source"]["snr"],
                   A_PM=point["source"]["A_PM"], n_PM=point["source"]["n_PM"]),
        params=cfg.param_names, truth=truth,
        seed=dict(source=os.path.basename(seed["path"]), model="0PA", stage=seed["stage"],
                  ov=seed["ov"], chi2=seed["chi2"], theta_9=seed["theta"],
                  ov_0pa_best=seed["ov_0pa_best"], chi2_0pa_best=seed["chi2_0pa_best"],
                  theta_9_0pa_best=seed["theta_0pa_best"]),
        reference=dict(pn_cv_de_cv=pn_de,
                       population_run=dict(ov_0pa=point["source"]["fit_0pa"]["ov_final"],
                                           ov_pn=point["source"]["fit_pn"]["ov_final"])),
        dev_delta_range=dev_deltas,
        cv_climb=fit, sigma=sigma, bias=bias, bias_over_sigma=bias / sigma,
        seconds=seconds))


# ---------------------------------------------------------------- driver

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-json", default="results_cv_de_cv_0pa.json",
                    help="the 0PA run this climb is seeded from")
    ap.add_argument("--seed-stage", default="cv3", choices=sorted(SEED_STAGES),
                    help="which point of that run to start at: cv3 the converged answer "
                         "(guarantees PN >= 0PA but starts at a maximum), de the DE winner "
                         "(room to climb, no guarantee), cv1 the single CV step")
    ap.add_argument("--pn-json", default="results_cv_de_cv_pn.json",
                    help="the CV->DE->CV PN run, reported alongside for comparison")
    ap.add_argument("--iters", type=int, default=None, help="LM iteration cap (default cv2_iters)")
    ap.add_argument("--out", default="results_pn_from_0pa.json")
    return ap.parse_args()


def load_pn_reference(path):
    """The PN CV->DE->CV answer, for the comparison line.  Absent is fine."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return dict(ov=float(d["stage3_cv"]["ov_final"]), chi2=float(d["stage3_cv"]["chi2"]),
                params=d["stage3_cv"]["params"])


def main():
    args = parse_args()
    cfg = Config(model="0PA+PN")
    apply_cv_controls(cfg, cvp)

    here = os.path.dirname(os.path.abspath(__file__))
    seed = load_0pa_answer(os.path.join(here, args.seed_json), args.seed_stage)
    pn_de = load_pn_reference(os.path.join(here, args.pn_json))

    point = load_point(cfg)
    describe_point(point)
    check_same_point(cfg, seed, point)
    print(f"[cfg] model = {cfg.model}, parameters = {', '.join(cfg.param_names)}", flush=True)
    print(f"[seed] {os.path.basename(seed['path'])} stage {seed['stage']}: "
          f"ov = {seed['ov']:.10f}, chi2 = {seed['chi2']:.6e}  "
          f"(that run's 0PA answer: {seed['ov_0pa_best']:.10f})", flush=True)

    t_all = time.time()
    S, ctx = build_context(cfg, point)
    truth = truth_vector(point, cfg.param_names)
    theta_seed = wrap_angles(cfg, pad_with_zero_deviations(seed["theta"]))
    dev_deltas = calibrate_if_pn(cfg, ctx, theta_seed)

    ov_seed = float(ctx["ov"](theta_seed))
    print(f"[seed] the same point read as a PN template: ov = {ov_seed:.10f} "
          f"(the 0PA run said {seed['ov']:.10f})", flush=True)

    print("\n=== CV climb from the 0PA answer, C_p = C_e = 0 ===", flush=True)
    iters = args.iters if args.iters is not None else cfg.cv2_iters
    fit, theta = cv_stage(ctx, theta_seed, iters, cfg.cv2_nm, "PN0")
    print(stage_line("CV climb", fit), flush=True)

    sigma = np.sqrt(np.abs(np.diag(cvp.cov_at(ctx, theta, f"{cfg.model} from-0PA"))))
    print_parameter_table(cfg, truth,
                          [(f"seed ({seed['stage']})", theta_seed), ("CV climb", theta)], sigma)
    report_ordering(seed, fit, pn_de)
    print(f"[done] total {(time.time() - t_all) / 60:.1f} min", flush=True)

    save(os.path.join(here, args.out),
         build_record(cfg, point, seed, pn_de, fit, theta, sigma, truth, dev_deltas,
                      time.time() - t_all))


if __name__ == "__main__":
    main()
