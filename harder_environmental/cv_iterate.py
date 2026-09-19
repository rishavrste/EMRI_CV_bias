"""Iterated exact Cutler-Vallisneri steps, seeded from a finished DE point.

The three-stage run in `cv_de_cv.py` ends with a damped LM climb, and on source 8 that
climb stalls: 19 iterations from the DE point move the overlap 0.7162 -> 0.7183 and then
stop with `lambda_exhausted`.  LM cannot escape, because a rejected step multiplies lambda
by LAMBDA_UP until the proposal is too small to matter.

This file tries the other thing.  It takes the same DE point and applies the *undamped* CV
step, `dtheta = Gamma^-1 <dh|s-h>`, over and over, recomputing the Fisher at each new point.
That is a fixed-point iteration on the CV formalism's own prediction rather than a descent:
chi2 is allowed to rise, and no step is ever rejected.  If the CV linearisation is valid
anywhere near the DE point, the iteration contracts onto the truth; if it is not, it will
wander or diverge, and the per-iteration record below is what says which happened.

The seed is a 0PA point, so running the 0PA+PN model here means recovering the deviation
from a vacuum starting guess with C_p = 0.  That also sidesteps the box that cripples the
plain PN run, whose stage-1 Fisher put C_p in [1613, 2563] while the truth is 0.

Bookkeeping: every iteration is recorded whole -- the point, the step, the step in units of
sigma, whatever was clipped at the physical boundary, the overlap, chi2 and the bias against
the injected truth.  Nothing is summarised away, because the question this run answers is
about the *trajectory*, not the endpoint.
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
import cv_de_cv as CD
from cv_de_cv import (build_context, calibrate_if_pn, cv_step_exact, describe_point,
                      jsonable, load_point, truth_vector, unwrap_bias, wrap_angles)

sys.path.insert(0, C.ENV_TESTS)
import cv_population as cvp          # noqa: E402  (path must be set first)

# Value given to a parameter the seed model did not carry.  C_p = 0 is the vacuum
# limit, i.e. "start by assuming no deviation and let the iteration find it".
SEED_DEFAULTS = {"C_p": 0.0}


# ---------------------------------------------------------------- the seed

def seed_record(path, stage):
    """The raw stage record the seed is taken from, with its provenance kept."""
    if not os.path.exists(path):
        raise SystemExit(f"[seed] no such results file: {path}")
    with open(path) as f:
        data = json.load(f)
    if stage not in data:
        raise SystemExit(f"[seed] {path} has no '{stage}' (keys: {sorted(data)})")
    return data, data[stage]


def seed_vector(data, stage_rec, want_names):
    """Re-order the seed onto `want_names`, filling parameters the seed model lacked.

    The seed file stores its own column list in `params`, so a 0PA seed can be lifted into
    the 9-parameter PN model without assuming either ordering.
    """
    have = dict(zip(data["params"], stage_rec["params"]))
    out, filled = [], []
    for n in want_names:
        if n in have:
            out.append(float(have[n]))
        elif n in SEED_DEFAULTS:
            out.append(float(SEED_DEFAULTS[n]))
            filled.append(n)
        else:
            raise SystemExit(f"[seed] seed has no '{n}' and no default for it")
    return np.array(out, dtype=float), filled


def describe_seed(path, stage, data, stage_rec, filled):
    print(f"[seed] {os.path.basename(path)} :: {stage}", flush=True)
    print(f"[seed]   model {data['config']['model']}, params {data['params']}", flush=True)
    print(f"[seed]   ov = {stage_rec.get('ov_final', float('nan')):.10f}  "
          f"chi2 = {stage_rec.get('chi2', float('nan')):.6e}", flush=True)
    if filled:
        print(f"[seed]   filled from defaults: "
              + ", ".join(f"{n} = {SEED_DEFAULTS[n]}" for n in filled), flush=True)


# ------------------------------------------------------------- the iteration

def step_sigma(fit):
    """The Fisher sigma cv_step_exact measured at the point the step started from."""
    return np.asarray(fit["sigma"], dtype=float)


def iteration_record(k, theta_in, fit, theta_out, cfg, truth):
    """Everything about one step, in the shape the results file keeps it."""
    sigma = step_sigma(fit)
    dtheta = np.asarray(theta_out, dtype=float) - np.asarray(theta_in, dtype=float)
    bias = unwrap_bias(cfg, theta_out, truth)
    return dict(
        iter=k,
        ov_start=fit["ov_start"], ov_final=fit["ov_final"], chi2=fit["chi2"],
        params=np.asarray(theta_out, dtype=float),
        params_unclipped=fit["params_unclipped"],
        clipped=fit["clipped"],
        dtheta=dtheta,
        dtheta_over_sigma=dtheta / np.where(sigma > 0, sigma, np.inf),
        sigma=sigma,
        bias=bias,
        bias_over_sigma=bias / np.where(sigma > 0, sigma, np.inf),
        seconds=fit["seconds"],
    )


def print_iteration(cfg, rec, best_ov):
    mark = "  <-- best" if rec["ov_final"] >= best_ov else ""
    print(f"  [it {rec['iter']:3d}] ov {rec['ov_start']:.10f} -> {rec['ov_final']:.10f}   "
          f"chi2 {rec['chi2']:.6e}   |dtheta/sigma|max = "
          f"{np.max(np.abs(rec['dtheta_over_sigma'])):.3e}   "
          f"{rec['seconds']:.0f} s{mark}", flush=True)
    if rec["clipped"]:
        print("           clipped: "
              + ", ".join(c["param"] for c in rec["clipped"]), flush=True)


def should_stop(cfg, history, best_ov, patience):
    """Stop on success, on a converged step, or after `patience` iterations off the best.

    The undamped step is not a descent, so a single bad iteration is not a reason to stop --
    only a run of them is.  Returns a reason string, or None to keep going.
    """
    last = history[-1]
    if last["ov_final"] >= cfg.overlap_target:
        return "overlap_target"
    if np.max(np.abs(last["dtheta_over_sigma"])) < 1e-3:
        return "step_below_1e-3_sigma"
    if not np.isfinite(last["ov_final"]):
        return "non_finite_overlap"
    recent = [h["ov_final"] for h in history[-patience:]]
    if len(history) >= patience and max(recent) < best_ov:
        return f"no_improvement_in_{patience}"
    return None


def iterate_cv(cfg, ctx, theta0, truth, n_iter, patience):
    """Apply the exact CV step `n_iter` times, keeping every intermediate point."""
    theta = np.asarray(theta0, dtype=float)
    history, best = [], dict(ov=-np.inf, iter=-1, params=theta.copy())
    for k in range(n_iter):
        fit, theta_next = cv_step_exact(ctx, theta, f"it{k}")
        rec = iteration_record(k, theta, fit, theta_next, cfg, truth)
        history.append(rec)
        print_iteration(cfg, rec, best["ov"])
        if rec["ov_final"] > best["ov"]:
            best = dict(ov=rec["ov_final"], chi2=rec["chi2"], iter=k,
                        params=np.asarray(theta_next, dtype=float).copy())
        theta = wrap_angles(cfg, theta_next)
        stop = should_stop(cfg, history, best["ov"], patience)
        if stop:
            print(f"  [stop] {stop} after {k + 1} iterations", flush=True)
            return history, best, stop
    return history, best, "max_iters"


# ------------------------------------------------------------------ report

def print_trajectory(cfg, history, truth, seed_ov):
    """The overlap trace, which is the whole point of the run."""
    print("\n=== overlap trajectory ===", flush=True)
    print(f"  seed  {seed_ov:.10f}", flush=True)
    for r in history:
        print(f"  it {r['iter']:3d}  {r['ov_final']:.10f}   chi2 {r['chi2']:.6e}", flush=True)


def print_final_table(cfg, truth, seed, best, sigma):
    print(f"\n{'param':>10} {'injected':>18} {'seed (DE)':>18} {'best iterate':>18} "
          f"{'sigma':>12} {'b/sigma':>11}", flush=True)
    bias = unwrap_bias(cfg, best["params"], truth)
    for i, n in enumerate(cfg.param_names):
        print(f"{n:>10} {truth[i]:>18.10g} {seed[i]:>18.10g} {best['params'][i]:>18.10g} "
              f"{sigma[i]:>12.4e} {bias[i] / sigma[i]:>+11.4f}", flush=True)


# ------------------------------------------------------------------ driver

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="0PA+PN", help="'0PA' or '0PA+PN'")
    ap.add_argument("--tag", default=None, help="suffix for the results file")
    ap.add_argument("--seed-results", default="results_cv_de_cv_0pa.json",
                    help="results file the seed point is read from")
    ap.add_argument("--seed-stage", default="stage2_de",
                    help="which stage of that file to seed from (stage2_de, stage3_cv, ...)")
    ap.add_argument("--iters", type=int, default=40, help="maximum CV steps")
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many iterations without beating the best overlap")
    return ap.parse_args()


def build_config(args):
    cfg = Config()
    cfg.model = args.model
    cfg.tag = args.tag or ("pn" if args.model == "0PA+PN" else "0pa")
    return cfg


def results_path(cfg):
    """Named for what the run is, not for which script wrote it."""
    return os.path.join(cfg.out_dir, f"results_cv_iter_{cfg.tag}.json")


def main():
    args = parse_args()
    cfg = build_config(args)
    apply_cv_controls(cfg, cvp)

    seed_path = os.path.join(cfg.out_dir, args.seed_results) \
        if not os.path.isabs(args.seed_results) else args.seed_results
    data, stage_rec = seed_record(seed_path, args.seed_stage)
    point = load_point(cfg)
    describe_point(point)
    print(f"[cfg] model = {cfg.model}, parameters = {', '.join(cfg.param_names)}", flush=True)

    theta0, filled = seed_vector(data, stage_rec, cfg.param_names)
    describe_seed(seed_path, args.seed_stage, data, stage_rec, filled)

    t_all = time.time()
    S, ctx = build_context(cfg, point)
    truth = truth_vector(point, cfg.param_names)
    dev_deltas = calibrate_if_pn(cfg, ctx, theta0)
    theta0 = wrap_angles(cfg, theta0)

    ov_seed = float(ctx["ov"](theta0))
    print(f"\n=== iterated exact CV steps from {args.seed_stage} ===", flush=True)
    print(f"    seed overlap in this model = {ov_seed:.10f}  "
          f"(seed file recorded {stage_rec.get('ov_final', float('nan')):.10f})", flush=True)

    history, best, stop = iterate_cv(cfg, ctx, theta0, truth, args.iters, args.patience)

    cov = cvp.cov_at(ctx, best["params"], f"{cfg.model} best iterate")
    sigma = np.sqrt(np.abs(np.diag(cov)))
    print_trajectory(cfg, history, truth, ov_seed)
    print_final_table(cfg, truth, theta0, best, sigma)

    record = dict(
        study="harder environmental: iterated exact CV steps from a DE point",
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        cli=vars(args),
        seed=dict(path=seed_path, stage=args.seed_stage,
                  model=data["config"]["model"], params=data["params"],
                  ov_in_seed_file=stage_rec.get("ov_final"),
                  ov_in_this_model=ov_seed,
                  filled_from_defaults={n: SEED_DEFAULTS[n] for n in filled},
                  theta=theta0),
        point={k: point["source"][k] for k in ("idx", "m1", "p0", "snr", "T", "dt",
                                               "A_PM", "n_PM")},
        params=cfg.param_names,
        truth=truth,
        population_run=dict(ov_0pa=point["source"]["fit_0pa"]["ov_final"],
                            ov_pn=point["source"]["fit_pn"]["ov_final"]),
        iterations=history,
        best=dict(**best, sigma=sigma, bias=unwrap_bias(cfg, best["params"], truth)),
        stop_reason=stop,
        n_iterations=len(history),
        seconds=time.time() - t_all,
    )
    out = results_path(cfg)
    with open(out, "w") as f:
        json.dump(jsonable(record), f, indent=1)
    print(f"\n[done] seed {ov_seed:.10f} -> best {best['ov']:.10f} "
          f"at iteration {best['iter']} ({stop})", flush=True)
    print(f"[done] population run reached {point['source']['fit_0pa']['ov_final']:.10f} (0PA), "
          f"{point['source']['fit_pn']['ov_final']:.10f} (0PA+PN)", flush=True)
    print(f"[done] total {(time.time() - t_all) / 3600:.2f} h", flush=True)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
