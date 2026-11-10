"""13-parameter Fisher at the fits of the arms: signal SNR 20, T = 2.5 yr, SEF stable derivatives.

A fit still running is taken at its latest point. Each Fisher is repeated with SEF's steps doubled, to
show how far the near-singular Gamma depends on the derivative step.
    -> results/fisher_at_best/fisher_{label}_{arm}.json

Run (GPU):  python fisher_at_best.py --points 0 4 --arms best_ladder
"""
import argparse
import json

import numpy as np

import config as C
from lm import fisher_and_gradient
from model import build_grid, build_point, evaluate, point_T_dt
from run_lm_arms import check_branch

FISHER_DIR = C.OUT_ROOT / "fisher_at_best"
PHASES = ("Phi_phi0", "Phi_r0")


def wrap_phase(x):
    """x mapped into (-pi, pi]."""
    return float(np.pi - np.mod(np.pi - x, 2.0 * np.pi))


def load_fit(point, arm):
    return json.loads((C.OUT_ROOT / arm / f"lm_{C.point_label(point)}.json").read_text())


def bias_vector(point, theta):
    sig_row = C.signal_row(point)
    b = [t - C.injected_value(sig_row, n) for n, t in zip(C.FIT_PARAMS, theta)]
    return [wrap_phase(v) if n in PHASES else float(v) for n, v in zip(C.FIT_PARAMS, b)]


def fisher_at(P, theta, deltas):
    dH, used = P["derivs"](theta, deltas)
    G, _ = fisher_and_gradient(P, dH, P["residual"](P["make_h"](theta)))
    return G, used


def run(Gd, point, arm):
    fit = load_fit(point, arm)
    source = fit.get("theta_final") or fit["theta_current"]
    theta = np.array([source[n] for n in C.FIT_PARAMS])
    P = build_point(Gd, C.signal_row(point), C.FIT_PARAMS)
    ev = evaluate(P, theta)
    G, deltas = fisher_at(P, theta, None)
    doubled = {n: 2.0 * float(v) for n, v in deltas.items()}
    G2, _ = fisher_at(P, theta, doubled)
    print(f"[{arm} {C.point_label(point)}] O={ev['overlap']:.10f} rho_s={ev['rho_s']:.6f} "
          f"running={fit.get('running')}", flush=True)
    FISHER_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(point=point, arm=arm, running=bool(fit.get("running")), names=C.FIT_PARAMS,
               theta=theta.tolist(), bias=bias_vector(point, theta), final_eval=ev, fisher=G.tolist(),
               fisher_doubled_steps=G2.tolist(), deltas={n: float(v) for n, v in deltas.items()})
    (FISHER_DIR / f"fisher_{C.point_label(point)}_{arm}.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=C.POINTS, required=True)
    ap.add_argument("--arms", nargs="+", choices=list(C.ARMS), required=True)
    args = ap.parse_args()
    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    Gd = build_grid(xp, use_gpu, C.T_FULL, point_T_dt(C.signal_row(args.points[0]))[1])
    for point in args.points:
        for arm in args.arms:
            try:
                run(Gd, point, arm)
            except Exception as exc:                        # noqa: BLE001  keep the batch going
                print(f"[FAIL {arm} {C.point_label(point)}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
