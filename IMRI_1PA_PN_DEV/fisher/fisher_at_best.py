"""Fisher matrix at the best fits of both templates, for the 1D and nD bias (as in IMRI_TAIL_dev).

    0pa : the 0PA best fit (config.BEST_FITS, best_fits.py), its 9 parameters
    pn  : the 0PA + PN best fit, all 11 (C_p, C_e included)
Gamma_ij = <d_i h|d_j h> at theta_final, SEF stable derivatives, the rishav inner product (DC bin
dropped), 2nd-generation TDI A and E, the point's stored T and dt (T 1 yr, dt 10, signal SNR 20); Gamma scales as
SNR^2. bias = theta_final - theta_inj, the phases wrapped into (-pi, pi].
Step search: a parameter near zero (config.NEAR_ZERO: the deviation coefficients at round-off level,
the spin at the a = 0 points) gets SEF's at-zero grid of absolute steps, config.ZERO_STEPS, in
model.derivs; theta unchanged.
    -> results/fisher_at_best/{template}/fisher_{label}.json
--steps-from SRC ...: step-sensitivity test. SEF's step search is skipped and the steps SEF chose at
point SRC (same template, read from its stored Fisher) are used as they are, once per SRC.
    -> results/fisher_at_best/{template}_steps/fisher_{label}_from{SRC}.json

Run (GPU, from fisher/):  python fisher_at_best.py --templates 0pa pn --points 0 1 2 [--steps-from 12 17]
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # IMRI_1PA_PN_DEV: config, model, lm
import config as C                                  # noqa: E402
from lm import fisher_and_gradient, sigma_from_fisher  # noqa: E402
from model import build_grid, build_point, evaluate, point_T_dt, zero_step_range  # noqa: E402
from run_lm_T_ladder import check_branch, free_gpu  # noqa: E402

def load_best_fit(template, point):
    return json.loads(C.BEST_FITS.read_text())[template][point]


def wrap_phase(x):
    """x mapped into (-pi, pi]."""
    return float(np.pi - np.mod(np.pi - x, 2.0 * np.pi))


def bias_vector(names, theta, theta_inj):
    b = np.asarray(theta) - np.asarray(theta_inj)
    return np.array([wrap_phase(v) if n in C.PHASES else v for n, v in zip(names, b)])


def stored_steps(template, point):
    """The steps SEF chose at point (its stored Fisher)."""
    path = C.FISHER_DIR / template / f"fisher_{C.point_label(point)}.json"
    return ast.literal_eval(json.loads(path.read_text())["deltas"])


def fisher_at(P, theta, delta_range, deltas=None):
    dH, deltas = P["derivs"](theta, deltas, None if deltas else (delta_range or None))
    G, _ = fisher_and_gradient(P, dH, P["residual"](P["make_h"](theta)))
    return G, deltas


def run(Gd, template, point, steps_from=None):
    best = load_best_fit(template, point)
    names = list(best["theta_final"])
    theta = np.array([best["theta_final"][n] for n in names])
    theta_inj = np.array([best["theta_inj"][n] for n in names])
    P = build_point(Gd, C.signal_row(point), names)
    ev = evaluate(P, theta)
    delta_range = zero_step_range(names, theta)
    tag = f"{template} {C.point_label(point)}" + (f" steps from {steps_from}" if steps_from else "")
    fixed = stored_steps(template, steps_from) if steps_from else None
    if delta_range and not fixed:
        print(f"[{tag}] at-zero step grid for {list(delta_range)}", flush=True)
    G, deltas = fisher_at(P, theta, delta_range, fixed)
    sigma = sigma_from_fisher(G)
    bias = bias_vector(names, theta, theta_inj)
    print(f"[{tag}] overlap={ev['overlap']!r} (stored {best['final_eval']['overlap']!r}) "
          f"rho_s={ev['rho_s']:.6f} cond={np.linalg.cond(G):.3e}", flush=True)
    print("  bias/sigma = " + ", ".join(f"{n}={b / s:+.4f}" for n, b, s in zip(names, bias, sigma)),
          flush=True)
    out_dir = C.FISHER_DIR / (f"{template}_steps" if steps_from else template)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fisher_{C.point_label(point)}" + (f"_from{steps_from}" if steps_from else "")
    (out_dir / f"{stem}.json").write_text(json.dumps(dict(
        grid=C.GRID, template=template, point=point, a_inj=best["a_inj"], e0_inj=best["e0_inj"],
        best_from=best["case"], setup=C.SETUP, drop_dc=C.DROP_DC, tdi="2nd generation, AE",
        T=Gd["T"], dt=Gd["dt"], names=names, theta=theta.tolist(), theta_inj=theta_inj.tolist(),
        bias=bias.tolist(), sigma=sigma.tolist(), fisher=G.tolist(),
        cond=float(np.linalg.cond(G)), final_eval=ev, stored_eval=best["final_eval"],
        steps_from=steps_from, deltas=repr(deltas), delta_range={n: v.tolist() for n, v in delta_range.items()}), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", nargs="+", choices=C.TEMPLATES, default=list(C.TEMPLATES))
    ap.add_argument("--points", nargs="+", choices=C.POINTS, default=C.POINTS)
    ap.add_argument("--steps-from", nargs="+", choices=C.POINTS, default=[None])
    args = ap.parse_args()
    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    grids = {}                                      # (T, dt) -> grid; the grid shares one
    for point in args.points:
        key = point_T_dt(C.signal_row(point))
        if key not in grids:
            grids.clear()
            free_gpu(xp)
            grids[key] = build_grid(xp, use_gpu, *key)
        for template in args.templates:
            for src in args.steps_from:
                try:
                    run(grids[key], template, point, src)
                except Exception as exc:                    # noqa: BLE001  keep the batch going
                    print(f"[FAIL {template} {C.point_label(point)} steps {src}] "
                          f"{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
