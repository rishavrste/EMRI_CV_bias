"""PN Fisher at one point's best fit with only the spin replaced, for a near zero (the a = 0 PE test).

theta = the template's best fit (config.BEST_FITS) with a set to each --spins value; nothing re-fitted,
so the residual and the overlap change with a. Step search as the pipeline does it (model.zero_step_range):
the absolute grid config.ZERO_STEPS for |a| < config.NEAR_ZERO["a"] (= 1e-2), SEF's relative search
otherwise, so a = +-0.01 take relative steps. Signal, inner product, TDI, T, dt: as fisher_at_best.py.
    -> results/fisher_at_best/{template}_spin_scan/fisher_{label}_a{spin}.json

Run (GPU, from fisher/):  python fisher_spin_scan.py --point 22 --spins 0 0.001 -0.001 0.01 -0.01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # EMRI_1PA_PN_DEV: config, model, lm
import config as C                                  # noqa: E402
from fisher_at_best import fisher_at, load_best_fit  # noqa: E402
from lm import sigma_from_fisher  # noqa: E402
from make_results_table import GRID_POINTS  # noqa: E402
from model import build_grid, build_point, evaluate, point_T_dt, zero_step_range  # noqa: E402
from run_lm_T_ladder import check_branch  # noqa: E402

SPINS = [0.0, 0.001, -0.001, 0.01, -0.01]


def theta_with_spin(best, names, spin):
    theta = np.array([best["theta_final"][n] for n in names])
    theta[names.index("a")] = spin
    return theta


def run_spin(P, best, names, template, point, spin, out_dir):
    theta = theta_with_spin(best, names, spin)
    ev = evaluate(P, theta)
    delta_range = zero_step_range(names, theta)
    G, deltas = fisher_at(P, theta, delta_range)
    sigma = sigma_from_fisher(G)
    tag = f"{template} {C.point_label(point)} a={spin:+g}"
    print(f"[{tag}] steps {'absolute' if 'a' in delta_range else 'relative'} for a, step_a={deltas['a']:.3e}  "
          f"overlap={ev['overlap']!r} chi2={ev['chi2']!r} rho_s={ev['rho_s']:.6f} "
          f"cond={np.linalg.cond(G):.3e}", flush=True)
    print("  sigma = " + ", ".join(f"{n}={s:.4e}" for n, s in zip(names, sigma)), flush=True)
    (out_dir / f"fisher_{C.point_label(point)}_a{spin:+g}.json").write_text(json.dumps(dict(
        grid=C.GRID, template=template, point=point, spin=spin, best_from=best["case"], setup=C.SETUP,
        drop_dc=C.DROP_DC, tdi="2nd generation, AE", names=names, theta=theta.tolist(),
        sigma=sigma.tolist(), fisher=G.tolist(), cond=float(np.linalg.cond(G)), eval=ev,
        deltas=repr(deltas), delta_range={n: v.tolist() for n, v in delta_range.items()}), indent=1))
    return sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", choices=C.TEMPLATES, default="pn")
    ap.add_argument("--point", choices=GRID_POINTS, default="22")
    ap.add_argument("--spins", nargs="+", type=float, default=SPINS)
    args = ap.parse_args()
    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    Gd = build_grid(xp, use_gpu, *point_T_dt(C.signal_row(args.point)))
    best = load_best_fit(args.template, args.point)
    names = list(best["theta_final"])
    P = build_point(Gd, C.signal_row(args.point), names)
    out_dir = C.FISHER_DIR / f"{args.template}_spin_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = {}
    for spin in args.spins:
        try:
            sigmas[spin] = run_spin(P, best, names, args.template, args.point, spin, out_dir)
        except Exception as exc:                    # noqa: BLE001  keep the scan going
            print(f"[FAIL a={spin:+g}] {type(exc).__name__}: {exc}", flush=True)
    if 0.0 in sigmas:
        print("\nsigma(a) / sigma(a=0):")
        for spin, s in sigmas.items():
            print(f"  a={spin:+g}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, s / sigmas[0.0])))


if __name__ == "__main__":
    main()
