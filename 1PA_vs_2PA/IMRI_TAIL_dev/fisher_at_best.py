"""Fisher matrix at the best fits of both templates, for the 1D and nD bias.

    1pa     : the SK_files 1PA-only best fit (config.BEST_1PA), fitted in its own 8 parameters,
              C_p = C_e = 0
    1pa_dev : the chosen 1PA + deviation best fit (config.BEST_DEV, best_fit_dev.py), all 10
Gamma_ij = <d_i h|d_j h> at theta_final, SEF stable derivatives, the config.SETUP inner product, the
stored T (0.25 yr) and the stored distance (dist_div 1, SNR 20); Gamma scales as SNR^2.
bias = theta_final - theta_inj, the phases wrapped into (-pi, pi].
Step search: SEF's default trial steps are relative to the parameter value, so a deviation
coefficient left at round-off level (idx4: C_p ~ 4e-16) gets steps ~ 1e-17, below round-off, and a
noise derivative. Coefficients with |C| < ZERO_C therefore get SEF's own at-zero grid
(ZERO_STEPS, as idx7 at C = 0 exactly); theta itself is unchanged.
    -> results/fisher_at_best/{template}/fisher_idx{i}.json

Run (GPU):  python fisher_at_best.py --template 1pa 1pa_dev --idx 0 1 2
"""
import argparse
import json

import numpy as np

import config as C
from lm import fisher_and_gradient, sigma_from_fisher
from model import build_grid, build_point, evaluate
from run_lm_T_ladder import check_branch

BEST = {"1pa": C.BEST_1PA, "1pa_dev": C.BEST_DEV}
ZERO_C = 1e-10                                  # |C| below this counts as zero for the step search
ZERO_STEPS = np.geomspace(1e-4, 1e-9, C.NDELTA)    # SEF's trial steps at a zero parameter


def load_best_fit(template, idx):
    return json.loads(BEST[template].read_text())[str(idx)]


def wrap_phase(x):
    """x mapped into (-pi, pi]."""
    return float(np.pi - np.mod(np.pi - x, 2.0 * np.pi))


def bias_vector(names, theta, theta_inj):
    b = np.asarray(theta) - np.asarray(theta_inj)
    return np.array([wrap_phase(v) if n in C.PHASES else v for n, v in zip(names, b)])


def zero_step_range(names, theta):
    """ZERO_STEPS for every deviation coefficient with |C| < ZERO_C."""
    return {n: ZERO_STEPS for n, v in zip(names, theta) if n in C.DEV_PARAMS and abs(v) < ZERO_C}


def fisher_at(P, theta, delta_range):
    dH, deltas = P["derivs"](theta, None, delta_range or None)
    G, _ = fisher_and_gradient(P, dH, P["residual"](P["make_h"](theta)))
    return G, deltas


def run(Gd, template, idx):
    best = load_best_fit(template, idx)
    names = list(best["theta_final"])
    theta = np.array([best["theta_final"][n] for n in names])
    theta_inj = np.array([best["theta_inj"][n] for n in names])
    P = build_point(Gd, C.ovl.signal_array(C.GRID)[idx], names)
    ev = evaluate(P, theta)
    delta_range = zero_step_range(names, theta)
    if delta_range:
        print(f"[{template} idx={idx}] at-zero step grid for {list(delta_range)}", flush=True)
    G, deltas = fisher_at(P, theta, delta_range)
    sigma = sigma_from_fisher(G)
    bias = bias_vector(names, theta, theta_inj)
    print(f"[{template} idx={idx}] overlap={ev['overlap']!r} (stored "
          f"{best['final_eval']['overlap']!r}) rho_s={ev['rho_s']:.6f} cond={np.linalg.cond(G):.3e}",
          flush=True)
    print("  bias/sigma = " + ", ".join(f"{n}={b / s:+.4f}" for n, b, s in zip(names, bias, sigma)),
          flush=True)
    out_dir = C.OUT_ROOT / "fisher_at_best" / template
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fisher_idx{idx}.json").write_text(json.dumps(dict(
        grid=C.GRID, template=template, idx=idx, a_inj=best["a_inj"], e0_inj=best["e0_inj"],
        setup=C.SETUP, drop_dc=C.DROP_DC, T=Gd["T"], names=names,
        theta=theta.tolist(), theta_inj=theta_inj.tolist(), bias=bias.tolist(),
        sigma=sigma.tolist(), fisher=G.tolist(), cond=float(np.linalg.cond(G)),
        final_eval=ev, stored_eval=best["final_eval"], deltas=repr(deltas),
        delta_range={n: v.tolist() for n, v in delta_range.items()}), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", nargs="+", choices=list(BEST), default=list(BEST))
    ap.add_argument("--idx", nargs="+", type=int, required=True)
    args = ap.parse_args()
    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    Gd = build_grid(xp, use_gpu)
    for template in args.template:
        for idx in args.idx:
            try:
                run(Gd, template, idx)
            except Exception as exc:                        # noqa: BLE001  keep the batch going
                print(f"[FAIL {template} idx={idx}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
