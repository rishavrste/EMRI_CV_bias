"""Injection, T-ladder LM fits and Fisher at the best fit for each spin (see spin_config.py).

Per spin, in results/a{spin}/:
    injection.json                the 17-entry injection row (p0 and dist solved here), its SNR
    {template}{tag}_lm_T{T}.json  one LM rung; rewritten after every accepted step
    {template}{tag}_fisher.json   Gamma, sigma and bias at the full-T best fit (SNR 20)
tag is "" for the default run (transplanted seed, T_FRACS), else "_" + --tag.
A finished file is not recomputed, so a re-submission picks up where a walltime kill stopped.

Run (GPU):  python run_spin_fits.py --spins -0.1 0.1 [--templates 0pa pn]
            [--seed injection] [--t-fracs 0.4 0.5 ... 1.0] [--tag inj_fine]
"""
import argparse
import json
import os
import time

import numpy as np

import spin_config as S
from spin_config import C
from fisher_at_best import bias_vector              # EMRI_1PA_PN_DEV/fisher
from lm import fisher_and_gradient, lm_climb, sigma_from_fisher
from model import add_args, build_grid, build_point, evaluate, point_T_dt
from run_lm_T_ladder import as_dict, check_branch, free_gpu


# --- output ----------------------------------------------------------------
def save(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(f"{path}.tmp", path)


def finished(path):
    return path.exists() and not json.loads(path.read_text()).get("running")


# --- injection -------------------------------------------------------------
def solve_p0(row):
    """p0 at which the 1PA trajectory (the signal's flags, no deviation) plunges at the stored T."""
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode.flux import SuperKludgeFlux
    from few.utils.utility import get_p_at_t

    col = C.COL
    traj = EMRIInspiral(func=SuperKludgeFlux)
    args = [row[col[n]] for n in ("m1", "m2", "a", "e0", "x0")]
    args += add_args(float(row[col["chi2"]]), C.SIGNAL_FLAGS, None)
    return float(get_p_at_t(traj_module=traj, t_out=float(row[col["T"]]), traj_args=args))


def make_injection(xp, use_gpu, spin):
    """idx22's row with a = spin, p0 plunging at T and dist giving SNR 20 -> injection.json."""
    path = S.RESULTS / S.label(spin) / "injection.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["row"])

    col = C.COL
    row = np.array(C.signal_row(S.BASE_POINT), dtype=float)
    row[col["a"]] = spin
    row[col["p0"]] = solve_p0(row)

    # rho scales as 1/dist: one signal at idx22's distance fixes the distance for SNR 20
    Gd = build_grid(xp, use_gpu, *point_T_dt(row))
    rho_ref = evaluate(build_point(Gd, row, C.PHYS), row_theta(row, C.PHYS))["rho_s"]
    dist_ref = float(row[col["dist"]])
    row[col["dist"]] = dist_ref * rho_ref / S.SNR
    rho = evaluate(build_point(Gd, row, C.PHYS), row_theta(row, C.PHYS))["rho_s"]
    del Gd
    free_gpu(xp)

    print(f"[INJ {S.label(spin)}] p0={row[col['p0']]!r} dist={row[col['dist']]!r} rho_s={rho:.6f}",
          flush=True)
    save(path, dict(spin=spin, base_point=S.BASE_POINT, row=row.tolist(), col=col,
                    p0_rule="1PA trajectory plunges at T (get_p_at_t)", rho_s=rho,
                    rho_at_dist_ref=rho_ref, dist_ref=dist_ref))
    return row


def row_theta(row, names):
    return np.array([C.injected_value(row, n) for n in names])


# --- seed ------------------------------------------------------------------
def transplanted_seed(row, template, names):
    """New injection + (idx22 best fit - idx22 injection) of the same template."""
    best = json.loads(C.BEST_FITS.read_text())[template][S.BASE_POINT]
    offset = np.array([best["theta_final"][n] - best["theta_inj"][n] for n in names])
    return row_theta(row, names) + offset


# --- fits ------------------------------------------------------------------
def run_rung(Gd, row, spin, template, names, theta0, start, tag):
    path = S.RESULTS / S.label(spin) / f"{template}{tag}_lm_T{Gd['T']:g}.json"
    if finished(path):
        d = json.loads(path.read_text())
        print(f"[SKIP {path.name} {S.label(spin)}] O={d['final_eval']['overlap']:.12f}", flush=True)
        return d

    P = build_point(Gd, row, names)
    what = f"{template}{tag} {S.label(spin)} T={Gd['T']:g}"
    t0 = time.time()
    first = evaluate(P, theta0)
    print(f"[START {what}] from {start} O={first['overlap']:.12f} chi2={first['chi2']:.6e} "
          f"rho_s={first['rho_s']:.6f}", flush=True)
    out = dict(spin=spin, template=template, T=Gd["T"], dt=Gd["dt"], start=start, params=names,
               setup=C.SETUP, drop_dc=C.DROP_DC, tdi="2nd generation, AE",
               theta_inj=as_dict(names, row_theta(row, names)), theta_start=as_dict(names, theta0),
               start_eval=first)

    def checkpoint(cur, sigma, history):
        save(path, dict(out, theta_current=as_dict(names, cur), history=history, running=True,
                        seconds=time.time() - t0))

    theta, sigma, info, history = lm_climb(P, theta0, what, on_iter=checkpoint)
    final = evaluate(P, theta)
    out.update(theta_final=as_dict(names, theta), final_eval=final,
               sigma_final=as_dict(names, sigma), info=info, history=history, running=False,
               seconds=time.time() - t0)
    save(path, out)
    print(f"[DONE {what}] O {first['overlap']:.12f} -> {final['overlap']:.12f}  "
          f"chi2 {first['chi2']:.6e} -> {final['chi2']:.6e}  iters={info['n_iter']} "
          f"stop={info['stop_reason']}  {out['seconds']:.0f} s", flush=True)
    print("theta_final = " + ", ".join(f"{n}={v!r}" for n, v in out["theta_final"].items()), flush=True)
    return out


def fisher_at_fit(Gd, row, spin, template, names, fit, tag):
    """Gamma at the full-T best fit, SEF's step search (spin: the at-zero grid, spin_config)."""
    path = S.RESULTS / S.label(spin) / f"{template}{tag}_fisher.json"
    if path.exists():
        return
    P = build_point(Gd, row, names)
    theta = np.array([fit["theta_final"][n] for n in names])
    dH, deltas = P["derivs"](theta, None)
    G, _ = fisher_and_gradient(P, dH, P["residual"](P["make_h"](theta)))
    sigma = sigma_from_fisher(G)
    bias = bias_vector(names, theta, row_theta(row, names))
    ev = evaluate(P, theta)
    print(f"[FISHER {template}{tag} {S.label(spin)}] O={ev['overlap']!r} cond={np.linalg.cond(G):.3e} "
          f"step_a={deltas['a']:.3e}", flush=True)
    print("  bias/sigma = " + ", ".join(f"{n}={b / s:+.4f}" for n, b, s in zip(names, bias, sigma)),
          flush=True)
    save(path, dict(spin=spin, template=template, T=Gd["T"], dt=Gd["dt"], names=names,
                    theta=theta.tolist(), theta_inj=row_theta(row, names).tolist(),
                    bias=bias.tolist(), sigma=sigma.tolist(), fisher=G.tolist(),
                    cond=float(np.linalg.cond(G)), eval=ev, deltas=repr(deltas)))


def first_seed(row, template, names, seed):
    if seed == "injection":
        return row_theta(row, names), "injection"
    return transplanted_seed(row, template, names), f"idx{S.BASE_POINT}_offset"


def climb(xp, use_gpu, row, spin, template, seed, t_fracs, tag):
    """The T ladder from the chosen seed, then the Fisher on the full-T grid."""
    names = C.fit_params(template)
    theta, start = first_seed(row, template, names, seed)
    t_full, dt = point_T_dt(row)
    for k, f in enumerate(t_fracs):
        Gd = build_grid(xp, use_gpu, f * t_full, dt)
        fit = run_rung(Gd, row, spin, template, names, theta, start, tag)
        if k == len(t_fracs) - 1:
            fisher_at_fit(Gd, row, spin, template, names, fit, tag)
        del Gd
        free_gpu(xp)
        theta, start = np.array([fit["theta_final"][n] for n in names]), f"T{f * t_full:g}_final"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spins", nargs="+", type=float, default=S.SPINS)
    ap.add_argument("--templates", nargs="+", choices=C.TEMPLATES, default=list(C.TEMPLATES))
    ap.add_argument("--seed", choices=("offset", "injection"), default="offset")
    ap.add_argument("--t-fracs", nargs="+", type=float, default=list(S.T_FRACS))
    ap.add_argument("--tag", default="", help="output name suffix; needed for a non-default run")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    for spin in args.spins:
        try:
            row = make_injection(xp, use_gpu, spin)
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL injection {S.label(spin)}] {type(exc).__name__}: {exc}", flush=True)
            continue
        for template in args.templates:
            try:
                climb(xp, use_gpu, row, spin, template, args.seed, args.t_fracs, tag)
            except Exception as exc:                        # noqa: BLE001
                print(f"[FAIL {template} {S.label(spin)}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
