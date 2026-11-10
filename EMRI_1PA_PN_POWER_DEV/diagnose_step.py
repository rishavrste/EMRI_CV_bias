"""Why does LM stop at a seed? Derivative steps, Fisher spectrum, gradient and trial steps at one seed.

Run (GPU):  python diagnose_step.py --point 4
"""
import argparse

import numpy as np

import config as C
from lm import fisher_and_gradient
from model import build_grid, build_point, point_T_dt
from run_lm_arms import seed_best


def trial_steps(P, cur, G, g, c0):
    """Actual chi2 change of damped steps; scan lam without the LM's accept logic."""
    d = np.abs(np.diag(G))
    for lam in 10.0 ** np.arange(-6, 7):
        delta = np.linalg.solve(G + lam * np.diag(d), g)
        c1 = P["chi2r"](cur + delta)
        pred = float(delta @ (g + lam * d * delta))
        print(f"lam {lam:.0e}: pred {pred:+.3e} actual {c0 - c1:+.3e} dn_p {delta[-2]:+.3e} dn_e {delta[-1]:+.3e}", flush=True)


def profile_n(P, cur, c0):
    """chi2 change along n alone, nothing else moved."""
    for name in ("n_p", "n_e"):
        i = C.FIT_PARAMS.index(name)
        for h in (-0.05, -0.01, -0.001, 0.001, 0.01, 0.05):
            v = cur.copy()
            v[i] += h
            print(f"{name} {h:+.3f}: dchi2 {P['chi2r'](v) - c0:+.3e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="4", choices=C.POINTS)
    args = ap.parse_args()
    xp, use_gpu = C.ovl.load_backend()
    sig_row = C.signal_row(args.point)
    Gd = build_grid(xp, use_gpu, C.T_FULL, point_T_dt(sig_row)[1])
    names = C.FIT_PARAMS
    P = build_point(Gd, sig_row, names)
    seed = seed_best(args.point)
    cur = np.array([seed[n] for n in names])
    dH, deltas = P["derivs"](cur, None)
    print("deltas", {k: float(v) for k, v in deltas.items()}, flush=True)
    r = P["residual"](P["make_h"](cur))
    c0 = P["inner"](r, r)
    G, g = fisher_and_gradient(P, dH, r)
    dd = np.sqrt(np.diag(G))
    Gs = G / np.outer(dd, dd)
    print("chi2", c0, "scaled eigenvalues", np.array2string(np.linalg.eigvalsh(Gs), precision=2), flush=True)
    for n, a, b in zip(names, dd, g / dd):
        print(f"{n:9s} sqrt(G_ii) {a:.3e}  g_i/sqrt(G_ii) {b:+.3e}", flush=True)
    i, j = names.index("C_p"), names.index("n_p")
    print("cos(C_p, n_p)", Gs[i, j], "cos(C_e, n_e)", Gs[i + 1, j + 1], flush=True)
    for keep, label in ((list(range(11)), "11 params"), (list(range(13)), "13 params")):
        s = np.linalg.solve(Gs[np.ix_(keep, keep)], (g / dd)[keep])
        print(label, "predicted chi2 drop (GN)", float((g / dd)[keep] @ s), flush=True)
    trial_steps(P, cur, G, g, c0)
    profile_n(P, cur, c0)


if __name__ == "__main__":
    main()
