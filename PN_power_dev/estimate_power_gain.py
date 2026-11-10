"""Phase-only estimate of what n_p, n_e add to the 0PA + PN template, at the existing PN best fits.

Both phases are sampled on one time grid (Phi_r at the same weight as Phi_phi, no noise weighting, no
amplitudes, no sky), so the numbers are a guide, not a Fisher:

    residual gain = |r|^2 after projecting out the PN tangent space / the same with n_p, n_e added
    sigma ratio   = marginal sigma with n_p, n_e free / without (the overall scale cancels)

r = 1PA signal phases at the source - 0PA + PN phases at the best fit.
Run:  python estimate_power_gain.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SuperKludgeFlux
from few.utils.constants import MTSUN_SI, YRSID_SI

from check_power_deviation import POINTS, args_for, term_e, term_p

CHECK_POINTS = (0, 8, 12, 4, 19, 24)
T_OBS = 2.4
N_TIMES = 1500
CHI2 = 0.95
BASE = ["m1", "m2", "a", "p0", "e0"]
STEPS = dict(m1=1e-6, m2=1e-6, a=1e-6, p0=1e-6, e0=1e-5, C_p=1.0, C_e=1.0, n_p=0.05, n_e=0.05)
RELATIVE = {"m1", "m2"}


def phases(ode, theta, flags_args):
    """Phi_phi, Phi_r on the uniform time grid, stacked; theta holds the BASE and deviation values."""
    m1, m2, a, p0, e0 = (theta[k] for k in BASE)
    q = m1 * m2 / (m1 + m2) ** 2
    scale = q / ((m1 + m2) * MTSUN_SI)
    ode.add_fixed_parameters(m1, m2, a, additional_args=flags_args)
    C_p, C_e, n_p, n_e = (theta.get(k, 0.0) for k in ("C_p", "C_e", "n_p", "n_e"))

    def f(_, y):
        p, e = y[:2]
        v = ode(np.array([p, e, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        return [v[0] + term_p(q, C_p, n_p, p, e), v[1] + term_e(q, C_e, n_e, p, e), v[3], v[5]]

    t = np.linspace(0.0, T_OBS * YRSID_SI, N_TIMES) * scale
    y = solve_ivp(f, (0.0, t[-1]), [p0, e0, 0.0, 0.0], method="DOP853", rtol=1e-11, atol=1e-12,
                  t_eval=t).y
    return np.concatenate([y[2], y[3]]) / q, (y[0, 0], y[0, -1])


def tangent(ode, theta, names, off):
    cols = []
    for k in names:
        h = STEPS[k] * (theta[k] if k in RELATIVE else 1.0)
        up, dn = dict(theta), dict(theta)
        up[k] += h
        dn[k] -= h
        cols.append((phases(ode, up, off)[0] - phases(ode, dn, off)[0]) / (2 * h))
    const_phi = np.r_[np.ones(N_TIMES), np.zeros(N_TIMES)]
    const_r = np.r_[np.zeros(N_TIMES), np.ones(N_TIMES)]
    return np.column_stack(cols + [const_phi, const_r])


def scaled(J):
    norm = np.linalg.norm(J, axis=0)
    return J / norm, norm


def leftover(J, r):
    Q = np.linalg.svd(scaled(J)[0], full_matrices=False)
    keep = Q[1] > 1e-11 * Q[1][0]
    U = Q[0][:, keep]
    return float(np.sum((r - U @ (U.T @ r)) ** 2)), int(keep.sum())


def marginal_sigma(J):
    Js, norm = scaled(J)
    return np.sqrt(np.diag(np.linalg.pinv(Js.T @ Js, rcond=1e-14))) / norm


def corr(J, i, j):
    G = np.linalg.pinv(scaled(J)[0].T @ scaled(J)[0], rcond=1e-14)
    return G[i, j] / np.sqrt(G[i, i] * G[j, j])


def study(ode, point):
    src, best = point["source"], point["best_fit"]["0PA+PN"]["params"]
    off = args_for(on=False)
    theta = {k: best[k] for k in BASE + ["C_p", "C_e"]} | {"n_p": 0.0, "n_e": 0.0}
    sig, _ = phases(ode, {k: src[k] for k in BASE}, args_for(on=False, chi2=CHI2, pa1=True))
    tpl, (p_start, p_end) = phases(ode, theta, off)
    r = sig - tpl
    names_pn = BASE + ["C_p", "C_e"]
    J_pn = tangent(ode, theta, names_pn, off)
    J_n = np.column_stack([J_pn, tangent(ode, theta, ["n_p", "n_e"], off)[:, :2]])
    r_pn, rank_pn = leftover(J_pn, r)
    r_n, rank_n = leftover(J_n, r)
    s_pn, s_n = marginal_sigma(J_pn), marginal_sigma(J_n)
    ratio = {k: s_n[i] / s_pn[i] for i, k in enumerate(names_pn)}
    ip, icp = J_pn.shape[1], names_pn.index("C_p")
    print(f"idx{point['idx']:<3d} a {src['a']:+.1f} e0 {src['e0']:.1f} C_p {theta['C_p']:+7.2f} "
          f"p {p_start:.2f}->{p_end:.2f} | rms r {np.sqrt(np.mean(r**2)):.3e} rad | residual gain "
          f"{r_pn / r_n:9.3g} rank {rank_pn}->{rank_n} | corr(C_p,n_p) {corr(J_n, icp, ip):+.5f} | "
          f"sigma ratio m2 {ratio['m2']:.1f} a {ratio['a']:.1f} p0 {ratio['p0']:.1f} "
          f"e0 {ratio['e0']:.1f} C_p {ratio['C_p']:.1f}", flush=True)


def main():
    ode = EMRIInspiral(func=SuperKludgeFlux).func
    points = json.loads(Path(POINTS).read_text())["points"]
    for idx in CHECK_POINTS:
        study(ode, points[idx])


if __name__ == "__main__":
    main()
