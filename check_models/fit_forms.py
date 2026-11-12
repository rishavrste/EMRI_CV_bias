"""Which low-parameter form of the 1PA deviation fits the SuperKludge 1PA rates best, across the EMRI and IMRI grids?

Every 1PA rate divided by its leading shape is exactly a cubic in v = p^-1/2 (sk_orders.py):
    r_X(v) = 1PA_X / shape_X,   shape_p = (1-e^2)^1.5 (8+7e^2) / p^3.5,  shape_e = e (1-e^2)^1.5 (304+121e^2) / p^4.5,
                                shape_phi = shape_r = (1-e^2)^1.5 / p^2
A deviation adds  q shape_X R_X(v)  to rate X; the forms below are choices of R_X with few free parameters.

Forms (number of free parameters):
    B        (4)  R_X = C_X
    PL       (8)  R_X = C_X (10/p)^n_X
    PL6      (6)  as PL with n shared by pdot, edot and by Omega_phi, Omega_r
    LIN      (8)  R_X = C_X (1 + k_X v)                       next PN order as a free linear slope
    J5       (5)  R_X = C_X (1 + k v)                         one slope for all channels
    PADE     (8)  R_X = C_X / (1 - k_X v)
    SVD_K    (K)  R_X = sum_j C_j b_jX(v), K = 1..4; b_jX are cubics in v with fixed coefficients, the top K
                  singular vectors of r_X over a in [-0.99, 0.99], chi2 in [0, 1], e in [0.05, 0.5], p in [5, 31]
                  (channels scaled to unit rms). The basis is analytic: its cubic coefficients are printed.

--rates   : at every grid point with a track (results/track_{grid}_idx{p}.npz, from pn_orders.py), least squares of
            each form to the four 1PA rates (uniform in t, each channel weighted by 1/rms of its 1PA rate);
            reports the relative rms remainder.      -> results/fit_forms_rates.json, logs/fit_forms_rates.log
--dephase : for the given forms and points, one trajectory at the injection with the fitted coefficients:
            Phi_phi, Phi_r at T against 1PA.        -> results/fit_forms_dephase_{grid}_idx{p}.json
Run:  python fit_forms.py --rates
      python fit_forms.py --dephase --grid IMRI --point 4 --forms B J5 SVD_2 SVD_3 SVD_4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
YR = 3.15581e7
CH = ("pdot", "edot", "Omega_phi", "Omega_r")

# --- SVD training box ----------------------------------------------------------------------
TRAIN_A = np.linspace(-0.99, 0.99, 23)
TRAIN_CHI2 = np.linspace(0.0, 1.0, 5)
TRAIN_E = np.linspace(0.05, 0.5, 6)
TRAIN_V = np.linspace(31 ** -0.5, 5 ** -0.5, 60)
K_MAX = 6


# --- shapes and ratios ------------------------------------------------------------------------
def shapes(p, e):
    f = (1 - e * e) ** 1.5
    return dict(pdot=f * (8 + 7 * e * e) / p ** 3.5, edot=e * f * (304 + 121 * e * e) / p ** 4.5,
                Omega_phi=f / p ** 2, Omega_r=f / p ** 2)


def ratios(a, p, e, chi2):
    """r_X = 1PA_X / shape_X for the four channels."""
    from sk_orders import CHANNELS
    s = shapes(p, e)
    return {ch: CHANNELS[ch][0](a, p, e, chi2) / s[ch] for ch in CH}


# --- SVD basis --------------------------------------------------------------------------------
def build_basis():
    """Top K_MAX singular vectors as cubic coefficients per channel, the singular values, and the channel scales."""
    rows = []
    p = TRAIN_V ** -2
    for a in TRAIN_A:
        for chi2 in TRAIN_CHI2:
            for e in TRAIN_E:
                r = ratios(a, p, np.full_like(p, e), chi2)
                rows.append(np.concatenate([r[ch] for ch in CH]))
    M = np.array(rows)
    nv = TRAIN_V.size
    scale = np.array([np.sqrt(np.mean(M[:, k * nv:(k + 1) * nv] ** 2)) for k in range(4)])
    Ms = M / np.repeat(scale, nv)
    _, sv, Vt = np.linalg.svd(Ms, full_matrices=False)
    V3 = np.vander(TRAIN_V, 4, increasing=True)
    beta = np.zeros((K_MAX, 4, 4))                     # [component, channel, power of v]
    for j in range(K_MAX):
        for k in range(4):
            beta[j, k] = np.linalg.lstsq(V3, Vt[j, k * nv:(k + 1) * nv] * scale[k], rcond=None)[0]
    return beta, sv, scale


def basis_eval(beta, K, v):
    """(K, 4, n) values of the first K basis functions at v."""
    V3 = np.vander(np.atleast_1d(v), 4, increasing=True).T          # (4, n)
    return np.einsum("jcm,mn->jcn", beta[:K], V3)


# --- forms ------------------------------------------------------------------------------------
def form_R(form, x, p, beta=None):
    """R_X(v) for the four channels, (4, n)."""
    v = p ** -0.5
    if form == "B":
        return np.outer(x, np.ones_like(v))
    if form == "PL":
        return np.array([x[k] * (10 / p) ** x[4 + k] for k in range(4)])
    if form == "PL6":
        n = (x[4], x[4], x[5], x[5])
        return np.array([x[k] * (10 / p) ** n[k] for k in range(4)])
    if form == "LIN":
        return np.array([x[k] * (1 + x[4 + k] * v) for k in range(4)])
    if form == "J5":
        return np.array([x[k] * (1 + x[4] * v) for k in range(4)])
    if form == "PADE":
        return np.array([x[k] / (1 - x[4 + k] * v) for k in range(4)])
    if form.startswith("SVD_"):
        K = int(form.split("_")[1])
        return np.einsum("j,jcn->cn", x, basis_eval(beta, K, v))
    raise ValueError(form)


N_PARAMS = dict(B=4, PL=8, PL6=6, LIN=8, J5=5, PADE=8, **{f"SVD_{K}": K for K in range(1, K_MAX + 1)})


def fit_form(form, p, e, r_true, beta):
    """Least squares of R_X to r_X along a track; returns (x, relative rms remainder per channel)."""
    w = np.array([1 / np.sqrt(np.mean(r_true[k] ** 2)) for k in range(4)])[:, None]
    sh = shapes(p, e)
    sh_arr = np.array([sh[ch] for ch in CH])                     # weight the ratio by the rate shape:
    sw = sh_arr / np.sqrt(np.mean(sh_arr ** 2, axis=1, keepdims=True))   # fit the rate, not the ratio
    target = r_true * sw

    def res(x):
        return ((target - form_R(form, x, p, beta) * sw) * w).ravel()

    c0 = np.array([np.mean(r_true[k]) for k in range(4)])
    x0 = dict(B=c0, PL=np.r_[c0, np.zeros(4)], PL6=np.r_[c0, 0, 0], LIN=np.r_[c0, np.zeros(4)], J5=np.r_[c0, 0],
              PADE=np.r_[c0, np.zeros(4)]).get(form)
    if form.startswith("SVD_"):
        K = int(form.split("_")[1])
        A = (basis_eval(beta, K, p ** -0.5) * sw[None] * w[None]).reshape(K, -1).T
        x = np.linalg.lstsq(A, (target * w).ravel(), rcond=None)[0]
    else:
        x = least_squares(res, x0, method="lm").x
    rem = (r_true - form_R(form, x, p, beta)) * sw
    rel = [float(np.sqrt(np.mean(rem[k] ** 2) / np.mean((r_true[k] * sw[k]) ** 2))) for k in range(4)]
    return x, rel


# --- tracks -----------------------------------------------------------------------------------
def load_track(grid, point):
    d = np.load(RES / f"track_{grid}_idx{point}.npz")
    return {k: d[k] for k in d.files}


def track_ratios(tr):
    r = ratios(float(tr["a"]), tr["p"], tr["e"], float(tr["chi2"]))
    return np.array([r[ch] for ch in CH])


# --- modes ------------------------------------------------------------------------------------
def run_rates(beta, sv):
    print("SVD singular values (normalised):", " ".join(f"{x:.3e}" for x in sv[:K_MAX] / sv[0]))
    print("cumulative energy:", " ".join(f"{x:.6f}" for x in np.cumsum(sv ** 2)[:K_MAX] / np.sum(sv ** 2)))
    for j in range(K_MAX):
        print(f"basis {j + 1}: " + "; ".join(f"{ch}: " + ", ".join(f"{b:+.5g}" for b in beta[j, k])
                                              for k, ch in enumerate(CH)))
    forms = list(N_PARAMS)
    out = {}
    print(f"\nrelative rms remainder of the four rates (max over channels), per form")
    print(f"{'point':10s} " + " ".join(f"{f:>8s}" for f in forms))
    for path in sorted(RES.glob("track_*_idx*.npz")):
        grid, idx = path.stem.split("_")[1], path.stem.split("idx")[1]
        tr = load_track(grid, idx)
        r = track_ratios(tr)
        row = {}
        for f in forms:
            x, rel = fit_form(f, tr["p"], tr["e"], r, beta)
            row[f] = dict(x=x.tolist(), rel=rel)
        out[f"{grid}_{idx}"] = dict(a=float(tr["a"]), p_range=[float(tr["p"][0]), float(tr["p"][-1])], forms=row)
        print(f"{grid[0]}{idx:>3s} a={float(tr['a']):+.1f} " +
              " ".join(f"{max(row[f]['rel']):8.4f}" for f in forms), flush=True)
    (RES / "fit_forms_rates.json").write_text(json.dumps(dict(singular_values=(sv / sv[0]).tolist(),
                                                              basis=beta.tolist(), points=out), indent=1))


def make_dev_flux(form, x, beta):
    from few.trajectory.ode.flux import SuperKludgeFlux

    class DevFlux(SuperKludgeFlux):
        def modify_rhs(self, ydot, y, **kwargs):
            super().modify_rhs(ydot, y, **kwargs)
            p, e = float(y[0]), float(y[1])
            sh = shapes(p, e)
            R = form_R(form, x, np.array([p]), beta)[:, 0]
            for slot, k in zip((0, 1, 3, 5), range(4)):
                ydot[slot] += self.massratio * sh[CH[k]] * R[k]

    DevFlux.__name__ = "SuperKludgeFlux"
    return DevFlux


def run_dephase(beta, grid, point, forms):
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode.flux import SuperKludgeFlux
    from families import ovl
    row = ovl.signal_array(grid)[int(point)]
    inj = {n: float(row[ovl.COL[n]]) for n in ("m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0", "chi2", "T")}
    tr = load_track(grid, point)
    r = track_ratios(tr)
    t_eval = np.linspace(0, tr["t"][-1], 400)

    def phases(flux_cls, flags):
        traj = EMRIInspiral(func=flux_cls)
        m = (inj["m1"], inj["m2"], inj["a"], inj["p0"], inj["e0"], 1.0)
        args = [inj["chi2"], *flags, 0, 0] + [0.0] * 8
        kw = dict(Phi_phi0=inj["Phi_phi0"], Phi_r0=inj["Phi_r0"], T=inj["T"], dt=10.0)
        t_end = traj(*m, *args, **kw)[0][-1]
        tt = t_eval[t_eval <= t_end]
        out = traj(*m, *args, **kw, upsample=True, new_t=tt)
        return out[4], out[6]

    sig = phases(SuperKludgeFlux, (1, 0, 0))
    res = {}
    for f in ["0PA"] + forms:
        if f == "0PA":
            ph = phases(SuperKludgeFlux, (0, 0, 0))
            x, rel = [], [1, 1, 1, 1]
        else:
            x, rel = fit_form(f, tr["p"], tr["e"], r, beta)
            ph = phases(make_dev_flux(f, x, beta), (0, 0, 0))
        n = min(ph[0].size, sig[0].size)
        d = dict(n_params=N_PARAMS.get(f, 0), x=list(map(float, x)), rate_rel=rel,
                 end_phi=float(sig[0][n - 1] - ph[0][n - 1]), end_r=float(sig[1][n - 1] - ph[1][n - 1]),
                 rms_phi=float(np.sqrt(np.mean((sig[0][:n] - ph[0][:n]) ** 2))),
                 rms_r=float(np.sqrt(np.mean((sig[1][:n] - ph[1][:n]) ** 2))))
        res[f] = d
        print(f"{grid} idx{point} {f:7s} ({d['n_params']}) rate rem max {max(rel):.4f}  end dPhi_phi {d['end_phi']:+10.4f}"
              f"  end dPhi_r {d['end_r']:+10.4f}  rms ({d['rms_phi']:.4f}, {d['rms_r']:.4f})", flush=True)
    (RES / f"fit_forms_dephase_{grid}_idx{point}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", action="store_true")
    ap.add_argument("--dephase", action="store_true")
    ap.add_argument("--grid", choices=("EMRI", "IMRI"))
    ap.add_argument("--point")
    ap.add_argument("--forms", nargs="+", default=["B", "J5", "SVD_2", "SVD_3", "SVD_4"])
    args = ap.parse_args()
    beta, sv, _ = build_basis()
    if args.rates:
        run_rates(beta, sv)
    if args.dephase:
        run_dephase(beta, args.grid, args.point, args.forms)


if __name__ == "__main__":
    main()
