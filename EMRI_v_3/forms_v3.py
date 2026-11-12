"""More general deviations than the SK-trained SVD basis: options 1, 2 and 3 (SuperKludge_r branch PN_v_3).

Every form is linear in its parameters x_k:   X += eta * S_X(p, e) * sum_k x_k g_kX(v),   v = p^-1/2,
(shifted forms, marked "v0", have g_kX(v, v0) with v0 = p0^-1/2 the start of the template's own track)
X = pdot, edot, Omega_phi, Omega_r, and S_X is either the leading 1PA shape ("lead") or the template's own 0PA rate
("0pa"). The 1PA rate it should absorb is eta * 1PA_X, so the expected x come from one weighted linear solve.

    B        (4) lead : C_X                                     reference (leading amplitudes only)
    V3       (5) lead : C_X + K3 v^3                            option 2 = flux.py on PN_v_3 (run through flux.py itself)
    REL5_0   (5) 0pa  : c_X + K v^3                             option 1 as first written (0PA-relative, v^0 and v^3)
    REL5_1   (5) 0pa  : c_X v + K v^4                           option 1, PN-consistent: 1PA/0PA starts at relative v^1
    REL8_1   (8) 0pa  : c_X v + k_X v^4                         option 1, slope per rate
    REL5_1S  (5) 0pa  : c_X (v - v0) + K v^4                    REL5_1 with the c_X terms zero at the start of the
                                                                track, v0 = p0^-1/2 of the template itself, so
                                                                they cannot act as a constant rescaling (mass ratio)
    REL5_1S2 (5) 0pa  : c_X (v - v0) + K (v^4 - v0^4)            the same with the K term shifted too
    SVDf{f}_{K}  lead : sum_j C_j b_jX(v)                       option 3: basis from SK 1PA cubics whose four PN-order
                                                                terms are each scaled by an independent U(1-f, 1+f);
                                                                f = 0 is the check_models SVD basis

--rates            : expected x at every grid track (check_models/results/track_*.npz), relative rms remainder of the
                     four rates; also the singular values of each perturbed basis      -> results/rates.json
--dephase --grid --point : one trajectory per form at the injection with the expected x, 1PA - form phases
                     (end and rms, as check_models/fit_forms.py)                       -> results/dephase_{grid}_idx{p}.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "check_models"
sys.path.insert(0, str(CHECK))
from sk_orders import CHANNELS, order_terms          # noqa: E402  (exact split of each SK 1PA rate into PN orders)

RES = HERE / "results"
SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
BRANCH = "PN_v_3"
CH = tuple(CHANNELS)                                   # pdot, edot, Omega_phi, Omega_r
SLOTS = (0, 1, 3, 5)                                   # their places in the ODE rhs
N_T = 400                                              # phase comparison grid

# option 3: perturbed SVD training box (the check_models box) and noise levels
TRAIN_A = np.linspace(-0.99, 0.99, 23)
TRAIN_CHI2 = np.linspace(0.0, 1.0, 5)
TRAIN_E = np.linspace(0.05, 0.5, 6)
TRAIN_V = np.linspace(31 ** -0.5, 5 ** -0.5, 60)
F_LEVELS = (0.0, 0.2, 0.5, 1.0)
N_DRAW = 4                                             # perturbed copies per (a, chi2, e) for f > 0
K_SVD = (3, 4, 5, 6)
SEED = 1


def branch_check():
    branch = subprocess.run(["git", "-C", str(SUPERKLUDGE), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print(f"SuperKludge_r branch: {branch}", flush=True)
    if branch != BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {BRANCH!r}")


# --- rates ------------------------------------------------------------------------------------
def lead_shapes(p, e):
    """(4, n) leading 1PA shapes (as flux.py)."""
    f = (1 - e * e) ** 1.5
    return np.array([f * (8 + 7 * e * e) / p ** 3.5, e * f * (304 + 121 * e * e) / p ** 4.5, f / p ** 2, f / p ** 2])


def rates_1pa(a, p, e, chi2):
    """(4, n) SK 1PA functions (the rate added is eta times this)."""
    return np.array([CHANNELS[ch][0](a, p, e, chi2) for ch in CH])


def rates_0pa(m1, m2, a, chi2, p, e):
    """(4, n) template 0PA rates from SuperKludgeFlux itself, in the units the deviation is added in."""
    from few.trajectory.ode.flux import SuperKludgeFlux
    fl = SuperKludgeFlux()
    fl.add_fixed_parameters(m1, m2, a, [chi2, 0, 0, 0, 0])
    out = np.array([fl(np.array([pi, ei, 1.0, 0.0, 0.0])).copy() for pi, ei in zip(p, e)])
    return out[:, SLOTS].T


# --- option 3: perturbed SVD basis ------------------------------------------------------------
def training_rows(f, rng):
    """Rows of r_X(v) = 1PA_X / shape_X over the box; for f > 0 each PN-order term is scaled by U(1-f, 1+f)."""
    p = TRAIN_V ** -2
    rows = []
    for a in TRAIN_A:
        for chi2 in TRAIN_CHI2:
            for e in TRAIN_E:
                ee = np.full_like(p, e)
                terms = np.array([order_terms(ch, a, p, ee, chi2) for ch in CH])      # (4 ch, 4 orders, nv)
                sh = lead_shapes(p, ee)
                for _ in range(1 if f == 0 else N_DRAW):
                    scale = rng.uniform(1 - f, 1 + f, size=(4, 4, 1)) if f > 0 else 1.0
                    rows.append(((terms * scale).sum(axis=1) / sh).ravel())
    return np.array(rows)


def build_basis(f):
    """Cubic coefficients beta[j, channel, power] of the top singular vectors, and the normalised singular values."""
    M = training_rows(f, np.random.default_rng(SEED))
    nv = TRAIN_V.size
    scale = np.array([np.sqrt(np.mean(M[:, k * nv:(k + 1) * nv] ** 2)) for k in range(4)])
    _, sv, Vt = np.linalg.svd(M / np.repeat(scale, nv), full_matrices=False)
    V3 = np.vander(TRAIN_V, 4, increasing=True)
    beta = np.array([[np.linalg.lstsq(V3, Vt[j, k * nv:(k + 1) * nv] * scale[k], rcond=None)[0] for k in range(4)]
                     for j in range(max(K_SVD))])
    return beta, sv / sv[0]


# --- forms ------------------------------------------------------------------------------------
def unit(k, g):
    """g(v[, v0]) in channel k only."""
    return lambda v, *v0: np.array([g(v, *v0) if c == k else 0 * v for c in range(4)])


def every(g):
    """g(v[, v0]) in all four channels."""
    return lambda v, *v0: np.array([g(v, *v0)] * 4)


def cubic(b):
    """b[channel, power] -> g(v) = sum_m b[X, m] v^m per channel."""
    return lambda v: np.einsum("cm,mn->cn", b, np.vander(np.atleast_1d(v), 4, increasing=True).T)


def make_forms(bases):
    one = lambda v: 1 + 0 * v
    forms = {
        "B": ("lead", [unit(k, one) for k in range(4)]),
        "V3": ("lead", [unit(k, one) for k in range(4)] + [every(lambda v: v ** 3)]),
        "REL5_0": ("0pa", [unit(k, one) for k in range(4)] + [every(lambda v: v ** 3)]),
        "REL5_1": ("0pa", [unit(k, lambda v: v) for k in range(4)] + [every(lambda v: v ** 4)]),
        "REL8_1": ("0pa", [unit(k, lambda v: v) for k in range(4)] + [unit(k, lambda v: v ** 4) for k in range(4)]),
        "REL5_1S": ("0pa", [unit(k, lambda v, v0: v - v0) for k in range(4)] + [every(lambda v, v0: v ** 4)], "v0"),
        "REL5_1S2": ("0pa", [unit(k, lambda v, v0: v - v0) for k in range(4)]
                     + [every(lambda v, v0: v ** 4 - v0 ** 4)], "v0"),
    }
    for f, beta in bases.items():
        for K in K_SVD:
            forms[f"SVDf{f}_{K}"] = ("lead", [cubic(beta[j]) for j in range(K)])
    return forms


def shifted(form):
    """True for forms whose basis also takes v0 (the start of the track)."""
    return len(form) > 2 and form[2] == "v0"


def design(form, v, v0=None):
    """(n_params, 4, n) basis functions g_kX(v) (or g_kX(v, v0) for shifted forms)."""
    if shifted(form) and v0 is None:
        raise ValueError("shifted form needs v0")
    extra = (v0,) if shifted(form) else ()
    return np.array([g(v, *extra) for g in form[1]])


def fit(form, S, v, target):
    """Expected x: least squares of S_X sum_k x_k g_kX to the 1PA rate, each rate weighted by 1/rms. Returns x, rel.
    v runs along the track from its start, so v0 = v[0] for shifted forms."""
    w = 1 / np.sqrt(np.mean(target ** 2, axis=1, keepdims=True))
    G = design(form, v, v[0]) * S[None] * w[None]
    x = np.linalg.lstsq(G.reshape(G.shape[0], -1).T, (target * w).ravel(), rcond=None)[0]
    rem = target - S * np.einsum("k,kcn->cn", x, design(form, v, v[0]))
    rel = np.sqrt(np.mean(rem ** 2, axis=1) / np.mean(target ** 2, axis=1))
    return x, rel


# --- tracks and injections --------------------------------------------------------------------
def load_track(grid, point):
    d = np.load(CHECK / "results" / f"track_{grid}_idx{point}.npz")
    return {k: (d[k] if d[k].ndim else float(d[k])) for k in d.files}


def track_inputs(tr, need_0pa):
    p, e = tr["p"], tr["e"]
    S = {"lead": lead_shapes(p, e)}
    if need_0pa:
        S["0pa"] = rates_0pa(tr["m1"], tr["m2"], tr["a"], tr["chi2"], p, e)
    return p ** -0.5, S, rates_1pa(tr["a"], p, e, tr["chi2"])


def injection(grid, point):
    from families import ovl
    row = ovl.signal_array(grid)[int(point)]
    return {n: float(row[ovl.COL[n]]) for n in ("m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0", "chi2", "T")}


# --- trajectories -----------------------------------------------------------------------------
def make_dev_flux(form, x):
    """SuperKludgeFlux (deviation off) that adds eta S_X sum_k x_k g_kX(v) itself."""
    from few.trajectory.ode.flux import SuperKludgeFlux
    kind, x = form[0], np.asarray(x)

    class DevFlux(SuperKludgeFlux):
        def modify_rhs(self, ydot, y, **kwargs):
            super().modify_rhs(ydot, y, **kwargs)
            p, e = float(y[0]), float(y[1])
            S = lead_shapes(p, e) if kind == "lead" else ydot[list(SLOTS)].copy()     # ydot is still 0PA here
            add = self.massratio * S * np.einsum("k,kc->c", x, design(form, np.array([p ** -0.5]))[..., 0])
            ydot[list(SLOTS)] += add

    DevFlux.__name__ = "SuperKludgeFlux"        # EMRIInspiral keys its SuperKludge set-up on this name
    return DevFlux


def phases(inj, flux_cls, args, t_eval):
    from few.trajectory.inspiral import EMRIInspiral
    traj = EMRIInspiral(func=flux_cls)
    m = (inj["m1"], inj["m2"], inj["a"], inj["p0"], inj["e0"], 1.0)
    kw = dict(Phi_phi0=inj["Phi_phi0"], Phi_r0=inj["Phi_r0"], T=inj["T"], dt=10.0)
    t_end = traj(*m, *args, **kw)[0][-1]
    out = traj(*m, *args, **kw, upsample=True, new_t=t_eval[t_eval <= t_end])
    return out[4], out[6]


def compare(sig, ph):
    n = min(ph[0].size, sig[0].size)
    return dict(end_phi=float(sig[0][n - 1] - ph[0][n - 1]), end_r=float(sig[1][n - 1] - ph[1][n - 1]),
                rms_phi=float(np.sqrt(np.mean((sig[0][:n] - ph[0][:n]) ** 2))),
                rms_r=float(np.sqrt(np.mean((sig[1][:n] - ph[1][:n]) ** 2))))


# --- runs -------------------------------------------------------------------------------------
def all_bases():
    out = {}
    for f in F_LEVELS:
        beta, sv = build_basis(f)
        out[f] = (beta, sv)
        print(f"f={f}: singular values " + " ".join(f"{s:.3e}" for s in sv[:7])
              + "  cumulative energy " + " ".join(f"{c:.6f}" for c in (np.cumsum(sv ** 2) / np.sum(sv ** 2))[:7]),
              flush=True)
    return out


def run_rates(forms, svs):
    names = list(forms)
    print("\nmax over the four rates of the relative rms remainder\npoint        " + " ".join(f"{n:>11s}" for n in names))
    table = {}
    for grid in ("EMRI", "IMRI"):
        for i in range(25):
            tr = load_track(grid, i)
            v, S, target = track_inputs(tr, need_0pa=True)
            row = {n: fit(forms[n], S[forms[n][0]], v, target) for n in names}
            table[f"{grid}_{i}"] = dict(a=tr["a"], **{n: dict(x=list(map(float, r[0])), rel=list(map(float, r[1])))
                                                     for n, r in row.items()})
            print(f"{grid[0]} {i:2d} a={tr['a']:+.1f} " + " ".join(f"{max(row[n][1]):11.4f}" for n in names), flush=True)
    (RES / "rates.json").write_text(json.dumps(dict(singular_values={str(f): list(map(float, sv)) for f, sv in svs.items()},
                                                    points=table), indent=1))


def run_dephase(forms, grid, point):
    from few.trajectory.ode.flux import SuperKludgeFlux
    inj, tr = injection(grid, point), load_track(grid, point)
    v, S, target = track_inputs(tr, need_0pa=True)
    t_eval = np.linspace(0, tr["t"][-1], N_T)
    flags = lambda pa1, dev, c=(): [inj["chi2"], pa1, 0, 0, dev, *c, *[0.0] * (5 - len(c))]
    sig = phases(inj, SuperKludgeFlux, flags(1, 0), t_eval)
    res = {"0PA": dict(n_params=0, x=[], rate_rel=[1.0] * 4, **compare(sig, phases(inj, SuperKludgeFlux, flags(0, 0), t_eval)))}
    for name, form in forms.items():
        x, rel = fit(form, S[form[0]], v, target)
        if name == "V3":            # through flux.py on PN_v_3 itself: [C_p, C_e, C_Omphi, C_Omr, K3]
            ph = phases(inj, SuperKludgeFlux, flags(0, 1, x), t_eval)
        else:
            ph = phases(inj, make_dev_flux(form, x), flags(0, 0), t_eval)
        res[name] = dict(n_params=len(form[1]), x=list(map(float, x)), rate_rel=list(map(float, rel)), **compare(sig, ph))
    for name, d in res.items():
        print(f"{grid} idx{point} {name:12s} ({d['n_params']}) rate rem max {max(d['rate_rel']):.4f}  end dPhi_phi "
              f"{d['end_phi']:+10.4f}  end dPhi_r {d['end_r']:+10.4f}  rms ({d['rms_phi']:.4f}, {d['rms_r']:.4f})",
              flush=True)
    (RES / f"dephase_{grid}_idx{point}.json").write_text(json.dumps(dict(injection=inj, forms=res), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", action="store_true")
    ap.add_argument("--dephase", action="store_true")
    ap.add_argument("--grid", choices=("EMRI", "IMRI"))
    ap.add_argument("--point")
    args = ap.parse_args()
    branch_check()
    RES.mkdir(exist_ok=True)
    bases = all_bases()
    forms = make_forms({f: b for f, (b, _) in bases.items()})
    if args.rates:
        run_rates(forms, {f: sv for f, (_, sv) in bases.items()})
    if args.dephase:
        run_dephase(forms, args.grid, args.point)


if __name__ == "__main__":
    main()
