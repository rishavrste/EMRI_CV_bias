"""Importance of each PN-order term of the SuperKludge 1PA (sk_orders.py) at every EMRI and IMRI grid point.

Along the 1PA trajectory at the injection (N_T samples uniform in t):
    rates : rms(term_i) / rms(full 1PA) for pdot, edot, Omega_phi, Omega_r, orders i = 0..3
            (terms can cancel, so the four fractions need not add to 1)
Dephasing at t = T against 0PA (Phi_phi, Phi_r), from trajectories that add to 0PA:
    single_i : only order i                    -> what order i alone does to the phase
    upto_i   : orders 0..i (upto_3 = full 1PA)  -> how the truncated 1PA converges to the full one
    -> results/pn_orders_{grid}_idx{p}.json, and the 1PA track (t, p, e) in results/track_{grid}_idx{p}.npz
Run:  python pn_orders.py --grid EMRI --point 4 > logs/pn_orders_EMRI_idx4.log
"""
import argparse
import json
from pathlib import Path

import numpy as np

from families import TEMPLATE_FLAGS, branch_check, ovl
from sk_orders import CHANNELS, ORDERS, make_order_flux, order_terms, self_check

HERE = Path(__file__).resolve().parent
YR = 3.15581e7
N_T = 1000
PHYS = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0", "chi2", "T"]


def injection(grid, idx):
    row = ovl.signal_array(grid)[idx]
    return {n: float(row[ovl.COL[n]]) for n in PHYS}


def trajectory(inj, flux_cls, t_eval=None):
    """(t, p, e, Phi_phi, Phi_r) of 0PA + whatever flux_cls adds."""
    from few.trajectory.inspiral import EMRIInspiral
    traj = EMRIInspiral(func=flux_cls)
    m = (inj["m1"], inj["m2"], inj["a"], inj["p0"], inj["e0"], 1.0)
    args = [inj["chi2"], *TEMPLATE_FLAGS, 0, 0] + [0.0] * 8
    kw = dict(Phi_phi0=inj["Phi_phi0"], Phi_r0=inj["Phi_r0"], T=inj["T"], dt=10.0)
    t_end = traj(*m, *args, **kw)[0][-1]
    t_new = np.linspace(0, t_end, N_T) if t_eval is None else t_eval[t_eval <= t_end]
    out = traj(*m, *args, **kw, upsample=True, new_t=t_new)
    return dict(t=out[0], p=out[1], e=out[2], Phi_phi=out[4], Phi_r=out[6])


def rate_fractions(inj, tr):
    """rms(term_i)/rms(full) per channel along the trajectory."""
    out = {}
    for ch in CHANNELS:
        terms = order_terms(ch, inj["a"], tr["p"], tr["e"], inj["chi2"])
        full = terms.sum(axis=0)
        rms_full = np.sqrt(np.mean(full ** 2))
        out[ch] = [float(np.sqrt(np.mean(terms[i] ** 2)) / rms_full) for i in ORDERS]
    return out


def dephasing(inj):
    """End-of-T Phi_phi, Phi_r of each order set minus 0PA, at the common end time."""
    sets = {f"single_{i}": (i,) for i in ORDERS}
    sets.update({f"upto_{i}": tuple(range(i + 1)) for i in ORDERS})
    ref = trajectory(inj, make_order_flux(()))
    tr_1pa = None
    out = {}
    for name, orders in sets.items():
        tr = trajectory(inj, make_order_flux(orders), t_eval=ref["t"])
        n = min(tr["t"].size, ref["t"].size)
        out[name] = dict(dPhi_phi=float(tr["Phi_phi"][n - 1] - ref["Phi_phi"][n - 1]),
                         dPhi_r=float(tr["Phi_r"][n - 1] - ref["Phi_r"][n - 1]),
                         t_end_yr=float(tr["t"][n - 1] / YR))
        if name == "upto_3":
            tr_1pa = tr
    return out, tr_1pa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, choices=("EMRI", "IMRI"))
    ap.add_argument("--point", required=True, type=int)
    args = ap.parse_args()
    branch_check()
    inj = injection(args.grid, args.point)
    self_check(inj["a"], inj["p0"], inj["e0"], inj["chi2"])
    deph, tr = dephasing(inj)
    rates = rate_fractions(inj, tr)
    tag = f"{args.grid} idx{args.point}"
    print(f"{tag}: a={inj['a']} e0={inj['e0']} p0={inj['p0']:.4f} T={inj['T']} -> p_end={tr['p'][-1]:.4f}", flush=True)
    for ch, f in rates.items():
        print(f"  rate {ch:9s} rms(term_i)/rms(1PA): " + "  ".join(f"{x:7.3f}" for x in f), flush=True)
    for name, d in deph.items():
        print(f"  {name:8s} dPhi_phi={d['dPhi_phi']:+11.4f}  dPhi_r={d['dPhi_r']:+11.4f}", flush=True)
    path = HERE / "results" / f"pn_orders_{args.grid}_idx{args.point}.json"
    path.write_text(json.dumps(dict(grid=args.grid, point=args.point, injection=inj,
                                    p_range=[float(tr["p"][0]), float(tr["p"][-1])],
                                    rate_fraction=rates, dephasing=deph), indent=1))
    np.savez(HERE / "results" / f"track_{args.grid}_idx{args.point}.npz", t=tr["t"], p=tr["p"], e=tr["e"],
             **{k: inj[k] for k in ("a", "chi2", "m1", "m2")})                # reused by fit_forms.py
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
