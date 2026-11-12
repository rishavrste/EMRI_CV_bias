"""LM seeds for the EMRI grid (CPU): injected physical parameters + REL5_1 coefficients fitted to the 1PA phases.

Per point:
    1. expected coefficients: the 1PA rate fit along the injected 1PA track (EMRI_v_3/forms_v3.fit)
    2. phase fit at the injection (EMRI_v_3/phase_fit_v3.fit, physical parameters fixed), starting from 1,
       integrating flux.py's own REL5_1 deviation (the one the GPU waveform uses)
    -> results/seeds.json  {point: {9 physical (injected), c_p, c_e, c_Omphi, c_Omr, K, + diagnostics}}
       rewritten after every point; points already in it are skipped.
Run:  python make_seeds.py [--points 0 1 ...] > logs/make_seeds.log
"""
import argparse
import json
import sys

import numpy as np

import config as C

sys.path.insert(0, str(C.EMRI_V3))
import forms_v3 as V            # noqa: E402
import phase_fit_v3 as PF       # noqa: E402


def flux_problem(point, forms):
    """phase_fit_v3.Problem for REL5_1, but integrating SuperKludgeFlux (flux.py's REL5_1) directly."""
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode.flux import SuperKludgeFlux
    P = PF.Problem(C.GRID, point, "REL5_1", forms)
    P.traj = EMRIInspiral(func=SuperKludgeFlux)
    return P


def seed_point(point, forms):
    P = flux_problem(point, forms)
    c_rate = PF.expected_coeffs(C.GRID, point, forms["REL5_1"])
    out, _, c = PF.fit(P, P.inj, c_rate, free_phys=False)
    PF.log_fit(f"{C.point_label(point)} at_injection", out)
    row = C.signal_row(point)
    seed = {n: float(row[C.COL[n]]) for n in C.PHYS}
    seed.update(dict(zip(C.DEV_SLOTS, map(float, c))))
    seed["diagnostics"] = dict(rate_fit=list(map(float, c_rate)), start=out["start"], final=out["final"],
                               nfev=out["nfev"])
    return seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", default=list(C.POINTS), choices=C.POINTS)
    args = ap.parse_args()
    V.branch_check()
    forms = V.make_forms({})
    seeds = json.loads(C.SEEDS_FILE.read_text()) if C.SEEDS_FILE.exists() else {}
    for point in args.points:
        if point in seeds:
            continue
        seeds[point] = seed_point(point, forms)
        C.SEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        C.SEEDS_FILE.write_text(json.dumps(seeds, indent=1))
        print(f"  seed idx{point}: " + ", ".join(f"{n}={seeds[point][n]:.6g}" for n in C.DEV_SLOTS), flush=True)


if __name__ == "__main__":
    main()
