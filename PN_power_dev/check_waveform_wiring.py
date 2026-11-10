"""Waveform check of the power-law deviation through the real response and SEF stack, one grid point.

    1. deviation on with n = 0 gives the same waveform as the old slot values 0, 0 (trivially the same
       args; kept as a guard on the name swap) and differs from 0PA once C_p != 0
    2. n_p, n_e change the waveform when C != 0, and do nothing when C = 0
    3. SEF's d/dn_p, d/dn_e, d/dC_p against central finite differences of make_h

EMRI_1PA_PN_DEV/model.py is used unchanged; only its slot names are swapped here.
Run:  python check_waveform_wiring.py --point 4 --T 0.25
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1] / "EMRI_1PA_PN_DEV"
sys.path.insert(0, str(STUDY))
import config as C                                                            # noqa: E402

C.DEV_SLOTS[:] = ["C_p", "C_e", "n_p", "n_e"]
C.DEV_INJ.update(n_p=0.0, n_e=0.0)
C.NEAR_ZERO.update(n_p=1e-10, n_e=1e-10)
from model import build_grid, build_point, point_T_dt                          # noqa: E402

NAMES = ["Phi_phi0", "C_p", "C_e", "n_p", "n_e"]
CHECK = {"C_p": [0.5, 2.0], "n_p": [0.01, 0.05], "n_e": [0.01, 0.05]}
SEF_STEPS = {"Phi_phi0": 1e-5, "C_p": 0.5, "C_e": 0.5, "n_p": 0.01, "n_e": 0.01}


def rel(a, b, inner):
    d = [x - y for x, y in zip(a, b)]
    return float(np.sqrt(inner(d, d) / inner(a, a)))


def fd(P, theta, i, step):
    up, dn = theta.copy(), theta.copy()
    up[i] += step
    dn[i] -= step
    return [(a - b) / (2 * step) for a, b in zip(P["make_h"](up), P["make_h"](dn))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="4")
    ap.add_argument("--T", type=float, default=0.25)
    args = ap.parse_args()
    sig_row = C.signal_row(args.point)
    xp, use_gpu = C.ovl.load_backend()
    Gd = build_grid(xp, use_gpu, args.T, point_T_dt(sig_row)[1])
    inner = Gd["inner"]
    best = json.loads((STUDY / "results/optimized_points.json").read_text())
    best = best["points"][int(args.point)]["best_fit"]["0PA+PN"]["params"]
    P = build_point(Gd, sig_row, NAMES)
    phi = float(sig_row[C.COL["Phi_phi0"]])
    th = lambda Cp, Ce, n1, n2: np.array([phi, Cp, Ce, n1, n2])               # noqa: E731
    Cp, Ce = best["C_p"], best["C_e"]
    h0, hC = P["make_h"](th(0, 0, 0, 0)), P["make_h"](th(Cp, Ce, 0, 0))
    print(f"1. C != 0, n = 0 vs C = 0: rel diff {rel(h0, hC, inner):.3e}", flush=True)
    print(f"2. C = 0, n = (0.7, -0.7) vs C = 0: rel diff "
          f"{rel(h0, P['make_h'](th(0, 0, .7, -.7)), inner):.3e}", flush=True)
    print(f"   C != 0, n_p = 0.5 vs n = 0: rel diff {rel(hC, P['make_h'](th(Cp, Ce, .5, 0)), inner):.3e}", flush=True)
    print(f"   C != 0, n_e = 0.5 vs n = 0: rel diff {rel(hC, P['make_h'](th(Cp, Ce, 0, .5)), inner):.3e}", flush=True)
    theta = th(Cp, Ce, 0.3, 0.3)
    dH, used = P["derivs"](theta, SEF_STEPS)
    for name, steps in CHECK.items():
        i = NAMES.index(name)
        for h in steps:
            print(f"3. d/d{name}: SEF (step {used[name]:g}) vs central FD (step {h:g}): rel diff "
                  f"{rel(fd(P, theta, i, h), dH[i], inner):.3e}", flush=True)


if __name__ == "__main__":
    main()
