"""Wiring check of the Fisher stack at one grid point, CPU, a short T (the wiring does not depend on T).

    1. A, E of the AET response (overlap.py RISHAV_GEN, used for s, h and the inner product) equal
       those of an AE response (what SEF's response is built with)
    2. the flags: the 1PA signal differs from the 0PA template at the injection, and the 0PA template
       with 1PA / primary / 2PA all off is what make_h builds
    3. the deviation switched on with C_p = C_e = 0 gives the 0PA template exactly (SEF always passes
       deviation_included = True)
    4. SEF's stable derivatives (2nd-generation A2/E2 TDI, A and E) against central finite
       differences of make_h over a range of steps, for a phase, a sky angle and C_p

Run (CPU, from fisher/):  python check_fisher_wiring.py --point 7 --T 0.25 [--deltas steps.json]
--deltas: SEF steps from an earlier run ({name: step}), skipping SEF's step search (~30 min on CPU).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # EMRI_1PA_PN_DEV: config, model, lm
import config as C                                  # noqa: E402
from fisher_at_best import load_best_fit           # noqa: E402
from model import add_args, build_grid, build_point, point_T_dt, waveform_AE  # noqa: E402
from run_lm_T_ladder import check_branch           # noqa: E402

CHECK_PARAMS = ["Phi_phi0", "qS", "C_p"]
FD_STEPS = {"Phi_phi0": [1e-6, 1e-4], "qS": [1e-6, 1e-5, 7e-5, 3e-4], "C_p": [1e-2, 1e-1, 0.86, 2.0]}


def rel(a, b, inner):
    """|a - b| / |a| in the noise-weighted norm."""
    d = [x - y for x, y in zip(a, b)]
    return float(np.sqrt(inner(d, d) / inner(a, a)))


def check_channels(Gd, p14, chi2, inner):
    resp_ae = C.ovl.build_response(Gd["T"], Gd["dt"], False, dict(C.GEN, tdi_chan="AE"))
    aet = waveform_AE(Gd, p14, add_args(chi2, C.TEMPLATE_FLAGS, None))
    ae = [np.asarray(c) for c in resp_ae(*[p14[n] for n in C.ARGS14],
                                         *add_args(chi2, C.TEMPLATE_FLAGS, None),
                                         T=Gd["T"], dt=Gd["dt"])[:2]]
    print(f"1. AET[:2] vs AE response: rel diff {rel(aet, ae, inner):.3e}  (GEN tdi_chan "
          f"{C.GEN['tdi_chan']})", flush=True)


def check_flags(Gd, P, p14, chi2, inner):
    s = P["s"]
    h0 = waveform_AE(Gd, p14, add_args(chi2, C.TEMPLATE_FLAGS, None))
    s2 = waveform_AE(Gd, p14, add_args(chi2, C.SIGNAL_FLAGS, None))
    ov = inner(s, h0) / np.sqrt(inner(s, s) * inner(h0, h0))
    print(f"2. flags signal {C.SIGNAL_FLAGS} template {C.TEMPLATE_FLAGS}; overlap(1PA signal, 0PA "
          f"template) at the injection {ov:.10f}; signal rebuilt vs P['s'] rel diff "
          f"{rel(s, s2, inner):.3e}", flush=True)
    return h0


def check_dev_zero(Gd, p14, chi2, h0, inner):
    on = waveform_AE(Gd, p14, add_args(chi2, C.TEMPLATE_FLAGS, {n: 0.0 for n in C.DEV_SLOTS}))
    print(f"3. 0PA, deviation on with C = 0 vs off: rel diff {rel(h0, on, inner):.3e}", flush=True)


def central_fd(P, theta, i, step):
    up, dn = theta.copy(), theta.copy()
    up[i] += step
    dn[i] -= step
    return [(a - b) / (2 * step) for a, b in zip(P["make_h"](up), P["make_h"](dn))]


def check_derivs(P, theta, names, inner, deltas):
    dH, deltas = P["derivs"](theta, deltas)
    print(f"   SEF steps: {deltas!r}", flush=True)
    for n in CHECK_PARAMS:
        i = names.index(n)
        fds = [central_fd(P, theta, i, h) for h in FD_STEPS[n]]
        for h, fd, nxt in zip(FD_STEPS[n], fds, fds[1:] + [None]):
            vs_next = f", vs the next step {rel(fd, nxt, inner):.3e}" if nxt is not None else ""
            print(f"4. d/d{n}: SEF (step {deltas[n]:.3g}) vs central FD (step {h:g}): "
                  f"rel diff {rel(fd, dH[i], inner):.3e}{vs_next}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", default="7")
    ap.add_argument("--T", type=float, default=0.25)
    ap.add_argument("--deltas", default=None)
    args = ap.parse_args()
    check_branch()
    sig_row = C.signal_row(args.point)
    Gd = build_grid(np, False, args.T, point_T_dt(sig_row)[1])
    inner, chi2 = Gd["inner"], float(sig_row[C.COL["chi2"]])
    p14 = {n: float(sig_row[C.COL[n]]) for n in C.ARGS14}
    best = load_best_fit("pn", args.point)
    names = list(best["theta_final"])
    P = build_point(Gd, sig_row, names)
    check_channels(Gd, p14, chi2, inner)
    h0 = check_flags(Gd, P, p14, chi2, inner)
    check_dev_zero(Gd, p14, chi2, h0, inner)
    deltas = json.loads(open(args.deltas).read()) if args.deltas else None
    check_derivs(P, np.array([best["theta_final"][n] for n in names]), names, inner, deltas)


if __name__ == "__main__":
    main()
