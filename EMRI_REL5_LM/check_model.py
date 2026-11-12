"""Check of the waveform path (response -> SuperKludgeWaveform -> flux.py REL5_1) before the GPU runs.

At one EMRI point with a short observation time (T_CHECK), through model.build_point exactly as run_lm.py uses it:
    1. GR limit : deviation_included = 1 with all five coefficients 0 gives the same template as plain 0PA
    2. response : non-zero coefficients change the template (chi2 against the 1PA signal moves)
    3. slots    : each coefficient alone changes the template (all five slots reach flux.py)
SEF derivatives are not built here. Needs a GPU (FEW has no CPU waveform backend on the login node).
Run:  python check_model.py > logs/check_model.log
"""
import numpy as np

import config as C
from model import build_point, evaluate, waveform_AE, add_args

T_CHECK = 0.1                 # yr
POINT = "0"


def grid_light(T, dt):
    """Response and inner product only (no SEF), on the GPU if there is one."""
    xp, use_gpu = C.ovl.load_backend()
    return dict(T=T, dt=dt, xp=xp, resp=C.ovl.build_response(T, dt, use_gpu, C.GEN),
                inner=C.ovl.make_inner(xp, dt, C.DROP_DC), sef=None)


def max_diff(h1, h2):
    return max(float(abs(a - b).max()) for a, b in zip(h1, h2))


def main():
    row = C.signal_row(POINT)
    dt = float(row[C.COL["dt"]])
    Gd = grid_light(T_CHECK, dt)
    P = build_point(Gd, row, C.FIT_PARAMS)
    inj = np.array([C.injected_value(row, n) for n in C.FIT_PARAMS])
    chi2 = float(row[C.COL["chi2"]])
    base = {n: float(row[C.COL[n]]) for n in C.ARGS14}

    plain = waveform_AE(Gd, base, add_args(chi2, C.TEMPLATE_FLAGS, None))
    zero_dev = P["make_h"](inj)
    d = max_diff(plain, zero_dev)
    print(f"[{'PASS' if d == 0 else 'FAIL'}] GR limit: max |h(dev on, C = 0) - h(0PA)| = {d:.3e}", flush=True)

    e0 = evaluate(P, inj)
    seed = inj.copy()
    seed[len(C.PHYS):] = [0.330241, 11.8206, 1.93908, 1.01968, 62.8424]       # idx0 free-fit coefficients
    e1 = evaluate(P, seed)
    ok = e1["chi2"] != e0["chi2"]
    print(f"[{'PASS' if ok else 'FAIL'}] response: chi2 0PA-at-injection {e0['chi2']:.6e} (O={e0['overlap']:.10f}) "
          f"-> with coefficients {e1['chi2']:.6e} (O={e1['overlap']:.10f}), rho_s={e0['rho_s']:.4f} at T={T_CHECK}",
          flush=True)

    for k, n in enumerate(C.DEV_SLOTS):
        one = inj.copy()
        one[len(C.PHYS) + k] = seed[len(C.PHYS) + k]
        h = P["make_h"](one)
        d = max_diff(h, zero_dev)
        print(f"[{'PASS' if d > 0 else 'FAIL'}] slot {n}: max |h change| = {d:.3e}", flush=True)


if __name__ == "__main__":
    main()
