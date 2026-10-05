"""PN re-climb by a C_p ramp, for fits stalled on the C = 0 ridge where a jump to the row C_p dephases
the template beyond LM's reach (make_seeds_row.py, job 645637 at idx7, 12, 17).

Rung 1 (T = first t-frac x the point's T):
    from the point's own 0PA ladder fit at that T, step C_p = +-RAMP_STEP, +-2 RAMP_STEP, ... out to
    +-RAMP_MAX with C_e = 0, re-fitting PHYS by LM at each step from the step before; a direction stops
    once 1 - O exceeds RAMP_LOSS x its value at C_p = 0. The best step then seeds the free PN climb
    (PHYS + C_p, C_e) on the same grid.
Later rungs: the T ladder of run_lm_T_ladder.py, each from the rung before.
    profile           -> results/{case}/ramp_T{T}_{label}.json   (rewritten after every step)
    free PN climbs    -> results/{case}/lm_T{T}_{label}.json, lm_{label}.json
case = config.case_name("pn", t_fracs, "ramp").

Run (GPU):  python run_cp_ramp.py --points 7 --t-fracs 0.9 1
"""
import argparse
import json
import time

import numpy as np

import config as C
from lm import lm_climb
from model import evaluate, build_point
from run_lm_T_ladder import as_dict, check_branch, climb_ladder, rung_stem

GRID_POINTS = [p for p in C.POINTS if p not in C.ADHOC]


# --- the 0PA starting point -------------------------------------------------
def zero_pa_fit(point, t_fracs, T):
    """The point's 0PA fit on rung 1 (run_lm_T_ladder.py, seed old), observation time T."""
    path = (C.OUT_ROOT / C.case_name("0pa", t_fracs) /
            f"{rung_stem(0, t_fracs[0], T, t_fracs)}_{C.point_label(point)}.json")
    th = json.loads(path.read_text())["theta_final"]
    return np.array([th[n] for n in C.PHYS]), path.name


# --- one step: PHYS re-fitted at fixed C_p ------------------------------------
def fit_at(Gd, point, theta, cp):
    P = build_point(Gd, C.signal_row(point), C.PHYS, fixed={"C_p": cp, "C_e": 0.0})
    tag = f"ramp T={Gd['T']} {C.point_label(point)} C_p={cp:+g}"
    th, _, info, _ = lm_climb(P, theta, tag, max_iters=C.RAMP_MAX_ITERS, rel_tol=C.RAMP_REL_TOL)
    ev = evaluate(P, th)
    print(f"[RAMP {tag}] O={ev['overlap']:.12f} chi2={ev['chi2']:.6e} iters={info['n_iter']} "
          f"stop={info['stop_reason']}", flush=True)
    return dict(C_p=cp, theta=as_dict(C.PHYS, th), eval=ev, info=info)


def scan_direction(Gd, point, step0, sign, profile, save):
    """Walk C_p away from 0 in one direction until RAMP_MAX or the template is lost."""
    loss0 = 1 - step0["eval"]["overlap"]
    theta = np.array([step0["theta"][n] for n in C.PHYS])
    n_steps = int(round(C.RAMP_MAX / C.RAMP_STEP))
    for k in range(1, n_steps + 1):
        step = fit_at(Gd, point, theta, sign * k * C.RAMP_STEP)
        profile.append(step)
        save()
        if 1 - step["eval"]["overlap"] > C.RAMP_LOSS * loss0:
            print(f"    direction {sign:+d} stopped at C_p={step['C_p']:+g}: 1-O > "
                  f"{C.RAMP_LOSS:g} x {loss0:.3e}", flush=True)
            return
        theta = np.array([step["theta"][n] for n in C.PHYS])


def ramp_seed(point, t_fracs):
    """start() for climb_ladder: run the C_p profile on rung 1's grid, seed from its best step."""
    def start(Gd, case):
        theta0, src = zero_pa_fit(point, t_fracs, Gd["T"])
        P0 = build_point(Gd, C.signal_row(point), C.PHYS)
        profile = [dict(C_p=0.0, theta=as_dict(C.PHYS, theta0), eval=evaluate(P0, theta0),
                        info=dict(source=src))]
        path = C.OUT_ROOT / case / f"ramp_T{Gd['T']:g}_{C.point_label(point)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        def save(done=False):
            path.write_text(json.dumps(dict(point=point, T=Gd["T"], step=C.RAMP_STEP,
                                            max=C.RAMP_MAX, loss=C.RAMP_LOSS, from_0pa=src,
                                            profile=sorted(profile, key=lambda s: s["C_p"]),
                                            done=done, seconds=time.time() - t0), indent=1))

        for sign in (+1, -1):
            scan_direction(Gd, point, profile[0], sign, profile, save)
        best = max(profile, key=lambda s: s["eval"]["overlap"])
        save(done=True)
        print(f"[RAMP BEST {C.point_label(point)} T={Gd['T']}] C_p={best['C_p']:+g} "
              f"O={best['eval']['overlap']:.12f}", flush=True)
        theta = dict(best["theta"], C_p=best["C_p"], C_e=0.0)
        return np.array([theta[n] for n in C.fit_params("pn")]), f"ramp_C_p{best['C_p']:+g}"
    return start


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=GRID_POINTS, required=True)
    ap.add_argument("--t-fracs", nargs="+", type=float, required=True,
                    help="observation times of the rungs as fractions of each point's T, in order")
    args = ap.parse_args()

    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    for point in args.points:
        try:
            climb_ladder(xp, use_gpu, point, "pn", args.t_fracs, "ramp",
                         start=ramp_seed(point, args.t_fracs))
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL C_p ramp {C.point_label(point)}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
