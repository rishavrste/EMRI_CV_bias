"""GPU LM climbs of the 0PA + REL5_1 template on the EMRI grid (14 parameters: 9 physical + 5 coefficients).

    full   : one LM climb at T = config.T_FULL from the seed (results/seeds.json, make_seeds.py)
    ladder : the T ladder config.LADDER, run only when "full" ends below the 0PA best-fit overlap
    reseed : (--coeff-seed-from J) one climb at T_FULL, physical parameters at the injection, deviation
             coefficients from point J's "full" final fit; used to look for another maximum at a point

    rung at T  -> results/{arm}/lm_T{T}_{label}.json
    last rung  -> results/{arm}/lm_{label}.json       label = idx4
Each JSON is rewritten after every accepted step, so a walltime kill still leaves the latest point.
Run (GPU):  python run_lm.py --points 4
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

import config as C
from lm import lm_climb
from model import build_grid, build_point, evaluate, point_T_dt


# --- output ----------------------------------------------------------------
def save(out):
    d = C.OUT_ROOT / out["arm"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{out['stem']}_{C.point_label(out['point'])}.json"
    with open(f"{path}.tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(f"{path}.tmp", path)


def as_dict(names, vec):
    return dict(zip(names, map(float, vec)))


def check_branch():
    """Stop if SuperKludge_r is not on the branch that carries REL5_1."""
    branch = subprocess.run(["git", "-C", str(C.SUPERKLUDGE), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print(f"SuperKludge_r branch: {branch}", flush=True)
    if branch != C.SUPERKLUDGE_BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {C.SUPERKLUDGE_BRANCH!r}")


# --- seeds and floor ---------------------------------------------------------
def seed(point):
    """Injected physical parameters + phase-fitted REL5_1 coefficients (make_seeds.py)."""
    s = json.loads(C.SEEDS_FILE.read_text())[point]
    return np.array([s[n] for n in C.FIT_PARAMS])


def floor_overlap(point):
    """Overlap a working climb must reach: the 0PA best fit of EMRI_1PA_PN_DEV (same TDI setup)."""
    return json.loads(C.OLD_BEST_FITS.read_text())[C.ACCEPT_FLOOR][point]["final_eval"]["overlap"]


def coeffs_from(point):
    """Deviation coefficients of another point's finished "full" climb."""
    r = json.loads((C.OUT_ROOT / "full" / f"lm_{C.point_label(point)}.json").read_text())
    if r.get("running"):
        raise SystemExit(f"full climb of {C.point_label(point)} is not finished")
    return np.array([r["theta_final"][n] for n in C.DEV_SLOTS])


# --- one rung --------------------------------------------------------------
def run_rung(Gd, point, theta0, arm, stem, start):
    """LM climb at one point and one observation time, from theta0."""
    names = C.FIT_PARAMS
    sig_row = C.signal_row(point)
    P = build_point(Gd, sig_row, names)
    tag = f"{arm} {stem} T={Gd['T']} {start} {C.point_label(point)}"
    t0 = time.time()

    first = evaluate(P, theta0)
    print(f"[START {tag}] O={first['overlap']:.12f} chi2={first['chi2']:.6e} "
          f"rho_s={first['rho_s']:.6f}", flush=True)
    out = dict(grid=C.GRID, signal="1PA", template=C.TEMPLATE, arm=arm, stem=stem, start=start,
               point=point, a_inj=float(sig_row[C.COL["a"]]), e0_inj=float(sig_row[C.COL["e0"]]),
               params=names, chi2_spin=float(sig_row[C.COL["chi2"]]), gen=str(C.GEN),
               drop_dc=C.DROP_DC, tdi="2nd generation, AE", setup=C.SETUP, dt=Gd["dt"], T=Gd["T"],
               theta_inj={n: C.injected_value(sig_row, n) for n in names},
               theta_start=as_dict(names, theta0), start_eval=first)

    def checkpoint(cur, sigma, history):
        save(dict(out, theta_current=as_dict(names, cur), sigma_current=as_dict(names, sigma),
                  history=history, running=True, seconds=time.time() - t0))

    theta, sigma, info, history = lm_climb(P, theta0, tag, on_iter=checkpoint)
    final = evaluate(P, theta)
    out.update(theta_final=as_dict(names, theta), final_eval=final,
               sigma_final=as_dict(names, sigma), info=info, history=history, running=False,
               seconds=time.time() - t0)
    save(out)
    print(f"[DONE {tag}] O {first['overlap']:.12f} -> {final['overlap']:.12f}  "
          f"chi2 {first['chi2']:.6e} -> {final['chi2']:.6e}  iters={info['n_iter']} "
          f"accepted={info['n_accept']} stop={info['stop_reason']}  {out['seconds']:.1f} s",
          flush=True)
    print("theta_final = " + ", ".join(f"{n}={v!r}" for n, v in out["theta_final"].items()), flush=True)
    print("bias        = " + ", ".join(f"{n}={out['theta_final'][n] - out['theta_inj'][n]:+.6g}" for n in C.PHYS),
          flush=True)
    return out


# --- arms ------------------------------------------------------------------
def free_gpu(xp):
    """Release the cached device memory of a grid that is no longer needed."""
    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()


def climb_arm(xp, use_gpu, point, arm, theta=None, start="seed"):
    rungs = C.ARMS[arm]
    dt = point_T_dt(C.signal_row(point))[1]
    theta = seed(point) if theta is None else theta
    for k, t in enumerate(rungs):
        Gd = build_grid(xp, use_gpu, t, dt)
        stem = "lm" if k == len(rungs) - 1 else f"lm_T{t:g}"
        out = run_rung(Gd, point, theta, arm, stem, start)
        del Gd
        free_gpu(xp)
        theta = np.array([out["theta_final"][n] for n in C.FIT_PARAMS])
        start = f"T{t:g}_final"
    return out


def climb_point(xp, use_gpu, point):
    """The full-T arm; then the ladder if that failed or ended below the 0PA floor."""
    floor = floor_overlap(point)
    try:
        ov = climb_arm(xp, use_gpu, point, "full")["final_eval"]["overlap"]
    except Exception as exc:                                # noqa: BLE001  fall through to the ladder
        print(f"[FAIL full {C.point_label(point)}] {type(exc).__name__}: {exc}", flush=True)
        ov = -1.0
    verdict = "worked" if ov >= floor else "did not work"
    print(f"[full {C.point_label(point)}] O={ov:.12f} vs 0PA floor {floor:.12f}: {verdict}", flush=True)
    if ov < floor:
        climb_arm(xp, use_gpu, point, "ladder")


def reseed(xp, use_gpu, points, src):
    """Reseed arm: injected physical parameters + the deviation coefficients of point src."""
    c = coeffs_from(src)
    for point in points:
        theta = seed(point)
        theta[len(C.PHYS):] = c
        print(f"[reseed {C.point_label(point)}] coefficients from {C.point_label(src)}: "
              + ", ".join(f"{n}={v:.6g}" for n, v in zip(C.DEV_SLOTS, c)), flush=True)
        out = climb_arm(xp, use_gpu, point, "reseed", theta, f"coeffs_{C.point_label(src)}")
        old = json.loads((C.OUT_ROOT / "full" / f"lm_{C.point_label(point)}.json").read_text())
        print(f"[reseed {C.point_label(point)}] O={out['final_eval']['overlap']:.12f} vs full "
              f"{old['final_eval']['overlap']:.12f}; m1 bias {out['theta_final']['m1'] - out['theta_inj']['m1']:+.4f} "
              f"vs full {old['theta_final']['m1'] - old['theta_inj']['m1']:+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=C.POINTS, required=True)
    ap.add_argument("--coeff-seed-from", choices=C.POINTS, default=None,
                    help="reseed arm only: take the deviation coefficients from this point's full fit")
    args = ap.parse_args()

    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    if args.coeff_seed_from is not None:
        reseed(xp, use_gpu, args.points, args.coeff_seed_from)
        return
    for point in args.points:
        try:
            climb_point(xp, use_gpu, point)
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL {C.point_label(point)}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
