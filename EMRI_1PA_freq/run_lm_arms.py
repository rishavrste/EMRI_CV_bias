"""LM climbs of the 0PA + PN + frequency template from the physical seed (physical_c.py).

    full   : one LM climb at T = config.T_FULL
    ladder : the T ladder config.LADDER, run only when "full" ends below the 0PA best-fit overlap

    rung at T  -> results/{arm}/lm_T{T}_{label}.json
    last rung  -> results/{arm}/lm_{label}.json       label = idx4
Each JSON is rewritten after every accepted step, so a walltime kill still leaves the latest point.

Seeds (config.SEED_ARMS): physical (arms full, ladder), pn_fit (full_pn, ladder_pn), 0pa_fit (full_0pa, ladder_0pa).

Run (GPU):  python run_lm_arms.py --points 4 --seed pn_fit
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
from physical_c import physical_c


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
    """Stop if SuperKludge_r is not on the branch that carries the frequency term."""
    branch = subprocess.run(["git", "-C", str(C.SUPERKLUDGE), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print(f"SuperKludge_r branch: {branch}", flush=True)
    if branch != C.SUPERKLUDGE_BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {C.SUPERKLUDGE_BRANCH!r}")


# --- seeds -----------------------------------------------------------------
def seed_physical(point):
    """The injected parameters, with C_p, C_e from the exact 1PA rates at the injection."""
    row = C.signal_row(point)
    return dict({n: float(row[C.COL[n]]) for n in C.PHYS}, **physical_c(point))


def seed_old_fit(template):
    """Seed from the physical parameters of an EMRI_1PA_PN_DEV best fit, with the physical C_p, C_e."""
    def seed(point):
        fit = json.loads(C.OLD_BEST_FITS.read_text())[template][point]["theta_final"]
        return dict({n: float(fit[n]) for n in C.PHYS}, **physical_c(point))
    return seed


SEEDS = {"physical": seed_physical, "pn_fit": seed_old_fit("pn"), "0pa_fit": seed_old_fit("0pa")}


def floor_overlap(point):
    """Overlap a working climb must reach: the C.ACCEPT_FLOOR best fit of EMRI_1PA_PN_DEV."""
    return json.loads(C.OLD_BEST_FITS.read_text())[C.ACCEPT_FLOOR][point]["final_eval"]["overlap"]


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
    return out


# --- one arm ---------------------------------------------------------------
def free_gpu(xp):
    """Release the cached device memory of a grid that is no longer needed."""
    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()


def rung_stem(k, t, rungs):
    """lm for the last rung, lm_T{T} otherwise."""
    return "lm" if k == len(rungs) - 1 else f"lm_T{t:g}"


def climb_arm(xp, use_gpu, point, arm):
    spec = C.ARMS[arm]
    dt = point_T_dt(C.signal_row(point))[1]
    seed = SEEDS[spec["seed"]](point)
    theta, start = np.array([seed[n] for n in C.FIT_PARAMS]), spec["seed"]
    for k, t in enumerate(spec["rungs"]):
        Gd = build_grid(xp, use_gpu, t, dt)
        out = run_rung(Gd, point, theta, arm, rung_stem(k, t, spec["rungs"]), start)
        del Gd
        free_gpu(xp)
        theta = np.array([out["theta_final"][n] for n in C.FIT_PARAMS])
        start = f"T{t:g}_final"
    return out


def climb_point(xp, use_gpu, point, seed):
    """The seed's full-T arm; then its ladder arm if that failed or ended below the floor."""
    full, ladder = C.SEED_ARMS[seed]
    floor = floor_overlap(point)
    try:
        ov = climb_arm(xp, use_gpu, point, full)["final_eval"]["overlap"]
    except Exception as exc:                                # noqa: BLE001  fall through to the ladder
        print(f"[FAIL {full} {C.point_label(point)}] {type(exc).__name__}: {exc}", flush=True)
        ov = -1.0
    verdict = "worked" if ov >= floor else "did not work"
    print(f"[{full} {C.point_label(point)}] O={ov:.12f} vs {C.ACCEPT_FLOOR} floor {floor:.12f}: {verdict}",
          flush=True)
    if ov < floor:
        climb_arm(xp, use_gpu, point, ladder)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=C.POINTS, required=True)
    ap.add_argument("--seed", choices=list(C.SEED_ARMS), default="physical")
    args = ap.parse_args()

    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    for point in args.points:
        try:
            climb_point(xp, use_gpu, point, args.seed)
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL {C.point_label(point)}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
