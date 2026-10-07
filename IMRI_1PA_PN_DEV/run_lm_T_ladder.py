"""LM climb up a ladder of observation times: IMRI grid, 1PA signal, 0PA and 0PA + PN deviation
(C_p, C_e) templates, 2nd-generation TDI, A and E.

Rung 1 starts at config.SEEDS[--seed] of the same template: "old", the 1st-generation best fit
(src/IMRI/results_combined.txt), or "sk", the SK_files 0PA LM best fit with qS, phiS at the injection
and C_p = C_e = 0 (make_seeds.py). Every later rung starts at the final point of the rung before.
The same source is observed for T = f x 1 yr for each f in --t-fracs in turn (config.T_FRACS).
Each point runs every template in --templates, in order.
    rung at T                     -> results/{case}/lm_T{T}_{label}.json
    last rung, at the stored T    -> results/{case}/lm_{label}.json      label = idx4
case = config.case_name(...). Each JSON is rewritten after every accepted step, so a walltime kill
still leaves the latest point.

Run (GPU):  python run_lm_T_ladder.py --points 0 1 2 3 4 --seed old
            python run_lm_T_ladder.py --points 4 --seed sk --templates pn --t-fracs 0.9 0.95 1
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
    d = C.OUT_ROOT / out["case"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{out['stem']}_{C.point_label(out['point'])}.json"
    with open(f"{path}.tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(f"{path}.tmp", path)


def as_dict(names, vec):
    return dict(zip(names, map(float, vec)))


def check_branch():
    """Stop if SuperKludge_r is not on the branch that carries the C_p / C_e slots."""
    branch = subprocess.run(["git", "-C", str(C.SUPERKLUDGE), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print(f"SuperKludge_r branch: {branch}", flush=True)
    if branch != C.SUPERKLUDGE_BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {C.SUPERKLUDGE_BRANCH!r}")


def seed_theta(point, names, template, seed="old"):
    """Rung-1 start: config.SEEDS[seed][template][point] (make_seeds.py)."""
    src = json.loads(C.SEEDS[seed].read_text())[template][point]["theta"]
    return np.array([src[n] for n in names])


# --- one rung --------------------------------------------------------------
def run_rung(Gd, point, names, theta0, case, stem, start):
    """LM climb at one point and one observation time, from theta0."""
    sig_row = C.signal_row(point)
    P = build_point(Gd, sig_row, names)
    theta_inj = np.array([C.injected_value(sig_row, n) for n in names])
    tag = f"{case} {stem} T={Gd['T']} {start} {C.point_label(point)}"
    t0 = time.time()

    first = evaluate(P, theta0)
    print(f"[START {tag}] O={first['overlap']:.12f} chi2={first['chi2']:.6e} "
          f"rho_s={first['rho_s']:.6f}", flush=True)
    out = dict(grid=C.GRID, signal="1PA", template=case.split("_")[1], case=case, stem=stem,
               start=start, point=point,
               a_inj=float(sig_row[C.COL["a"]]), e0_inj=float(sig_row[C.COL["e0"]]),
               params=names, chi2_spin=float(sig_row[C.COL["chi2"]]),
               dist=float(sig_row[C.COL["dist"]]), gen=str(C.GEN), drop_dc=C.DROP_DC,
               tdi="2nd generation, AE", setup=C.SETUP, dt=Gd["dt"], T=Gd["T"],
               theta_inj=as_dict(names, theta_inj), theta_start=as_dict(names, theta0),
               start_eval=first)

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


# --- the ladder ------------------------------------------------------------
def free_gpu(xp):
    """Release the cached device memory of a grid that is no longer needed."""
    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()


def rung_stem(k, f, t, t_fracs):
    """lm for the last rung at the stored T, lm_T{T} otherwise."""
    return "lm" if f == 1 and k == len(t_fracs) - 1 else f"lm_T{t:g}"


def climb_ladder(xp, use_gpu, point, template, t_fracs, seed_src):
    names = C.fit_params(template)
    case = C.case_name(template, t_fracs, seed_src)
    t_point, dt = point_T_dt(C.signal_row(point))
    seed, seed_from = seed_theta(point, names, template, seed_src), seed_src
    for k, f in enumerate(t_fracs):
        t = f * t_point
        Gd = build_grid(xp, use_gpu, t, dt)
        out = run_rung(Gd, point, names, seed, case, rung_stem(k, f, t, t_fracs), seed_from)
        del Gd
        free_gpu(xp)
        seed, seed_from = np.array([out["theta_final"][n] for n in names]), f"T{t:g}_final"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=C.POINTS, required=True, help="grid cells 0..24")
    ap.add_argument("--t-fracs", nargs="+", type=float, default=list(C.T_FRACS),
                    help="observation times of the rungs as fractions of each point's T, in order")
    ap.add_argument("--templates", nargs="+", choices=C.TEMPLATES, default=list(C.TEMPLATES))
    ap.add_argument("--seed", choices=list(C.SEEDS), default="old",
                    help="rung-1 start: old (1st-generation best fit) or sk (SK_files 0PA best fit)")
    args = ap.parse_args()

    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    for point in args.points:
        for template in args.templates:
            try:
                climb_ladder(xp, use_gpu, point, template, args.t_fracs, args.seed)
            except Exception as exc:                        # noqa: BLE001  keep the batch going
                print(f"[FAIL T-ladder {template} {C.point_label(point)}] "
                      f"{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
