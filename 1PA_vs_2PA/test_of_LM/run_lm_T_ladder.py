"""LM climb up a ladder of observation times, each rung seeded at the previous rung's final.

The same source (2PA signal, --template 1pa/0pa) is observed for each T in --t-steps in turn:
rung 1 starts at --start (the injection, or the current SK_files best fit
LM_iterations/best_fit_{GRID}_{pa}.json), every later rung at the final point of the rung before.
    rung at T           -> {case}/lm_T{T}_idx{i}.json
    last rung, if its T is the stored observation time
                        -> {case}/lm_idx{i}.json   (the layout make_best_fit.py merges)
case = {GRID}_{pa}[_fixchi2][_distdiv{D}]_Tladder_{start}_{T1-T2-...}.

Run (GPU):  python run_lm_T_ladder.py --grid IMRI --template 0pa --idx 3 --start best --t-steps 0.75 1
            python run_lm_T_ladder.py --grid IMRI --template 0pa --idx 3 --start inj \
                                      --t-steps 0.2 0.4 0.6 0.8 1
"""
import argparse
import json

import config as C
from model import build_grid, grid_T_dt
from run_lm_from_inj import run_point
from run_lm_short_T import free_gpu


def ladder_case(fix_chi2, dist_div, start, t_steps):
    return (C.case_name(fix_chi2, dist_div) + f"_Tladder_{start}_"
            + "-".join(f"{t:g}" for t in t_steps))


def best_fit_seed(idx, names):
    """theta_final of idx in the current SK_files best fit, in the order of names."""
    path = C.LM_ITERATIONS / f"best_fit_{C.GRID}_{C.TEMPLATE_PA}.json"
    theta = json.loads(path.read_text())[str(idx)]["theta_final"]
    return [theta[n] for n in names]


def rung_stem(t, t_grid, last):
    return "lm" if last and t == t_grid else f"lm_T{t:g}"


def climb_ladder(xp, use_gpu, idx, fix_chi2, dist_div, start, t_steps):
    names = C.fit_params(fix_chi2)
    case = ladder_case(fix_chi2, dist_div, start, t_steps)
    t_grid, _ = grid_T_dt()
    seed, seed_from = (best_fit_seed(idx, names), "best_fit") if start == "best" else (None, "inj")

    outs = []
    for k, t in enumerate(t_steps):
        Gd = build_grid(xp, use_gpu, T=t)
        out = run_point(Gd, idx, fix_chi2, dist_div, False, theta0=seed, case=case,
                        stem=rung_stem(t, t_grid, k == len(t_steps) - 1), start=seed_from)
        del Gd
        free_gpu(xp)
        seed, seed_from = [out["theta_final"][n] for n in names], f"T{t:g}_final"
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=C.GRID, help="SK_files grid (IMRI, IMRI_TAIL, EMRI)")
    ap.add_argument("--idx", nargs="+", type=int, required=True)
    ap.add_argument("--template", choices=("1pa", "0pa"), default=C.TEMPLATE_PA,
                    help="analysis template (0pa never fits chi2)")
    ap.add_argument("--start", choices=("inj", "best"), required=True,
                    help="first-rung seed: the injection or the SK_files best fit")
    ap.add_argument("--t-steps", nargs="+", type=float, required=True,
                    help="observation times of the rungs [yr], in climbing order")
    ap.add_argument("--fix-chi2", action="store_true", help="hold chi2 at its injected value")
    ap.add_argument("--dist-div", type=float, default=1.0, help="SNR x dist_div")
    args = ap.parse_args()
    C.GRID, C.TEMPLATE_PA = args.grid, args.template

    xp, use_gpu = C.ovl.load_backend()
    for idx in args.idx:
        try:
            climb_ladder(xp, use_gpu, idx, args.fix_chi2, args.dist_div, args.start, args.t_steps)
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL T-ladder idx={idx}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
