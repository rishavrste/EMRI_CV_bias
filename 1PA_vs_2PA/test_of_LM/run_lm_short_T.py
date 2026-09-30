"""Two-stage LM climb from the injection with a shorter observation time as a stepping stone.

Stage 1: the same source (2PA signal, --template 1pa/0pa) observed for --t-short years (default 0.2),
         LM from the injection.                          -> {case}/lm_T{t_short}_idx{i}.json
Stage 2: the stored observation time (0.25 yr), LM seeded at the stage-1 final.
                                                         -> {case}/lm_idx{i}.json
The stage-2 file has the lm_idx{i}.json layout, so make_best_fit.py can merge it like any other
test_of_LM run. Stage 2 always runs; whether it beats the current best is judged afterwards.
case = IMRI_TAIL_{1pa,0pa}[_fixchi2][_distdiv{D}]_Tseed{t_short}.

Run (GPU):  python run_lm_short_T.py --idx 9 --fix-chi2 --dist-div 10 [--t-short 0.2] [--template 0pa]
"""
import argparse

import config as C
from model import build_grid
from run_lm_from_inj import run_point


def free_gpu(xp):
    """Release the cached device memory of a grid that is no longer needed."""
    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()


def climb_two_stage(xp, use_gpu, idx, fix_chi2, dist_div, t_short):
    case = C.case_name(fix_chi2, dist_div) + f"_Tseed{t_short:g}"
    names = C.fit_params(fix_chi2)

    Gd = build_grid(xp, use_gpu, T=t_short)
    short = run_point(Gd, idx, fix_chi2, dist_div, False, case=case, stem=f"lm_T{t_short:g}")
    del Gd
    free_gpu(xp)

    seed = [short["theta_final"][n] for n in names]
    Gd = build_grid(xp, use_gpu)
    full = run_point(Gd, idx, fix_chi2, dist_div, False, theta0=seed, case=case, stem="lm",
                     start=f"T{t_short:g}_final")
    del Gd
    free_gpu(xp)
    return short, full


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idx", nargs="+", type=int, required=True)
    ap.add_argument("--fix-chi2", action="store_true", help="hold chi2 at its injected value")
    ap.add_argument("--dist-div", type=float, default=1.0, help="SNR x dist_div")
    ap.add_argument("--t-short", type=float, default=0.2, help="stage-1 observation time [yr]")
    ap.add_argument("--template", choices=("1pa", "0pa"), default=C.TEMPLATE_PA,
                    help="analysis template (0pa never fits chi2)")
    args = ap.parse_args()
    C.TEMPLATE_PA = args.template

    xp, use_gpu = C.ovl.load_backend()
    for idx in args.idx:
        try:
            climb_two_stage(xp, use_gpu, idx, args.fix_chi2, args.dist_div, args.t_short)
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL two-stage idx={idx}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
