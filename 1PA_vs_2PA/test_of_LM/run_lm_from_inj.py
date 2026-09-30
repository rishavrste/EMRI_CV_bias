"""LM climb from the injected point: IMRI_TAIL, 2PA signal, 1PA template, SNR 20 (stored distance).

The 8 free parameters (m1, m2, a, p0, e0, Phi_phi0, Phi_r0, chi2) start at the injection
(chi2 = 0.95); --fix-chi2 holds the secondary spin at 0.95 and --fix-phases holds Phi_phi0 and
Phi_r0 at their injected values (5 free parameters with both). Distance,
sky angles and Phi_theta0 stay at the injected values; --dist-div D divides the distance of
signal and template by D (SNR 20 D; chi2 = <s-h|s-h> scales by D^2, the overlap does not).
The evaluation stack is SK_files `overlap.py`'s `doc` setup, so overlaps compare directly with
SK_files.

Output: results/IMRI_TAIL_1pa[_fixchi2][_fixphases][_distdiv{D}]/lm_idx{i}.json, rewritten after every
accepted step so a walltime kill still leaves the latest point.

--regularise steps through the regularised inverse of lm.py (results folder suffix _reg).

Run (GPU):  python run_lm_from_inj.py --idx 6 9 [--fix-chi2] [--fix-phases] [--dist-div 1000]
                                      [--regularise]
"""
import argparse
import json
import os
import time

import numpy as np

import config as C
from lm import lm_climb
from model import build_grid, build_point, evaluate


def case_dir(out):
    d = C.OUT_ROOT / out.get("case", C.case_name(out["fix_chi2"], out["dist_div"], out["fix_phases"]))
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(out):
    path = case_dir(out) / f"{out.get('stem', 'lm')}_idx{out['idx']}.json"
    with open(f"{path}.tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(f"{path}.tmp", path)


def as_dict(names, vec):
    return dict(zip(names, map(float, vec)))


def point_record(idx, sig_row, names, theta0, first, Gd, fix_chi2, dist_div, fix_phases):
    return dict(grid=C.GRID, signal="2PA", template=C.TEMPLATE_PA.upper(), start="inj", idx=idx,
                a_inj=float(sig_row[C.COL["a"]]), e0_inj=float(sig_row[C.COL["e0"]]),
                params=names, fix_chi2=fix_chi2, chi2_spin_inj=float(sig_row[C.COL["chi2"]]),
                fix_phases=fix_phases, regularise=C.REGULARISE, dist_div=dist_div, dist=float(sig_row[C.COL["dist"]]) / dist_div,
                gen=str(C.GEN), dt=Gd["dt"], T=Gd["T"],
                theta_inj=as_dict(names, theta0), theta_start=as_dict(names, theta0),
                start_eval=first)


def injected_theta(sig_row, names):
    return np.array([float(sig_row[C.COL[n]]) for n in names])


def run_point(Gd, idx, fix_chi2, dist_div, fix_phases, theta0=None, case=None, stem="lm",
              start="inj"):
    """LM climb at one grid point, from the injection unless theta0 is given. case / stem
    override the results folder and file prefix ({case}/{stem}_idx{i}.json)."""
    sig_row = C.ovl.signal_array(C.GRID)[idx]
    names = C.fit_params(fix_chi2, fix_phases)
    P = build_point(Gd, sig_row, names, dist_div)
    theta_inj = injected_theta(sig_row, names)
    theta0 = theta_inj if theta0 is None else np.asarray(theta0, dtype=float)
    case = case or C.case_name(fix_chi2, dist_div, fix_phases)
    tag = f"{case} {stem} T={Gd['T']} {start} idx={idx}"
    t0 = time.time()

    first = evaluate(P, theta0)
    print(f"[START {tag}] O={first['overlap']:.12f} chi2={first['chi2']:.6e} "
          f"rho_s={first['rho_s']:.6f}", flush=True)
    out = point_record(idx, sig_row, names, theta_inj, first, Gd, fix_chi2, dist_div, fix_phases)
    out.update(case=case, stem=stem, start=start, theta_start=as_dict(names, theta0))

    def checkpoint(cur, sigma, history):
        save(dict(out, theta_current=as_dict(names, cur), sigma_current=as_dict(names, sigma),
                  history=history, running=True, seconds=time.time() - t0))

    theta, sigma, info, history = lm_climb(P, theta0, tag, on_iter=checkpoint)
    final = evaluate(P, theta)
    out.update(theta_final=as_dict(names, theta), final_eval=final,
               sigma_final=as_dict(names, sigma),
               info=info, history=history, running=False, seconds=time.time() - t0)
    save(out)
    print(f"[DONE {tag}] O {first['overlap']:.12f} -> {final['overlap']:.12f}  "
          f"chi2 {first['chi2']:.6e} -> {final['chi2']:.6e}  iters={info['n_iter']} "
          f"accepted={info['n_accept']} stop={info['stop_reason']}  {out['seconds']:.1f} s",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idx", nargs="+", type=int, default=list(C.IDX_DEFAULT))
    ap.add_argument("--fix-chi2", action="store_true",
                    help="hold the secondary spin at its injected value (7 free parameters)")
    ap.add_argument("--fix-phases", action="store_true",
                    help="hold Phi_phi0 and Phi_r0 at their injected values")
    ap.add_argument("--dist-div", type=float, default=1.0,
                    help="divide signal and template distance by this (SNR x dist_div)")
    ap.add_argument("--regularise", action="store_true",
                    help="step with Sigma_reg = A^-1 + eps diag(A^-1), eps = ||A A^-1 - 1||_F")
    ap.add_argument("--max-iters", type=int, default=None, help=f"default: {C.LM_MAX_ITERS}")
    args = ap.parse_args()
    if args.max_iters is not None:
        C.LM_MAX_ITERS = args.max_iters
    C.REGULARISE = args.regularise

    xp, use_gpu = C.ovl.load_backend()
    Gd = build_grid(xp, use_gpu)
    for idx in args.idx:
        try:
            run_point(Gd, idx, args.fix_chi2, args.dist_div, args.fix_phases)
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            case = C.case_name(args.fix_chi2, args.dist_div, args.fix_phases)
            print(f"[FAIL {case} inj idx={idx}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
