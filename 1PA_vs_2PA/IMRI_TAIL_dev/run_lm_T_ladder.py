"""LM climb up a ladder of observation times: IMRI_TAIL, 2PA signal, 1PA + deviation (C_p, C_e)
template.

Rung 1 starts at --seed (config.SEEDS: the injection with C_p = C_e = 0, the 1PA-only best fit, or
the final point of an earlier run); every later rung starts at the final point of the
rung before. The same source is observed for each T in --t-steps in turn. With --then-free (needs
--fix-chi2) each point is climbed once more with chi2 free at the last T, from the fixed-chi2
result, written under seed "dev{seed}".
    rung at T                            -> results/{case}/lm_T{T}_idx{i}.json
    last rung, at the stored T (0.25 yr) -> results/{case}/lm_idx{i}.json
case = config.case_name(...). Each JSON is rewritten after every accepted step, so a walltime kill
still leaves the latest point.

Run (GPU):  python run_lm_T_ladder.py --idx 9 --fix-chi2 --dist-div 10 --t-steps 0.2 0.25
            python run_lm_T_ladder.py --idx 0 1 --dist-div 10 --t-steps 0.25 --seed devinj
            python run_lm_T_ladder.py --idx 15 --fix-chi2 --dist-div 10 --t-steps 0.25 \
                   --seed 1pabest --then-free
            python run_lm_T_ladder.py --idx 20 --fix-chi2 --chi2-at seed --dist-div 10 \
                   --t-steps 0.242 0.25 --seed 1pabest      # chi2 held at the 1PA best-fit value
            python run_lm_T_ladder.py --idx 21 --fix-chi2 --chi2-at seed --dist-div 10 \
                   --t-steps 0.242 0.25 --seed 1pabest --dev-start -8 0   # start at C_p = -8
            python run_lm_T_ladder.py --idx 20 --dist-div 10 --t-steps 0.25 --seed devchi2seed
                   # chi2 free, from the end of the chi2-held-at-1PA ladder
            python run_lm_T_ladder.py --idx 2 7 --dist-div 10 --t-steps 0.25 --seed best
                   # re-climb from the chosen best fit (best_fit_dev.py)
            python run_lm_T_ladder.py --template 1pa --idx 2 --dist-div 10 --t-steps 0.25 \
                   --seed 1pabest     # plain 1PA template, from the SK_files 1PA best fit
            python run_lm_T_ladder.py --template 1pa --idx 0 1 --dist-div 10 --t-steps 0.25 \
                   --seed 1pabest --tag final      # the final fits (config.free_case)
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

import config as C
from lm import lm_climb
from model import build_grid, build_point, evaluate, grid_T_dt


# --- output ----------------------------------------------------------------
def save(out):
    d = C.OUT_ROOT / out["case"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{out['stem']}_idx{out['idx']}.json"
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


# --- seeds ----------------------------------------------------------------
def seed_source(idx, seed, template=C.TEMPLATE):
    """{name: value} at the seed for grid point idx and template; empty for the injection."""
    if seed == "inj":
        return {}
    if seed == "1pabest":
        with open(C.BEST_1PA) as f:
            return json.load(f)[str(idx)]["theta_final"]
    if seed == C.SEED_BEST:
        return json.loads(C.BEST_DEV.read_text())[str(idx)]["theta_final"]
    if seed == C.SEED_ROWMEAN:
        return rowmean_seed(idx, template)
    if seed == C.SEED_DEVFINAL:
        return json.loads(C.BEST_FREE["1pa_dev"].read_text())[str(idx)]["theta_final"]
    with open(C.OUT_ROOT / C.SEED_CASES[seed] / f"lm_idx{idx}.json") as f:
        out = json.load(f)
    if out.get("running"):
        raise RuntimeError(f"seed {seed} idx={idx} has not finished")
    if out.get("chi2_fixed") is None:
        return out["theta_final"]
    return {**out["theta_final"], "chi2": out["chi2_fixed"]}


def rowmean_seed(idx, template=C.TEMPLATE):
    """Injection of idx plus the mean offset of its e0-neighbours idx -/+ 5 that are on the grid
    (final fits of template)."""
    best = json.loads(C.BEST_FREE[template].read_text())
    nbrs = [best[str(j)] for j in (idx - 5, idx + 5) if str(j) in best]
    names = C.fit_params(False, template)
    inj = {n: C.injected_value(C.ovl.signal_array(C.GRID)[idx], n) for n in names}
    return {n: inj[n] + sum(b["theta_final"][n] - b["theta_inj"][n] for b in nbrs) / len(nbrs)
            for n in names}


def seed_theta(sig_row, idx, names, seed, dev_start=None, template=C.TEMPLATE):
    """Seed vector: the source's values, the injected value for anything it lacks. dev_start
    (C_p, C_e) overrides the deviation coefficients."""
    src = seed_source(idx, seed, template)
    if dev_start is not None:
        src = {**src, **dict(zip(C.DEV_PARAMS, dev_start))}
    return np.array([src.get(n, C.injected_value(sig_row, n)) for n in names])


# --- one rung --------------------------------------------------------------
def run_rung(Gd, idx, names, theta0, dist_div, case, stem, start, chi2_fixed=None,
             template=C.TEMPLATE):
    """LM climb at one grid point and one observation time, from theta0. Without chi2 in names
    the template chi2 is chi2_fixed (None: the injected value)."""
    sig_row = C.ovl.signal_array(C.GRID)[idx]
    P = build_point(Gd, sig_row, names, dist_div, chi2_fixed)
    theta_inj = np.array([C.injected_value(sig_row, n) for n in names])
    tag = f"{case} {stem} T={Gd['T']} {start} idx={idx}"
    t0 = time.time()

    first = evaluate(P, theta0)
    print(f"[START {tag}] O={first['overlap']:.12f} chi2={first['chi2']:.6e} "
          f"rho_s={first['rho_s']:.6f}", flush=True)
    out = dict(grid=C.GRID, signal="2PA", template=template,
               deviation=[n for n in C.DEV_PARAMS if n in names],
               case=case, stem=stem, start=start, idx=idx,
               a_inj=float(sig_row[C.COL["a"]]), e0_inj=float(sig_row[C.COL["e0"]]),
               params=names, chi2_spin_inj=float(sig_row[C.COL["chi2"]]), chi2_fixed=chi2_fixed,
               dist_div=dist_div,
               dist=float(sig_row[C.COL["dist"]]) / dist_div, gen=str(C.GEN), drop_dc=C.DROP_DC,
               setup=C.SETUP, dt=Gd["dt"], T=Gd["T"], theta_inj=as_dict(names, theta_inj),
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


# --- the ladder ------------------------------------------------------------
def free_gpu(xp):
    """Release the cached device memory of a grid that is no longer needed."""
    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()


def rung_stem(t, t_grid, last):
    return "lm" if last and t == t_grid else f"lm_T{t:g}"


def fixed_chi2(idx, fix_chi2, seed_key, chi2_at):
    """Template chi2 when it is not fitted: the seed's value with chi2_at="seed", else None
    (the injected value)."""
    if not fix_chi2 or chi2_at == "inj":
        return None
    return float(seed_source(idx, seed_key)["chi2"])


def seed_label(seed_key, dev_start):
    """seed key, plus the starting deviation when it is not the source's: 1pabest_Cp-8_Ce0"""
    if dev_start is None:
        return seed_key
    return seed_key + "".join(f"_{n.replace('_', '')}{v:g}" for n, v in zip(C.DEV_PARAMS, dev_start))


def climb_ladder(xp, use_gpu, idx, fix_chi2, dist_div, t_steps, seed_key, chi2_at="inj",
                 dev_start=None, template=C.TEMPLATE, tag=""):
    names = C.fit_params(fix_chi2, template)
    case = C.case_name(fix_chi2, dist_div, t_steps, seed_label(seed_key, dev_start), chi2_at,
                       template, tag)
    chi2_fixed = fixed_chi2(idx, fix_chi2, seed_key, chi2_at)
    t_grid, _ = grid_T_dt()
    sig_row = C.ovl.signal_array(C.GRID)[idx]
    seed = seed_theta(sig_row, idx, names, seed_key, dev_start, template)
    seed_from = seed_label(seed_key, dev_start)

    for k, t in enumerate(t_steps):
        Gd = build_grid(xp, use_gpu, T=t)
        out = run_rung(Gd, idx, names, seed, dist_div, case,
                       rung_stem(t, t_grid, k == len(t_steps) - 1), seed_from, chi2_fixed,
                       template)
        del Gd
        free_gpu(xp)
        seed, seed_from = np.array([out["theta_final"][n] for n in names]), f"T{t:g}_final"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idx", nargs="+", type=int, required=True)
    ap.add_argument("--t-steps", nargs="+", type=float, required=True,
                    help="observation times of the rungs [yr], in climbing order")
    ap.add_argument("--fix-chi2", action="store_true", help="hold chi2 fixed (see --chi2-at)")
    ap.add_argument("--chi2-at", choices=["inj", "seed"], default="inj",
                    help="with --fix-chi2: hold chi2 at the injected value or at the seed's value")
    ap.add_argument("--dev-start", nargs=2, type=float, metavar=("C_P", "C_E"),
                    help="start the first rung at these deviation coefficients instead of the seed's")
    ap.add_argument("--dist-div", type=float, default=1.0, help="SNR x dist_div")
    ap.add_argument("--seed", choices=C.SEEDS, default="inj", help="start of the first rung")
    ap.add_argument("--template", choices=C.TEMPLATES, default=C.TEMPLATE,
                    help="1pa_dev (C_p, C_e free) or 1pa (plain 1PA, C_p = C_e = 0)")
    ap.add_argument("--tag", default="", help="suffix of the case name (final: the final fits)")
    ap.add_argument("--then-free", action="store_true",
                    help="after the fixed-chi2 ladder, climb again with chi2 free at the last T")
    args = ap.parse_args()
    if args.chi2_at == "seed" and not (args.fix_chi2 and args.seed != "inj"):
        ap.error("--chi2-at seed needs --fix-chi2 and a --seed other than inj")
    if args.template == "1pa" and (args.dev_start is not None or args.then_free):
        ap.error("--dev-start and --then-free are for the 1pa_dev template")
    if args.then_free and not (args.fix_chi2 and f"dev{args.seed}" in C.SEED_CASES):
        ap.error(f"--then-free needs --fix-chi2 and config.SEED_CASES['dev{args.seed}']")

    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    for idx in args.idx:
        try:
            climb_ladder(xp, use_gpu, idx, args.fix_chi2, args.dist_div, args.t_steps, args.seed,
                         args.chi2_at, args.dev_start, args.template, args.tag)
            if args.then_free:
                climb_ladder(xp, use_gpu, idx, False, args.dist_div, args.t_steps[-1:],
                             f"dev{args.seed}")
        except Exception as exc:                            # noqa: BLE001  keep the batch going
            print(f"[FAIL T-ladder idx={idx}] {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
