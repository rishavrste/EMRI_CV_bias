"""Compute and cache Fisher matrices (+ per-point injection SNR) for every
optimized best-fit case in src/IMRI/results_combined.txt and
src/EMRI/results_compiled.txt.

Requires a GPU node with `few`, `fastlisaresponse`, `lisatools`, `stableemrifisher`
installed (the same venv used by the gauss_cv_* scripts). The deviation wiring is
branch-specific (see fisher_common2.py docstring), so this must be run ONCE per
SuperKludge_r branch:

    # while the 'hybrid' branch is installed (covers 0PA, PN, simple):
    python compute_fishers.py --branch hybrid

    # while the 'dev_a_pe' branch is installed (covers simple_pe only):
    python compute_fishers.py --branch dev_a_pe

Output layout (resumable -- already-saved .npy files are skipped):
    Fisher_results/<system>/Fisher_<point>_<model>.npy   (n_params x n_params)
    Fisher_results/<system>/SNR_<point>.npy              (scalar, shared across models)

Use --only system:point:model[,system:point:model,...] to (re)compute a subset,
and --dry-run to just list what would be computed.
"""
import argparse
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from parse_results import parse_all  # noqa: E402


def out_paths(case):
    d = os.path.join(HERE, case["system"])
    os.makedirs(d, exist_ok=True)
    fisher_path = os.path.join(d, f"Fisher_{case['point']}_{case['model']}.npy")
    snr_path = os.path.join(d, f"SNR_{case['point']}.npy")
    return fisher_path, snr_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True, choices=["hybrid", "dev_a_pe"],
                     help="SuperKludge_r branch currently installed; only cases "
                          "wired for this branch are computed.")
    ap.add_argument("--system", default="all", choices=["all", "IMRI", "EMRI"])
    ap.add_argument("--only", default="", help="comma list of system:point:model to force-recompute")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU (no cupy)")
    args = ap.parse_args()

    cases = parse_all()
    cases = [c for c in cases if c["branch"] == args.branch]
    if args.system != "all":
        cases = [c for c in cases if c["system"] == args.system]

    only = set()
    if args.only:
        for tok in args.only.split(","):
            sys_, pt, mdl = tok.strip().split(":")
            only.add((sys_, pt, mdl))

    todo = []
    for c in cases:
        fisher_path, snr_path = out_paths(c)
        need_fisher = only and (c["system"], c["point"], c["model"]) in only
        need_fisher = need_fisher or (not only and not os.path.exists(fisher_path))
        need_snr = not os.path.exists(snr_path) or ((c["system"], c["point"], c["model"]) in only)
        if need_fisher or need_snr:
            todo.append((c, need_fisher, need_snr, fisher_path, snr_path))

    print(f"[compute_fishers] branch={args.branch} system={args.system}: "
          f"{len(cases)} cases total, {len(todo)} to (re)compute")
    for c, nf, ns, fp, sp in todo:
        print(f"  {c['system']:5s} {c['point']:15s} {c['model']:10s} "
              f"fisher={'yes' if nf else 'no'} snr={'yes' if ns else 'no'}")
    if args.dry_run:
        return

    import fisher_common2 as fc
    use_gpu = fc.USE_GPU and not args.cpu

    failures = []
    for c, need_fisher, need_snr, fisher_path, snr_path in todo:
        tag = f"{c['system']}/{c['point']}/{c['model']}"
        try:
            if need_snr:
                print(f"[{tag}] computing injection SNR ...", flush=True)
                snr = fc.compute_injection_snr(c, use_gpu=use_gpu)
                np.save(snr_path, np.array(snr))
                print(f"[{tag}] SNR = {snr:.4f} -> {snr_path}", flush=True)
            if need_fisher:
                print(f"[{tag}] computing Fisher matrix ({len(c['param_names'])} params) ...",
                      flush=True)
                fisher = fc.compute_fisher(c, use_gpu=use_gpu)
                np.save(fisher_path, fisher)
                print(f"[{tag}] saved Fisher {fisher.shape} -> {fisher_path}", flush=True)
        except Exception:
            print(f"[{tag}] FAILED:\n{traceback.format_exc()}", flush=True)
            failures.append(tag)

    print(f"\n[compute_fishers] done. {len(failures)} failures: {failures}")


if __name__ == "__main__":
    main()
