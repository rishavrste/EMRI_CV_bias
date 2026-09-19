"""Freeze one source of a population run into hard_point_<idx>.json.

The runner does not need this file -- cv_de_cv.point_from_run rebuilds the same structure
straight from the results json -- but freezing it pins the numbers a run was launched
against, so a later re-run of the population cannot silently change what "source 13" means.
One file per source, named after it, because two hard points in the same directory must
never be able to stand in for each other.
"""

import argparse
import json
import os
import sys

import config as C
from config import Config

sys.path.insert(0, C.HERE)
from cv_de_cv import point_from_run     # noqa: E402  (path must be set first)


def summarise(point):
    """The one line that says which source landed in the file."""
    s = point["source"]
    return (f"idx {s['idx']}: m1 = {s['m1']:.4e}, p0 = {s['p0']:.4f}, T = {s['T']:.3f} yr, "
            f"A_PM = {s['A_PM']:.4e}, SNR = {s['snr']:.2f}, "
            f"population ov = {s['fit_0pa']['ov_final']:.6f} (0PA) / "
            f"{s['fit_pn']['ov_final']:.6f} (0PA+PN)")


def freeze(idx, run, overwrite):
    """Write the frozen point, refusing to clobber an existing one unless told to."""
    cfg = Config(point_idx=idx, point_run=run)
    if os.path.exists(cfg.point_json) and not overwrite:
        raise SystemExit(f"[freeze] {os.path.basename(cfg.point_json)} exists; "
                         "pass --overwrite to replace it")
    point = point_from_run(cfg)
    point["frozen_note"] = f"frozen from {cfg.point_results} by freeze_point.py"
    with open(cfg.point_json, "w") as f:
        json.dump(point, f, indent=1)
    print(f"[freeze] {summarise(point)}")
    print(f"[freeze] -> {cfg.point_json}")
    return cfg.point_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point-idx", type=int, required=True)
    ap.add_argument("--point-run", default="", help="'' is the live population run")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    freeze(args.point_idx, args.point_run, args.overwrite)


if __name__ == "__main__":
    main()
