"""Redo the last 0PA ladder step of run_2 idx11 and idx19 with finer rungs (user, 2026-10-09).

In lm_e0fixed.py both 0PA ladders were fine up to T = 0.95 and broke on the single step to
T = 1: idx11 started the T = 1 rung at ov 0.428 and finished at 0.842 (Phi_phi0 pinned at 0),
idx19 finished at 0.99999726 while every earlier rung ended near 1 - 1e-8.

Here: 0PA only, e0 fixed at 0, LM only, from each source's T = 0.95 0PA fit, up the rungs
T_FRACS_FINE.  PN is not re-run.

idx2 (job 654143) failed the same way and was added afterwards (user, 2026-10-10).

Output: results_lm_e0fixed_redo0pa{_TAG}.json, rewritten after every rung.

Run:  python lm_e0fixed_redo_0pa.py                          (idx 11, 19 -- job 656173)
      REDO_IDXS=2 REDO_TAG=idx2 python lm_e0fixed_redo_0pa.py
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import cv_population as cvp
import lm_e0fixed as lme


# --- controls --------------------------------------------------------------
REDO = {2: "results_lm_e0fixed_ladder_d.json",       # idx -> the run holding its 0PA ladder
        11: "results_lm_e0fixed_ladder_b.json",
        19: "results_lm_e0fixed_ladder_a.json"}
SEED_FRAC = 0.95
T_FRACS_FINE = (0.96, 0.97, 0.98, 0.99, 1.0)

# REDO_IDXS narrows REDO (e.g. "2"); REDO_TAG keeps a later batch from overwriting this one's JSON.
_SEL = os.environ.get("REDO_IDXS", "").strip()
WANT = {int(x) for x in _SEL.replace(",", " ").split()} if _SEL else set(REDO)
_TAG = os.environ.get("REDO_TAG", "").strip()
JSON_PATH = os.path.join(lme.HERE, f"results_lm_e0fixed_redo0pa{('_' + _TAG) if _TAG else ''}.json")


# --- setup -----------------------------------------------------------------
def seed_rung(idx):
    """The T = SEED_FRAC rung of idx's 0PA ladder in its earlier run."""
    with open(os.path.join(lme.HERE, REDO[idx])) as f:
        rec = next(s for s in json.load(f)["sources"] if s["idx"] == idx)
    return next(r for r in rec["0pa"]["ladder"] if r["t_frac"] == SEED_FRAC)


def fine_ladder(src, seed, save_rungs, params=lme.PARAMS_0PA, label="0pa"):
    """lme.ladder, but starting from `seed` (a T = SEED_FRAC rung) instead of the truth."""
    theta, start, rungs = np.array(seed["theta_final"]), f"T{SEED_FRAC:g}_final (earlier run)", []
    for f in T_FRACS_FINE:
        T = f * src["T"]
        print(f"    --- {label} ({len(params)} params) rung T = {f:g} x {src['T']:.4f} "
              f"= {T:.4f} yr, from {start} ---", flush=True)
        S = cvp.build_source(lme.with_T(src, T))
        rung = lme.climb(S, src, params, theta, f"{src['idx']}/{label}/T{f:g}", start)
        rungs.append(dict(rung, t_frac=f))
        del S
        lme.free_gpu()
        theta, start = np.array(rung["theta_final"]), f"T{f:g}_final"
        save_rungs(rungs)
    return rungs


# --- driver ----------------------------------------------------------------
def save_json(records):
    payload = dict(
        study="environmental_CV_population_run_2_e0fixed_redo_0pa",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        params_0pa=lme.PARAMS_0PA, e0_fixed=0.0, lm_max_iters=lme.LM_MAX_ITERS,
        seed_frac=SEED_FRAC, t_fracs=list(T_FRACS_FINE),
        seeded_from={str(k): v for k, v in REDO.items()},
        note=("0PA only. Ladder continued from the earlier run's T=0.95 0PA rung in steps of "
              "0.01 to T=1. sigma_lm_last_iter is fishinv of the last LM iteration's Fisher."),
        sources=records)
    with open(f"{JSON_PATH}.tmp", "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(f"{JSON_PATH}.tmp", JSON_PATH)


def run_source(src, records):
    seed = seed_rung(src["idx"])
    print("=" * 78, flush=True)
    print(f"[SRC {src['idx']:2d}] p0 = {src['p0']:.6f}  T = {src['T']:.4f} yr  -> 0PA from the "
          f"T = {SEED_FRAC:g} fit (ov {seed['ov_final']:.12f})", flush=True)
    t0 = time.time()
    rec = dict(idx=src["idx"], m1=src["m1"], p0=src["p0"], T=src["T"], dt=src["dt"],
               A_PM=src["A_PM"], n_PM=src["n_PM"], dist=src["dist"],
               source_params=src["source_params"], seed_rung=seed)
    records.append(rec)

    def save(rungs):
        rec.update({"0pa": dict(rungs[-1], ladder=rungs)}, seconds=time.time() - t0)
        save_json(records)

    fine_ladder(src, seed, save)


def main():
    lme.check_branch()
    with open(lme.POP) as f:
        sources = [s for s in json.load(f)["sources"] if s["idx"] in WANT]
    print(f"[info] redo 0PA for {[s['idx'] for s in sources]}, rungs {T_FRACS_FINE}, "
          f"LM_MAX_ITERS = {lme.LM_MAX_ITERS}\n", flush=True)
    records, t_all = [], time.time()
    for src in sources:
        try:
            run_source(src, records)
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            print(f"[FAIL SRC {src['idx']}] {type(exc).__name__}: {exc}", flush=True)
    save_json(records)
    for r in records:
        if "0pa" in r:
            print(f"{r['idx']:>3}  ov 0PA {r['0pa']['ov_final']:.10f}  chi2 {r['0pa']['chi2']:.4e}  "
                  f"stop {r['0pa']['stop_reason']}")
    print(f"\n[saved] {JSON_PATH}  ({time.time() - t_all:.1f} s total)")


if __name__ == "__main__":
    main()
