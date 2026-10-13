"""Redo the last PN ladder step of run_2 idx22 with finer rungs (user, 2026-10-10).

In ladder_e (job 654144) idx22's PN truth ladder reached ov 0.9999992509 at T = 0.95 (C_p 155)
and broke on the single step to T = 1 (start 0.748, end 0.99947774, lambda_exhausted); the
other seed, the 0PA fit + C_p = 0, stalled at C_p = 0 after 2 iterations.

Here: PN only, e0 fixed at 0, LM only, from the T = 0.95 rung of the PN truth ladder, up the
rungs T_FRACS_FINE of lm_e0fixed_redo_0pa.  0PA is not re-run.

Output: results_lm_e0fixed_redopn.json, rewritten after every rung.

Run:  python lm_e0fixed_redo_pn.py
"""

import json
import os
import time
from datetime import datetime, timezone

import lm_e0fixed as lme
import lm_e0fixed_redo_0pa as r0


# --- controls --------------------------------------------------------------
REDO = {22: "results_lm_e0fixed_ladder_e.json"}      # idx -> the run holding its PN truth ladder
JSON_PATH = os.path.join(lme.HERE, "results_lm_e0fixed_redopn.json")


# --- setup -----------------------------------------------------------------
def seed_rung(idx):
    """The T = SEED_FRAC rung of idx's PN truth ladder in its earlier run."""
    with open(os.path.join(lme.HERE, REDO[idx])) as f:
        rec = next(s for s in json.load(f)["sources"] if s["idx"] == idx)
    rungs = rec["pn"]["by_seed"]["truth_ladder"]["ladder"]
    return next(r for r in rungs if r["t_frac"] == r0.SEED_FRAC)


# --- driver ----------------------------------------------------------------
def save_json(records):
    payload = dict(
        study="environmental_CV_population_run_2_e0fixed_redo_pn",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        params_pn=lme.PARAMS_PN, e0_fixed=0.0, lm_max_iters=lme.LM_MAX_ITERS,
        seed_frac=r0.SEED_FRAC, t_fracs=list(r0.T_FRACS_FINE),
        seeded_from={str(k): v for k, v in REDO.items()},
        note=("PN only. Ladder continued from the earlier run's T=0.95 PN truth-ladder rung in "
              "steps of 0.01 to T=1. sigma_lm_last_iter is fishinv of the last LM iteration's Fisher."),
        sources=records)
    with open(f"{JSON_PATH}.tmp", "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(f"{JSON_PATH}.tmp", JSON_PATH)


def run_source(src, records):
    seed = seed_rung(src["idx"])
    print("=" * 78, flush=True)
    print(f"[SRC {src['idx']:2d}] p0 = {src['p0']:.6f}  T = {src['T']:.4f} yr  -> PN from the "
          f"T = {r0.SEED_FRAC:g} truth-ladder fit (ov {seed['ov_final']:.12f}, "
          f"C_p {seed['theta_final'][-1]:.4f})", flush=True)
    t0 = time.time()
    rec = dict(idx=src["idx"], m1=src["m1"], p0=src["p0"], T=src["T"], dt=src["dt"],
               A_PM=src["A_PM"], n_PM=src["n_PM"], dist=src["dist"],
               source_params=src["source_params"], seed_rung=seed)
    records.append(rec)

    def save(rungs):
        rec.update({"pn": dict(rungs[-1], ladder=rungs)}, seconds=time.time() - t0)
        save_json(records)

    r0.fine_ladder(src, seed, save, params=lme.PARAMS_PN, label="pn")


def main():
    lme.check_branch()
    with open(lme.POP) as f:
        sources = [s for s in json.load(f)["sources"] if s["idx"] in REDO]
    print(f"[info] redo PN for {[s['idx'] for s in sources]}, rungs {r0.T_FRACS_FINE}, "
          f"LM_MAX_ITERS = {lme.LM_MAX_ITERS}\n", flush=True)
    records, t_all = [], time.time()
    for src in sources:
        try:
            run_source(src, records)
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            print(f"[FAIL SRC {src['idx']}] {type(exc).__name__}: {exc}", flush=True)
    save_json(records)
    for r in records:
        if "pn" in r:
            print(f"{r['idx']:>3}  ov PN {r['pn']['ov_final']:.10f}  chi2 {r['pn']['chi2']:.4e}  "
                  f"C_p {r['pn']['theta_final'][-1]:.4f}  stop {r['pn']['stop_reason']}")
    print(f"\n[saved] {JSON_PATH}  ({time.time() - t_all:.1f} s total)")


if __name__ == "__main__":
    main()
