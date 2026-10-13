"""run_1 population, 2nd-generation TDI, LM only -- re-optimised with e0 FIXED at 0.

Same sources, injection, TDI, channels (A, E), dt and SNR-200 distance as lm_second_gen.py.
What changes (user, 2026-10-08):

* e0 is no longer a fit parameter.  The population is circular, and with e0 free the fit
  parked it on its e0 >= 0 boundary at ~5e-5 to 1e-4, with a meaningless b/sigma there and,
  most likely, a narrower valley for LM to crawl along.  Templates now take e0 = 0 from the
  source.  0PA = 7 params, 0PA + PN = 8 params.
* LM_MAX_ITERS = 300 (was 150): the unconverged climbs were still rising at 150.
* No Nelder-Mead (user: do not add it).

Two kinds of source, as before:

converged in run_1 (19 sources)
    one full-T LM per model, from the e0-free 2nd-gen fit of that model
    (results_lm_second_gen_converged.json) with e0 dropped.

not converged in run_1 (idx 8, 13, 14, 16)
    0PA: a finer T-ladder, rung 1 from the injected truth, every later rung from the rung
         before.  The old ladder (0.5, 0.75, 0.9, 0.95, 1) left the start of each rung far
         below its optimum (idx13: 0.78 at T=0.9, 0.84 at T=1).
    PN:  two seeds, the higher final overlap kept (user, 2026-10-08) -- its own T-ladder
         from the truth with C_p = 0, and a full-T climb from the 0PA T=1 fit with C_p = 0.
         The second guarantees PN >= 0PA (the e0-free ladder alone ended below 0PA at 8, 13, 14).

The C_p finite-difference step is calibrated at the truth per source, as in run_1.

Output: results_lm_e0fixed_{TAG}.json here, rewritten after every climb.

Run:  ENV_POP_IDXS=0,2,...  ENV_POP_TAG=converged  python lm_e0fixed.py
      ENV_POP_IDXS=8,14     ENV_POP_TAG=ladder_a   python lm_e0fixed.py
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import cv_population as cvp
from lm_second_gen import (CONVERGED_OV, POP, RUN1_RESULTS, TARGET_SNR, check_branch,
                           free_gpu, load_run1, rescale_to_snr, with_changes)


# --- controls --------------------------------------------------------------
LM_MAX_ITERS = 300
cvp.LM_MAX_ITERS = LM_MAX_ITERS             # lm_climb reads the module constant

PARAMS_0PA = [p for p in cvp.PARAMS_0PA if p != "e0"]
PARAMS_PN = PARAMS_0PA + ["C_p"]

# 0PA T-ladder (fractions of the source's T), user-approved 2026-10-08
T_FRACS = tuple(np.round(np.arange(0.5, 1.0001, 0.05), 2))           # 0.5, 0.55, ..., 1
T_FRACS_BY_IDX = {16: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)}

HERE = os.path.dirname(os.path.abspath(__file__))
PREV_2NDGEN = os.path.join(HERE, "results_lm_second_gen_converged.json")

_TAG = os.environ.get("ENV_POP_TAG", "").strip()
JSON_PATH = os.path.join(HERE, f"results_lm_e0fixed{('_' + _TAG) if _TAG else ''}.json")


# --- helpers ---------------------------------------------------------------
def load_prev_2ndgen():
    """The e0-free 2nd-gen fits of the converged sources, keyed by idx."""
    with open(PREV_2NDGEN) as f:
        return {r["idx"]: r for r in json.load(f)["sources"]}


def drop_e0(names, vec):
    """vec over `names` with the e0 entry removed."""
    return np.array([v for n, v in zip(names, vec) if n != "e0"], dtype=float)


def truth_of(src, params):
    t = [float(src["source_params"][n]) for n in PARAMS_0PA]
    return np.array(t + [0.0] * (len(params) - len(PARAMS_0PA)))


def model_ctx(S, params, truth_pn, label):
    """0PA context, or the PN context with its C_p step calibrated at the truth.

    The same as cvp.calibrated_pn_ctx, but for the e0-fixed parameter list.
    """
    ctx = cvp.build_ctx(S, params)
    if params != PARAMS_PN:
        return ctx, {}
    t0 = time.time()
    dev_dr = cvp.calibrate_dev_deltas(ctx, truth_pn, tag=f" {label}")
    ctx["delta_range"].update(dev_dr)
    print(f"    [DELTA] calibration took {time.time() - t0:.1f} s", flush=True)
    return ctx, {k: [float(x) for x in v] for k, v in dev_dr.items()}


# --- one LM climb ----------------------------------------------------------
def climb(S, src, params, theta0, label, start):
    """One LM climb from theta0, recorded with its start and end point."""
    truth = truth_of(src, params)
    ctx, dev_dr = model_ctx(S, params, truth_of(src, PARAMS_PN), label)
    theta0 = np.asarray(theta0, dtype=float)
    ov_start, chi2_start = float(ctx["ov"](theta0)), float(ctx["chi2r"](theta0))
    print(f"    [START {label}] from {start}: ov = {ov_start:.12f}  chi2 = {chi2_start:.6e}",
          flush=True)
    t0 = time.time()
    theta, ov, sigma, info = cvp.lm_climb(ctx, theta0, tag=f" {label}")
    chi2 = float(ctx["chi2r"](theta))
    print(f"    [DONE  {label}] ov {ov_start:.12f} -> {float(ov):.12f}  chi2 {chi2_start:.6e} -> "
          f"{chi2:.6e}  its={info['n_iter']} accepted={info['n_accept']} "
          f"stop={info['stop_reason']}  {time.time() - t0:.0f} s", flush=True)
    print("    theta_final = " + ", ".join(f"{n}={v!r}" for n, v in zip(params, map(float, theta))),
          flush=True)
    return dict(start=start, T=float(S["T"]), snr=float(S["snr"]),
                params=params, theta_start=[float(x) for x in theta0],
                theta_final=[float(x) for x in theta], bias=[float(x) for x in theta - truth],
                # fishinv of the last LM iteration's Fisher, NOT a Fisher at the final point
                sigma_lm_last_iter=[float(x) for x in sigma],
                ov_start=ov_start, chi2_start=chi2_start, ov_final=float(ov), chi2=chi2,
                dev_delta_range=dev_dr, seconds=time.time() - t0, **info)


# --- the two kinds of source -----------------------------------------------
def run_converged(src, prev, save):
    """Full T, LM from the e0-free 2nd-gen fit of each model, e0 dropped."""
    S = cvp.build_source(src)
    out = {}
    for model, params in (("0pa", PARAMS_0PA), ("pn", PARAMS_PN)):
        p = prev[model]
        print(f"    --- {model} ({len(params)} params, from the e0-free 2nd-gen fit) ---",
              flush=True)
        out[model] = climb(S, src, params, drop_e0(p["params"], p["theta_final"]),
                           f"{src['idx']}/{model}", "2nd-gen e0-free fit, e0 dropped")
        save(out)
    del S
    free_gpu()
    return out


def ladder(src, params, fracs, label, save_rungs):
    """LM up the T-ladder `fracs` from the injected truth (C_p = 0 for PN), each rung from
    the rung before.  Returns the rungs; save_rungs(rungs) is called after every rung."""
    seed, start, rungs = truth_of(src, params), "truth", []
    for f in fracs:
        T = f * src["T"]
        print(f"    --- {label} ({len(params)} params) rung T = {f:g} x {src['T']:.4f} "
              f"= {T:.4f} yr, from {start} ---", flush=True)
        S = cvp.build_source(with_changes(src, T=T))
        rung = climb(S, src, params, seed, f"{src['idx']}/{label}/T{f:g}", start)
        rungs.append(dict(rung, t_frac=f))
        del S
        free_gpu()
        seed, start = np.array(rung["theta_final"]), f"T{f:g}_final"
        save_rungs(rungs)
    return rungs


def run_ladder(src, save):
    """0PA T-ladder from the truth.  PN from two seeds, the higher overlap kept:
    its own T-ladder from the truth (C_p = 0), and a full-T climb from the 0PA T=1 fit + C_p = 0
    (the second guarantees PN >= 0PA)."""
    fracs = T_FRACS_BY_IDX.get(src["idx"], T_FRACS)
    out = {}

    def keep(model, **extra):
        def _save(rungs):
            out[model] = dict(rungs[-1], ladder=rungs, **extra)
            save(out)
        return _save

    rungs_0pa = ladder(src, PARAMS_0PA, fracs, "0pa", keep("0pa"))

    by_seed = {}
    rungs_pn = ladder(src, PARAMS_PN, fracs, "pn_truth", keep("pn_from_truth"))
    by_seed["truth_ladder"] = dict(rungs_pn[-1], ladder=rungs_pn)

    print(f"    --- pn ({len(PARAMS_PN)} params) full T, from the 0PA T=1 fit + C_p = 0 ---",
          flush=True)
    S = cvp.build_source(src)
    by_seed["0pa_fit"] = climb(S, src, PARAMS_PN, np.append(rungs_0pa[-1]["theta_final"], 0.0),
                               f"{src['idx']}/pn_0pa", "0PA T1 fit + C_p=0")
    del S
    free_gpu()

    best = max(by_seed, key=lambda k: by_seed[k]["ov_final"])
    print(f"    [PN SEED] truth ladder ov = {by_seed['truth_ladder']['ov_final']:.12f}, "
          f"0PA fit ov = {by_seed['0pa_fit']['ov_final']:.12f} -> {best}", flush=True)
    out.pop("pn_from_truth", None)
    out["pn"] = dict(by_seed[best], seed=best, by_seed=by_seed)
    save(out)
    return out


# --- driver ----------------------------------------------------------------
def save_json(records):
    payload = dict(
        study="environmental_CV_population_run_1_second_gen_e0fixed",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        population=os.path.relpath(POP, HERE), run1_results=os.path.relpath(RUN1_RESULTS, HERE),
        prev_2ndgen=os.path.basename(PREV_2NDGEN),
        tdi="2nd generation", nchannels=cvp.NCHANNELS, freq_band=[cvp.F_MIN, cvp.F_MAX],
        target_snr=TARGET_SNR, params_0pa=PARAMS_0PA, params_pn=PARAMS_PN, e0_fixed=0.0,
        lm_max_iters=LM_MAX_ITERS, t_fracs=[float(f) for f in T_FRACS],
        t_fracs_by_idx={str(k): list(v) for k, v in T_FRACS_BY_IDX.items()},
        note=("e0 fixed at 0, LM only (no Nelder-Mead), dist rescaled to SNR 200 under 2nd-gen "
              "TDI. Converged run_1 sources: one full-T climb per model from the e0-free 2nd-gen "
              "fit with e0 dropped. Non-converged: 0PA T-ladder from the truth; PN = best of its "
              "own T-ladder from the truth and a full-T climb from the 0PA T=1 fit with C_p=0. "
              "sigma_lm_last_iter is fishinv of the last LM "
              "iteration's Fisher, not a Fisher at the final point."),
        sources=records)
    with open(f"{JSON_PATH}.tmp", "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(f"{JSON_PATH}.tmp", JSON_PATH)


def run_source(src, fit1, prev, records):
    idx = src["idx"]
    converged = fit1["fit_0pa"]["ov_final"] > CONVERGED_OV
    print("=" * 78, flush=True)
    print(f"[SRC {idx:2d}] m1 = {src['m1']:.6e}  p0 = {src['p0']:.6f}  T = {src['T']:.4f} yr  "
          f"-> {'LM from 2nd-gen fit' if converged else 'T-ladder from truth'}", flush=True)
    t0 = time.time()
    src, snr_run1_dist = rescale_to_snr(src)
    rec = dict(idx=idx, m1=src["m1"], p0=src["p0"], T=src["T"], dt=src["dt"],
               A_PM=src["A_PM"], n_PM=src["n_PM"], dist=src["dist"],
               snr_2ndgen_at_run1_dist=snr_run1_dist,
               run1_converged=bool(converged), mode="lm" if converged else "T_ladder",
               run1_ov_0pa=fit1["fit_0pa"]["ov_final"], run1_ov_pn=fit1["fit_pn"]["ov_final"],
               source_params=src["source_params"])
    if converged:
        rec.update(prev_e0free_ov_0pa=prev[idx]["0pa"]["ov_final"],
                   prev_e0free_ov_pn=prev[idx]["pn"]["ov_final"])
    records.append(rec)

    def save(fits):
        rec.update(fits, seconds=time.time() - t0)
        save_json(records)

    if converged:
        run_converged(src, prev[idx], save)
    else:
        run_ladder(src, save)


def print_summary(records):
    print("\n" + "=" * 78)
    print(f"{'idx':>3} {'mode':>8} {'ov 0PA':>14} {'chi2 0PA':>11} {'ov PN':>14} {'chi2 PN':>11} "
          f"{'stop 0PA':>17} {'stop PN':>17}")
    for r in records:
        if "0pa" not in r or "pn" not in r:
            continue
        a, b = r["0pa"], r["pn"]
        print(f"{r['idx']:>3} {r['mode']:>8} {a['ov_final']:>14.10f} {a['chi2']:>11.4e} "
              f"{b['ov_final']:>14.10f} {b['chi2']:>11.4e} "
              f"{a['stop_reason']:>17} {b['stop_reason']:>17}")


def main():
    check_branch()
    sources, fits = load_run1()
    prev = load_prev_2ndgen()
    print(f"[info] {len(sources)} sources {[s['idx'] for s in sources]}, e0 fixed at 0, "
          f"LM_MAX_ITERS = {LM_MAX_ITERS}; out -> {os.path.basename(JSON_PATH)}\n", flush=True)
    records, t_all = [], time.time()
    for src in sources:
        try:
            run_source(src, fits[src["idx"]], prev, records)
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            print(f"[FAIL SRC {src['idx']}] {type(exc).__name__}: {exc}", flush=True)
    print_summary(records)
    save_json(records)
    print(f"\n[saved] {JSON_PATH}  ({time.time() - t_all:.1f} s total)")


if __name__ == "__main__":
    main()
