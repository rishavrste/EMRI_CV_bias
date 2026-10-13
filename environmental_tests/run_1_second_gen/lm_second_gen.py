"""run_1 population refit with 2nd-generation TDI: LM only, from the run_1 best fits.

Same 23 circular sources as run_1 (run_1/population.json), same injection, same recovery
models (0PA, 8 params; 0PA + PN, 9 params), A and E only, dt = 10.  What changes:

* TDI is 2nd generation (A2TDISens, E2TDISens), via the cv_population.py copied next to this
  file -- identical to run_2/cv_population.py, and to run_1's except for the TDI generation.
* dist is rescaled per source so the injection again has SNR = 200 under the 2nd-gen PSD.
  dist is not a fit parameter, so the run_1 best fits remain valid starting points.
* No Nelder-Mead, no environmental Fisher, no Fisher at the best fit.  Just LM.

Two kinds of source:

converged (run_1 0PA overlap > CONVERGED_OV, 19 of them)
    one LM climb at the full T, 0PA from run_1's fit_0pa and PN from run_1's fit_pn.

not converged (idx 8, 13, 14, 16)
    their run_1 best fits sit in a wrong basin, and at full T even the injected truth has
    overlap -0.30 to 0.09 against the 0PA template.  So climb a T-ladder instead: the same
    source observed for T = f x T_src for f in T_FRACS, rung 1 from the injected truth
    (C_p = 0 for PN), every later rung from the previous rung's final point.

The C_p finite-difference step is calibrated at the truth, per source and per rung (it
depends on T), exactly as run_1 did.

Output: results_lm_second_gen.json here, rewritten after every climb.

Run:  python lm_second_gen.py
      ENV_POP_IDXS=8,13,14,16  ENV_POP_TAG=ladder  python lm_second_gen.py
"""

import copy
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

import cv_population as cvp


# --- controls --------------------------------------------------------------
TARGET_SNR = 200.0                        # as run_1, now under the 2nd-gen PSD
CONVERGED_OV = 0.99                       # run_1's own convergence cut (analysis_report.txt)
T_FRACS = (0.5, 0.75, 0.9, 0.95, 1.0)     # ladder for the non-converged sources (user, 2026-10-07)
SUPERKLUDGE = "/home/svu/e1583490/packages_to_install/SuperKludge_r"
SUPERKLUDGE_BRANCH = "environ_shubham"

HERE = os.path.dirname(os.path.abspath(__file__))
RUN1 = os.path.join(HERE, "..", "run_1")
POP = os.path.join(RUN1, "population.json")
RUN1_RESULTS = os.path.join(RUN1, "results_env_cv.json")

_TAG = os.environ.get("ENV_POP_TAG", "").strip()
JSON_PATH = os.path.join(HERE, f"results_lm_second_gen{('_' + _TAG) if _TAG else ''}.json")


# --- setup -----------------------------------------------------------------
def check_branch():
    """Stop if SuperKludge_r is not on the branch that carries the environmental term."""
    branch = subprocess.run(["git", "-C", SUPERKLUDGE, "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print(f"SuperKludge_r branch: {branch}", flush=True)
    if branch != SUPERKLUDGE_BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {SUPERKLUDGE_BRANCH!r}")


def load_run1():
    """run_1 sources and their run_1 fits, keyed by idx, optionally narrowed by ENV_POP_IDXS."""
    with open(POP) as f:
        sources = json.load(f)["sources"]
    with open(RUN1_RESULTS) as f:
        fits = {r["idx"]: r for r in json.load(f)["sources"]}
    sel = os.environ.get("ENV_POP_IDXS", "").strip()
    if sel:
        want = {int(x) for x in sel.replace(",", " ").split()}
        sources = [s for s in sources if s["idx"] in want]
    return sources, fits


def with_changes(src, T=None, dist=None):
    """A copy of a population entry with T and / or dist replaced."""
    out = copy.deepcopy(src)
    if T is not None:
        out["T"] = T
    if dist is not None:
        out["dist"] = dist
        out["source_params"]["dist"] = dist
    return out


def free_gpu():
    if cvp.use_gpu:
        cvp.xp.get_default_memory_pool().free_all_blocks()


def rescale_to_snr(src):
    """dist that gives SNR = TARGET_SNR under the 2nd-gen PSD.  SNR scales as 1 / dist."""
    S = cvp.build_source(src)
    snr_run1_dist = float(S["snr"])
    del S
    free_gpu()
    dist = src["dist"] * snr_run1_dist / TARGET_SNR
    print(f"    [DIST] SNR at run_1 dist {src['dist']:.6f} Gpc = {snr_run1_dist:.4f} (2nd gen) "
          f"-> dist = {dist:.6f} Gpc", flush=True)
    return with_changes(src, dist=dist), snr_run1_dist


# --- one LM climb ----------------------------------------------------------
def model_ctx(S, params, truth_pn, label):
    """0PA context, or the PN context with its C_p step calibrated at the truth."""
    if params == cvp.PARAMS_PN:
        ctx, dev_dr = cvp.calibrated_pn_ctx(S, truth_pn, label)
        return ctx, {k: [float(x) for x in v] for k, v in dev_dr.items()}
    return cvp.build_ctx(S, params), {}


def climb(S, params, theta0, truth, label, start):
    """One LM climb from theta0, recorded with its start and end point."""
    truth_pn = np.concatenate([truth[:len(cvp.PARAMS_0PA)],
                               np.zeros(len(cvp.PARAMS_PN) - len(cvp.PARAMS_0PA))])
    ctx, dev_dr = model_ctx(S, params, truth_pn, label)
    theta0 = np.asarray(theta0, dtype=float)
    ov_start, chi2_start = float(ctx["ov"](theta0)), float(ctx["chi2r"](theta0))
    print(f"    [START {label}] from {start}: ov = {ov_start:.12f}  chi2 = {chi2_start:.6e}",
          flush=True)
    t0 = time.time()
    theta, ov, sigma, info = cvp.lm_climb(ctx, theta0, tag=f" {label}")
    chi2 = float(ctx["chi2r"](theta))
    bias = theta - truth
    print(f"    [DONE  {label}] ov {ov_start:.12f} -> {float(ov):.12f}  chi2 {chi2_start:.6e} -> "
          f"{chi2:.6e}  its={info['n_iter']} accepted={info['n_accept']} "
          f"stop={info['stop_reason']}  {time.time() - t0:.0f} s", flush=True)
    print("    theta_final = " + ", ".join(f"{n}={v!r}" for n, v in zip(params, map(float, theta))),
          flush=True)
    return dict(start=start, T=float(S["T"]), snr=float(S["snr"]),
                params=params, theta_start=[float(x) for x in theta0],
                theta_final=[float(x) for x in theta], bias=[float(x) for x in bias],
                # fishinv of the last LM iteration's Fisher, NOT a Fisher at the final point
                sigma_lm_last_iter=[float(x) for x in sigma],
                ov_start=ov_start, chi2_start=chi2_start, ov_final=float(ov), chi2=chi2,
                dev_delta_range=dev_dr, seconds=time.time() - t0, **info)


# --- the two kinds of source -----------------------------------------------
def truth_of(src, params):
    t = [float(src["source_params"][n]) for n in cvp.PARAMS_0PA]
    return np.array(t + [0.0] * (len(params) - len(cvp.PARAMS_0PA)))


def run_converged(src, fit1, save):
    """Full T, LM from run_1's best fit of each model."""
    S = cvp.build_source(src)
    out = {}
    for model, params, key in (("0pa", cvp.PARAMS_0PA, "fit_0pa"), ("pn", cvp.PARAMS_PN, "fit_pn")):
        print(f"    --- {model} ({len(params)} params, from the run_1 best fit) ---", flush=True)
        out[model] = climb(S, params, fit1[key]["params"], truth_of(src, params),
                           f"{src['idx']}/{model}", f"run_1 {key}")
        save(out)
    del S
    free_gpu()
    return out


def run_ladder(src, save):
    """T-ladder over T_FRACS: rung 1 from the truth, every later rung from the rung before."""
    out = {}
    for model, params in (("0pa", cvp.PARAMS_0PA), ("pn", cvp.PARAMS_PN)):
        truth = truth_of(src, params)
        seed, start, rungs = truth, "truth", []
        for f in T_FRACS:
            T = f * src["T"]
            print(f"    --- {model} ({len(params)} params) rung T = {f:g} x {src['T']:.4f} "
                  f"= {T:.4f} yr, from {start} ---", flush=True)
            S = cvp.build_source(with_changes(src, T=T))
            rung = climb(S, params, seed, truth, f"{src['idx']}/{model}/T{f:g}", start)
            rungs.append(dict(rung, t_frac=f))
            del S
            free_gpu()
            seed, start = np.array(rung["theta_final"]), f"T{f:g}_final"
            out[model] = dict(rungs[-1], ladder=rungs)
            save(out)
    return out


# --- driver ----------------------------------------------------------------
def save_json(records):
    payload = dict(
        study="environmental_CV_population_run_1_second_gen",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        population=os.path.relpath(POP, HERE), run1_results=os.path.relpath(RUN1_RESULTS, HERE),
        tdi="2nd generation", nchannels=cvp.NCHANNELS, freq_band=[cvp.F_MIN, cvp.F_MAX],
        target_snr=TARGET_SNR, params_0pa=cvp.PARAMS_0PA, params_pn=cvp.PARAMS_PN,
        t_fracs=list(T_FRACS), converged_ov=CONVERGED_OV,
        note=("LM only (no Nelder-Mead), dist rescaled to SNR 200 under 2nd-gen TDI. Converged "
              "run_1 sources: one full-T climb per model from the run_1 best fit. Non-converged: "
              "T-ladder over t_fracs from the truth. sigma_lm_last_iter is fishinv of the last "
              "LM iteration's Fisher, not a Fisher at the final point."),
        sources=records)
    with open(f"{JSON_PATH}.tmp", "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(f"{JSON_PATH}.tmp", JSON_PATH)


def run_source(src, fit1, records):
    idx = src["idx"]
    converged = fit1["fit_0pa"]["ov_final"] > CONVERGED_OV
    print("=" * 78, flush=True)
    print(f"[SRC {idx:2d}] m1 = {src['m1']:.6e}  p0 = {src['p0']:.6f}  T = {src['T']:.4f} yr  "
          f"run_1 ov 0PA = {fit1['fit_0pa']['ov_final']:.10f}  "
          f"-> {'LM from run_1 best fit' if converged else 'T-ladder from truth'}", flush=True)
    t0 = time.time()
    src, snr_run1_dist = rescale_to_snr(src)
    rec = dict(idx=idx, m1=src["m1"], p0=src["p0"], T=src["T"], dt=src["dt"],
               A_PM=src["A_PM"], n_PM=src["n_PM"], dist=src["dist"],
               dist_run1=fit1["dist"], snr_2ndgen_at_run1_dist=snr_run1_dist,
               run1_converged=bool(converged), mode="lm" if converged else "T_ladder",
               run1_ov_0pa=fit1["fit_0pa"]["ov_final"], run1_ov_pn=fit1["fit_pn"]["ov_final"],
               source_params=src["source_params"])
    records.append(rec)

    def save(fits):
        rec.update(fits, seconds=time.time() - t0)
        save_json(records)

    if converged:
        run_converged(src, fit1, save)
    else:
        run_ladder(src, save)


def print_summary(records):
    print("\n" + "=" * 78)
    print(f"{'idx':>3} {'mode':>8} {'run_1 ov 0PA':>14} {'ov 0PA':>14} {'run_1 ov PN':>14} "
          f"{'ov PN':>14} {'stop 0PA':>17} {'stop PN':>17}")
    for r in records:
        if "0pa" not in r or "pn" not in r:
            continue
        print(f"{r['idx']:>3} {r['mode']:>8} {r['run1_ov_0pa']:>14.10f} {r['0pa']['ov_final']:>14.10f} "
              f"{r['run1_ov_pn']:>14.10f} {r['pn']['ov_final']:>14.10f} "
              f"{r['0pa']['stop_reason']:>17} {r['pn']['stop_reason']:>17}")


def main():
    check_branch()
    sources, fits = load_run1()
    print(f"[info] {len(sources)} sources {[s['idx'] for s in sources]}, 2nd-gen TDI, "
          f"T-ladder {T_FRACS}; out -> {os.path.basename(JSON_PATH)}\n", flush=True)
    records, t_all = [], time.time()
    for src in sources:
        try:
            run_source(src, fits[src["idx"]], records)
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            print(f"[FAIL SRC {src['idx']}] {type(exc).__name__}: {exc}", flush=True)
    print_summary(records)
    save_json(records)
    print(f"\n[saved] {JSON_PATH}  ({time.time() - t_all:.1f} s total)")


if __name__ == "__main__":
    main()
