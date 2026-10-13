"""Fisher of each recovery model at its own best fit, for the combined population.

    FISH_RUN=run_2 [FISH_IDXS="0 2 3"] python refit_fisher.py

Same recipe as ../run_1/refit_fisher.py -- order-8 derivatives over a 24-point stability
ladder, taken at each model's *own* best fit, C_p step recalibrated per source at the truth
(C_p = 0) -- but for the e0-fixed fits (0PA 7 params, PN 8) of run_1_second_gen and run_2.

Consistency with the fits: the source is rebuilt by the very cv_population.build_source the
fits used (run_1_second_gen's and run_2's copies are byte-identical): 2nd-generation TDI,
A and E with A2/E2TDISens, band [1e-5, 0.1] Hz with the DC bin dropped, dt and T of the
source, and the dist each run actually used (run_1_second_gen rescaled it to SNR 200).
Templates have the environment off and e0 = 0, as in the fits.  Nothing is refitted.

One JSON per source, results/<run>_idxNN.json, written as it finishes.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

import results_inputs as RI

sys.path.insert(0, os.path.join(RI.ENV, "run_2"))
import cv_population as CP          # noqa: E402  identical in run_1_second_gen and run_2
import lm_e0fixed as LME            # noqa: E402  for check_branch

# The whole point of this script.  Set before any source is built: build_source hands them to
# the SEF instance and fisher_derivs reads them as module globals at call time.
DER_ORDER = 8
NDELTA = 24
CP.DER_ORDER = DER_ORDER
CP.NDELTA = NDELTA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


# --- selection -------------------------------------------------------------
def selected():
    """The sources of FISH_RUN this job handles, optionally narrowed by FISH_IDXS."""
    run = os.environ.get("FISH_RUN", "").strip()
    if run not in RI.RUNS:
        raise SystemExit(f"FISH_RUN must be one of {list(RI.RUNS)}, got {run!r}")
    sources = RI.load_run(run)
    sel = os.environ.get("FISH_IDXS", "").strip()
    if not sel:
        return sources
    want = {int(x) for x in sel.replace(",", " ").split()}
    missing = want - {s["idx"] for s in sources}
    if missing:
        raise ValueError(f"FISH_IDXS names sources not in {run}: {sorted(missing)}")
    return [s for s in sources if s["idx"] in want]


# --- Fisher ----------------------------------------------------------------
def model_ctx(S, names, truth_pn, label):
    """The recovery context, with the C_p step calibrated at the truth where there is one."""
    ctx = CP.build_ctx(S, names)
    if "C_p" not in names:
        return ctx, None
    dev_dr = CP.calibrate_dev_deltas(ctx, np.asarray(truth_pn, dtype=float), tag=f" {label}")
    ctx["delta_range"].update(dev_dr)
    return ctx, {k: [float(x) for x in v] for k, v in dev_dr.items()}


def jacobi_scale(M):
    """`d` and `M_hat = D^-1 M D^-1`, `d_i = sqrt(|M_ii|)`: unit diagonal, same inertia."""
    d = np.sqrt(np.abs(np.diag(M)))
    if np.any(d <= 0):
        raise ValueError(f"non-positive diagonal at {np.where(d <= 0)[0]}: a derivative "
                         f"column is dead")
    return d, M / np.outer(d, d)


def fisher_at(ctx, theta, label):
    """The Fisher at `theta` and its inverse, with the definiteness and inversion checks done
    in the scaled basis (see ../run_1/refit_fisher.py for the reasoning)."""
    t0 = time.time()
    G, _, _ = ctx["fisher_derivs"](np.asarray(theta, dtype=float), None)
    G = 0.5 * (G + G.T)

    d, G_hat = jacobi_scale(G)
    G_hat = 0.5 * (G_hat + G_hat.T)
    C_inv = np.linalg.inv(G_hat)
    C_hat = 0.5 * (C_inv + C_inv.T)
    C = C_hat / np.outer(d, d)

    ev_hat = np.linalg.eigvalsh(C_hat)
    cond_raw, cond_scaled = float(np.linalg.cond(G)), float(np.linalg.cond(G_hat))
    resid = float(np.abs(G_hat @ C_hat - np.eye(len(d))).max())
    resid_ratio = float(resid / (cond_scaled * np.finfo(float).eps))
    n_eff = int((ev_hat > ev_hat.max() * np.finfo(float).eps).sum())

    print(f"[FISH] {label}: cond {cond_raw:.3e} -> {cond_scaled:.3e} scaled, "
          f"min eig(C_hat) = {ev_hat.min():.3e}, inv residual = {resid:.2e} "
          f"({resid_ratio:.2f} x float64 limit), rank {n_eff}/{len(d)}, "
          f"{time.time() - t0:.1f} s", flush=True)
    if ev_hat.min() <= 0:
        print(f"[WARN] {label}: covariance indefinite even after rescaling", flush=True)
    if resid_ratio > 10.0:
        print(f"[WARN] {label}: inversion residual {resid:.2e} is {resid_ratio:.0f}x the "
              f"float64 limit -- covariance is not reliable", flush=True)
    return dict(G=G, C=C, C_hat=C_hat, scale=d, cond_raw=cond_raw, cond_scaled=cond_scaled,
                min_eig_scaled=float(ev_hat.min()), inv_residual=resid,
                inv_residual_ratio=resid_ratio, numerical_rank=n_eff,
                pos_def=bool(ev_hat.min() > 0))


def sigma_at(ctx, theta, label):
    """sigma, the Fisher, the covariance, the correlation matrix and the conditioning."""
    f = fisher_at(ctx, theta, label)
    sig = np.sqrt(np.abs(np.diag(f["C_hat"]))) / f["scale"]
    corr = f["C"] / np.outer(sig, sig)

    def mat(M):
        return [[float(x) for x in row] for row in M]

    return dict(sigma=[float(x) for x in sig], fisher=mat(f["G"]), cov=mat(f["C"]),
                corr=mat(corr), scale=[float(x) for x in f["scale"]],
                cond_raw=f["cond_raw"], cond_scaled=f["cond_scaled"],
                cov_min_eig_scaled=f["min_eig_scaled"], inv_residual=f["inv_residual"],
                inv_residual_ratio=f["inv_residual_ratio"],
                numerical_rank=f["numerical_rank"], cov_pos_def=f["pos_def"])


# --- per source ------------------------------------------------------------
def fit_record(src, model, S):
    """One model's entry: the fit as found, the new sigma, and the bias."""
    label = f"[{src['name']} {model}]"
    fit = src["fits"][model]
    names, theta, truth = fit["params"], fit["theta"], src["truth"][model]

    t0 = time.time()
    ctx, dev_dr = model_ctx(S, names, src["truth"]["0PA+PN"], label)
    ov_check = float(ctx["ov"](np.asarray(theta, dtype=float)))
    if abs(ov_check - fit["ov_final"]) > 1e-11:
        print(f"[WARN] {label}: overlap at the fit re-evaluates to {ov_check:.12f}, "
              f"stored {fit['ov_final']:.12f}", flush=True)
    fish = sigma_at(ctx, theta, label)

    bias = RI.signed_bias_of(names, truth, theta)
    rec = dict(model=model, params=names, origin=fit["origin"], seed=fit["seed"],
               stop_reason=fit["stop_reason"],
               truth=[float(x) for x in truth], fit=[float(x) for x in theta],
               ov_final=fit["ov_final"], ov_recheck=ov_check, chi2=fit["chi2"],
               bias=[bias[n] for n in names],
               bias_over_sigma=[bias[n] / s if s > 0 else None
                                for n, s in zip(names, fish["sigma"])],
               dev_delta_range=dev_dr, fisher_seconds=time.time() - t0)
    rec.update(fish)
    return rec


def run_source(src):
    """Both models for one source, sharing one signal and one SEF."""
    t0 = time.time()
    print(f"\n=== {src['name']} (p0 = {src['p0']:.3f}) ===", flush=True)
    S = CP.build_source(src["source"])
    print(f"[{src['name']}] snr = {S['snr']:.3f}, context in {time.time() - t0:.1f} s",
          flush=True)
    models = {m: fit_record(src, m, S) for m in RI.MODELS}
    del S
    LME.free_gpu()
    s = src["source"]
    return dict(run=src["run"], idx=src["idx"], name=src["name"],
                p0=float(s["p0"]), m1=float(s["m1"]), T=float(s["T"]), dt=float(s["dt"]),
                dist=float(s["dist"]), A_PM=float(s["A_PM"]), n_PM=float(s["n_PM"]),
                tdi="2nd generation", nchannels=CP.NCHANNELS, freq_band=[CP.F_MIN, CP.F_MAX],
                e0_fixed=0.0, der_order=DER_ORDER, Ndelta=NDELTA,
                generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                models=models, seconds=time.time() - t0)


def write_source(rec):
    path = os.path.join(OUT, f"{rec['run']}_idx{rec['idx']:02d}.json")
    with open(f"{path}.tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(f"{path}.tmp", path)
    print(f"[SAVE] {path} ({rec['seconds']:.1f} s)", flush=True)


def main():
    LME.check_branch()
    os.makedirs(OUT, exist_ok=True)
    sources = selected()
    print(f"[INFO] der_order = {DER_ORDER}, Ndelta = {NDELTA}, tdi = 2nd generation, "
          f"nchannels = {CP.NCHANNELS}, band = [{CP.F_MIN}, {CP.F_MAX}]", flush=True)
    print(f"[INFO] {len(sources)} sources: {[s['name'] for s in sources]}", flush=True)
    t0, failed = time.time(), []
    for src in sources:
        try:
            write_source(run_source(src))
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            print(f"[FAIL] {src['name']}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(src["name"])
    print(f"\n[DONE] {len(sources) - len(failed)}/{len(sources)} sources in "
          f"{time.time() - t0:.1f} s; failed {failed}", flush=True)


if __name__ == "__main__":
    main()
