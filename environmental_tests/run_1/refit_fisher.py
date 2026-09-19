"""Recompute each recovery model's Fisher at its own best fit, on a finer stencil.

    ENV_FISH_IDXS="0 2 3" python refit_fisher.py

The population run took its sigmas with `der_order = 6` and `Ndelta = 12`.  Nothing here
refits anything: the best-fit vectors are the ones those runs already found, read through
`results_inputs`.  All that changes is the derivative stencil used for the Fisher that turns
an absolute bias into a bias in sigma -- order 8 over a 24-point stability ladder -- so the
b/sigma column stops depending on a choice of step size.

The Fisher is taken at each model's *own* best fit, not at the injected truth, which is the
same convention the population run used (`cov_at`).  For the PN model the C_p finite-
difference step is recalibrated first, because the ladder is Ndelta points long and so is
not the one the earlier run measured.

One JSON per source is written as it finishes, so a job that runs out of walltime leaves
behind everything it did complete.
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import cv_population as CP
import results_inputs as RI

# The whole point of this script.  Set before anything calls into cv_population: its
# functions read these as module globals at call time.
DER_ORDER = 8
NDELTA = 24
CP.DER_ORDER = DER_ORDER
CP.NDELTA = NDELTA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def selected(sources):
    """The sources this job handles, so the population can be split across PBS jobs."""
    sel = os.environ.get("ENV_FISH_IDXS", "").strip()
    if not sel:
        return sources
    want = {int(x) for x in sel.replace(",", " ").split()}
    missing = want - {s["idx"] for s in sources}
    if missing:
        raise ValueError(f"ENV_FISH_IDXS names sources that are not in the analysis: "
                         f"{sorted(missing)}")
    return [s for s in sources if s["idx"] in want]


def model_ctx(S, model, truth_pn, label):
    """The recovery context for one model, with C_p's step calibrated where there is one."""
    if model == "0PA":
        return CP.build_ctx(S, CP.PARAMS_0PA), None
    ctx, dev_dr = CP.calibrated_pn_ctx(S, np.asarray(truth_pn, dtype=float), label)
    return ctx, {k: [float(x) for x in v] for k, v in dev_dr.items()}


def jacobi_scale(M):
    """`d` and `M_hat = D^-1 M D^-1` with `D = diag(d)`, `d_i = sqrt(|M_ii|)`.

    M_hat has a unit diagonal, which is the whole point: the raw matrices here hold
    m1 ~ 1e6 beside C_p ~ 1e-4, so their condition number is dominated by the choice of
    units rather than by any real degeneracy, and at cond ~ 1e22 a float64 eigenvalue near
    zero is indistinguishable from noise.

    This is a congruence, not a similarity, so it does not preserve the eigenvalues -- but
    by Sylvester's law of inertia it does preserve their *signs*.  A definiteness test in
    the scaled basis therefore answers the question asked of the unscaled matrix, and
    answers it at a condition number some ten orders lower.  Quadratic forms are preserved
    exactly: b^T M b = (D b)^T M_hat (D b).
    """
    d = np.sqrt(np.abs(np.diag(M)))
    if np.any(d <= 0):
        raise ValueError(f"non-positive diagonal at {np.where(d <= 0)[0]}: a derivative "
                         f"column is dead")
    return d, M / np.outer(d, d)


def fisher_at(ctx, theta, label):
    """The Fisher at `theta`, the covariance from it, and what the rescaling reveals.

    Everything that can be done in the scaled basis is done there and mapped back at the
    end.  `cv_population.cov_at` already inverts this way; what it does not do is test
    definiteness in the scaled basis, which is why the population run reported covariances
    as indefinite that are in fact positive definite by a wide margin.
    """
    t0 = time.time()
    G, _, _ = ctx["fisher_derivs"](np.asarray(theta, dtype=float), None)
    G = 0.5 * (G + G.T)

    d, G_hat = jacobi_scale(G)
    G_hat = 0.5 * (G_hat + G_hat.T)
    C_inv = np.linalg.inv(G_hat)
    C_hat = 0.5 * (C_inv + C_inv.T)
    C = C_hat / np.outer(d, d)

    # The inertia test, in the basis where it means something.
    ev_hat = np.linalg.eigvalsh(C_hat)
    cond_raw, cond_scaled = float(np.linalg.cond(G)), float(np.linalg.cond(G_hat))

    # Does the inverse actually invert?  This is the direct test of whether the covariance
    # can be trusted, and it is stronger than quoting a condition number: cond says how much
    # error the inversion *could* amplify, the residual says how much it *did*.
    #
    # The residual is judged against `cond_scaled * eps`, not against a fixed number.  That
    # product is the accuracy float64 can deliver at this conditioning, so a ratio of order
    # one means the inversion is as good as the arithmetic allows and a small absolute
    # residual at a large cond is a success, not a failure.  An absolute threshold gets this
    # backwards and flags the floor itself.
    resid = float(np.abs(G_hat @ C_hat - np.eye(len(d))).max())
    resid_ratio = float(resid / (cond_scaled * np.finfo(float).eps))
    n_eff = int((ev_hat > ev_hat.max() * np.finfo(float).eps).sum())

    print(f"[FISH] {label}: cond {cond_raw:.3e} -> {cond_scaled:.3e} scaled, "
          f"min eig(C_hat) = {ev_hat.min():.3e}, inv residual = {resid:.2e} "
          f"({resid_ratio:.2f} x float64 limit), rank {n_eff}/{len(d)}, "
          f"{time.time() - t0:.1f} s")
    if ev_hat.min() <= 0:
        print(f"[WARN] {label}: covariance indefinite even after rescaling")
    if resid_ratio > 10.0:
        print(f"[WARN] {label}: inversion residual {resid:.2e} is {resid_ratio:.0f}x the "
              f"float64 limit -- covariance is not reliable")
    return dict(G=G, C=C, G_hat=G_hat, C_hat=C_hat, scale=d,
                cond_raw=float(cond_raw), cond_scaled=float(cond_scaled),
                min_eig_scaled=float(ev_hat.min()),
                inv_residual=resid, inv_residual_ratio=resid_ratio, numerical_rank=n_eff,
                pos_def=bool(ev_hat.min() > 0))


def sigma_at(ctx, theta, label):
    """sigma, the Fisher, the correlation matrix and the conditioning, at `theta`."""
    f = fisher_at(ctx, np.asarray(theta, dtype=float), label)
    # sigma_i = sqrt(C_ii) taken as sqrt(C_hat_ii)/d_i: the same number, formed in the
    # basis where the intermediate is order unity.
    sig = np.sqrt(np.abs(np.diag(f["C_hat"]))) / f["scale"]
    corr = f["C"] / np.outer(sig, sig)
    return dict(sigma=[float(x) for x in sig],
                fisher=[[float(x) for x in row] for row in f["G"]],
                cov=[[float(x) for x in row] for row in f["C"]],
                corr=[[float(x) for x in row] for row in corr],
                scale=[float(x) for x in f["scale"]],
                cond_raw=f["cond_raw"], cond_scaled=f["cond_scaled"],
                cov_min_eig_scaled=f["min_eig_scaled"],
                inv_residual=f["inv_residual"],
                inv_residual_ratio=f["inv_residual_ratio"],
                numerical_rank=f["numerical_rank"],
                cov_pos_def=f["pos_def"])


def fit_record(src, model, S):
    """One model's entry for one source: the fit as found, the new sigma, and the bias."""
    idx = src["idx"]
    label = f"[SRC {idx:2d} {model}]"
    fit = src["fits"][model]
    names, theta = fit["params"], fit["theta"]
    truth = src["truth"][model]

    ctx, dev_dr = model_ctx(S, model, src["truth"]["0PA+PN"], label)
    t0 = time.time()
    fish = sigma_at(ctx, theta, label)

    bias = RI.bias_of(names, truth, theta)
    bos = {n: (abs(bias[n] / s) if s > 0 else None)
           for n, s in zip(names, fish["sigma"])}

    rec = dict(model=model, params=names, origin=fit["origin"],
               truth=[float(x) for x in truth], fit=[float(x) for x in theta],
               ov_final=fit["ov_final"], chi2=fit["chi2"],
               bias=[bias[n] for n in names],
               bias_over_sigma=[bos[n] for n in names],
               dev_delta_range=dev_dr, fisher_seconds=time.time() - t0)
    rec.update(fish)
    return rec


def run_source(src):
    """Both models for one source, sharing one signal and one SEF."""
    idx = src["idx"]
    t0 = time.time()
    print(f"\n=== source {idx} (p0 = {src['source']['p0']:.3f}) ===", flush=True)
    S = CP.build_source(src["source"])
    print(f"[SRC {idx:2d}] snr = {S['snr']:.3f}, context in {time.time() - t0:.1f} s",
          flush=True)

    models = {m: fit_record(src, m, S) for m in RI.MODELS}
    return dict(idx=idx, converged=src["converged"],
                not_converged_why=src["not_converged_why"],
                p0=float(src["source"]["p0"]), m1=float(src["source"]["m1"]),
                T=float(src["source"]["T"]), dt=float(src["source"]["dt"]),
                A_PM=float(src["source"]["A_PM"]), n_PM=float(src["source"]["n_PM"]),
                snr=float(S["snr"]), snr_channels=S["snr_channels"],
                der_order=DER_ORDER, Ndelta=NDELTA,
                generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                models=models, seconds=time.time() - t0)


def write_source(rec):
    """One file per source, written the moment it is done."""
    path = os.path.join(OUT, f"idx{rec['idx']:02d}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[SAVE] {path} ({rec['seconds']:.1f} s)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    sources = selected(RI.load_sources())
    print(f"[INFO] der_order = {DER_ORDER}, Ndelta = {NDELTA}", flush=True)
    print(f"[INFO] {len(sources)} sources: {[s['idx'] for s in sources]}", flush=True)

    t0 = time.time()
    for src in sources:
        try:
            write_source(run_source(src))
        except Exception as exc:
            print(f"[FAIL] source {src['idx']}: {type(exc).__name__}: {exc}", flush=True)
            raise
    print(f"\n[DONE] {len(sources)} sources in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
