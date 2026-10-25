"""summary/fisher_bestfit.py --snr 200 for the 2nd-generation TDI best fits.

Each set's best fits are read from ../results_lm_second_gen_{a,b}.json (lm_second_gen.py) and
re-scored with the same 2nd-gen setup they were fitted with: A and E only (no T channel), band
capped at 0.05 Hz, dist rescaled to SNR 200.  Per model: overlap, chi2 and a fresh Fisher at the
best fit, and the bias it implies.  Output has the same layout as summary/fisher_bestfit*.json,
so the plot / report / export scripts next to this file read it unchanged.

    bias            = theta_bf - theta_inj
    sigma           = sqrt(diag Gamma^-1), Gamma = Fisher at the best fit (8x8 for PN, so the
                      intrinsic sigmas are marginalised over C_p, C_e)
    bias_over_sigma = bias / sigma

Written after every model, so a walltime kill still leaves valid output.
Run (GPU):  python fisher_bestfit_2g.py   -> fisher_bestfit_2g.json
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SECOND_GEN = os.path.dirname(HERE)
sys.path.insert(0, SECOND_GEN)

from lm_second_gen import (TEMPLATES, check_branch, check_same_source, load_best_fits,  # noqa: E402
                           rescale_to_snr, second_gen_configs, free_gpu, ROOT)
from env_cv import SPECS, jsonable, log, make_model, observation_time, sigma_of     # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "summary"))
from fisher_bestfit import channel_snr, fisher_health                              # noqa: E402

LM_RESULTS = [os.path.join(SECOND_GEN, f"results_lm_second_gen_{x}.json") for x in "ab"]
OUT = os.path.join(HERE, "fisher_bestfit_2g.json")


def load_lm_fits():
    return {s["tag"]: s for p in LM_RESULTS for s in json.load(open(p))["sets"]}


def score_model(cfg, scoring, T, template, fit):
    """Overlap, chi2 and Fisher of one template at its 2nd-gen best fit."""
    model = make_model(cfg, scoring, T, SPECS[template](cfg))
    assert model["names"] == fit["names"], (model["names"], fit["names"])
    theta = np.array(fit["theta"], float)
    theta_inj = np.array(fit["theta_inj"], float)
    t0 = time.time()
    ov, chi2 = model["ov"](theta), model["chi2"](theta)
    G, _, _ = model["fisher"](theta)
    sigma = sigma_of(theta, G)
    bias = theta - theta_inj
    log(f"[FIT ] {model['name']:<22} ov = {ov:.12f}  (LM {fit['overlap']:.12f})  "
        f"chi2 = {chi2:.6e}  [{time.time() - t0:.0f} s]")
    for n, b, s in zip(model["names"], bias, sigma):
        log(f"         {n:>9}  bias = {b:+.6e}  sigma = {s:.6e}  b/s = {b / s:+.3f}")
    return dict(names=model["names"], theta_inj=theta_inj, theta=theta, overlap=ov,
                mismatch=1 - ov, chi2=chi2, fisher=G, sigma=sigma, bias=bias,
                bias_over_sigma=bias / sigma, fisher_health=fisher_health(G),
                overlap_lm=fit["overlap"])


def score_set(tag, stored, lm_set, results):
    """Regenerate the set's 2nd-gen signal at SNR 200 once, then score both templates."""
    log(f"\n{'=' * 90}\n[SET ] {tag}  ({stored['folder']})")
    cfgs = second_gen_configs(tag)
    check_same_source(*cfgs, stored)
    T = observation_time(cfgs[-1])
    scoring, d0, snr0, d = rescale_to_snr(cfgs, T)
    assert np.isclose(d, lm_set["dist"], rtol=1e-8), (d, lm_set["dist"])
    out = dict(tag=tag, folder=stored["folder"], injection=dict(stored["injection"], dist=d),
               disk=stored["disk"], T=T, snr=scoring["snr"], snr_native=snr0,
               dist_native=d0, dist=d, channel_snr=channel_snr(scoring), models={})
    results.append(out)
    for template, cfg in zip(TEMPLATES, cfgs):
        out["models"][template] = score_model(cfg, scoring, T, template,
                                              lm_set["models"][template])
        save(results)
    v, p = out["models"]["vacuum"], out["models"]["pn"]
    log(f"[SET ] {tag}: mismatch 0PA {v['mismatch']:.3e}, PN {p['mismatch']:.3e}, "
        f"ratio {v['mismatch'] / p['mismatch']:.2f}x")
    del scoring
    free_gpu()


def save(results):
    with open(f"{OUT}.tmp", "w") as f:
        json.dump(jsonable(results), f, indent=1)
    os.replace(f"{OUT}.tmp", OUT)


def main():
    check_branch()
    stored, lm = load_best_fits(), load_lm_fits()
    results = []
    for tag in lm:
        score_set(tag, stored[tag], lm[tag], results)
    log(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
