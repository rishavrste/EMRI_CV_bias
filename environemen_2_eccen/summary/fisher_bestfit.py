"""Re-score every best fit of the five sets in one pipeline: overlap, chi2, SNR and a fresh
Fisher matrix at the 0PA and 0PA+PN best-fit points, and the bias they imply.

The best fits are read from summary.json (make_summary.py; phases already wrapped to within
pi of the injection, which is the same waveform).  Each set's signal is regenerated once from
its own config and shared by both templates, so the two models of a set are scored against
the identical injection, PSD and band.  Set 4 came from test.ipynb, which has no config;
configs/set4_config_*.json rebuild it (run_5's config with e0 = 0.0125).

    bias            = theta_bf - theta_inj
    sigma           = sqrt(diag Gamma^-1), Gamma = Fisher at the best fit (8x8 for PN, so
                      the intrinsic sigmas are marginalised over C_p, C_e)
    bias_over_sigma = bias / sigma

With --snr S every set is first renormalised to SNR = S by moving the (pinned, unfitted)
distance, d = d0 * SNR0 / S, for signal and template alike.  That scales chi2 and the Fisher by
(S/SNR0)^2 and leaves the best fit and the overlap exactly where they were, so the stored best
fits are still the optima; the run checks it.  The per-channel (A, E, T) SNR is recorded for
every set: a large T share would mean the signal aliases above Nyquist.

Results are written after every model, so a walltime kill still leaves valid output.
Run:  python fisher_bestfit.py            -> fisher_bestfit.json         (native distance)
      python fisher_bestfit.py --snr 200  -> fisher_bestfit_snr200.json  (GPU; plots:
      plot_fisher_bestfit.py)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from env_cv import (Config, SPECS, build_scoring, make_model, observation_time,  # noqa: E402
                    sigma_of, jsonable, log)


# set tag -> (0PA config, 0PA+PN config), paths relative to environemen_2_eccen/
CONFIGS = {
    "set 1": ("config_de_env_vs_vacuum.json", "config_de_env_vs_pn.json"),
    "set 2": ("new_runs/config_de_0pa.json", "new_runs/config_de_0pa_pn.json"),
    "set 3": ("new_run_3/config_de_0pa.json", "new_run_3/config_de_0pa_pn.json"),
    "set 4": ("summary/configs/set4_config_0pa.json", "summary/configs/set4_config_0pa_pn.json"),
    "set 5": ("run_5/config_de_0pa.json", "run_5/config_de_0pa_pn.json"),
}
TEMPLATES = ("vacuum", "pn")


def load_best_fits():
    return {s["tag"]: s for s in json.load(open(os.path.join(HERE, "summary.json")))}


def check_same_source(cfg_v, cfg_p, stored):
    """Both configs and the stored summary must describe one injection, or the rescore
    would silently compare against a different source."""
    for n in stored["injection"]:
        assert np.isclose(cfg_v.injection[n], cfg_p.injection[n]), n
        assert np.isclose(cfg_p.injection[n], stored["injection"][n]), n
    assert np.isclose(cfg_v.disk["Sigma0"], cfg_p.disk["Sigma0"])


def fisher_health(G):
    eig = np.linalg.eigvalsh(G)
    return dict(min_eig=float(eig.min()), max_eig=float(eig.max()),
                cond=float(eig.max() / abs(eig.min())), positive_definite=bool(eig.min() > 0))


def score_model(cfg, scoring, T, template, stored_model):
    """Overlap, chi2 and Fisher of one template at its stored best fit."""
    spec = SPECS[template](cfg)
    model = make_model(cfg, scoring, T, spec)
    assert model["names"] == stored_model["names"], (model["names"], stored_model["names"])

    theta = np.array(stored_model["theta"], float)
    theta_inj = np.array(stored_model["theta_inj"], float)
    t0 = time.time()
    ov, chi2 = model["ov"](theta), model["chi2"](theta)
    G, _, _ = model["fisher"](theta)
    sigma = sigma_of(theta, G)
    bias = theta - theta_inj
    log(f"[FIT ] {model['name']:<22} ov = {ov:.12f}  (stored {stored_model['overlap']:.12f})  "
        f"chi2 = {chi2:.6e}  [{time.time() - t0:.0f} s]")
    for n, b, s in zip(model["names"], bias, sigma):
        log(f"         {n:>9}  bias = {b:+.6e}  sigma = {s:.6e}  b/s = {b / s:+.3f}")
    return dict(names=model["names"], theta_inj=theta_inj, theta=theta, overlap=ov,
                mismatch=1 - ov, chi2=chi2, fisher=G, sigma=sigma, bias=bias,
                bias_over_sigma=bias / sigma, fisher_health=fisher_health(G),
                overlap_stored=stored_model["overlap"], sigma_stored=stored_model["sigma"])


def channel_snr(scoring):
    """SNR carried by each channel alone (the PSD is diagonal in A, E, T)."""
    sig, out = scoring["signal"], {}
    for k, name in enumerate("AET"[:sig.shape[0]]):
        one = sig * 0
        one[k] = sig[k]
        out[name] = float(np.sqrt(scoring["ip"](one, one)))
    total = sum(v ** 2 for v in out.values())
    log("[CHAN] " + "  ".join(f"{n}: SNR {v:.4f} ({100 * v ** 2 / total:.4f}% of SNR^2)"
                              for n, v in out.items()))
    return out


def renormalise(cfgs, scoring, T, target):
    """Move the distance so the SNR is `target`; returns the new scoring and distances."""
    d0 = cfgs[0].injection["dist"]
    d = d0 * scoring["snr"] / target
    for cfg in cfgs:
        cfg.injection["dist"] = d
    scoring = build_scoring(cfgs[-1], T)
    log(f"[SNR ] dist {d0:g} -> {d:.6f} Gpc: SNR {scoring['snr']:.6f} (target {target:g})")
    return scoring, d0, d


def score_set(tag, stored, results, target_snr, out_path):
    """Regenerate the set's signal once, then score both templates against it."""
    log(f"\n{'=' * 90}\n[SET ] {tag}  ({stored['folder']})")
    cfg_v, cfg_p = (Config(os.path.join(ROOT, p)) for p in CONFIGS[tag])
    check_same_source(cfg_v, cfg_p, stored)
    T = observation_time(cfg_p)
    scoring = build_scoring(cfg_p, T)
    log(f"[CTX ] SNR = {scoring['snr']:.6f} at dist = {cfg_p.injection['dist']:g}")
    snr0, d0, d = scoring["snr"], cfg_p.injection["dist"], cfg_p.injection["dist"]
    if target_snr:
        scoring, d0, d = renormalise((cfg_v, cfg_p), scoring, T, target_snr)
    out = dict(tag=tag, folder=stored["folder"], injection=dict(stored["injection"], dist=d),
               disk=stored["disk"], T=T, snr=scoring["snr"], snr_native=snr0,
               dist_native=d0, dist=d, channel_snr=channel_snr(scoring), models={})
    results.append(out)
    for template, cfg in zip(TEMPLATES, (cfg_v, cfg_p)):
        out["models"][template] = score_model(cfg, scoring, T, template,
                                              stored["models"][template])
        save(results, out_path)
    v, p = out["models"]["vacuum"], out["models"]["pn"]
    log(f"[SET ] {tag}: mismatch 0PA {v['mismatch']:.3e}, PN {p['mismatch']:.3e}, "
        f"ratio {v['mismatch'] / p['mismatch']:.2f}x")


def save(results, out_path):
    with open(out_path, "w") as f:
        json.dump(jsonable(results), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=None,
                    help="renormalise every set to this SNR via the distance")
    args = ap.parse_args()
    suffix = f"_snr{args.snr:g}" if args.snr else ""
    out_path = os.path.join(HERE, f"fisher_bestfit{suffix}.json")
    stored = load_best_fits()
    results = []
    for tag in CONFIGS:
        score_set(tag, stored[tag], results, args.snr, out_path)
    log(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
