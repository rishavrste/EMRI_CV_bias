"""The five eccentric environmental sets refit with 2nd-generation TDI: LM only, from the best fits.

Same injections and templates as before (summary/fisher_bestfit.py CONFIGS: 0PA = vacuum spec,
0PA + PN = pn spec, env_cv.py).  What changes:

* TDI is 2nd generation (A2TDISens, E2TDISens), A and E only.
* The band is capped at FMAX = 0.05 Hz for every set: at dt = 5 the 2nd-gen PSD null at ~0.06 Hz
  is otherwise in band and one bin dominates SNR and chi2.  Set 4 already used 0.05 Hz.
* dist is rescaled per set so the injection has SNR = 200 under the 2nd-gen PSD.  dist is not
  a fit parameter, so the stored best fits remain valid starting points.
* One LM climb per model from the stored best fit (summary/summary.json).  No Nelder-Mead, no
  Fisher at the final point.

Output: results_lm_second_gen[_TAG].json here, rewritten after every climb.

Run:  python lm_second_gen.py
      ENV_SETS="set 1,set 2" ENV_TAG=a  python lm_second_gen.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from env_cv import (PARAM_NAMES, Config, SPECS, FastKerrEccentricEquatorialAccretionFlux,  # noqa: E402
                    GenerateEMRIWaveform, ResponseWrapper, EqualArmlengthOrbits, generate_PSD,
                    get_sensitivity, inner_product, jsonable, lm_climb, log, make_model,
                    observation_time, use_gpu, xp)
from lisatools.sensitivity import A2TDISens, E2TDISens  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "summary"))
from fisher_bestfit import CONFIGS, TEMPLATES, check_same_source, load_best_fits  # noqa: E402


# --- controls --------------------------------------------------------------
TARGET_SNR = 200.0
FMAX = 0.05                               # below the 2nd-gen PSD null at ~0.06 Hz
CHANNELS = [A2TDISens, E2TDISens]
SUPERKLUDGE = "/home/svu/e1583490/packages_to_install/SuperKludge_r"
SUPERKLUDGE_BRANCH = "Schwad_eccen_Environ"

_TAG = os.environ.get("ENV_TAG", "").strip()
JSON_PATH = os.path.join(HERE, f"results_lm_second_gen{('_' + _TAG) if _TAG else ''}.json")


# --- setup -----------------------------------------------------------------
def check_branch():
    """Stop if SuperKludge_r is not on the branch that carries the eccentric disk term."""
    branch = subprocess.run(["git", "-C", SUPERKLUDGE, "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    log(f"SuperKludge_r branch: {branch}")
    if branch != SUPERKLUDGE_BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {SUPERKLUDGE_BRANCH!r}")


def selected_sets():
    sel = os.environ.get("ENV_SETS", "").strip()
    return [s.strip() for s in sel.split(",")] if sel else list(CONFIGS)


def second_gen_configs(tag):
    """Both configs of a set, switched to A and E with the band capped at FMAX."""
    cfgs = [Config(os.path.join(ROOT, p)) for p in CONFIGS[tag]]
    for cfg in cfgs:
        cfg.nchannels, cfg.fmax = len(CHANNELS), FMAX
    return cfgs


def build_scoring(cfg, T):
    """env_cv.build_scoring with 2nd-generation TDI: injection, inner product, response."""
    noise_kwargs = [{"sens_fn": ch} for ch in CHANNELS]
    response_kwargs = dict(
        Tobs=T, t0=10000.0, dt=cfg.dt, index_lambda=8, index_beta=7, flip_hx=True,
        is_ecliptic_latitude=False, remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend="cuda12x" if use_gpu else "cpu",
        order=20, tdi="2nd generation", tdi_chan="AE")

    generator = GenerateEMRIWaveform(FastKerrEccentricEquatorialAccretionFlux,
                                     sum_kwargs=dict(pad_output=True, odd_len=True),
                                     return_list=False, use_gpu=use_gpu)
    response = ResponseWrapper(waveform_gen=generator, **response_kwargs)

    args = [cfg.injection[n] for n in PARAM_NAMES] + [cfg.disk["Sigma0"], cfg.disk["h0"],
                                                      cfg.disk["Sigma_p"]]
    signal = xp.array(response(*args))[:cfg.nchannels, :]

    PSD = xp.array(generate_PSD(waveform=signal, dt=cfg.dt, noise_PSD=get_sensitivity,
                                channels=CHANNELS, noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    freq = xp.fft.rfftfreq(signal.shape[-1], cfg.dt)
    mask = ((freq > cfg.fmin) & (freq < cfg.fmax))[1:]   # DC dropped, as inner_product wants

    def ip(x, y):
        v = inner_product(x, y, PSD, cfg.dt, freq_mask=mask, use_gpu=use_gpu)
        return float(v.get() if hasattr(v, "get") else v)

    return dict(signal=signal, ip=ip, signal_norm=ip(signal, signal),
                snr=float(np.sqrt(ip(signal, signal))), channels=CHANNELS,
                noise_kwargs=noise_kwargs, response_kwargs=response_kwargs)


def free_gpu():
    if use_gpu:
        xp.get_default_memory_pool().free_all_blocks()


def rescale_to_snr(cfgs, T):
    """Move dist so the injection has SNR = TARGET_SNR under the 2nd-gen PSD (SNR ~ 1 / dist)."""
    d0 = cfgs[-1].injection["dist"]
    snr0 = build_scoring(cfgs[-1], T)["snr"]
    free_gpu()
    d = d0 * snr0 / TARGET_SNR
    for cfg in cfgs:
        cfg.injection["dist"] = d
    scoring = build_scoring(cfgs[-1], T)
    log(f"[DIST] SNR at dist {d0:g} Gpc = {snr0:.4f} (2nd gen) -> dist = {d:.6f} Gpc, "
        f"SNR = {scoring['snr']:.6f}")
    return scoring, d0, snr0, d


# --- one LM climb ----------------------------------------------------------
def climb(cfg, scoring, T, template, stored_model, label):
    """One LM climb from the stored best fit, recorded with its start and end point."""
    model = make_model(cfg, scoring, T, SPECS[template](cfg))
    assert model["names"] == stored_model["names"], (model["names"], stored_model["names"])
    theta0 = np.array(stored_model["theta"], float)
    theta_inj = np.array(stored_model["theta_inj"], float)
    ov_start, chi2_start = model["ov"](theta0), model["chi2"](theta0)
    log(f"[START {label}] ov = {ov_start:.12f}  chi2 = {chi2_start:.6e}  "
        f"(1st-gen stored ov {stored_model['overlap']:.12f})")
    t0 = time.time()
    theta, sigma, history = lm_climb(cfg, model, scoring, theta0, tag=f" {label}")
    ov, chi2 = model["ov"](theta), model["chi2"](theta)
    log(f"[DONE  {label}] ov {ov_start:.12f} -> {ov:.12f}  chi2 {chi2_start:.6e} -> {chi2:.6e}  "
        f"its = {len(history)}  {time.time() - t0:.0f} s")
    log("    theta_final = [" + ", ".join(repr(float(v)) for v in theta) + "]")
    return dict(names=model["names"], theta_inj=theta_inj, theta_start=theta0, theta=theta,
                bias=theta - theta_inj,
                # fishinv of the last LM iteration's Fisher, NOT a Fisher at the final point
                sigma_lm_last_iter=sigma, ov_start=ov_start, chi2_start=chi2_start,
                overlap=ov, mismatch=1 - ov, chi2=chi2, overlap_1st_gen=stored_model["overlap"],
                history=history, seconds=time.time() - t0)


# --- driver ----------------------------------------------------------------
def save_json(records):
    payload = dict(
        study="environemen_2_eccen second_gen_TDI", generated_utc=datetime.now(timezone.utc).isoformat(),
        tdi="2nd generation", channels="AE", freq_band_max=FMAX, target_snr=TARGET_SNR,
        superkludge_branch=SUPERKLUDGE_BRANCH,
        note=("LM only (no Nelder-Mead) from the 1st-gen best fits in summary/summary.json, dist "
              "rescaled to SNR 200 under 2nd-gen TDI, band capped at 0.05 Hz. sigma_lm_last_iter "
              "is fishinv of the last LM iteration's Fisher, not a Fisher at the final point."),
        sets=records)
    with open(f"{JSON_PATH}.tmp", "w") as f:
        json.dump(jsonable(payload), f, indent=1)
    os.replace(f"{JSON_PATH}.tmp", JSON_PATH)


def run_set(tag, stored, records):
    log(f"\n{'=' * 90}\n[SET ] {tag}  ({stored['folder']})")
    cfgs = second_gen_configs(tag)
    check_same_source(*cfgs, stored)
    T = observation_time(cfgs[-1])
    scoring, d0, snr0, d = rescale_to_snr(cfgs, T)
    rec = dict(tag=tag, folder=stored["folder"], injection=dict(stored["injection"], dist=d),
               disk=stored["disk"], T=T, dt=cfgs[-1].dt, snr=scoring["snr"],
               dist_native=d0, snr_2ndgen_at_native_dist=snr0, dist=d, models={})
    records.append(rec)
    for template, cfg in zip(TEMPLATES, cfgs):
        rec["models"][template] = climb(cfg, scoring, T, template, stored["models"][template],
                                        f"{tag}/{template}")
        save_json(records)
    del scoring
    free_gpu()


def print_summary(records):
    log("\n" + "=" * 90)
    log(f"{'set':>6} {'model':>7} {'ov 1st gen':>16} {'ov start':>16} {'ov final':>16} {'chi2':>12}")
    for r in records:
        for template, m in r["models"].items():
            log(f"{r['tag']:>6} {template:>7} {m['overlap_1st_gen']:>16.12f} {m['ov_start']:>16.12f} "
                f"{m['overlap']:>16.12f} {m['chi2']:>12.4e}")


def main():
    check_branch()
    stored = load_best_fits()
    sets = selected_sets()
    log(f"[info] sets {sets}, 2nd-gen TDI AE, fmax {FMAX} Hz, SNR {TARGET_SNR:g}; "
        f"out -> {os.path.basename(JSON_PATH)}")
    records, t_all = [], time.time()
    for tag in sets:
        try:
            run_set(tag, stored[tag], records)
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            log(f"[FAIL SET {tag}] {type(exc).__name__}: {exc}")
    print_summary(records)
    save_json(records)
    log(f"\n[saved] {JSON_PATH}  ({time.time() - t_all:.1f} s total)")


if __name__ == "__main__":
    main()
