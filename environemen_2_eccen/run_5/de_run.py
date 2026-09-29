"""DE (+-5 sigma about the injection) -> LM, for the notebook's environmental source.

The same two-stage fit as ../de_env_vs_pn.py, driven by a config in this folder:

1. Differential evolution in a box of +-sigma_range Fisher sigma about the
   injection, deviations centred on zero.  The parameters listed under
   bounds.full_period instead span one whole period, [inj - pi, inj + pi].
2. A Levenberg-Marquardt CV climb from the DE winner.

Then the azimuthal and radial dephasing of the injection, the DE winner and the
final fit against the environmental signal (dephasing.py).

Run:  python de_run.py config_de_0pa.json      (0PA = vacuum GR)
      python de_run.py config_de_0pa_pn.json   (0PA + 2.5PN deviation)
"""

import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # env_cv and de_env_vs_pn live one up

from env_cv import (Config, SPECS, build_scoring, make_model, baseline_overlap,  # noqa: E402
                    observation_time, sigma_of, fisher_box, run_climb, save,
                    report, log)
from de_env_vs_pn import Objective, run_de                                        # noqa: E402
from dephasing import dephasing_at, format_dephasing                              # noqa: E402


def search_box(cfg, model, theta, sigma):
    """The +-sigma Fisher box, with the periodic parameters opened to one period."""
    bounds = fisher_box(cfg, model, theta, sigma)
    for name in cfg.raw["bounds"].get("full_period", []):
        j = model["names"].index(name)
        bounds[j] = (theta[j] - np.pi, theta[j] + np.pi)
        log(f"[BOX ] {name:>9}  full period  ->  [{bounds[j][0]:.10g}, {bounds[j][1]:.10g}]")
    return bounds


def dephasing_table(cfg, template, model, spec, T, stages):
    """{stage: dephasing} for every (name, theta) stage, logged as it goes."""
    out = {}
    for name, theta in stages:
        out[name] = dephasing_at(cfg, template, model["names"], theta, spec["tail"], T)
        log(format_dephasing(name, out[name]))
    return out


def main(config_path):
    t_start = time.time()
    cfg = Config(config_path)
    log(f"[CONF] {config_path}")
    template = cfg.raw["template"]

    T = observation_time(cfg)
    scoring = build_scoring(cfg, T)
    spec = SPECS[template](cfg)
    model = make_model(cfg, scoring, T, spec)
    log(f"[MODEL] {model['name']}: fitting {model['names']}")
    if cfg.pn:
        log(f"[MODEL] ASSUMED secondary spin chi2 = {cfg.pn['chi2_secondary']}, "
            f"evolve_1PA = {cfg.pn['evolve_1PA']}")

    theta_seed = model["theta_inj"]
    ov0 = baseline_overlap(cfg, scoring, model)
    chi2_0 = model["chi2"](theta_seed)
    log(f"[CTX ] SNR = {scoring['snr']:.6f}, overlap = {ov0:.12f}, chi2 = {chi2_0:.6e}")

    log(f"[BOX ] Fisher at the injection, box = +-{cfg.sigma_range:g} sigma")
    G, _, _ = model["fisher"](theta_seed)
    sigma_seed = sigma_of(theta_seed, G)
    bounds = search_box(cfg, model, theta_seed, sigma_seed)

    record = dict(config=cfg.summary(), model=model["name"], names=model["names"],
                  when=datetime.now(timezone.utc).isoformat(), T=T,
                  snr=scoring["snr"], theta_inj=theta_seed, sigma_inj=sigma_seed,
                  bounds=bounds, overlap_seed=ov0, chi2_seed=chi2_0, status="running")

    def checkpoint(objective):
        record["de_partial"] = dict(nfev=objective.n, best_chi2=objective.best,
                                    best_params=objective.best_x)
        save(cfg, record)

    save(cfg, record)
    de = run_de(cfg, model, bounds, Objective(model["chi2"], checkpoint, cfg.report_every))
    record.update(de=de, status="de_done")
    save(cfg, record)

    lm = run_climb(cfg, model, scoring, de["params"], label="LM  ")
    record.update(lm=lm, status="lm_done")
    save(cfg, record)

    record["dephasing"] = dephasing_table(
        cfg, template, model, spec, T,
        [("injection", theta_seed), ("DE", de["params"]), ("DE+LM", lm["params"])])
    record.update(status="done", seconds=time.time() - t_start)
    save(cfg, record)

    report(model, scoring,
           [("seed (inj)", theta_seed, ov0, chi2_0),
            ("DE", de["params"], de["overlap"], de["chi2"]),
            ("DE+LM", lm["params"], lm["overlap"], lm["chi2"])],
           lm["sigma"], theta_seed)
    log(f"[saved] {cfg.out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
