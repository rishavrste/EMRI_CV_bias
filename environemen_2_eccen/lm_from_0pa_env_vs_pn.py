"""LM climb of the 0PA + 2.5PN-deviation template, started at the 0PA best fit.

Same case as de_env_vs_pn.py -- same environmental injection, same template,
same band -- but no global search.  The climb starts where the vacuum run
finished, with both deviation coefficients at zero, which is the house
`from_0PA` seed:

    * It guarantees the answer ends at least as good as the vacuum fit, since
      vacuum GR is nested at dev = 0.
    * It is the seed that characteristically STALLS, because at the 0PA minimum
      the residual is orthogonal to the template derivatives and
      <dh/d(dev) | r> ~ 0.  There the LM step vanishes for any damping.  That is
      a basin problem, not a step-size one, so the Nelder-Mead escape in the
      config is doing real work here and should be left on.

Comparing this endpoint with the DE run's says whether the DE winner is the
global optimum or merely a better basin: DE above, a local climb from a known
good point here, and the higher overlap wins.

Run:  python lm_from_0pa_env_vs_pn.py [config.json]
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

from env_cv import (Config, SPECS, build_scoring, make_model, baseline_overlap,
                    observation_time, run_climb, save, report, log)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config_lm_from_0pa_env_vs_pn.json")


def load_seed(cfg, model):
    """The starting vector: the 0PA best fit, with the deviations appended at zero.

    Either given outright in the config, or read from the vacuum run's results
    JSON.  Reading it back rather than hard-coding it means this script always
    starts from whatever the vacuum run actually converged to.
    """
    s = cfg.seed_cfg
    if s.get("params"):
        theta = np.asarray(s["params"], dtype=float)
        log(f"[SEED] from the config: {theta}")
        return theta, dict(source="config")

    path = s["file"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(cfg.path)), path)
    with open(path) as f:
        prev = json.load(f)

    stage = prev[s["stage"]]
    intrinsic = np.asarray(stage["params"], dtype=float)
    if len(intrinsic) != len(cfg.infer):
        raise ValueError(f"{path} stage '{s['stage']}' has {len(intrinsic)} parameters, "
                         f"but this config infers {len(cfg.infer)}: {cfg.infer}")

    theta = np.concatenate([intrinsic, np.asarray(s["deviation"], dtype=float)])
    if len(theta) != model["npar"]:
        raise ValueError(f"seed has {len(theta)} entries, model wants {model['npar']} "
                         f"({model['names']})")

    log(f"[SEED] {os.path.basename(path)} stage '{s['stage']}': "
        f"vacuum overlap was {stage['overlap']:.12f}, chi2 {stage['chi2']:.6e}")
    log("[SEED] " + ", ".join(f"{n}={v:.10g}" for n, v in zip(model["names"], theta)))
    return theta, dict(source=path, stage=s["stage"],
                       vacuum_overlap=stage["overlap"], vacuum_chi2=stage["chi2"],
                       vacuum_params=intrinsic.tolist())


def main(config_path=DEFAULT_CONFIG):
    t_start = time.time()
    cfg = Config(config_path)
    log(f"[CONF] {config_path}")

    T = observation_time(cfg)
    scoring = build_scoring(cfg, T)
    spec = SPECS[cfg.raw.get("template", "pn")](cfg)
    model = make_model(cfg, scoring, T, spec)
    log(f"[MODEL] {model['name']}: fitting {model['names']}")
    if cfg.pn:
        log(f"[MODEL] ASSUMED secondary spin chi2 = {cfg.pn['chi2_secondary']}, "
            f"evolve_1PA = {cfg.pn['evolve_1PA']}")

    # Measured, not assumed: how much of the mismatch is the waveform class
    # rather than the disk.  See the note at the top of env_cv.py.
    ov_baseline = baseline_overlap(cfg, scoring, model)
    log(f"[CTX ] SNR = {scoring['snr']:.6f}")

    theta0, seed_info = load_seed(cfg, model)
    seed_stage = ("seed (0PA)", theta0, model["ov"](theta0), model["chi2"](theta0))
    log(f"[SEED] this template at that point: ov = {seed_stage[2]:.12f}, "
        f"chi2 = {seed_stage[3]:.6e}")

    record = dict(config=cfg.summary(), model=model["name"], names=model["names"],
                  when=datetime.now(timezone.utc).isoformat(), T=T,
                  snr=scoring["snr"], theta_inj=model["theta_inj"],
                  overlap_baseline=ov_baseline, seed=seed_info, theta_seed=theta0,
                  overlap_seed=seed_stage[2], chi2_seed=seed_stage[3],
                  status="running")
    save(cfg, record)

    lm = run_climb(cfg, model, scoring, theta0, label="LM  ")
    record.update(lm=lm, status="done", seconds=time.time() - t_start)
    save(cfg, record)

    # The bias is measured against the injection, not against the seed: the seed
    # is already displaced from the truth by the vacuum fit's own bias.
    report(model, scoring,
           [seed_stage, ("LM", lm["params"], lm["overlap"], lm["chi2"])],
           lm["sigma"], model["theta_inj"])
    if lm["overlap"] < seed_stage[2] - 1e-12:
        log("[WARN] the climb ended BELOW its seed.  Vacuum GR is nested at "
            "deviation = 0, so this is a convergence artifact on the flat "
            "deviation ridge, not physics.")
    log(f"[saved] {cfg.out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
