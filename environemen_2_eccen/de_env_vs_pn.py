"""Fit the environmentally perturbed EMRI with a 0PA + 2.5PN-deviation template.

The companion of de_env_vs_vacuum.py, same case, same injection, same band --
the only change is the template.  Instead of vacuum GR it is SuperKludge 0PA
with the 2.5PN flux deviation switched on, so the fit vector carries two extra
parameters, C_p (dev_1) and C_e (dev_2).  The question is how much of the disk's
imprint that extra freedom can absorb, and where it drags the intrinsics.

Two stages, as before:

1. Differential evolution in a box of +-sigma_range Fisher sigma about the
   injection (deviations centred on zero, i.e. on GR).  DE is a population
   search, so unlike a local climber it can cross the ridge left by the ~22 rad
   of accumulated dephasing.
2. A Levenberg-Marquardt CV climb from the DE winner, using SEF's stable
   derivatives.  This is where the final digits come from.

Read the result against the baseline check this prints first: the disk lives
only in KerrEccEqAccFlux and the deviation only in SuperKludgeFlux, so signal
and template are different waveform classes and part of any PN gain is the class
difference rather than the disk.

Run:  python de_env_vs_pn.py [config.json]
"""

import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import differential_evolution

from env_cv import (Config, SPECS, build_scoring, make_model, baseline_overlap,
                    observation_time, sigma_of, fisher_box, run_climb, save,
                    report, log)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config_de_env_vs_pn.json")


class Objective:
    """chi2, with a progress line and a JSON checkpoint every report_every calls.

    scipy's differential_evolution is silent while it runs and one call here is a
    full waveform plus the LISA response, so without this there is no telling a
    slow run from a hung one -- nor costing the next one.
    """

    def __init__(self, chi2, checkpoint, report_every):
        self.chi2, self.checkpoint, self.report_every = chi2, checkpoint, report_every
        self.n, self.best, self.best_x, self.t0 = 0, np.inf, None, time.time()

    def __call__(self, theta):
        value = self.chi2(theta)
        self.n += 1
        if value < self.best:
            self.best, self.best_x = value, np.array(theta, dtype=float)
        if self.n % self.report_every == 0:
            elapsed = time.time() - self.t0
            log(f"    [DE] {self.n:>7} evals  {elapsed / 3600:6.2f} h  "
                f"{elapsed / self.n:5.2f} s/eval  best chi2 = {self.best:.6e}")
            self.checkpoint(self)
        return value


def run_de(cfg, model, bounds, objective):
    """Differential evolution in the Fisher box, seeded with the injection.

    x0 = the injection (with the deviations at zero) puts GR in the initial
    population, so DE can only improve on the GR starting overlap.
    """
    log(f"[DE  ] ndim = {model['npar']}, "
        + ", ".join(f"{k} = {v}" for k, v in cfg.de.items()))
    t0 = time.time()
    res = differential_evolution(objective, bounds, x0=model["theta_inj"], **cfg.de)
    theta = np.asarray(res.x, dtype=float)
    log(f"[DE  ] {res.nit} iters, {res.nfev} evals, {(time.time() - t0) / 3600:.2f} h "
        f"({res.message})")
    return dict(params=theta, nit=int(res.nit), nfev=int(res.nfev),
                success=bool(res.success), message=str(res.message),
                overlap=model["ov"](theta), chi2=model["chi2"](theta),
                seconds=time.time() - t0)


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

    theta_seed = model["theta_inj"]
    ov0 = baseline_overlap(cfg, scoring, model)
    seed_stage = ("seed (GR)", theta_seed, ov0, model["chi2"](theta_seed))
    log(f"[CTX ] SNR = {scoring['snr']:.6f}, overlap = {ov0:.12f}, "
        f"chi2 = {seed_stage[3]:.6e}")

    log(f"[BOX ] Fisher at the seed, box = +-{cfg.sigma_range:g} sigma")
    G, _, _ = model["fisher"](theta_seed)
    sigma_seed = sigma_of(theta_seed, G)
    bounds = fisher_box(cfg, model, theta_seed, sigma_seed)

    record = dict(config=cfg.summary(), model=model["name"], names=model["names"],
                  when=datetime.now(timezone.utc).isoformat(), T=T,
                  snr=scoring["snr"], theta_inj=theta_seed, sigma_inj=sigma_seed,
                  bounds=bounds, overlap_seed=ov0, chi2_seed=seed_stage[3],
                  status="running")

    def checkpoint(objective):
        record["de_partial"] = dict(nfev=objective.n, best_chi2=objective.best,
                                    best_params=objective.best_x)
        save(cfg, record)

    save(cfg, record)
    de = run_de(cfg, model, bounds, Objective(model["chi2"], checkpoint, cfg.report_every))
    record.update(de=de, status="de_done")
    save(cfg, record)                   # DE survives a kill during the climb below

    lm = run_climb(cfg, model, scoring, de["params"], label="LM  ")
    record.update(lm=lm, status="done", seconds=time.time() - t_start)
    save(cfg, record)

    report(model, scoring,
           [seed_stage,
            ("DE", de["params"], de["overlap"], de["chi2"]),
            ("DE+LM", lm["params"], lm["overlap"], lm["chi2"])],
           lm["sigma"], theta_seed)
    log(f"[saved] {cfg.out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
