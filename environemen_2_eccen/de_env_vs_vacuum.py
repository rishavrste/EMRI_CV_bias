"""Fit an environmentally perturbed EMRI with a vacuum GR template.

Same case as test.ipynb.  The injection is an eccentric EMRI evolved through a
Duque et al. accretion disk; the template is the *same* waveform class with the
disk arguments dropped.  The offset of the best fit from the injection is the
systematic bias a vacuum search would incur.

Two stages:

1. Differential evolution in a box of +-sigma_range Fisher sigma about the
   injection, polished by scipy's L-BFGS-B.  DE is a population search, so
   unlike a local climber it can cross the ridge left by the ~22 rad of
   accumulated dephasing.
2. A Levenberg-Marquardt CV climb from the DE winner, using SEF's stable
   derivatives.  This is where the final digits come from.

Everything that defines the run -- source, disk, fit vector, band, DE and LM
controls -- lives in a JSON config; nothing here needs editing to run a new
case.  Results are written to JSON on every progress line, so a walltime kill
still leaves the best point found so far on disk.

Run:  python de_env_vs_vacuum.py [config.json]
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import differential_evolution

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import FastKerrEccentricEquatorialAccretionFlux
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqAccFlux
from few.utils.constants import YRSID_SI, MTSUN_SI
from few.utils.geodesic import get_separatrix

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product, fishinv
from stableemrifisher.fisher import StableEMRIFisher

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    xp, use_gpu = cp, True
except Exception as exc:
    xp, use_gpu = np, False
    print(f"[INFO] no usable GPU ({type(exc).__name__}), falling back to NumPy")


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config_de_env_vs_vacuum.json")

# The 14 positional waveform arguments, in the order the generator wants them.
PARAM_NAMES = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
               "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]

MSUN_G = 1.98892e33
C_CM_S = 2.99792458e10


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ config
class Config:
    """The JSON config, with the few derived quantities it implies.

    Keys beginning with an underscore are comments and are ignored.
    """

    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)

        self.path = path
        self.case = raw.get("case", "env-vs-vacuum")
        self.injection = raw["injection"]
        self.infer = raw["infer"]
        self.npar = len(self.infer)

        w = raw["waveform"]
        self.T, self.T_safety = w.get("T"), w["T_safety"]
        self.dt, self.nchannels = w["dt"], w["nchannels"]
        self.fmin, self.fmax = w["fmin"], w["fmax"]

        self.Ndelta = raw["fisher"]["Ndelta"]
        self.der_order = raw["fisher"]["der_order"]

        b = raw["bounds"]
        self.sigma_range = b["sigma_range"]
        self.floors, self.ceilings = b["floors"], b["ceilings"]
        self.p0_pad = b["p0_separatrix_pad"]

        de = {k: v for k, v in raw["de"].items() if not k.startswith("_")}
        self.report_every = de.pop("report_every")
        de["mutation"] = tuple(de["mutation"])       # JSON has no tuples
        self.de = de

        lm = raw["lm"]
        self.lambda0, self.lm_max_iters = lm["lambda0"], lm["max_iters"]
        self.max_inner, self.rel_tol = lm["max_inner"], lm["rel_tol"]
        self.recompute_deltas_every = lm["recompute_deltas_every"]
        self.overlap_target = lm["overlap_target"]

        # Sigma enters the ODE in geometric units (G = c = M = 1), where it
        # carries a length^2 as well as a mass -- dividing by the solar mass
        # alone leaves it ~1e11 times too small.
        r_g_cm = self.injection["m1"] * MTSUN_SI * C_CM_S
        self.sigma_cgs_to_geo = r_g_cm ** 2 / (self.injection["m1"] * MSUN_G)
        d = raw["disk"]
        self.disk = dict(Sigma0=d["Sigma0_cgs"] * self.sigma_cgs_to_geo,
                         h0=d["h0"], Sigma_p=d["Sigma_p"])

        # A relative output path belongs next to the config that named it.
        out = raw["output"]
        self.out_path = out if os.path.isabs(out) else os.path.join(
            os.path.dirname(os.path.abspath(path)), out)

        self.theta_inj = np.array([self.injection[n] for n in self.infer], float)

    def summary(self):
        return dict(case=self.case, config=self.path, injection=self.injection,
                    disk=self.disk, infer=self.infer, T=self.T, dt=self.dt,
                    nchannels=self.nchannels, fmin=self.fmin, fmax=self.fmax,
                    Ndelta=self.Ndelta, der_order=self.der_order,
                    sigma_range=self.sigma_range, de=self.de,
                    lambda0=self.lambda0, lm_max_iters=self.lm_max_iters)


def wave_params(cfg, theta, with_disk):
    """A fit vector -> the positional arguments the response wants.

    Anything outside cfg.infer stays pinned at its injected value, so the
    template differs from the signal only in what we are fitting and in the
    missing disk.
    """
    p = dict(cfg.injection, **dict(zip(cfg.infer, theta)))
    tail = [cfg.disk["Sigma0"], cfg.disk["h0"], cfg.disk["Sigma_p"]] if with_disk else []
    return [p[n] for n in PARAM_NAMES] + tail


def plunge_time(cfg):
    """Observation time from the trajectory, when the config asks for plunge.

    The environmental trajectory sets T so that both waveforms share one T.
    """
    traj = EMRIInspiral(func=KerrEccEqAccFlux)
    t = traj(*[cfg.injection[n] for n in ("m1", "m2", "a", "p0", "e0", "xI0")],
             cfg.disk["Sigma0"], cfg.disk["h0"], cfg.disk["Sigma_p"], T=10)[0]
    return float(t[-1]) / YRSID_SI


# ------------------------------------------------------------------ context
def build_context(cfg, T):
    """Signal, template, overlap, chi2 and the Fisher provider, as one dict.

    The signal is generated once and the PSD built from it, so every number
    below is scored on the same band and the same noise curve.
    """
    channels = [A1TDISens, E1TDISens, T1TDISens][:cfg.nchannels]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    response_kwargs = dict(
        Tobs=T, t0=10000.0, dt=cfg.dt, index_lambda=8, index_beta=7, flip_hx=True,
        is_ecliptic_latitude=False, remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend="cuda12x" if use_gpu else "cpu",
        order=20, tdi="1st generation", tdi_chan={2: "AE", 3: "AET"}[cfg.nchannels])

    generator = GenerateEMRIWaveform(FastKerrEccentricEquatorialAccretionFlux,
                                     sum_kwargs=dict(pad_output=True, odd_len=True),
                                     return_list=False, use_gpu=use_gpu)
    response = ResponseWrapper(waveform_gen=generator, **response_kwargs)
    n = cfg.nchannels

    def template(theta):
        """Vacuum template: the same waveform class, handed no disk arguments."""
        return xp.array(response(*wave_params(cfg, theta, with_disk=False)))[:n, :]

    signal = xp.array(response(*wave_params(cfg, cfg.theta_inj, with_disk=True)))[:n, :]

    PSD = xp.array(generate_PSD(waveform=signal, dt=cfg.dt, noise_PSD=get_sensitivity,
                                channels=channels, noise_kwargs=noise_kwargs,
                                use_gpu=use_gpu))
    freq = xp.fft.rfftfreq(signal.shape[-1], cfg.dt)
    mask = ((freq > cfg.fmin) & (freq < cfg.fmax))[1:]   # DC dropped, as inner_product wants

    def ip(x, y):
        v = inner_product(x, y, PSD, cfg.dt, freq_mask=mask, use_gpu=use_gpu)
        return float(v.get() if hasattr(v, "get") else v)

    signal_norm = ip(signal, signal)

    def overlap(theta):
        h = template(theta)
        return ip(signal, h) / np.sqrt(signal_norm * ip(h, h))

    def chi2(theta):
        """<r|r>.  A template that fails to generate is scored as unusable rather
        than raised, so one bad trial vector cannot take the run down."""
        try:
            r = signal - template(theta)
            return ip(r, r)
        except Exception as exc:
            log(f"    [warn] template failed at {theta}: {type(exc).__name__}: {exc}")
            return 1e30

    sef = StableEMRIFisher(
        waveform_class=FastKerrEccentricEquatorialAccretionFlux,
        waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=response_kwargs,
        stats_for_nerds=False, use_gpu=use_gpu, deriv_type="stable",
        noise_model=get_sensitivity, noise_kwargs=noise_kwargs, channels=channels,
        T=T, dt=cfg.dt, stability_plot=False, der_order=cfg.der_order,
        Ndelta=cfg.Ndelta, plunge_check=True, return_derivatives=True)

    def fisher(theta, deltas=None):
        """(Gamma, dh/dtheta, deltas) for the vacuum template at theta.

        add_param_args={} is what makes it the vacuum model: SEF differentiates
        the same waveform template() builds.  Passing `deltas` back in reuses the
        stencil, which is the expensive half of the call.  fmin/fmax go per call
        because plunge_check may shorten T, and a prebuilt mask would then be the
        wrong length.
        """
        p = dict(cfg.injection, **dict(zip(cfg.infer, theta)))
        F = sef(wave_params={k: p[k] for k in PARAM_NAMES}, param_names=cfg.infer,
                add_param_args={}, deltas=deltas, live_dangerously=False,
                stability_plot=False, der_order=cfg.der_order,
                Ndelta=(cfg.Ndelta if deltas is None else None),
                fmin=cfg.fmin, fmax=cfg.fmax)
        return np.asarray(F[-1], dtype=float), xp.array(F[0]), sef.deltas

    return dict(template=template, signal=signal, ip=ip, overlap=overlap, chi2=chi2,
                fisher=fisher, snr=float(np.sqrt(signal_norm)))


def sigma_of(theta, G):
    """1-sigma Fisher errors, in cfg.infer order."""
    return np.sqrt(np.abs(np.diag(fishinv(theta[0], G, index_of_M=0))))


def fisher_box(cfg, theta, sigma):
    """+-sigma_range about theta, clipped to what the waveform can actually take."""
    floors = dict(cfg.floors,
                  p0=float(get_separatrix(cfg.injection["a"], cfg.injection["e0"],
                                          cfg.injection["xI0"])) + cfg.p0_pad)
    bounds = []
    for name, value, s in zip(cfg.infer, theta, sigma):
        lo = max(value - cfg.sigma_range * s, floors.get(name, -np.inf))
        hi = min(value + cfg.sigma_range * s, cfg.ceilings.get(name, np.inf))
        log(f"[BOX ] {name:>9}  sigma = {s:.6e}  ->  [{lo:.10g}, {hi:.10g}]")
        bounds.append((lo, hi))
    return bounds


# ---------------------------------------------------------------- optimisers
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


def run_de(cfg, ctx, bounds, objective):
    """Differential evolution in the Fisher box, seeded with the injection.

    x0 = the injection puts the truth in the initial population, so DE can only
    improve on the starting overlap.
    """
    log(f"[DE  ] ndim = {cfg.npar}, " + ", ".join(f"{k} = {v}" for k, v in cfg.de.items()))
    t0 = time.time()
    res = differential_evolution(objective, bounds, x0=cfg.theta_inj, **cfg.de)
    theta = np.asarray(res.x, dtype=float)
    log(f"[DE  ] {res.nit} iters, {res.nfev} evals, {(time.time() - t0) / 3600:.2f} h "
        f"({res.message})")
    return dict(params=theta, nit=int(res.nit), nfev=int(res.nfev),
                success=bool(res.success), message=str(res.message),
                overlap=ctx["overlap"](theta), chi2=ctx["chi2"](theta),
                seconds=time.time() - t0)


def lm_climb(cfg, ctx, theta0):
    """Adaptive Levenberg-Marquardt CV climb from the DE winner.

    Each iteration is one damped Cutler-Vallisneri step,

        dtheta = (Gamma + lam diag Gamma)^-1 g ,   g_i = <d_i h | s - h>

    with Nielsen gain-ratio updates on lam: a step that lowers chi2 relaxes the
    damping, one that does not raises it and is retried.  DE has already found
    the basin, so this should converge in a few iterations -- a long walk here
    means DE had not actually converged.

    Returns (theta, sigma, history).
    """
    theta = np.array(theta0, dtype=float)
    lam, nu, deltas = cfg.lambda0, 2.0, None
    sigma, history = np.ones(cfg.npar), []

    for it in range(cfg.lm_max_iters):
        if it % cfg.recompute_deltas_every == 0:
            deltas = None                             # fresh stencil, reused in between
        G, dH, deltas = ctx["fisher"](theta, deltas)

        residual = ctx["signal"] - ctx["template"](theta)
        gradient = np.array([ctx["ip"](dH[j], residual) for j in range(cfg.npar)])
        sigma = sigma_of(theta, G)
        chi2, overlap = ctx["ip"](residual, residual), ctx["overlap"](theta)
        history.append(dict(it=it, lam=lam, overlap=overlap, chi2=chi2))

        if overlap > cfg.overlap_target:
            log(f"    [LM] it {it:>3}  target reached, ov = {overlap:.12f}")
            break

        # raise the damping until the step actually lowers chi2
        scale = np.abs(np.diag(G)) + 1e-30
        step, improvement = None, None
        for _ in range(cfg.max_inner):
            try:
                trial = np.linalg.solve(G + lam * np.diag(scale), gradient)
            except np.linalg.LinAlgError:
                lam, nu = lam * nu, nu * 2.0
                continue
            predicted = float(trial @ (gradient + lam * scale * trial))
            reduction = chi2 - ctx["chi2"](theta + trial)
            gain = reduction / predicted if predicted > 0 else -1.0
            if gain > 0.0:                            # accept, relax the damping
                lam, nu = lam * max(1 / 3, 1 - (2 * gain - 1) ** 3), 2.0
                step, improvement = trial, reduction / chi2
                break
            lam, nu = lam * nu, nu * 2.0              # reject, damp harder

        log(f"    [LM] it {it:>3}  lam = {lam:.1e}  ov = {overlap:.12f}  "
            f"chi2 = {chi2:.6e}  rel = {0.0 if improvement is None else improvement:.2e}")

        if step is None:
            # No damping helped.  After DE that means the winner is already at the
            # optimum, not that the climber is lost.
            log(f"    [LM] no damped step lowers chi2; stopping at iteration {it}")
            break
        theta += step
        if improvement < cfg.rel_tol:
            log(f"    [LM] converged: chi2 moved by less than {cfg.rel_tol:g}")
            break

    return theta, sigma, history


def run_lm(cfg, ctx, theta0):
    """The LM stage, packaged like run_de so both land in the same record."""
    log(f"[LM  ] climbing from the DE winner, lam0 = {cfg.lambda0:g}, "
        f"up to {cfg.lm_max_iters} iterations")
    t0 = time.time()
    theta, sigma, history = lm_climb(cfg, ctx, theta0)
    return dict(params=theta, sigma=sigma, history=history,
                overlap=ctx["overlap"](theta), chi2=ctx["chi2"](theta),
                seconds=time.time() - t0)


# ----------------------------------------------------------------- reporting
def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def save(cfg, record):
    with open(cfg.out_path, "w") as f:
        json.dump(jsonable(record), f, indent=1)


def report(cfg, ctx, stages, sigma):
    """Copy-ready: each stage's overlap, then the final per-parameter bias.

    `stages` is [(name, theta, overlap, chi2), ...] in the order they ran, the
    first being the injection itself.  sigma is the Fisher error at the final
    point, which is the right unit for a bias.
    """
    start, final = stages[0], stages[-1]
    gain = (1 - start[2]) / max(1 - final[2], 1e-300)

    log("\n" + "-" * 78)
    log(f"[RES ] SNR = {ctx['snr']:.6f}")
    for name, _, overlap, chi2 in stages:
        log(f"[RES ] {name:<10} ov = {overlap:.12f}  chi2 = {chi2:.6e}")
    log(f"[RES ] mismatch {1 - start[2]:.6e} -> {1 - final[2]:.6e}  ({gain:.2f}x reduction)")

    log("\n" + f"{'param':>10}" + "".join(f" {n:>18}" for n, *_ in stages)
        + f" {'bias':>15} {'sigma':>13} {'bias/sigma':>11}")
    for j, name in enumerate(cfg.infer):
        bias = final[1][j] - start[1][j]              # the box was centred on the injection
        log(f"{name:>10}" + "".join(f" {theta[j]:>18.10g}" for _, theta, *_ in stages)
            + f" {bias:>+15.6e} {sigma[j]:>13.5e} {bias / sigma[j]:>+11.3f}")
    log("\nbest-fit vector [" + ", ".join(f"{v:.10e}" for v in final[1]) + "]")


def main(config_path=DEFAULT_CONFIG):
    t_start = time.time()
    cfg = Config(config_path)
    log(f"[CONF] {config_path}")

    T = cfg.T if cfg.T is not None else cfg.T_safety * plunge_time(cfg)
    log(f"[CONF] T = {T:.6f} yr ({'given' if cfg.T is not None else f'{cfg.T_safety:g} x plunge'}), "
        f"dt = {cfg.dt} s, {cfg.nchannels} channels, "
        f"band [{cfg.fmin:.1e}, {cfg.fmax:.1e}] Hz")

    ctx = build_context(cfg, T)
    theta_inj = cfg.theta_inj
    injected = ("injection", theta_inj, ctx["overlap"](theta_inj), ctx["chi2"](theta_inj))
    log(f"[CTX ] SNR = {ctx['snr']:.6f}, overlap = {injected[2]:.12f}, "
        f"chi2 = {injected[3]:.6e}")

    log(f"[BOX ] Fisher at the injection, box = +-{cfg.sigma_range:g} sigma")
    G, _, _ = ctx["fisher"](theta_inj)
    sigma_inj = sigma_of(theta_inj, G)
    bounds = fisher_box(cfg, theta_inj, sigma_inj)

    record = dict(config=cfg.summary(),
                  when=datetime.now(timezone.utc).isoformat(), T=T,
                  snr=ctx["snr"], theta_inj=theta_inj, sigma_inj=sigma_inj,
                  bounds=bounds, overlap_inj=injected[2], chi2_inj=injected[3],
                  status="running")

    def checkpoint(objective):
        record["de_partial"] = dict(nfev=objective.n, best_chi2=objective.best,
                                    best_params=objective.best_x)
        save(cfg, record)

    save(cfg, record)
    de = run_de(cfg, ctx, bounds, Objective(ctx["chi2"], checkpoint, cfg.report_every))
    record.update(de=de, status="de_done")
    save(cfg, record)                   # DE survives a kill during the climb below

    lm = run_lm(cfg, ctx, de["params"])
    record.update(lm=lm, status="done", seconds=time.time() - t_start)
    save(cfg, record)

    report(cfg, ctx, [injected,
                      ("DE", de["params"], de["overlap"], de["chi2"]),
                      ("DE+LM", lm["params"], lm["overlap"], lm["chi2"])], lm["sigma"])
    log(f"[saved] {cfg.out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
