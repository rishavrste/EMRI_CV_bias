"""Shared machinery for fitting the environmentally perturbed EMRI.

The injection never changes: an eccentric EMRI evolved through a Duque et al.
accretion disk.  Only the *template* changes, and a template is described here
by a small `spec` -- its waveform class, the extra arguments it takes, and the
extra parameters it adds to the fit vector.  Two specs are provided:

    vacuum_spec  the same waveform class as the signal, handed no disk
                 arguments.  This is the baseline bias.
    pn_spec      SuperKludge 0PA with the 2.5PN flux deviation switched on,
                 fitting C_p (dev_1) and C_e (dev_2) on top of the intrinsics.

THE STRUCTURAL CAVEAT, stated once here because it shapes every result below:
the disk lives only in KerrEccEqAccFlux and the PN deviation only in
SuperKludgeFlux, so the signal and the PN template come from DIFFERENT waveform
classes.  A PN fit therefore absorbs the disk *plus* whatever the two vacuum
baselines differ by.  `baseline_overlap()` measures that second part directly;
run it before reading anything into a recovered C_p.

Everything a run needs lives in a JSON config; the drivers that import this
module are thin.
"""

import json
import os
import time

import numpy as np

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import (FastKerrEccentricEquatorialAccretionFlux,
                                   SuperKludgeWaveform)
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
except Exception as exc:                                    # noqa: BLE001
    xp, use_gpu = np, False
    print(f"[INFO] no usable GPU ({type(exc).__name__}), falling back to NumPy")

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
        self.raw = raw
        self.case = raw.get("case", "env")
        self.injection = raw["injection"]
        self.infer = raw["infer"]                 # the intrinsic parameters only

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

        if "de" in raw:
            de = {k: v for k, v in raw["de"].items() if not k.startswith("_")}
            self.report_every = de.pop("report_every")
            de["mutation"] = tuple(de["mutation"])       # JSON has no tuples
            self.de = de

        lm = raw["lm"]
        self.lambda0, self.lm_max_iters = lm["lambda0"], lm["max_iters"]
        self.max_inner, self.rel_tol = lm["max_inner"], lm["rel_tol"]
        self.recompute_deltas_every = lm["recompute_deltas_every"]
        self.overlap_target = lm["overlap_target"]

        nm = raw.get("nm", {})
        self.nm_enabled = nm.get("enabled", False)
        self.nm_maxiter, self.nm_step = nm.get("maxiter", 1000), nm.get("step", 2.0)

        self.pn = raw.get("pn", {})
        self.seed_cfg = raw.get("seed", {})

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
        return {k: v for k, v in self.raw.items() if not k.startswith("_")} | dict(
            config=self.path, disk_geometric=self.disk)


def plunge_time(cfg):
    """Observation time from the trajectory, when the config asks for plunge.

    The *environmental* trajectory sets T, so signal and template share one T
    no matter which template is being fitted.
    """
    traj = EMRIInspiral(func=KerrEccEqAccFlux)
    t = traj(*[cfg.injection[n] for n in ("m1", "m2", "a", "p0", "e0", "xI0")],
             cfg.disk["Sigma0"], cfg.disk["h0"], cfg.disk["Sigma_p"], T=10)[0]
    return float(t[-1]) / YRSID_SI


def observation_time(cfg):
    T = cfg.T if cfg.T is not None else cfg.T_safety * plunge_time(cfg)
    log(f"[CONF] T = {T:.6f} yr "
        f"({'given' if cfg.T is not None else f'{cfg.T_safety:g} x plunge'}), "
        f"dt = {cfg.dt} s, {cfg.nchannels} channels, "
        f"band [{cfg.fmin:.1e}, {cfg.fmax:.1e}] Hz")
    return T


# ------------------------------------------------------------- template specs
def vacuum_spec(cfg):
    """The same waveform class as the signal, with the disk arguments dropped."""
    return dict(name="vacuum GR",
                waveform_class=FastKerrEccentricEquatorialAccretionFlux,
                extra_names=[],
                tail=lambda theta: [],
                apa=lambda theta: {})


def pn_spec(cfg):
    """SuperKludge 0PA + 2.5PN flux deviation.

    SuperKludgeFlux reads its additional_args POSITIONALLY:

      [0] chi2 (secondary spin)  [1] evolve_1PA   [2] evolve_primary
      [3] evolve_2PA             [4] deviation_included   <- master switch
      [5] C_p = dev_1   [6] C_e = dev_2      <- 2.5PN, additive on pdot/edot
      [7] del_0_p       [8] del_0_e          <- multiplicative family, held at 0

    Get the order wrong and the wrong model is fitted silently, so the tail and
    the SEF dict are written side by side here and nowhere else.  SEF appends
    add_param_args in dict order, so that dict order IS the slot assignment.
    """
    chi2_sec = cfg.pn["chi2_secondary"]
    evolve_1PA = cfg.pn["evolve_1PA"]

    def tail(theta):
        C_p, C_e = theta[-2], theta[-1]
        return [chi2_sec, evolve_1PA, False, False, True, C_p, C_e, 0.0, 0.0]

    def apa(theta):
        C_p, C_e = theta[-2], theta[-1]
        return {"chi2": chi2_sec, "evolve_1PA": evolve_1PA, "evolve_primary": False,
                "evolve_2PA": False, "deviation_included": True,
                "dev_1": C_p, "dev_2": C_e, "del_0_p": 0.0, "del_0_e": 0.0}

    return dict(name="0PA + 2.5PN deviation", waveform_class=SuperKludgeWaveform,
                extra_names=["dev_1", "dev_2"], tail=tail, apa=apa)


SPECS = {"vacuum": vacuum_spec, "pn": pn_spec}


# ------------------------------------------------------------------ scoring
def build_scoring(cfg, T):
    """The injection, the inner product it defines, and the response settings.

    The signal is generated once and the PSD built from it, so every number in
    the run is scored on the same band and the same noise curve.
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

    p = dict(cfg.injection)
    args = [p[n] for n in PARAM_NAMES] + [cfg.disk["Sigma0"], cfg.disk["h0"],
                                          cfg.disk["Sigma_p"]]
    signal = xp.array(response(*args))[:cfg.nchannels, :]

    PSD = xp.array(generate_PSD(waveform=signal, dt=cfg.dt, noise_PSD=get_sensitivity,
                                channels=channels, noise_kwargs=noise_kwargs,
                                use_gpu=use_gpu))
    freq = xp.fft.rfftfreq(signal.shape[-1], cfg.dt)
    mask = ((freq > cfg.fmin) & (freq < cfg.fmax))[1:]   # DC dropped, as inner_product wants

    def ip(x, y):
        v = inner_product(x, y, PSD, cfg.dt, freq_mask=mask, use_gpu=use_gpu)
        return float(v.get() if hasattr(v, "get") else v)

    return dict(signal=signal, ip=ip, signal_norm=ip(signal, signal),
                snr=float(np.sqrt(ip(signal, signal))), channels=channels,
                noise_kwargs=noise_kwargs, response_kwargs=response_kwargs)


# ------------------------------------------------------------------- model
def make_model(cfg, scoring, T, spec):
    """Assemble one template family into the dict the climbers expect.

    Bundling template / overlap / chi2 / fisher behind one interface is what
    lets the LM and NM code below be written once and serve every spec.
    """
    names = list(cfg.infer) + list(spec["extra_names"])
    response_kwargs = scoring["response_kwargs"]

    generator = GenerateEMRIWaveform(spec["waveform_class"],
                                     sum_kwargs=dict(pad_output=True, odd_len=True),
                                     return_list=False, use_gpu=use_gpu)
    response = ResponseWrapper(waveform_gen=generator, **response_kwargs)

    def base(theta):
        """The 14 standard parameters at theta; anything not fitted stays injected.

        Names outside PARAM_NAMES -- the deviation coefficients -- are model
        arguments and reach the waveform through tail/apa instead.
        """
        p = dict(cfg.injection)
        p.update({n: v for n, v in zip(names, theta) if n in PARAM_NAMES})
        return p

    def template(theta):
        args = [base(theta)[n] for n in PARAM_NAMES] + list(spec["tail"](theta))
        return xp.array(response(*args))[:cfg.nchannels, :]

    def overlap(theta):
        h = template(theta)
        return scoring["ip"](scoring["signal"], h) / np.sqrt(
            scoring["signal_norm"] * scoring["ip"](h, h))

    def chi2(theta):
        """<r|r>.  A template that fails to generate is scored as unusable rather
        than raised, so one bad trial vector cannot take the run down."""
        try:
            r = scoring["signal"] - template(theta)
            return scoring["ip"](r, r)
        except Exception as exc:                            # noqa: BLE001
            log(f"    [warn] {spec['name']} template failed at {theta}: "
                f"{type(exc).__name__}: {exc}")
            return 1e30

    sef = StableEMRIFisher(
        waveform_class=spec["waveform_class"],
        waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=response_kwargs,
        stats_for_nerds=False, use_gpu=use_gpu, deriv_type="stable",
        noise_model=get_sensitivity, noise_kwargs=scoring["noise_kwargs"],
        channels=scoring["channels"], T=T, dt=cfg.dt, stability_plot=False,
        der_order=cfg.der_order, Ndelta=cfg.Ndelta, plunge_check=True,
        return_derivatives=True)

    def fisher(theta, deltas=None):
        """(Gamma, dh/dtheta, deltas) for THIS template at theta.

        Passing the previous `deltas` back in reuses the stencil, which is the
        expensive half of an SEF call; None forces a fresh stability scan.
        fmin/fmax go per call rather than a prebuilt mask because plunge_check
        may shorten T, and a prebuilt mask would then be the wrong length.
        """
        p = base(theta)
        F = sef(wave_params={n: p[n] for n in PARAM_NAMES}, param_names=names,
                add_param_args=spec["apa"](theta), deltas=deltas,
                live_dangerously=False, stability_plot=False,
                der_order=cfg.der_order,
                Ndelta=(cfg.Ndelta if deltas is None else None),
                fmin=cfg.fmin, fmax=cfg.fmax)
        return np.asarray(F[-1], dtype=float), xp.array(F[0]), sef.deltas

    return dict(name=spec["name"], names=names, npar=len(names),
                template=template, ov=overlap, chi2=chi2, fisher=fisher,
                theta_inj=np.concatenate([cfg.theta_inj,
                                          np.zeros(len(spec["extra_names"]))]))


def baseline_overlap(cfg, scoring, model):
    """How much of this template's mismatch is the waveform class, not the disk.

    At dev = 0 the PN template is plain SuperKludge 0PA, so overlapping it with
    the injection's own parameters isolates the class difference.  Compare it
    with the vacuum template's overlap at the same point: if the two agree, a
    later PN gain is about the disk; if this one is markedly worse, that much of
    any recovered deviation is absorbing the class difference instead.
    """
    ov = model["ov"](model["theta_inj"])
    log(f"[BASE] {model['name']} at the injected parameters (deviation = 0): "
        f"overlap = {ov:.12f}")
    return ov


# ------------------------------------------------------------------- bounds
def sigma_of(theta, G):
    """1-sigma Fisher errors, in model["names"] order."""
    return np.sqrt(np.abs(np.diag(fishinv(theta[0], G, index_of_M=0))))


def fisher_box(cfg, model, theta, sigma):
    """+-sigma_range about theta, clipped to what the waveform can actually take."""
    floors = dict(cfg.floors,
                  p0=float(get_separatrix(cfg.injection["a"], cfg.injection["e0"],
                                          cfg.injection["xI0"])) + cfg.p0_pad)
    bounds = []
    for name, value, s in zip(model["names"], theta, sigma):
        lo = max(value - cfg.sigma_range * s, floors.get(name, -np.inf))
        hi = min(value + cfg.sigma_range * s, cfg.ceilings.get(name, np.inf))
        log(f"[BOX ] {name:>9}  sigma = {s:.6e}  ->  [{lo:.10g}, {hi:.10g}]")
        bounds.append((lo, hi))
    return bounds


# ---------------------------------------------------------------- optimisers
def lm_climb(cfg, model, scoring, theta0, tag=""):
    """Adaptive Levenberg-Marquardt CV climb.

    Each iteration is one damped Cutler-Vallisneri step,

        dtheta = (Gamma + lam diag Gamma)^-1 g ,   g_i = <d_i h | s - h>

    with Nielsen gain-ratio updates on lam: a step that lowers chi2 relaxes the
    damping, one that does not raises it and is retried.

    Returns (theta, sigma, history).
    """
    theta = np.array(theta0, dtype=float)
    lam, nu, deltas = cfg.lambda0, 2.0, None
    sigma, history = np.ones(model["npar"]), []

    for it in range(cfg.lm_max_iters):
        if it % cfg.recompute_deltas_every == 0:
            deltas = None                             # fresh stencil, reused in between
        G, dH, deltas = model["fisher"](theta, deltas)

        residual = scoring["signal"] - model["template"](theta)
        gradient = np.array([scoring["ip"](dH[j], residual) for j in range(model["npar"])])
        sigma = sigma_of(theta, G)
        chi2 = scoring["ip"](residual, residual)
        overlap = model["ov"](theta)
        history.append(dict(it=it, lam=lam, overlap=overlap, chi2=chi2))

        if overlap > cfg.overlap_target:
            log(f"    [LM{tag}] it {it:>3}  target reached, ov = {overlap:.12f}")
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
            reduction = chi2 - model["chi2"](theta + trial)
            gain = reduction / predicted if predicted > 0 else -1.0
            if gain > 0.0:                            # accept, relax the damping
                lam, nu = lam * max(1 / 3, 1 - (2 * gain - 1) ** 3), 2.0
                step, improvement = trial, reduction / chi2
                break
            lam, nu = lam * nu, nu * 2.0              # reject, damp harder

        log(f"    [LM{tag}] it {it:>3}  lam = {lam:.1e}  ov = {overlap:.12f}  "
            f"chi2 = {chi2:.6e}  rel = {0.0 if improvement is None else improvement:.2e}")

        if step is None:
            log(f"    [LM{tag}] no damped step lowers chi2; stopping at iteration {it}")
            break
        theta = theta + step
        if improvement < cfg.rel_tol:
            log(f"    [LM{tag}] converged: chi2 moved by less than {cfg.rel_tol:g}")
            break

    return theta, sigma, history


def nm_refine(cfg, model, theta0, sigma, tag=""):
    """One Nelder-Mead escape, in sigma-scaled coordinates.

    Where <dh/dtheta | r> ~ 0 the LM step vanishes for ANY damping -- that is a
    basin problem, not a step-size one, which is what this is for.  The result is
    accepted only if it improves the overlap, so the escape can never make the
    answer worse.
    """
    from scipy.optimize import minimize

    scale = np.where(np.abs(sigma) > 0, np.abs(sigma), 1.0)
    ov0 = model["ov"](theta0)

    def objective(u):
        return model["chi2"](theta0 + u * scale)

    log(f"    [NM{tag}] escaping from ov = {ov0:.12f}, up to {cfg.nm_maxiter} evals")
    res = minimize(objective, np.zeros(model["npar"]), method="Nelder-Mead",
                   options=dict(maxiter=cfg.nm_maxiter, maxfev=cfg.nm_maxiter,
                                xatol=1e-10, fatol=1e-12,
                                initial_simplex=_simplex(model["npar"], cfg.nm_step)))
    theta = theta0 + np.asarray(res.x, dtype=float) * scale
    ov = model["ov"](theta)
    if ov > ov0:
        log(f"    [NM{tag}] accepted: ov {ov0:.12f} -> {ov:.12f}")
        return theta, True
    log(f"    [NM{tag}] rejected: ov {ov0:.12f} -> {ov:.12f}, keeping the LM point")
    return np.array(theta0, dtype=float), False


def _simplex(npar, step):
    """A simplex `step` sigma on a side, centred on the current point."""
    s = np.zeros((npar + 1, npar))
    s[1:] = step * np.eye(npar)
    return s


def run_climb(cfg, model, scoring, theta0, label="climb"):
    """The house recipe: CV -> (if stalled below target) one NM escape -> CV again.

    Returns a record ready to drop into the results JSON.
    """
    t0 = time.time()
    log(f"[{label}] {model['name']}, {model['npar']} parameters, "
        f"lam0 = {cfg.lambda0:g}, up to {cfg.lm_max_iters} iterations")
    log(f"[{label}] seed: " + ", ".join(f"{n}={v:.10g}"
                                        for n, v in zip(model["names"], theta0)))

    theta, sigma, history = lm_climb(cfg, model, scoring, theta0, tag="1")
    ov = model["ov"](theta)
    escaped = False

    if cfg.nm_enabled and ov <= cfg.overlap_target:
        theta_nm, escaped = nm_refine(cfg, model, theta, sigma, tag="")
        if escaped:
            theta, sigma, h2 = lm_climb(cfg, model, scoring, theta_nm, tag="2")
            history = history + [dict(nm=True)] + h2
            ov = model["ov"](theta)

    return dict(params=theta, sigma=sigma, history=history, nm_used=escaped,
                overlap=ov, chi2=model["chi2"](theta), seconds=time.time() - t0)


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


def report(model, scoring, stages, sigma, theta_ref):
    """Copy-ready: each stage's overlap, then the final per-parameter bias.

    `stages` is [(name, theta, overlap, chi2), ...] in the order they ran, the
    first being the seed.  `theta_ref` is what the bias is measured against --
    the injection, including zero for any deviation coefficient.  sigma is the
    Fisher error at the final point, which is the right unit for a bias.
    """
    start, final = stages[0], stages[-1]
    gain = (1 - start[2]) / max(1 - final[2], 1e-300)

    log("\n" + "-" * 90)
    log(f"[RES ] model = {model['name']},  SNR = {scoring['snr']:.6f}")
    for name, _, overlap, chi2 in stages:
        log(f"[RES ] {name:<12} ov = {overlap:.12f}  chi2 = {chi2:.6e}")
    log(f"[RES ] mismatch {1 - start[2]:.6e} -> {1 - final[2]:.6e}  ({gain:.2f}x reduction)")

    log("\n" + f"{'param':>10}" + "".join(f" {n:>18}" for n, *_ in stages)
        + f" {'bias':>15} {'sigma':>13} {'bias/sigma':>11}")
    for j, name in enumerate(model["names"]):
        bias = final[1][j] - theta_ref[j]
        log(f"{name:>10}" + "".join(f" {theta[j]:>18.10g}" for _, theta, *_ in stages)
            + f" {bias:>+15.6e} {sigma[j]:>13.5e} {bias / sigma[j]:>+11.3f}")
    log("\nbest-fit vector [" + ", ".join(f"{v:.10e}" for v in final[1]) + "]")
