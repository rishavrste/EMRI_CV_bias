"""Step 2 of the environmental EMRI study: environmental Fisher + CV climbs over the population.

Reads population.json (step 1) and, for each source, does three things:

The population is **circular**, and that sets both the injection and the recovery model.
The environmental deviation is a torque on Ldot.  On a circular orbit L = L_circ(p) forces
Ldot = (dL/dp) pdot and Edot = Omega_phi Ldot, so the torque and its consistent energy loss
together are exactly `pdot -> env * pdot`, which is how flux.py applies it.  (Scaling Ldot
alone would be a torque that does no work; at e = 0 it gives the opposite sign.)  This is a
circular-only identity -- at finite e the deviation has to be applied to Edot and Ldot before
the Jacobian, with flux_output_convention = "ELQ".

Two recovery parameters are dropped because the circular orbit kills them outright.  Phi_r0
is the phase of a radial oscillation that does not exist: shifting it by 2 radians leaves the
waveform bit-for-bit identical.  C_e enters as `edot += massratio * e * C_e * ...`, which
carries an explicit factor of e, and e stays pinned at 0 for the whole inspiral -- measured
mismatch 2.22e-16, i.e. zero.  Keeping either one makes Gamma singular.  See PARAMS_0PA.

1. The **environmental Fisher**, 10 x 10, at the injected truth with the environment ON and
   the parameters `PARAMS_0PA + [A_PM, n_PM]` all free.  Inverting it and normalising gives the
   correlation coefficients corr(A_PM, theta_i) for the 8 vacuum parameters -- the x-axis of
   the step-3 scatter.  A_GC / n_GC are NOT freed: at these p the GC term is numerically dead
   (the population carries A_GC = 1e-12, n_GC = 4, a fractional flux change of ~1e-8) and
   freeing it makes the matrix singular.

2. The **0PA CV climb**, 8 free parameters, template with no deviation and no environment,
   seeded at the injected truth.

3. The **0PA + PN CV climb**, 9 free parameters (the 8 plus C_p, the additive 2.5PN
   deviation on pdot), run from BOTH seeds every time: once from the injected truth and
   once from the 0PA best fit with C_p = 0.

   The deviation enters as `pdot += massratio * C_p * ...`, so a PN template at C_p = 0
   is the 0PA template exactly: the 0PA model is a strict subspace of the PN model and the PN
   maximum can never lie below the 0PA maximum.  Rather than use that as a patch -- swapping in
   the 0PA point whenever the PN climb loses, which manufactures an exact tie and hides the
   failure -- both seeds are always run and both are kept.  The spread between them,
   `seed_spread`, is the honest per-source statement of how converged the PN answer is:
   agreement within the evaluation noise means the answer is real, disagreement means it is
   not.  The higher-overlap climb is the one that enters the plot.

Both climbs use the LM-damped Cutler-Vallisneri iteration of the grid runs
(`dtheta = Gamma^-1 <dh|s-h>` as a Gauss-Newton step, Levenberg-Marquardt damping with
Nielsen gain-ratio lambda updates), arranged as LM -> Nelder-Mead -> LM exactly as the grid
runs do.  The Fisher and the derivatives are recomputed at every LM iteration;
RECOMPUTE_DELTAS_EVERY only controls how often SEF redoes its stable-delta scan, which is the
expensive half.

The finite-difference step for C_p is **measured per source**, not assumed.  SEF
gives a parameter sitting at zero a fixed absolute ladder, geomspace(1e-4, 1e-9), calibrated
to a parameter of order unity; C_p enters pdot multiplied by the mass ratio and is integrated
over ~1e5 radians, so its usable step is set by the source.  In the v1 run the resulting
sigma_C_p came out as 7e-3, 2.7e0 and 1.5e6 on three similar sources and one PN covariance was
indefinite.  `calibrate_dev_deltas` now sweeps delta, measures the mismatch it produces, and
keeps the window where that mismatch is in [DEV_MISMATCH_LO, DEV_MISMATCH_HI].  Covariances
come from `cov_precond` rather than `fishinv`, which only rescales m1.

Every climb records **why it stopped** -- `stop_reason` is one of overlap_target, rel_tol,
lambda_exhausted or max_iters -- along with `n_accept` and `lam_final`, per stage and for the
climb as a whole.  In the first run of this script every single climb terminated on a lambda
blowup while printing a converged-looking `rel`, and the JSON could not tell you so.

The bias is `best_fit - injected`, and the sigma it is divided by comes from each recovery
model's OWN Fisher recomputed at its OWN best fit (8 x 8 at fit_0pa, 9 x 9 at fit_pn),
which is what the CV formalism assumes.

Channels: A and E only.  dt = 10, T and dist come from population.json.

Output
------
environmental_tests/results_env_cv.json, written after every source, so a walltime kill
still leaves a valid file.

Run:  python cv_population.py
      ENV_POP_IDXS=0,1,2  ENV_POP_TAG=a  python cv_population.py
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
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
    print(f"[INFO] No usable GPU ({type(exc).__name__}), falling back to NumPy on CPU.")

print(f"[INFO] use_gpu = {use_gpu}")


# --- controls --------------------------------------------------------------
# CV iteration: same values as the grid runs and as test.ipynb section 10.
OVERLAP_TARGET = 0.9999999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-2, 150, 30, 1e-9
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_UP = 1e-12, 1e8, 10.0   # the clamp; see lm_climb
NM_MAXITER, NM_STEP = 1000, 2.0            # Nelder-Mead between the two LM climbs

# The finite-difference step for C_p is measured per source rather than assumed; see
# calibrate_dev_deltas.  The band is the mismatch a single step is required to produce:
# well above the waveform evaluation noise (~1e-5 fractional in chi2) and still linear.
DEV_MISMATCH_LO, DEV_MISMATCH_HI = 1e-9, 1e-5
DEV_LADDER = np.geomspace(1e-20, 1e2, 45)
RECOMPUTE_DELTAS_EVERY = 10                 # stable-delta scan, NOT the Fisher
NDELTA = 12
DER_ORDER = 6

NCHANNELS = 2                              # A, E only
F_MIN, F_MAX = 1e-5, 1e-1

param_names_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]

# The population is circular (e0 = 0), and that decides the recovery model.  Phi_r0 is the
# phase of a radial oscillation that does not exist: shifting it by 2 radians leaves the
# waveform bit-for-bit identical, so its Fisher row is exactly zero and Gamma is singular.
# It is dropped.  e0 survives -- a step of 1e-4 still moves the waveform by 3.5e-5 relative,
# so eccentricity is measurable at e0 = 0 -- but it sits on its physical boundary, which is
# why every candidate vector goes through `project_physical` and why SEF switches itself to
# forward differences there (its minmax lower bound for e0 is 0.01).
PARAMS_0PA = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0"]
# C_e is NOT free: `edot += massratio * e * C_e * ...` carries a factor of e, and on this
# circular population e is pinned at 0 for the whole inspiral, so the direction is exactly
# dead (measured mismatch 2.22e-16 on every source of the first run).  Restore it the moment
# the population becomes eccentric.
PARAMS_PN = PARAMS_0PA + ["C_p"]           # 0PA + PN recovery model
ENV_FREE = ["A_PM", "n_PM"]                # free in the environmental Fisher
PARAMS_ENV = PARAMS_0PA + ENV_FREE

# Physical bounds, applied to any point a climb proposes.  The waveform raises outright on
# e0 < 0 ("e0 is negative. It must be positive."), and with the injected e0 at exactly zero
# roughly half the CV steps want to cross it, so without this a source dies mid-climb.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0,
           "qS": 0.0, "phiS": 0.0, "Phi_phi0": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99,
           "qS": np.pi, "phiS": 2 * np.pi, "Phi_phi0": 2 * np.pi}

HERE = os.path.dirname(os.path.abspath(__file__))
POP = os.path.join(HERE, "population.json")

_TAG = os.environ.get("ENV_POP_TAG", "").strip()
JSON_PATH = os.path.join(HERE, f"results_env_cv{('_' + _TAG) if _TAG else ''}.json")

channels = [A1TDISens, E1TDISens, T1TDISens]


def project_physical(names, vec):
    """A proposed point pulled back inside the physical box.

    A Gauss-Newton step knows nothing about e0 >= 0 or |a| < 1; it is free to propose a point
    the waveform generator refuses to evaluate.  Clipping the proposal is what a bounded
    optimiser does, and it keeps a single bad step from killing the whole source.  Parameters
    with no entry in PHYS_LO / PHYS_HI (C_p) are left alone.
    """
    out = np.array(vec, dtype=float)
    for i, nm in enumerate(names):
        out[i] = min(max(out[i], PHYS_LO.get(nm, -np.inf)), PHYS_HI.get(nm, np.inf))
    return out


def _to_float(x):
    return float(x.get()) if hasattr(x, "get") else float(x)


def make_freq_mask(n, dt):
    """Boolean mask over rfftfreq with the DC bin dropped, which is what inner_product wants."""
    f = xp.fft.rfftfreq(n, dt)
    return ((f > F_MIN) & (f < F_MAX))[1:]


def cov_precond(F):
    """Jacobi-preconditioned inverse: rescale every parameter by 1/sqrt(F_ii) before inverting,
    then undo. Algebraically exact, and unlike fishinv (which only rescales m1) it copes with
    A_PM ~ 1e-4 sitting in the same matrix as m1 ~ 1e7."""
    d = np.sqrt(np.diag(F))
    if np.any(d <= 0):
        raise ValueError(f"non-positive Fisher diagonal at {np.where(d <= 0)[0]}: a derivative column is dead")
    C = np.linalg.inv(F / np.outer(d, d)) / np.outer(d, d)
    return 0.5 * (C + C.T)


# --- per-source context ----------------------------------------------------
def build_source(src):
    """Response, injected environmental signal, PSD, inner product and the SEF instance.

    Everything that depends on T and dt lives here, so it is rebuilt per source.
    """
    T, dt = src["T"], src["dt"]
    source_params = {n: float(src["source_params"][n]) for n in param_names_14}
    A_PM, n_PM, A_GC, n_GC = src["A_PM"], src["n_PM"], src["A_GC"], src["n_GC"]

    # SuperKludge tail, everything off except the environmental block -- as in test.ipynb.
    CHI2 = 0.0
    EVOLVE_1PA = EVOLVE_PRIMARY = EVOLVE_2PA = False

    def env_tail(environmental_included):
        """The 12 extra SuperKludgeFlux arguments, positional, as ResponseWrapper wants them."""
        return [CHI2, EVOLVE_1PA, EVOLVE_PRIMARY, EVOLVE_2PA,
                False, 0.0, 0.0,
                environmental_included, A_PM, n_PM, A_GC, n_GC]

    def env_apa(environmental_included):
        """The same 12 arguments as a dict, which is how SEF wants them."""
        return dict(chi2=CHI2, evolve_1PA=EVOLVE_1PA, evolve_primary=EVOLVE_PRIMARY,
                    evolve_2PA=EVOLVE_2PA, deviation_included=False, C_p=0.0, C_e=0.0,
                    environmental_included=environmental_included,
                    A_PM=A_PM, n_PM=n_PM, A_GC=A_GC, n_GC=n_GC)

    response_kwargs = dict(
        Tobs=T, t0=10000.0, dt=dt, index_lambda=8, index_beta=7, flip_hx=True,
        is_ecliptic_latitude=False, remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend="cuda12x" if use_gpu else "cpu",
        order=20, tdi="1st generation", tdi_chan={2: "AE", 3: "AET"}[NCHANNELS])

    waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform,
                                          sum_kwargs=dict(pad_output=True, odd_len=True),
                                          return_list=False, use_gpu=use_gpu)
    waveform_response = ResponseWrapper(waveform_gen=waveform_model, **response_kwargs)

    # The injected signal: environment ON.  Every template below has it off.
    signal = xp.array(waveform_response(
        *([source_params[n] for n in param_names_14] + env_tail(True))))[0:NCHANNELS, :]

    noise_kwargs = [{"sens_fn": ch} for ch in channels[:NCHANNELS]]
    PSD = xp.array(generate_PSD(waveform=signal, dt=dt, noise_PSD=get_sensitivity,
                                channels=channels[:NCHANNELS],
                                noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    freq_mask = make_freq_mask(len(signal[0]), dt)

    def ip(a, b):
        return _to_float(inner_product(a, b, PSD, dt, freq_mask=freq_mask, use_gpu=use_gpu))

    sef = StableEMRIFisher(
        waveform_class=SuperKludgeWaveform,
        waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=response_kwargs,
        stats_for_nerds=False, use_gpu=use_gpu, deriv_type="stable",
        noise_model=get_sensitivity, noise_kwargs=noise_kwargs, channels=channels[:NCHANNELS],
        T=T, dt=dt, stability_plot=False, der_order=DER_ORDER, Ndelta=NDELTA,
        plunge_check=True, return_derivatives=False)

    snr = np.sqrt(ip(signal, signal))
    per_chan = [float(np.sqrt(_to_float(inner_product(
        signal[i:i + 1], signal[i:i + 1], PSD[i:i + 1, :], dt,
        freq_mask=freq_mask, use_gpu=use_gpu)))) for i in range(NCHANNELS)]

    return dict(source_params=source_params, signal=signal, ip=ip, sef=sef,
                freq_mask=freq_mask, dt=dt, T=T, snr=snr, snr_channels=per_chan,
                env_apa=env_apa, env_tail=env_tail,
                waveform_response=waveform_response)


# --- 1. environmental Fisher and the A_PM correlations ---------------------
def env_fisher(S, label):
    """Fisher at the injected truth, environment on, A_PM and n_PM free."""
    t0 = time.time()
    F = S["sef"](wave_params=S["source_params"], param_names=PARAMS_ENV,
                 add_param_args=S["env_apa"](True), freq_mask=S["freq_mask"],
                 live_dangerously=False, stability_plot=False,
                 der_order=DER_ORDER, Ndelta=NDELTA)
    F = np.asarray(F, dtype=float)
    F = 0.5 * (F + F.T)
    C = cov_precond(F)
    sig = np.sqrt(np.diag(C))
    corr = C / np.outer(sig, sig)
    print(f"[FISH] {label} env {len(PARAMS_ENV)}x{len(PARAMS_ENV)}: cond = {np.linalg.cond(F):.3e}, {time.time() - t0:.1f} s")
    return F, C, sig, corr


# --- 2/3. recovery models --------------------------------------------------
def build_ctx(S, model_params):
    """Everything the CV iteration needs for one recovery model.

    Templates never carry the environment; the PN model carries C_p as a free parameter.
    """
    source_params, waveform_response = S["source_params"], S["waveform_response"]
    signal, ip, sef = S["signal"], S["ip"], S["sef"]

    def template_tail(vec):
        dev = dict(zip(model_params, vec))
        on = ("C_p" in dev) or ("C_e" in dev)
        return [0.0, False, False, False,
                on, dev.get("C_p", 0.0), dev.get("C_e", 0.0),
                False, 0.0, 0.0, 0.0, 0.0]

    def template_apa(vec):
        dev = dict(zip(model_params, vec))
        on = ("C_p" in dev) or ("C_e" in dev)
        apa = S["env_apa"](False)
        apa.update(deviation_included=on, C_p=dev.get("C_p", 0.0), C_e=dev.get("C_e", 0.0))
        return apa

    def make0(vec):
        p = dict(source_params)
        p.update({n: v for n, v in zip(model_params, vec) if n in param_names_14})
        args = [p[n] for n in param_names_14] + template_tail(vec)
        return xp.array(waveform_response(*args))[0:NCHANNELS, :]

    def ov(vec):
        h = make0(vec)
        return ip(signal, h) / np.sqrt(ip(signal, signal) * ip(h, h))

    def chi2r(vec):
        try:
            r = signal - make0(vec)
            return ip(r, r)
        except Exception:
            return 1e30

    # Populated by calibrate_dev_deltas once the context exists; empty means "use the SEF
    # default grid", which is right for the vacuum parameters and wrong for C_p.
    delta_range = {}

    def fisher_derivs(vec, dl):
        wp = {n: source_params[n] for n in param_names_14}
        wp.update({n: v for n, v in zip(model_params, vec) if n in param_names_14})
        F = sef(wave_params=wp, param_names=model_params,
                add_param_args=template_apa(vec), deltas=dl, freq_mask=S["freq_mask"],
                delta_range=(dict(delta_range) if delta_range else None),
                live_dangerously=False, stability_plot=False,
                der_order=DER_ORDER, Ndelta=(NDELTA if dl is None else None),
                return_derivatives=True)
        return np.asarray(F[-1], dtype=float), xp.array(F[0]), sef.deltas

    return dict(make0=make0, ov=ov, chi2r=chi2r, ip=ip, s=signal,
                fisher_derivs=fisher_derivs, params=model_params,
                delta_range=delta_range)


def calibrate_dev_deltas(ctx, theta0, tag=""):
    """Measure the finite-difference step for C_p instead of assuming it.

    SEF gives a parameter sitting at zero a fixed absolute ladder, geomspace(1e-4, 1e-9)
    (fisher.py, the `wave_params[param_name] == 0.0` branch).  That ladder is calibrated to a
    parameter of order unity and has nothing to do with C_p: the deviation enters pdot
    multiplied by the mass ratio (~3e-5) and is then integrated over ~1e5 radians of orbital
    phase, so the step that yields a usable derivative is set by the source, not by a constant.
    In the v1 run the resulting sigma_C_p came out as 7e-3, 2.7e0 and 1.5e6 on three similar
    sources -- nine orders of magnitude of disagreement, and one indefinite covariance.

    So sweep delta, measure the mismatch between the template at C_p = 0 and at C_p = delta,
    and keep the window where that mismatch lands in [DEV_MISMATCH_LO, DEV_MISMATCH_HI]:
    large enough to sit well above the waveform evaluation noise, small enough to stay linear.
    The scan is cut off at the first step that overshoots the band, so a mismatch that
    saturates or folds back at large delta cannot be mistaken for a valid window.

    Returns {name: ndarray of steps}; a parameter with no usable window is left out, and SEF
    falls back to its own grid for it.

    Basically the same as StableEMRIFisher._calibrate_dev_deltas, but with a per-source context and
    a different way of handling the results.
    """
    make0, ip, names_model = ctx["make0"], ctx["ip"], ctx["params"]
    h0 = make0(np.asarray(theta0, dtype=float))
    n0 = np.sqrt(ip(h0, h0))
    out = {}
    for nm in ("C_p", "C_e"):    # C_e is skipped below while it is not in PARAMS_PN
        if nm not in names_model:
            continue
        j = names_model.index(nm)
        mism = np.full(len(DEV_LADDER), np.nan)
        for k, d in enumerate(DEV_LADDER):
            v = np.array(theta0, dtype=float)
            v[j] = theta0[j] + d
            try:
                h = make0(v)
                mism[k] = 1.0 - ip(h0, h) / (n0 * np.sqrt(ip(h, h)))
            except Exception:
                break
            if mism[k] > DEV_MISMATCH_HI:      # past the top of the band, stop climbing
                break
        ok = np.isfinite(mism)
        band = np.where(ok & (mism >= DEV_MISMATCH_LO) & (mism <= DEV_MISMATCH_HI))[0]
        if band.size:
            d_hi, d_lo = float(DEV_LADDER[band[-1]]), float(DEV_LADDER[band[0]])
        elif ok.any() and mism[np.argmax(ok)] > DEV_MISMATCH_HI:
            # Even the smallest step in the ladder overshoots: this direction is far more
            # sensitive than the ladder can resolve.  Falling back to the SEF default here
            # would be the worst possible choice -- its smallest step is 1e-9, which is
            # eleven decades coarser than what this parameter needs -- so take the bottom of
            # the ladder and say so.
            d_hi, d_lo = float(DEV_LADDER[1]), float(DEV_LADDER[0])
            print(f"    [WARN]{tag} {nm}: hypersensitive -- the smallest ladder step "
                  f"{DEV_LADDER[0]:.1e} already gives mismatch "
                  f"{mism[np.argmax(ok)]:.2e} > {DEV_MISMATCH_HI:.0e}; using the bottom of "
                  f"the ladder, and treat this parameter's sigma with suspicion", flush=True)
        else:
            span = (f"{np.nanmin(mism):.2e} to {np.nanmax(mism):.2e}" if ok.any() else "nothing finite")
            print(f"    [WARN]{tag} {nm}: no step reaches mismatch {DEV_MISMATCH_LO:.0e} "
                  f"(measured {span}) -- this direction is numerically dead, leaving it on "
                  f"the SEF default grid", flush=True)
            continue
        if d_hi / d_lo < 10.0:                 # single-point window, widen it a little
            d_hi, d_lo = d_hi * 3.0, d_lo / 3.0
        out[nm] = np.geomspace(d_hi, d_lo, NDELTA)
        print(f"    [DELTA]{tag} {nm}: steps {d_hi:.3e} -> {d_lo:.3e}  "
              f"(mismatch {mism[band[-1]]:.2e} -> {mism[band[0]]:.2e}, "
              f"{band.size} of {len(DEV_LADDER)} ladder points in band)", flush=True)
    return out


def lm_climb(ctx, theta0, tag=""):
    """LM-damped Cutler-Vallisneri climb.

    lambda is clamped to [LAMBDA_MIN, LAMBDA_MAX] and a rejected step multiplies it by a fixed
    LAMBDA_UP.

    Returns (theta, overlap, sigma, info); `info` carries n_iter, n_accept, lam_final and
    stop_reason -- one of overlap_target, rel_tol, lambda_exhausted, max_iters.
    """
    ov, chi2r, fisher_derivs, s, ip_, make0 = (ctx["ov"], ctx["chi2r"], ctx["fisher_derivs"],
                                               ctx["s"], ctx["ip"], ctx["make0"])
    names = ctx["params"]
    npar = len(theta0)
    cur = project_physical(names, theta0)
    lam, dl, sigma = LAMBDA0, None, np.ones(npar)
    n_iter, n_accept, stop = 0, 0, "max_iters"
    for it in range(LM_MAX_ITERS):
        n_iter = it + 1
        if it % RECOMPUTE_DELTAS_EVERY == 0:
            dl = None
        G, dH, deltas = fisher_derivs(cur, dl)
        if dl is None:
            dl = deltas
        h = make0(cur)
        r = s - h
        g = np.array([ip_(dH[j], r) for j in range(npar)])
        sigma = np.sqrt(np.abs(np.diag(fishinv(cur[0], G, index_of_M=0))))
        c0 = ip_(r, r)
        ovc = ip_(s, h) / np.sqrt(ip_(s, s) * ip_(h, h))
        if ovc > OVERLAP_TARGET:
            stop = "overlap_target"
            break
        dvec = np.abs(np.diag(G)) + 1e-30
        delta, ok, rel, nxt = np.zeros(npar), False, 0.0, cur
        for attempt in range(2):
            if attempt == 1:
                lam = LAMBDA0          # one restart from a sane damping before giving up
            for _ in range(MAX_INNER):
                try:
                    delta = np.linalg.solve(G + lam * np.diag(dvec), g)
                except np.linalg.LinAlgError:
                    lam = min(lam * LAMBDA_UP, LAMBDA_MAX)
                    continue
                pred = float(delta @ (g + lam * dvec * delta))
                cand = project_physical(names, cur + delta)
                c1 = chi2r(cand)
                rho = (c0 - c1) / pred if pred > 0 else -1.0
                if rho > 0.0:
                    lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), LAMBDA_MIN)
                    rel = (c0 - c1) / c0
                    ok, nxt = True, cand
                    break
                lam = min(lam * LAMBDA_UP, LAMBDA_MAX)
            if ok:
                break
        print(f"    CV{tag} it {it:>3} lam={lam:.1e} ov={ovc:.10f} chi2={c0:.3e} rel={rel:.1e}",
              flush=True)
        if not ok:
            stop = "lambda_exhausted"
            break
        cur = nxt
        n_accept += 1
        if rel < REL_TOL:
            stop = "rel_tol"
            break
    info = dict(n_iter=n_iter, n_accept=n_accept, lam_final=float(lam), stop_reason=stop)
    return cur, ov(cur), sigma, info


def nm_refine(ctx, theta0, sigma, maxiter, tag=""):
    """Nelder-Mead in units of sigma, run between the two LM climbs exactly as the grid runs
    do.  It is accepted only if it actually raises the overlap, so it can never make a climb
    worse -- its job is to walk the simplex off a point where LM's linear step has stalled."""
    chi2r, ov, names = ctx["chi2r"], ctx["ov"], ctx["params"]
    n = len(theta0)
    simplex = np.vstack([np.zeros(n)] + [NM_STEP * np.eye(n)[i] for i in range(n)])
    res = minimize(lambda x: chi2r(project_physical(names, theta0 + x * sigma)), np.zeros(n),
                   method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, maxiter=maxiter,
                                xatol=1e-4, fatol=1e-4, adaptive=True))
    cand = project_physical(names, theta0 + res.x * sigma)
    ov0, ov1 = float(ov(theta0)), float(ov(cand))
    ok = bool(ov1 > ov0)
    print(f"    NM{tag} nfev={res.nfev} ov {ov0:.10f} -> {ov1:.10f} (accepted={ok})", flush=True)
    return (cand if ok else theta0), dict(nfev=int(res.nfev), ov_before=ov0, ov_after=ov1,
                                          accepted=ok)


def cv_from(ctx, tag, start):
    """LM, then -- if the overlap target is not reached -- one Nelder-Mead refinement and a
    second LM, which is the arrangement the grid runs use.

    Every stage records why it stopped, so a climb that died on a lambda blowup is
    distinguishable in the JSON from one that actually converged.  They used to print and
    store the same thing.
    """
    theta = np.array(start, dtype=float)
    ov_start = float(ctx["ov"](theta))
    print(f"    start overlap = {ov_start:.10f}", flush=True)
    t0 = time.time()
    stages = []
    theta, ov1, sigma, info = lm_climb(ctx, theta, tag=f" {tag}#1")
    stages.append(dict(stage="LM1", ov=float(ov1), **info))
    if ov1 < OVERLAP_TARGET:
        theta, nm = nm_refine(ctx, theta, sigma, NM_MAXITER, tag=f" {tag}")
        stages.append(dict(stage="NM", ov=nm["ov_after"] if nm["accepted"] else nm["ov_before"],
                           nfev=nm["nfev"], accepted=nm["accepted"]))
        theta, ov1, sigma, info = lm_climb(ctx, theta, tag=f" {tag}#2")
        stages.append(dict(stage="LM2", ov=float(ov1), **info))
    return dict(ov_start=ov_start, ov_final=float(ov1),
                chi2=float(ctx["chi2r"](theta)), params=theta.copy(),
                start=np.array(start, dtype=float),
                stages=stages, stop_reason=stages[-1]["stop_reason"],
                n_iter=sum(st.get("n_iter", 0) for st in stages),
                n_accept=sum(st.get("n_accept", 0) for st in stages),
                lam_final=float(stages[-1]["lam_final"]),
                nm_accepted=any(st.get("accepted") for st in stages),
                seconds=time.time() - t0)


def cov_at(ctx, theta, label):
    """The recovery model's own Fisher, recomputed at its own best fit -- the sigma denominator."""
    t0 = time.time()
    G, _, _ = ctx["fisher_derivs"](np.asarray(theta, dtype=float), None)
    G = 0.5 * (G + G.T)
    # Jacobi preconditioning, not fishinv: this matrix holds m1 ~ 1e6 next to C_p ~ 1e-4 and
    # fishinv only rescales m1, which is how the v1 run produced an indefinite PN covariance.
    try:
        C = cov_precond(G)
    except ValueError as exc:
        print(f"[WARN] {label}: cov_precond failed ({exc}); falling back to fishinv")
        C = fishinv(theta[0], G, index_of_M=0)
    C = 0.5 * (C + C.T)
    ev = np.linalg.eigvalsh(C)
    if ev.min() <= 0:
        print(f"[WARN] {label}: covariance not positive definite, min eigenvalue = {ev.min():.3e}")
    print(f"[FISH] {label}: cond = {np.linalg.cond(G):.3e}, {time.time() - t0:.1f} s")
    return C


# --- JSON records ----------------------------------------------------------
CLIMB_FIELDS = ("ov_start", "ov_final", "chi2", "n_iter", "n_accept", "lam_final",
                "stop_reason", "stages", "nm_accepted", "seconds")


def climb_json(fit, sigma=None, bias=None):
    """One climb as plain JSON.

    Defined once and used for the 0PA fit, the kept PN fit and each of the two PN seeds, so
    the field list cannot drift between them.  `sigma` and `bias` are attached only to the
    fits that carry a Fisher; the per-seed records are the climb alone.
    """
    out = {k: fit[k] for k in CLIMB_FIELDS}
    out["params"] = [float(x) for x in fit["params"]]
    if sigma is not None:
        out["sigma"] = [float(x) for x in sigma]
        out["bias"] = [float(x) for x in bias]
        out["bias_over_sigma"] = [float(b / s) for b, s in zip(bias, sigma)]
    return out


def save_json(records, pop):
    payload = dict(
        study="environmental_CV_population",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        population=os.path.basename(POP), population_generated=pop.get("generated"),
        nchannels=NCHANNELS, freq_band=[F_MIN, F_MAX],
        params_0pa=PARAMS_0PA, params_pn=PARAMS_PN, params_env=PARAMS_ENV, env_free=ENV_FREE,
        note=("env Fisher at the injected truth with A_PM, n_PM free; 0PA climbed from the "
              "injected truth, 0PA+PN climbed from BOTH the injected truth and the 0PA best "
              "fit with the higher-overlap one kept and both stored under fit_pn.by_seed; "
              "each climb is LM -> Nelder-Mead -> LM with lambda clamped to "
              f"[{LAMBDA_MIN:.0e}, {LAMBDA_MAX:.0e}] and a stop_reason recorded per stage; "
              "C_p finite-difference step calibrated per source by measured mismatch "
              f"in [{DEV_MISMATCH_LO:.0e}, {DEV_MISMATCH_HI:.0e}] and stored in "
              "dev_delta_range; covariances from cov_precond, not fishinv; "
              "sigma from each model's own Fisher at its own best fit"),
        sources=records)
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)


# --- the three steps, one function each ------------------------------------
def apm_correlations(corr_env):
    """corr(A_PM, theta_i) for the vacuum parameters -- the scatter plot's x axis."""
    i_apm = PARAMS_ENV.index("A_PM")
    corr_apm = [float(corr_env[i, i_apm]) for i in range(len(PARAMS_0PA))]
    print(f"    corr(A_PM, .) : " +
          "  ".join(f"{p}={c:+.4f}" for p, c in zip(PARAMS_0PA, corr_apm)), flush=True)
    return corr_apm


def calibrated_pn_ctx(S, truth_pn, label):
    """The PN recovery context with its C_p step measured for this source.

    The calibration has to happen before any Fisher of the PN model is taken: the climb, the
    sigma and the marginalisation onto the vacuum parameters all read those steps.
    """
    ctx_pn = build_ctx(S, PARAMS_PN)
    t0 = time.time()
    dev_dr = calibrate_dev_deltas(ctx_pn, truth_pn, tag=f" {label}")
    ctx_pn["delta_range"].update(dev_dr)
    print(f"    [DELTA] calibration took {time.time() - t0:.1f} s, "
          f"{len(dev_dr)} of 2 parameters calibrated", flush=True)
    return ctx_pn, dev_dr


def climb_pn_both_seeds(ctx_pn, idx, truth_pn, fit_0pa):
    """Climb the PN model from the injected truth AND from the 0PA best fit, every time.

    Both are kept.  The spread between them is what says whether the PN answer is converged;
    picking the winner does not make a disagreement go away, so the spread is reported rather
    than papered over.  The winner is what enters the plot.
    """
    print(f"    --- 0PA + PN ({len(PARAMS_PN)} params, from the injected truth) ---", flush=True)
    pn_truth = cv_from(ctx_pn, f"{idx}/PN(truth)", truth_pn)
    print(f"    --- 0PA + PN ({len(PARAMS_PN)} params, from the 0PA best fit) ---", flush=True)
    seed_0pa = np.concatenate([fit_0pa["params"], np.zeros(len(PARAMS_PN) - len(PARAMS_0PA))])
    pn_0pa = cv_from(ctx_pn, f"{idx}/PN(from_0PA)", seed_0pa)

    by_seed = {"from_truth": pn_truth, "from_0PA": pn_0pa}
    seed = "from_truth" if pn_truth["ov_final"] >= pn_0pa["ov_final"] else "from_0PA"
    fit_pn = by_seed[seed]
    spread = abs(pn_truth["ov_final"] - pn_0pa["ov_final"])
    print(f"    PN seeds: truth ov = {pn_truth['ov_final']:.12f} "
          f"({pn_truth['stop_reason']}), 0PA ov = {pn_0pa['ov_final']:.12f} "
          f"({pn_0pa['stop_reason']}), spread = {spread:.3e}, kept {seed}", flush=True)
    if fit_pn["ov_final"] < fit_0pa["ov_final"]:
        print(f"    [WARN] both PN seeds end below the 0PA overlap "
              f"{fit_0pa['ov_final']:.12f} -- this source's PN row is not converged",
              flush=True)
    return fit_pn, seed, spread, by_seed


# --- reporting -------------------------------------------------------------
def print_bias_table(corr_apm, bias_0pa, sig_0pa, bias_pn, sig_pn):
    print(f"\n    {'param':>10} | {'corr(A_PM)':>10} | {'0PA bias/sig':>13} | {'PN bias/sig':>13}")
    print("    " + "-" * 56, flush=True)
    for i, p in enumerate(PARAMS_0PA):
        print(f"    {p:>10} | {corr_apm[i]:>+10.4f} | {bias_0pa[i] / sig_0pa[i]:>13.4f} | "
              f"{bias_pn[i] / sig_pn[i]:>13.4f}", flush=True)


def print_climb_lines(fit_0pa, fit_pn, pn_seed, seed_spread):
    """The two result lines: overlap, chi2 and how the climb ended, for each model."""
    print(f"    0PA    ov = {fit_0pa['ov_final']:.12f}  chi2 = {fit_0pa['chi2']:.6e}  "
          f"({fit_0pa['n_iter']} its, {fit_0pa['n_accept']} accepted, "
          f"{fit_0pa['seconds']:.0f} s, stop {fit_0pa['stop_reason']}, "
          f"lam {fit_0pa['lam_final']:.1e})", flush=True)
    print(f"    0PA+PN ov = {fit_pn['ov_final']:.12f}  chi2 = {fit_pn['chi2']:.6e}  "
          f"({fit_pn['n_iter']} its, {fit_pn['n_accept']} accepted, "
          f"{fit_pn['seconds']:.0f} s, seed {pn_seed}, stop {fit_pn['stop_reason']}, "
          f"lam {fit_pn['lam_final']:.1e}, seed spread {seed_spread:.3e})", flush=True)


def print_summary(records):
    """The end-of-run table.  Every column here is a thing the v1 run could not tell you."""
    print("\n" + "=" * 78)
    print(f"{'idx':>3} {'m1':>11} {'SNR':>7} {'ov 0PA':>14} {'ov PN':>14} "
          f"{'|b/s| 0PA':>10} {'|b/s| PN':>10} {'stop 0PA':>17} {'stop PN':>17} "
          f"{'PN seed':>10} {'spread':>9} {'ok':>3}")
    for r in records:
        p_, q = r["fit_pn"], r["fit_0pa"]
        print(f"{r['idx']:>3} {r['m1']:>11.4e} {r['snr']:>7.3f} {q['ov_final']:>14.10f} "
              f"{p_['ov_final']:>14.10f} "
              f"{max(abs(x) for x in q['bias_over_sigma']):>10.3f} "
              f"{max(abs(x) for x in p_['bias_over_sigma'][:9]):>10.3f} "
              f"{q['stop_reason']:>17} {p_['stop_reason']:>17} {p_['seed']:>10} "
              f"{p_['seed_spread']:>9.2e} "
              f"{'yes' if p_['converged_above_0pa'] else 'NO':>3}")

    stops = Counter([r["fit_0pa"]["stop_reason"] for r in records] +
                    [r["fit_pn"]["stop_reason"] for r in records])
    print(f"\n[STOP] over {2 * len(records)} kept climbs: " +
          ", ".join(f"{k} x{v}" for k, v in stops.most_common()))
    n_seed0 = sum(r["fit_pn"]["seed"] == "from_0PA" for r in records)
    n_bad = sum(not r["fit_pn"]["converged_above_0pa"] for r in records)
    spreads = [r["fit_pn"]["seed_spread"] for r in records]
    print(f"[PN] the 0PA seed won on {n_seed0} / {len(records)} sources; "
          f"seed spread median {np.median(spreads):.2e}, max {max(spreads):.2e}; "
          f"{n_bad} sources end below their own 0PA")


# --- driver ----------------------------------------------------------------
def load_population():
    """population.json, optionally narrowed by ENV_POP_IDXS so a run can be split by hand."""
    with open(POP) as f:
        pop = json.load(f)
    sources = pop["sources"]
    sel = os.environ.get("ENV_POP_IDXS", "").strip()
    if sel:
        want = {int(x) for x in sel.replace(",", " ").split()}
        sources = [s for s in sources if s["idx"] in want]
    return pop, sources


def run_source(src):
    """One EMRI end to end: env Fisher, both recovery climbs, sigmas, one JSON record."""
    idx = src["idx"]
    label = f"[SRC {idx:2d}]"
    t_src = time.time()

    S = build_source(src)
    A, E = S["snr_channels"]
    print(f"{label} m1 = {src['m1']:.6e}  p0 = {src['p0']:.6f}  T = {S['T']:.4f} yr  "
          f"dt = {S['dt']:g}  N = {len(S['signal'][0]):,d}", flush=True)
    print(f"{label} A_PM = {src['A_PM']:.6e}  SNR = {S['snr']:.4f} "
          f"(A = {A:.4f}, E = {E:.4f})", flush=True)

    # 1. environmental Fisher -> the A_PM correlations (the scatter x-axis)
    F_env, C_env, sig_env, corr_env = env_fisher(S, label)
    corr_apm = apm_correlations(corr_env)

    truth_0pa = np.array([S["source_params"][n] for n in PARAMS_0PA], dtype=float)
    # one zero per free deviation parameter -- derived from PARAMS_PN, not hardcoded, so
    # that dropping or restoring C_e cannot desync this vector from the context again
    truth_pn = np.concatenate([truth_0pa, np.zeros(len(PARAMS_PN) - len(PARAMS_0PA))])

    # 2. 0PA
    print(f"    --- 0PA ({len(PARAMS_0PA)} params, from the injected truth) ---", flush=True)
    ctx_0pa = build_ctx(S, PARAMS_0PA)
    fit_0pa = cv_from(ctx_0pa, f"{idx}/0PA", truth_0pa)

    # 3. 0PA + PN, from both seeds, with its finite-difference steps measured first
    ctx_pn, dev_dr = calibrated_pn_ctx(S, truth_pn, label)
    fit_pn, pn_seed, seed_spread, by_seed = climb_pn_both_seeds(ctx_pn, idx, truth_pn, fit_0pa)

    # sigma from each model's own Fisher at its own best fit
    sig_0pa = np.sqrt(np.diag(cov_at(ctx_0pa, fit_0pa["params"], f"{label} 0PA    @ best fit")))
    sig_pn = np.sqrt(np.diag(cov_at(ctx_pn, fit_pn["params"], f"{label} 0PA+PN @ best fit")))
    bias_0pa = fit_0pa["params"] - truth_0pa
    bias_pn = fit_pn["params"] - truth_pn

    print_bias_table(corr_apm, bias_0pa, sig_0pa, bias_pn, sig_pn)
    print_climb_lines(fit_0pa, fit_pn, pn_seed, seed_spread)

    record = dict(
        idx=idx, m1=src["m1"], p0=src["p0"], dist=src["dist"], T=S["T"], dt=S["dt"],
        A_PM=src["A_PM"], n_PM=src["n_PM"],
        snr=float(S["snr"]), snr_channels=[float(x) for x in S["snr_channels"]],
        source_params=S["source_params"],
        truth_0pa=[float(x) for x in truth_0pa], truth_pn=[float(x) for x in truth_pn],
        fisher_env=[[float(v) for v in row] for row in F_env],
        cov_env=[[float(v) for v in row] for row in C_env],
        sigma_env=[float(x) for x in sig_env],
        corr_env=[[float(v) for v in row] for row in corr_env],
        corr_A_PM=corr_apm,
        dev_delta_range={k: [float(x) for x in v] for k, v in dev_dr.items()},
        fit_0pa=climb_json(fit_0pa, sig_0pa, bias_0pa),
        fit_pn=dict(climb_json(fit_pn, sig_pn, bias_pn),
                    seed=pn_seed, seed_spread=float(seed_spread),
                    converged_above_0pa=bool(fit_pn["ov_final"] >= fit_0pa["ov_final"]),
                    by_seed={k: climb_json(v) for k, v in by_seed.items()}),
        seconds=time.time() - t_src)

    del S
    if use_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    return record


def main():
    pop, sources = load_population()
    print(f"[info] {len(sources)} sources {[s['idx'] for s in sources]}, "
          f"{len(sources)} env Fishers + {3 * len(sources)} CV climbs; "
          f"out -> {os.path.basename(JSON_PATH)}\n", flush=True)

    records, t_all = [], time.time()
    for src in sources:
        print("=" * 78, flush=True)
        records.append(run_source(src))
        save_json(records, pop)          # written after every source: a walltime kill is safe
        print(f"[SRC {records[-1]['idx']:2d}] done in {records[-1]['seconds']:.1f} s, "
              f"{len(records)}/{len(sources)} written to {os.path.basename(JSON_PATH)}",
              flush=True)

    print_summary(records)
    save_json(records, pop)
    print(f"\n[saved] {JSON_PATH}  ({time.time() - t_all:.1f} s total)")


if __name__ == "__main__":
    main()
