"""Hybrid CV / Gauss-Newton + Nelder-Mead climb from the 1PA truth to the 0PA MAP.

For each EMRI case we start at the 1PA injected parameters and:
  1. climb with adaptive Levenberg-Marquardt CV steps (Fisher gradient),
     re-optimising the finite-difference deltas every RECOMPUTE_DELTAS_EVERY points;
  2. stop early once the 0PA<->1PA overlap exceeds OVERLAP_TARGET;
  3. if the CV climb stalls below the target (a local maximum), escape with
     Nelder-Mead from the stalled point, then hand back to CV -- repeating up to
     MAX_ROUNDS times.

Run:  python gauss_cv_hybrid.py
"""

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
    xp = cp
except ImportError:
    xp = np
    print("[INFO] CuPy not found, using NumPy instead.")

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
F_MIN = 1e-5
NDELTA = 12                    # SEF stability-search grid size
RECOMPUTE_DELTAS_EVERY = 12     # re-optimise finite-diff steps every N CV points
OVERLAP_TARGET = 0.999         # stop the CV climb once overlap exceeds this

# adaptive Levenberg-Marquardt (Nielsen gain ratio)
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-2, 150, 30, 1e-7

# Nelder-Mead escape (objective and simplex are in units of the local sigma)
NM_MAXITER, NM_STEP, NM_XATOL, NM_FATOL = 1000, 3.0, 1e-3, 1e-2

MAX_ROUNDS = 3                 # CV -> NM -> CV -> NM -> ...

use_gpu = True
nchannels = 3
param_names_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
params_to_infer = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]

CASES = [
    dict(name="idx0", dt=5.0, T=1.0, chi2=0.0, nm_overlap=0.9982389175309955,
         signal_param={"m1": 1000000.0, "m2": 10.0, "a": 0.9, "p0": 7.5, "e0": 0.5,
                       "xI0": 1.0, "dist": 5.0, "qS": 0.7853981633974483, "phiS": 1.0,
                       "qK": 1.0, "phiK": 1.0471975511965976, "Phi_phi0": 0.9,
                       "Phi_theta0": 0.5, "Phi_r0": 0.4, "dev_1": 0.0, "dev_2": 0.0}),
    dict(name="idx9", dt=10.0, T=2.5, chi2=0.95, nm_overlap=0.9980349241125692,
         signal_param={"m1": 1000000.0, "m2": 10.0, "a": 0.9, "p0": 9.07414088, "e0": 0.2,
                       "xI0": 1.0, "dist": 5.0, "qS": 1.04719755, "phiS": 0.785398163,
                       "qK": 0.628318531, "phiK": 0.523598776, "Phi_phi0": 0.1,
                       "Phi_theta0": 0.2, "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0}),
    dict(name="idx13", dt=10.0, T=2.5, chi2=0.95, nm_overlap=0.9997570193374152,
         signal_param={"m1": 1000000.0, "m2": 10.0, "a": 0.5, "p0": 9.97066819, "e0": 0.3,
                       "xI0": 1.0, "dist": 5.0, "qS": 1.04719755, "phiS": 0.785398163,
                       "qK": 0.628318531, "phiK": 0.523598776, "Phi_phi0": 0.1,
                       "Phi_theta0": 0.2, "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0}),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_float(x):
    return float(x.get()) if hasattr(x, "get") else float(x)


def make_freq_mask(n, dt, fmin):
    return (xp.fft.rfftfreq(n, dt) > fmin)[1:]


def highpass_clip(waveform, dt, fmin):
    n = waveform.shape[-1]
    freq = xp.fft.rfftfreq(n, dt)
    return xp.fft.irfft(xp.fft.rfft(waveform, axis=-1) * (freq >= fmin), n=n, axis=-1)


def build_context(case):
    """Return the per-case callables and the injected 1PA signal."""
    sp, dt, T, chi2 = case["signal_param"], case["dt"], case["T"], case["chi2"]
    t0 = 10000.0
    channels = [A1TDISens, E1TDISens, T1TDISens][:nchannels]
    tdi_chan = {2: "AE", 3: "AET"}[nchannels]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    def rkw():
        return dict(Tobs=T, t0=t0, dt=dt, index_lambda=8, index_beta=7, flip_hx=True,
                    is_ecliptic_latitude=False, remove_garbage="zero",
                    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                    force_backend="cuda12x" if use_gpu else "cpu",
                    order=20, tdi="1st generation", tdi_chan=tdi_chan)

    wfm = GenerateEMRIWaveform(SuperKludgeWaveform,
                               sum_kwargs=dict(pad_output=True, odd_len=True),
                               return_list=False, use_gpu=use_gpu)
    wresp = ResponseWrapper(waveform_gen=wfm, **rkw())

    def resp_params(vec, evolve_1pa):
        p14 = {n: sp[n] for n in param_names_14}
        p14.update(dict(zip(params_to_infer, vec)))
        return [p14[n] for n in param_names_14] + [
            chi2, evolve_1pa, False, False, False, sp["dev_1"], sp["dev_2"]]

    def make0(vec):
        return xp.array(wresp(*resp_params(vec, False)))[:nchannels, :]

    true_inf = np.array([sp[n] for n in params_to_infer])
    s = highpass_clip(xp.array(wresp(*resp_params(true_inf, True)))[:nchannels, :], dt, F_MIN)
    PSD = xp.array(generate_PSD(waveform=s, dt=dt, noise_PSD=get_sensitivity,
                                channels=channels, noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    fmask = make_freq_mask(s.shape[-1], dt, F_MIN)

    def ip(a, b):
        return _to_float(inner_product(a, b, PSD=PSD, dt=dt, freq_mask=fmask, use_gpu=use_gpu))

    def ov(vec):
        h = make0(vec)
        return ip(s, h) / np.sqrt(ip(s, s) * ip(h, h))

    def chi2r(vec):
        try:
            r = s - make0(vec)
            return ip(r, r)
        except Exception:
            return 1e30

    sef = StableEMRIFisher(
        waveform_class=SuperKludgeWaveform,
        waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=rkw(),
        stats_for_nerds=False, use_gpu=use_gpu, deriv_type="stable",
        noise_model=get_sensitivity, noise_kwargs=noise_kwargs, channels=channels,
        T=T, dt=dt, stability_plot=False, der_order=6, Ndelta=NDELTA,
        plunge_check=True, return_derivatives=True)
    apa = {"chi2": chi2, "evolve_1PA": False, "evolve_primary": False,
           "evolve_2PA": False, "deviation_included": False, "dev_1": 0.0, "dev_2": 0.0}

    def fisher_derivs(vec, dl):
        wp = {n: sp[n] for n in param_names_14}
        wp.update(dict(zip(params_to_infer, vec)))
        F = sef(wave_params={n: wp[n] for n in param_names_14}, param_names=params_to_infer,
                add_param_args=apa, deltas=dl, live_dangerously=False, stability_plot=False,
                der_order=8, Ndelta=(NDELTA if dl is None else None))
        return np.asarray(F[-1], dtype=float), xp.array(F[0]), sef.deltas

    snr = np.sqrt(ip(s, s))
    return dict(make0=make0, ov=ov, chi2r=chi2r, ip=ip, s=s,
                fisher_derivs=fisher_derivs, true_inf=true_inf, snr=snr)


# ---------------------------------------------------------------------------
# the two optimisers
# ---------------------------------------------------------------------------
def lm_climb(ctx, theta0):
    """Adaptive Levenberg-Marquardt CV climb. Returns (theta, overlap, chi2, sigma, why)."""
    ov, chi2r, fisher_derivs, s, ip = (ctx["ov"], ctx["chi2r"], ctx["fisher_derivs"],
                                       ctx["s"], ctx["ip"])
    make0 = ctx["make0"]
    npar = len(theta0)
    cur = np.array(theta0, dtype=float)
    lam, nu, dl, sigma = LAMBDA0, 2.0, None, None
    for it in range(LM_MAX_ITERS):
        if it % RECOMPUTE_DELTAS_EVERY == 0:
            dl = None                                  # re-optimise deltas at this point
        G, dH, deltas = fisher_derivs(cur, dl)
        if dl is None:
            dl = deltas
        h = make0(cur)
        r = s - h
        g = np.array([ip(dH[j], r) for j in range(npar)])
        sigma = np.sqrt(np.abs(np.diag(fishinv(cur[0], G, index_of_M=0))))
        c0 = ip(r, r)
        ovc = ip(s, h) / np.sqrt(ip(s, s) * ip(h, h))
        if ovc > OVERLAP_TARGET:
            return cur, ovc, c0, sigma, "target"

        dvec = np.abs(np.diag(G)) + 1e-30
        delta, ok, rel = np.zeros(npar), False, 0.0
        for _ in range(MAX_INNER):
            try:
                delta = np.linalg.solve(G + lam * np.diag(dvec), g)
            except np.linalg.LinAlgError:
                lam *= nu; nu *= 2.0; continue
            c1 = chi2r(cur + delta)
            pred = float(delta @ (g + lam * dvec * delta))
            rho = (c0 - c1) / pred if pred > 0 else -1.0
            if rho > 0.0:
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0; rel = (c0 - c1) / c0; ok = True; break
            lam *= nu; nu *= 2.0
        if not ok:
            return cur, ovc, c0, sigma, "reject"
        cur = cur + delta
        if rel < REL_TOL:
            return cur, ov(cur), chi2r(cur), sigma, "stall"
    return cur, ov(cur), chi2r(cur), sigma, "maxiter"


def nm_escape(ctx, theta0, sigma):
    """Nelder-Mead in sigma-scaled coordinates, minimising chi2 = <s-h|s-h>."""
    chi2r = ctx["chi2r"]
    n = len(theta0)

    def obj(x):
        return chi2r(theta0 + x * sigma)

    x0 = np.zeros(n)
    simplex = np.vstack([x0] + [NM_STEP * np.eye(n)[i] for i in range(n)])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, maxiter=NM_MAXITER,
                                xatol=NM_XATOL, fatol=NM_FATOL, adaptive=True))
    return theta0 + res.x * sigma, res


# ---------------------------------------------------------------------------
# per-case hybrid driver
# ---------------------------------------------------------------------------
def run_case(case):
    name = case["name"]
    ctx = build_context(case)
    theta = np.array(ctx["true_inf"], dtype=float)
    ov_truth = ctx["ov"](theta)
    print(f"[{name}] SNR={ctx['snr']:.2f}  overlap(0PA@truth)={ov_truth:.4f}")

    for rnd in range(MAX_ROUNDS):
        theta, ovc, c0, sigma, why = lm_climb(ctx, theta)
        print(f"[{name}]  round {rnd}  CV : overlap={ovc:.6f}  chi2={c0:.4e}  ({why})")
        if ovc > OVERLAP_TARGET:
            break
        # stalled below target -> Nelder-Mead escape from here, then CV again
        theta_nm, res = nm_escape(ctx, theta, sigma)
        ov_nm = ctx["ov"](theta_nm)
        if ov_nm > ovc:                       # only accept NM if it helped
            theta = theta_nm
        print(f"[{name}]  round {rnd}  NM : overlap={ctx['ov'](theta):.6f}  "
              f"(nfev={res.nfev}, accepted={ov_nm > ovc})")

    final_ov = ctx["ov"](theta)
    return dict(name=name, snr=float(ctx["snr"]), ov_truth=float(ov_truth),
                ov_final=float(final_ov), params=theta.copy(),
                true=ctx["true_inf"], nm_overlap=case.get("nm_overlap"))


def main():
    results = []
    for case in CASES:
        print("=" * 70)
        results.append(run_case(case))

    print("\n" + "=" * 70)
    print(f"{'case':6} {'SNR':>7} {'ov@truth':>10} {'ov@final':>12} {'ov@NM':>10} {'verdict':>9}")
    for r in results:
        nm = r["nm_overlap"]
        verdict = "SUCCESS" if r["ov_final"] > OVERLAP_TARGET else "STALL"
        nm_s = f"{nm:.6f}" if nm is not None else "-"
        print(f"{r['name']:6} {r['snr']:>7.2f} {r['ov_truth']:>10.4f} "
              f"{r['ov_final']:>12.6f} {nm_s:>10} {verdict:>9}")

    print("\nsystematic bias  (final - truth):")
    for r in results:
        print(f"[{r['name']}]")
        for nm_, tv, cv in zip(params_to_infer, r["true"], r["params"]):
            print(f"    {nm_:10s} true={tv: .8e}  final={cv: .8e}  bias={cv - tv: .3e}")


if __name__ == "__main__":
    main()
