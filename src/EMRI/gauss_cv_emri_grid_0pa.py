"""EMRI grid (least-biased) 0PA fit: 1PA injection -> 0PA MAP via LM-damped CV.

Least-biased grid points from the 0PA-vs-2PA critical-SNR analysis (a=+0.9 column):
    idx4  : a=+0.9, e0=0.1
    idx14 : a=+0.9, e0=0.3

Signal = 1PA injection; template = 0PA (9 params, no deviation). The CV climb STARTS
FROM THE INJECTION itself (not a pre-found best fit) and ascends to the 0PA MLP:
adaptive-LM CV; if it stalls below the overlap limit, one 1000-step Nelder-Mead escape,
then one more CV.

All runs: dt=10, T=2.5, nchannels=3, no noise, chi2 (secondary spin) = 0.95.

Run:  python gauss_cv_emri_grid_0pa.py
"""

import json
import os
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
    xp = cp
except ImportError:
    xp = np
    print("[INFO] CuPy not found, using NumPy instead.")

# --- controls --------------------------------------------------------------
F_MIN = 1e-5
NDELTA = 12
RECOMPUTE_DELTAS_EVERY = 5
OVERLAP_TARGET = 0.9999999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-2, 150, 30, 1e-9
NM_MAXITER, NM_STEP = 1000, 2.0

use_gpu = True
nchannels = 3
param_names_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
params_to_infer = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]

# start = the 1PA INJECTION (intrinsic + phase params); CV climbs to the 0PA MLP.
CASES = [
    dict(name="idx4", dt=10.0, T=2.5, chi2=0.95,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 9.08679788774945, "e0": 0.1,
                       "xI0": 1.0, "dist": 13.7253796777895, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0}),
    dict(name="idx14", dt=10.0, T=2.5, chi2=0.95,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 9.048439858487455, "e0": 0.3,
                       "xI0": 1.0, "dist": 14.134243467540808, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0}),
]


# 0PA-vs-2PA recovered MAP (from the critical-SNR grid) -- used as the CV start.
START_MAP = {
    "idx4":  [9.9999312902e+05, 1.0000034144e+01, 8.9999831452e-01, 9.0868360349e+00,
              9.9999256695e-02, 1.0471975512e+00, 7.8539816340e-01, 1.0118033613e-01, 2.9999226055e-01],
    "idx14": [9.9999725137e+05, 1.0000063704e+01, 9.0000202970e-01, 9.0484577531e+00,
              2.9999822161e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0347980230e-01, 3.0892018459e-01],
}


# --- helpers ---------------------------------------------------------------
def _to_float(x):
    return float(x.get()) if hasattr(x, "get") else float(x)


def make_freq_mask(n, dt, fmin):
    return (xp.fft.rfftfreq(n, dt) > fmin)[1:]


def highpass_clip(w, dt, fmin):
    n = w.shape[-1]
    f = xp.fft.rfftfreq(n, dt)
    return xp.fft.irfft(xp.fft.rfft(w, axis=-1) * (f >= fmin), n=n, axis=-1)


def build_context(case):
    sp, dt, T, chi2 = case["signal_param"], case["dt"], case["T"], case["chi2"]
    channels = [A1TDISens, E1TDISens, T1TDISens][:nchannels]
    tdi_chan = {2: "AE", 3: "AET"}[nchannels]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    def rkw():
        return dict(Tobs=T, t0=10000.0, dt=dt, index_lambda=8, index_beta=7, flip_hx=True,
                    is_ecliptic_latitude=False, remove_garbage="zero",
                    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                    force_backend="cuda12x" if use_gpu else "cpu",
                    order=20, tdi="1st generation", tdi_chan=tdi_chan)

    wfm = GenerateEMRIWaveform(SuperKludgeWaveform,
                               sum_kwargs=dict(pad_output=True, odd_len=True),
                               return_list=False, use_gpu=use_gpu)
    wresp = ResponseWrapper(waveform_gen=wfm, **rkw())

    def resp_args(vec, evolve_1pa):
        """14 EMRI params + [chi2, evolve_1PA, evolve_primary=F, evolve_2PA=F, deviation_included=F]."""
        p = {n: sp[n] for n in param_names_14}
        p.update(dict(zip(params_to_infer, vec)))
        return [p[n] for n in param_names_14] + [chi2, evolve_1pa, False, False, False]

    def make0(vec):                                    # 0PA template
        return xp.array(wresp(*resp_args(vec, False)))[:nchannels, :]

    true_inf = np.array([sp[n] for n in params_to_infer])
    s = highpass_clip(xp.array(wresp(*resp_args(true_inf, True)))[:nchannels, :], dt, F_MIN)
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

    def fisher_derivs(vec, dl):
        wp = {n: sp[n] for n in param_names_14}
        wp.update(dict(zip(params_to_infer, vec)))
        apa = {"chi2": chi2, "evolve_1PA": False, "evolve_primary": False,
               "evolve_2PA": False, "deviation_included": False}
        F = sef(wave_params={n: wp[n] for n in param_names_14}, param_names=params_to_infer,
                add_param_args=apa, deltas=dl, live_dangerously=False, stability_plot=False,
                der_order=8, Ndelta=(NDELTA if dl is None else None))
        return np.asarray(F[-1], dtype=float), xp.array(F[0]), sef.deltas

    return dict(make0=make0, ov=ov, chi2r=chi2r, ip=ip, s=s,
                fisher_derivs=fisher_derivs, snr=float(np.sqrt(ip(s, s))))


# --- optimisers ------------------------------------------------------------
def lm_climb(ctx, theta0, tag=""):
    ov, chi2r, fisher_derivs, s, ip, make0 = (ctx["ov"], ctx["chi2r"], ctx["fisher_derivs"],
                                              ctx["s"], ctx["ip"], ctx["make0"])
    npar = len(theta0)
    cur = np.array(theta0, dtype=float)
    lam, nu, dl, sigma = LAMBDA0, 2.0, None, np.ones(npar)
    for it in range(LM_MAX_ITERS):
        if it % RECOMPUTE_DELTAS_EVERY == 0:
            dl = None
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
            break
        dvec = np.abs(np.diag(G)) + 1e-30
        delta, ok, rel = np.zeros(npar), False, 0.0
        for _ in range(MAX_INNER):
            try:
                delta = np.linalg.solve(G + lam * np.diag(dvec), g)
            except np.linalg.LinAlgError:
                lam *= nu; nu *= 2.0; continue
            pred = float(delta @ (g + lam * dvec * delta))
            rho = (c0 - chi2r(cur + delta)) / pred if pred > 0 else -1.0
            if rho > 0.0:
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3); nu = 2.0
                rel = (c0 - chi2r(cur + delta)) / c0; ok = True; break
            lam *= nu; nu *= 2.0
        print(f"    CV{tag} it {it:>3} lam={lam:.1e} ov={ovc:.7f} chi2={c0:.3e} rel={rel:.1e}")
        if not ok:
            break
        cur = cur + delta
        if rel < REL_TOL:
            break
    return cur, ov(cur), sigma


def nm_refine(ctx, theta0, sigma, maxiter):
    chi2r, ov = ctx["chi2r"], ctx["ov"]
    n = len(theta0)
    simplex = np.vstack([np.zeros(n)] + [NM_STEP * np.eye(n)[i] for i in range(n)])
    res = minimize(lambda x: chi2r(theta0 + x * sigma), np.zeros(n), method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, maxiter=maxiter,
                                xatol=1e-4, fatol=1e-4, adaptive=True))
    cand = theta0 + res.x * sigma
    ok = ov(cand) > ov(theta0)
    print(f"    NM  nfev={res.nfev} ov {ov(theta0):.7f} -> {ov(cand):.7f} (accepted={ok})")
    return cand if ok else theta0


# --- per-case driver: start FROM THE INJECTION ------------------------------
def run_case(case):
    ctx = build_context(case)
    start = np.array(START_MAP[case["name"]], dtype=float)   # <-- 0PA-vs-2PA recovered MAP
    print(f"[{case['name']}] SNR={ctx['snr']:.2f}  overlap(MAP seed) = {ctx['ov'](start):.7f}")
    theta, ov1, sigma = lm_climb(ctx, start, tag="#1")
    if ov1 < OVERLAP_TARGET:
        theta = nm_refine(ctx, theta, sigma, NM_MAXITER)
        theta, ov1, sigma = lm_climb(ctx, theta, tag="#2")
    return dict(name=case["name"], snr=ctx["snr"], ov_start=float(ctx["ov"](start)),
                ov_final=float(ctx["ov"](theta)), chi2=float(ctx["chi2r"](theta)),
                start=start.copy(), params=theta.copy())


JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results_emri_grid_0pa.json")


def _jsonable(results):
    out = {"model": "0PA", "system": "EMRI_grid", "start": "0PA-vs-2PA recovered MAP",
           "n_params": len(params_to_infer), "param_names": params_to_infer,
           "generated_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    for r in results:
        out["cases"].append({
            "name": r["name"], "snr": r["snr"], "ov_start": r["ov_start"],
            "ov_final": r["ov_final"], "chi2": r["chi2"],
            "start_params": [float(x) for x in r["start"]],
            "params": [float(x) for x in r["params"]]})
    return out


def save_json(results):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(results), f, indent=2)


def main():
    results = []
    for case in CASES:
        print("=" * 70)
        results.append(run_case(case))
        save_json(results)

    print("\n" + "=" * 70)
    print("model = 0PA (EMRI grid, 9 params, start = 0PA-vs-2PA MAP)")
    print(f"{'case':6} {'SNR':>8} {'ov@seed':>11} {'ov@CV':>11} {'chi2':>11}")
    for r in results:
        print(f"{r['name']:6} {r['snr']:>8.2f} {r['ov_start']:>11.6f} {r['ov_final']:>11.6f} "
              f"{r['chi2']:>11.3e}")

    print("\noptimized 9-param points  [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0]:")
    for r in results:
        print(f"[{r['name']}] ov={r['ov_final']:.6f} = [{', '.join(f'{v:.8e}' for v in r['params'])}]")

    save_json(results)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
