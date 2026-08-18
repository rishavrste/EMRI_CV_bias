"""EMRI 0PA + PN (2.5PN) deviation, refined with CV from TWO starts.

PN (2.5PN) deviation (hybrid-branch SuperKludgeFlux): additive pdot/edot correction,
    additional_args[5]=C_p=dev_1, [6]=C_e=dev_2, deviation_included=True.
    Layout: [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, del_0_p, del_0_e]

For each EMRI case the CV method (adaptive-LM; if it stalls below the overlap limit, one
1000-step Nelder-Mead escape then one more CV) is run from BOTH:
    * start_0pa : the NM 0PA best fit with dev_1 = dev_2 = 0
    * start_dev : the NM 0PA+PN-deviation best fit
and both endpoints are reported so you can see which start wins.  The external NM reference
overlaps (0PA and 0PA+deviation) are carried through for comparison.

Per-case dt, T, chi2 (secondary spin) are taken from the signal (idx0: dt=5,T=1.0,chi2=0.0;
idx9/idx13: dt=10,T=2.5,chi2=0.95).  nchannels=3, no noise.

Run:  python gauss_cv_emri_pn_dev.py
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

# --- deviation wiring: PN (2.5PN) = C_p (idx 5), C_e (idx 6) ----------------
DEV_NAME = "PN"


def dev_tail(dev_1, dev_2):
    """[C_p, C_e, del_0_p, del_0_e] slots of additional_args for the PN (2.5PN) deviation."""
    return [dev_1, dev_2, 0.0, 0.0]


def dev_apa(dev_1, dev_2):
    """add_param_args entries after deviation_included, ordered to hit idx 5,6,7,8."""
    return {"dev_1": dev_1, "dev_2": dev_2, "del_0_p": 0.0, "del_0_e": 0.0}


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
params_to_infer = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0",
                   "dev_1", "dev_2"]

CASES = [
    dict(name="idx0", dt=5.0, T=1.0, chi2=0.0,
         zeropa_overlap=0.99999443, nm_dev_overlap=0.9986067657080662,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 7.5, "e0": 0.5, "xI0": 1.0,
                       "dist": 5.0, "qS": 0.7853981633974483, "phiS": 1.0, "qK": 1.0,
                       "phiK": 1.0471975511965976, "Phi_phi0": 0.9, "Phi_theta0": 0.5,
                       "Phi_r0": 0.4, "dev_1": 0.0, "dev_2": 0.0},
         start_0pa=[1.00014867e+06, 1.00010513e+01, 9.00096370e-01,
                    7.49942614e+00, 4.99974688e-01, 7.83388290e-01,
                    9.99626083e-01, 9.09269038e-01, 3.96038198e-01, 0.0, 0.0],
         start_dev=[1001634.5052297474, 10.013531592050047, 0.9008881957061771,
                    7.494406049462866, 0.49963984824180485, 0.7920618620465177,
                    1.011565734126598, 0.965833226490635, 0.51047866455633,
                    0.2465764435421102, -13.05973884773176]),
    dict(name="idx9", dt=10.0, T=2.5, chi2=0.95,
         zeropa_overlap=0.99818007, nm_dev_overlap=0.9995428908954582,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 9.07414088, "e0": 0.2,
                       "xI0": 1.0, "dist": 13.8096925, "qS": 1.04719755, "phiS": 0.785398163,
                       "qK": 0.628318531, "phiK": 0.523598776, "Phi_phi0": 0.1,
                       "Phi_theta0": 0.2, "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start_0pa=[9.99994929e+05, 1.00000755e+01, 9.00001697e-01,
                    9.07417360e+00, 1.99997410e-01, 1.04653506e+00,
                    7.83187637e-01, 1.59230827e-01, 9.90108700e-02, 0.0, 0.0],
         start_dev=[1000007.3064101462, 9.999410375134975, 0.9000127192599089,
                    9.074071159805102, 0.20000475732861356, 1.0482850200266476,
                    0.7844199402867075, 0.12661137120309485, 0.24181029046804597,
                    -20.17003070886153, -0.8428739349130099]),
    dict(name="idx13", dt=10.0, T=2.5, chi2=0.95,
         zeropa_overlap=None, nm_dev_overlap=0.9998995470850759,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.5, "p0": 9.97066819, "e0": 0.3,
                       "xI0": 1.0, "dist": 9.78949272, "qS": 1.04719755, "phiS": 0.785398163,
                       "qK": 0.628318531, "phiK": 0.523598776, "Phi_phi0": 0.1,
                       "Phi_theta0": 0.2, "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start_0pa=[1.00003623e+06, 1.00001339e+01, 5.00052129e-01,
                    9.97041651e+00, 2.99991839e-01, 1.04533235e+00,
                    7.83864901e-01, 1.13205146e-01, 2.54180379e-01, 0.0, 0.0],
         start_dev=[1000130.8327781362, 10.00079770659335, 0.5001362939488679,
                    9.969961366053813, 0.2999545544870821, 1.0474335312621117,
                    0.784031893602174, 0.12631188468377094, 0.28624276030600854,
                    4.890018986449651, -4.471832004638747]),
]


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

    def resp_args(vec, evolve_1pa, deviation_on):
        p = {n: sp[n] for n in param_names_14}
        p.update(dict(zip(params_to_infer, vec)))
        return [p[n] for n in param_names_14] + [
            chi2, evolve_1pa, False, False, deviation_on, *dev_tail(vec[9], vec[10])]

    def make0(vec):                                    # 0PA + deviation template
        return xp.array(wresp(*resp_args(vec, False, True)))[:nchannels, :]

    true_inf = np.array([sp[n] for n in params_to_infer])
    s = highpass_clip(xp.array(wresp(*resp_args(true_inf, True, False)))[:nchannels, :], dt, F_MIN)
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
               "evolve_2PA": False, "deviation_included": True}
        apa.update(dev_apa(vec[9], vec[10]))
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


def cv_from(ctx, name, label, start):
    theta = np.array(start, dtype=float)
    print(f"\n[{name}] --- start = {label} (overlap {ctx['ov'](theta):.7f}) ---")
    theta, ov1, sigma = lm_climb(ctx, theta, tag=f" {label}#1")
    if ov1 < OVERLAP_TARGET:
        theta = nm_refine(ctx, theta, sigma, NM_MAXITER)
        theta, ov1, sigma = lm_climb(ctx, theta, tag=f" {label}#2")
    return dict(ov_start=float(ctx["ov"](start)), ov_final=float(ctx["ov"](theta)),
                chi2=float(ctx["chi2r"](theta)), params=theta.copy(),
                start=np.array(start, dtype=float))


# --- per-case driver: run CV from BOTH starts -------------------------------
def run_case(case):
    ctx = build_context(case)
    out = {"name": case["name"], "snr": ctx["snr"],
           "zeropa_overlap": case["zeropa_overlap"], "nm_dev_overlap": case["nm_dev_overlap"]}
    out["from_0PA"] = cv_from(ctx, case["name"], "from_0PA", case["start_0pa"])
    out["from_NMdev"] = cv_from(ctx, case["name"], "from_NMdev", case["start_dev"])
    return out


JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"results_emri_{DEV_NAME}.json")


def _jsonable(results):
    out = {"model": f"0PA + {DEV_NAME} deviation", "deviation": DEV_NAME, "system": "EMRI",
           "n_params": len(params_to_infer), "param_names": params_to_infer,
           "generated_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    for r in results:
        entry = {"name": r["name"], "snr": r["snr"],
                 "zeropa_overlap": r["zeropa_overlap"], "nm_dev_overlap": r["nm_dev_overlap"]}
        best_label, best_ov = None, -1.0
        for label in ("from_0PA", "from_NMdev"):
            d = r[label]
            entry[label] = {"ov_start": d["ov_start"], "ov_final": d["ov_final"],
                            "chi2": d["chi2"],
                            "start_params": [float(x) for x in d["start"]],
                            "params": [float(x) for x in d["params"]]}
            if d["ov_final"] > best_ov:
                best_ov, best_label = d["ov_final"], label
        entry["best_start"], entry["best_overlap"] = best_label, best_ov
        out["cases"].append(entry)
    return out


def save_json(results):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(results), f, indent=2)


def main():
    results = []
    for case in CASES:
        print("=" * 70)
        results.append(run_case(case))
        save_json(results)          # incremental: partial results survive a later crash

    print("\n" + "=" * 70)
    print(f"model = 0PA + {DEV_NAME} deviation  (EMRI, 11 params)")
    print(f"{'case':6} {'SNR':>8} {'start':>10} {'ov@start':>11} {'ov@CV':>11} "
          f"{'ov@0PA':>10} {'ov@NMdev':>10}")
    for r in results:
        z = r["zeropa_overlap"]
        zs = f"{z:10.6f}" if z is not None else f"{'n/a':>10}"
        for label in ("from_0PA", "from_NMdev"):
            d = r[label]
            print(f"{r['name']:6} {r['snr']:>8.2f} {label:>10} {d['ov_start']:>11.6f} "
                  f"{d['ov_final']:>11.6f} {zs} {r['nm_dev_overlap']:>10.6f}")

    print("\noptimized 11-param points  [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, dev_1, dev_2]:")
    for r in results:
        for label in ("from_0PA", "from_NMdev"):
            v = r[label]["params"]
            print(f"[{r['name']} {label}] ov={r[label]['ov_final']:.6f} = [{', '.join(f'{x:.8e}' for x in v)}]")

    save_json(results)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
