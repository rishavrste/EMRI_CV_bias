"""EMRI diverse grid points: 0PA + SIMPLE-PE deviation, CV from TWO starts.

Four grid points: idx0 (-0.9, 0.1), idx6 (-0.5, 0.2), idx12 (0.0, 0.3), idx18 (0.5, 0.4).

Simple-PE deviation (dev_a_pe-branch SuperKludgeFlux): multiplicative rescaling applied
    DIRECTLY to the adiabatic p, e trajectory rates,
        additional_args[5] = del_0_p = dev_1 :  pdot -> (1 + eta * del_0_p) * pdot
        additional_args[6] = del_0_e = dev_2 :  edot -> (1 + eta * del_0_e) * edot
    deviation_included = True (index 4).  Layout on this branch:
        [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, del_0_p, del_0_e]

For each point CV (adaptive-LM; if it stalls, one 1000-step Nelder-Mead escape then one more
CV) is run from BOTH:
    * from_MAP    : the 0PA-vs-2PA recovered MAP        (+ dev=0)
    * from_0PAcv  : our CV-optimized 0PA best fit       (+ dev=0)

All runs: dt=10, T=2.5, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'dev_a_pe' branch.

Run:  python gauss_cv_emri_grid_diverse_simple_pe.py
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

# --- deviation wiring: SIMPLE-PE = del_0_p (idx 5), del_0_e (idx 6) ---------
DEV_NAME = "simple_pe"


def dev_tail(dev_1, dev_2):
    """[del_0_p, del_0_e] slots (idx 5,6) of additional_args -- direct p,e rescaling."""
    return [dev_1, dev_2]


def dev_apa(dev_1, dev_2):
    """add_param_args entries after deviation_included, ordered to hit idx 5,6."""
    return {"dev_1": dev_1, "dev_2": dev_2}


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

# start_map    = 0PA-vs-2PA recovered MAP (9 params);  start_0pacv = our CV 0PA best fit (9 params).
CASES = [
    dict(name="idx0", ov_0pa=0.999986147968984,
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.9, "p0": 13.0673756813688, "e0": 0.1,
                       "xI0": 1.0, "dist": 2.8282909665832117, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start_map=[1.0000448842e+06, 1.0000062758e+01, -8.9985997562e-01, 1.3066900351e+01,
                    9.9981605459e-02, 1.0471975512e+00, 7.8539816340e-01, 1.1010061766e-01,
                    2.6898423615e-01],
         start_0pacv=[1000044.7720278731, 10.00005294590481, -0.8998601592219645,
                      13.066899681550376, 0.09998241980944299, 1.0460138218995223,
                      0.7853309107578463, 0.10231277993160921, 0.28107103756891927]),
    dict(name="idx6", ov_0pa=0.9999843699680888,
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.5, "p0": 12.218336776485106, "e0": 0.2,
                       "xI0": 1.0, "dist": 4.09963816070808, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start_map=[1.0000578633e+06, 1.0000071718e+01, -4.9985782093e-01, 1.2217803482e+01,
                    1.9998784680e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0170443578e-01,
                    2.9990713717e-01],
         start_0pacv=[1000053.6810327651, 10.000072690976687, -0.49986579385681296,
                      12.217836372258024, 0.19998845561480535, 1.04616365320541,
                      0.7851889874278619, 0.10143227642198388, 0.2875618108647028]),
    dict(name="idx12", ov_0pa=0.9999614279079387,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.0, "p0": 11.105928547343611, "e0": 0.3,
                       "xI0": 1.0, "dist": 6.511129386938633, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start_map=[1.0000514197e+06, 1.0000104290e+01, 9.8809396009e-05, 1.1105511492e+01,
                    2.9999155917e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0021045440e-01,
                    3.0000005703e-01],
         start_0pacv=[1000051.4782712947, 10.000104989414359, 9.899365889262895e-05,
                      11.105511081442879, 0.29999153793538824, 1.0457118389778928,
                      0.7849727681788038, 0.10003194166108595, 0.2913994760867554]),
    dict(name="idx18", ov_0pa=0.9997665959425159,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.5, "p0": 9.941538910913811, "e0": 0.4,
                       "xI0": 1.0, "dist": 10.065969264365489, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start_map=[1.0000453842e+06, 1.0000171160e+01, 5.0006050192e-01, 9.9412389174e+00,
                    3.9999268941e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0011050683e-01,
                    3.0378736561e-01],
         start_0pacv=[1000043.1904769371, 10.00017873801177, 0.5000583381308088,
                      9.941254326282897, 0.39999262975456285, 1.045400821566479,
                      0.7841933045239787, 0.12124668316943134, 0.2675510956952596]),
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
    sp, dt, T, chi2 = case["signal_param"], 10.0, 2.5, 0.95
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


# --- per-case driver: run CV from BOTH starts (each + dev=0) -----------------
def run_case(case):
    ctx = build_context(case)
    out = {"name": case["name"], "snr": ctx["snr"], "ov_0pa": case["ov_0pa"],
           "a": case["signal_param"]["a"], "e0": case["signal_param"]["e0"]}
    out["from_MAP"] = cv_from(ctx, case["name"], "from_MAP",
                              list(case["start_map"]) + [0.0, 0.0])
    out["from_0PAcv"] = cv_from(ctx, case["name"], "from_0PAcv",
                                list(case["start_0pacv"]) + [0.0, 0.0])
    return out


JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"results_emri_grid_diverse_{DEV_NAME}.json")


def _jsonable(results):
    out = {"model": f"0PA + {DEV_NAME} deviation", "deviation": DEV_NAME,
           "system": "EMRI_grid_diverse", "branch": "dev_a_pe",
           "n_params": len(params_to_infer), "param_names": params_to_infer,
           "generated_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    for r in results:
        entry = {"name": r["name"], "a": r["a"], "e0": r["e0"], "snr": r["snr"],
                 "ov_0pa": r["ov_0pa"]}
        best_label, best_ov = None, -1.0
        for label in ("from_MAP", "from_0PAcv"):
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
        save_json(results)

    print("\n" + "=" * 70)
    print(f"model = 0PA + {DEV_NAME} deviation  (EMRI diverse grid, 11 params)")
    print(f"{'case':6} {'a':>5} {'e0':>4} {'start':>11} {'ov@start':>11} {'ov@CV':>11} "
          f"{'chi2':>11} {'ov@0PA':>11}")
    for r in results:
        for label in ("from_MAP", "from_0PAcv"):
            d = r[label]
            print(f"{r['name']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {label:>11} {d['ov_start']:>11.6f} "
                  f"{d['ov_final']:>11.6f} {d['chi2']:>11.3e} {r['ov_0pa']:>11.6f}")

    print("\noptimized 11-param points  [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, dev_1, dev_2]:")
    for r in results:
        for label in ("from_MAP", "from_0PAcv"):
            v = r[label]["params"]
            print(f"[{r['name']} {label}] ov={r[label]['ov_final']:.6f} = [{', '.join(f'{x:.8e}' for x in v)}]")

    save_json(results)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
