"""EMRI diverse grid points (set 2): 0PA / 0PA+PN / 0PA+simple, from TWO starts.

Six grid points spanning ALL 5 spins and ALL 5 eccentricities (distinct from every point
used before):
    idx3  : a=+0.5, e0=0.1     idx5  : a=-0.9, e0=0.2     idx11 : a=-0.5, e0=0.3
    idx17 : a= 0.0, e0=0.4     idx20 : a=-0.9, e0=0.5     idx24 : a=+0.9, e0=0.5

For each point we run THREE template models and, for each, CV from TWO starts:
    models : 0PA (9 params) | 0PA+PN (11) | 0PA+simple (11)
    starts : from_injection  (the 1PA injected parameters)
             from_MAP        (the 0PA-vs-2PA recovered best fit)
=> 6 points x 3 models x 2 starts = 36 CV climbs.

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN     : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2   (additive 2.5PN pdot/edot)
    simple : del_0_p (idx 7) = dev_1, del_0_e (idx 8) = dev_2   (multiplicative Edot/Ldot)
Layout: [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, del_0_p, del_0_e]

NOTE: starting from the 1PA injection usually stalls (the 0PA template is dephased there);
the from_MAP start is the reliable one.  Both are run so the difference is visible.

All runs: dt=10, T=2.5, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.  WARNING: 36 climbs -> long runtime.

Run:  python gauss_cv_emri_grid_diverse2.py
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


# --- deviation wiring (hybrid branch), per model ---------------------------
def pn_tail(d1, d2):     return [d1, d2, 0.0, 0.0]                                  # C_p(5), C_e(6)
def pn_apa(d1, d2):      return {"dev_1": d1, "dev_2": d2, "del_0_p": 0.0, "del_0_e": 0.0}
def simple_tail(d1, d2): return [0.0, 0.0, d1, d2]                                  # del_0_p(7), del_0_e(8)
def simple_apa(d1, d2):  return {"C_p": 0.0, "C_e": 0.0, "dev_1": d1, "dev_2": d2}

PARAMS_9 = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
PARAMS_11 = PARAMS_9 + ["dev_1", "dev_2"]

MODELS = {
    "0PA":    dict(params=PARAMS_9,  deviation=False, tail=None,        apa=None),
    "PN":     dict(params=PARAMS_11, deviation=True,  tail=pn_tail,     apa=pn_apa),
    "simple": dict(params=PARAMS_11, deviation=True,  tail=simple_tail, apa=simple_apa),
}
STARTS = ["from_injection", "from_MAP"]


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
DT, T, CHI2 = 10.0, 2.5, 0.95

# map9 = the 0PA-vs-2PA recovered MAP (9 params).  from_injection uses signal_param instead.
POINTS = [
    dict(name="idx3",
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.5, "p0": 9.987641937579998, "e0": 0.1,
                       "xI0": 1.0, "dist": 9.414980595910297, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000170522e+06, 1.0000058073e+01, 5.0003157677e-01, 9.9874990568e+00,
               9.9988957308e-02, 1.0471975512e+00, 7.8539816340e-01, 1.0234223391e-01,
               2.9899186891e-01]),
    dict(name="idx5",
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.9, "p0": 13.122559320333108, "e0": 0.2,
                       "xI0": 1.0, "dist": 3.0324274273280873, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000643634e+06, 1.0000080419e+01, -8.9981777156e-01, 1.3121923538e+01,
               1.9998689559e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0302734196e-01,
               2.9768138682e-01]),
    dict(name="idx11",
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.5, "p0": 12.255354927771608, "e0": 0.3,
                       "xI0": 1.0, "dist": 4.4616428166698565, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000650384e+06, 1.0000114447e+01, -4.9984623079e-01, 1.2254775202e+01,
               2.9999001830e-01, 1.0471975512e+00, 7.8539816340e-01, 9.9544931049e-02,
               3.0023172742e-01]),
    dict(name="idx17",
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.0, "p0": 11.104224019622757, "e0": 0.4,
                       "xI0": 1.0, "dist": 6.926568915250337, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000609991e+06, 1.0000165343e+01, 1.1059446669e-04, 1.1103755181e+01,
               3.9999181636e-01, 1.0471975512e+00, 7.8539816340e-01, 9.9998881545e-02,
               2.9999465806e-01]),
    dict(name="idx20",
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.9, "p0": 13.302718414742197, "e0": 0.5,
                       "xI0": 1.0, "dist": 4.121669070558761, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[9.9989240810e+05, 9.9996423909e+00, -9.0017655175e-01, 1.3303409652e+01,
               5.0001323590e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0079272165e-01,
               3.0114256316e-01]),
    dict(name="idx24",
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 8.926976081703573, "e0": 0.5,
                       "xI0": 1.0, "dist": 14.334976856664628, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000091405e+06, 1.0000146791e+01, 9.0001031561e-01, 8.9269416917e+00,
               4.9999666944e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0596605716e-01,
               2.9984530381e-01]),
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


def build_context(case, mcfg):
    sp = case["signal_param"]
    params = mcfg["params"]
    dev_on = mcfg["deviation"]
    dev_tail, dev_apa = mcfg["tail"], mcfg["apa"]
    channels = [A1TDISens, E1TDISens, T1TDISens][:nchannels]
    tdi_chan = {2: "AE", 3: "AET"}[nchannels]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    def rkw():
        return dict(Tobs=T, t0=10000.0, dt=DT, index_lambda=8, index_beta=7, flip_hx=True,
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
        p.update(dict(zip(params, vec)))
        tail = [CHI2, evolve_1pa, False, False, deviation_on]
        if deviation_on:
            tail = tail + dev_tail(vec[9], vec[10])
        return [p[n] for n in param_names_14] + tail

    def make0(vec):
        return xp.array(wresp(*resp_args(vec, False, dev_on)))[:nchannels, :]

    true_inf = np.array([sp[n] for n in params])
    s = highpass_clip(xp.array(wresp(*resp_args(true_inf, True, False)))[:nchannels, :], DT, F_MIN)
    PSD = xp.array(generate_PSD(waveform=s, dt=DT, noise_PSD=get_sensitivity,
                                channels=channels, noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    fmask = make_freq_mask(s.shape[-1], DT, F_MIN)

    def ip(a, b):
        return _to_float(inner_product(a, b, PSD=PSD, dt=DT, freq_mask=fmask, use_gpu=use_gpu))

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
        T=T, dt=DT, stability_plot=False, der_order=6, Ndelta=NDELTA,
        plunge_check=True, return_derivatives=True)

    def fisher_derivs(vec, dl):
        wp = {n: sp[n] for n in param_names_14}
        wp.update(dict(zip(params, vec)))
        apa = {"chi2": CHI2, "evolve_1PA": False, "evolve_primary": False,
               "evolve_2PA": False, "deviation_included": dev_on}
        if dev_on:
            apa.update(dev_apa(vec[9], vec[10]))
        F = sef(wave_params={n: wp[n] for n in param_names_14}, param_names=params,
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


def cv_from(ctx, tag, start):
    theta = np.array(start, dtype=float)
    print(f"    start overlap = {ctx['ov'](theta):.7f}")
    theta, ov1, sigma = lm_climb(ctx, theta, tag=f" {tag}#1")
    if ov1 < OVERLAP_TARGET:
        theta = nm_refine(ctx, theta, sigma, NM_MAXITER)
        theta, ov1, sigma = lm_climb(ctx, theta, tag=f" {tag}#2")
    return dict(ov_start=float(ctx["ov"](start)), ov_final=float(ctx["ov"](theta)),
                chi2=float(ctx["chi2r"](theta)), params=theta.copy(),
                start=np.array(start, dtype=float))


def build_start(case, mcfg, sname):
    base = ([case["signal_param"][n] for n in PARAMS_9] if sname == "from_injection"
            else list(case["map9"]))
    if mcfg["deviation"]:
        base = list(base) + [0.0, 0.0]
    return np.array(base, dtype=float)


# --- driver ----------------------------------------------------------------
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results_emri_grid_diverse2.json")


def _jsonable(records):
    pts, order = {}, []
    for r in records:
        if r["point"] not in pts:
            pts[r["point"]] = {"name": r["point"], "a": r["a"], "e0": r["e0"],
                               "snr": r["snr"], "runs": {}}
            order.append(r["point"])
        pts[r["point"]]["runs"].setdefault(r["model"], {})[r["start"]] = {
            "ov_start": r["ov_start"], "ov_final": r["ov_final"], "chi2": r["chi2"],
            "start_params": r["start_params"], "params": r["params"]}
    return {"system": "EMRI_grid_diverse2", "branch": "hybrid",
            "models": list(MODELS), "starts": STARTS,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "points": [pts[n] for n in order]}


def save_json(records):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(records), f, indent=2)


def main():
    records = []
    for case in POINTS:
        for mname, mcfg in MODELS.items():
            ctx = build_context(case, mcfg)
            for sname in STARTS:
                print("=" * 70)
                print(f"[{case['name']} | model={mname} | {sname}]  SNR={ctx['snr']:.2f}")
                d = cv_from(ctx, f"{case['name']}/{mname}/{sname}", build_start(case, mcfg, sname))
                records.append(dict(
                    point=case["name"], a=case["signal_param"]["a"], e0=case["signal_param"]["e0"],
                    model=mname, start=sname, snr=ctx["snr"],
                    ov_start=d["ov_start"], ov_final=d["ov_final"], chi2=d["chi2"],
                    start_params=[float(x) for x in d["start"]],
                    params=[float(x) for x in d["params"]]))
                save_json(records)

    print("\n" + "=" * 70)
    print("EMRI diverse grid set 2: 0PA / PN / simple, from injection & MAP")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'model':>7} {'start':>15} "
          f"{'ov@start':>11} {'ov@CV':>11} {'chi2':>11}")
    for r in records:
        print(f"{r['point']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {r['model']:>7} {r['start']:>15} "
              f"{r['ov_start']:>11.6f} {r['ov_final']:>11.6f} {r['chi2']:>11.3e}")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
