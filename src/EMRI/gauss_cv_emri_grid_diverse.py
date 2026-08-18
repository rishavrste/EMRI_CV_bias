"""EMRI diverse grid points: 0PA / 0PA+PN / 0PA+simple, from TWO starts.

Four grid points spanning distinct spin and eccentricity (all different from the a=+0.9
cases used before):
    idx0  : a=-0.9, e0=0.1      idx6  : a=-0.5, e0=0.2
    idx12 : a= 0.0, e0=0.3      idx18 : a=+0.5, e0=0.4

For each point we run THREE template models and, for each, CV from TWO starts:
    models : 0PA (9 params) | 0PA+PN (11) | 0PA+simple (11)
    starts : from_injection  (the 1PA injected parameters)
             from_MAP        (the 0PA-vs-2PA recovered best fit)
=> 4 points x 3 models x 2 starts = 24 CV climbs.

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN     : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2   (additive 2.5PN pdot/edot)
    simple : del_0_p (idx 7) = dev_1, del_0_e (idx 8) = dev_2   (multiplicative Edot/Ldot)
Layout: [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, del_0_p, del_0_e]

NOTE: starting from the 1PA injection usually stalls (the 0PA template is dephased there);
the from_MAP start is the reliable one.  Both are run so the difference is visible.

All runs: dt=10, T=2.5, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.  WARNING: 24 climbs -> long runtime.

Run:  python gauss_cv_emri_grid_diverse.py
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
    dict(name="idx0",
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.9, "p0": 13.0673756813688, "e0": 0.1,
                       "xI0": 1.0, "dist": 2.8282909665832117, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000448842e+06, 1.0000062758e+01, -8.9985997562e-01, 1.3066900351e+01,
               9.9981605459e-02, 1.0471975512e+00, 7.8539816340e-01, 1.1010061766e-01,
               2.6898423615e-01]),
    dict(name="idx6",
         signal_param={"m1": 1e6, "m2": 10.0, "a": -0.5, "p0": 12.218336776485106, "e0": 0.2,
                       "xI0": 1.0, "dist": 4.09963816070808, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000578633e+06, 1.0000071718e+01, -4.9985782093e-01, 1.2217803482e+01,
               1.9998784680e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0170443578e-01,
               2.9990713717e-01]),
    dict(name="idx12",
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.0, "p0": 11.105928547343611, "e0": 0.3,
                       "xI0": 1.0, "dist": 6.511129386938633, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000514197e+06, 1.0000104290e+01, 9.8809396009e-05, 1.1105511492e+01,
               2.9999155917e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0021045440e-01,
               3.0000005703e-01]),
    dict(name="idx18",
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.5, "p0": 9.941538910913811, "e0": 0.4,
                       "xI0": 1.0, "dist": 10.065969264365489, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1.0000453842e+06, 1.0000171160e+01, 5.0006050192e-01, 9.9412389174e+00,
               3.9999268941e-01, 1.0471975512e+00, 7.8539816340e-01, 1.0011050683e-01,
               3.0378736561e-01]),
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
    """Build signal + template + Fisher machinery for one (point, model)."""
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

    def make0(vec):                                    # template: 0PA (dev_on=False) or 0PA+dev
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
                         "results_emri_grid_diverse.json")


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
    return {"system": "EMRI_grid_diverse", "branch": "hybrid",
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
            ctx = build_context(case, mcfg)   # signal SNR is the same across models
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
    print("EMRI diverse grid: 0PA / PN / simple, from injection & MAP")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'model':>7} {'start':>15} "
          f"{'ov@start':>11} {'ov@CV':>11} {'chi2':>11}")
    for r in records:
        print(f"{r['point']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {r['model']:>7} {r['start']:>15} "
              f"{r['ov_start']:>11.6f} {r['ov_final']:>11.6f} {r['chi2']:>11.3e}")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
