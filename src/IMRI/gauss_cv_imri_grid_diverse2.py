"""IMRI diverse grid points (set 2): 0PA / 0PA+PN / 0PA+simple, from the MAP start only.

Six fresh IMRI (m1=1e6, m2=1e3) grid points -- none used before -- again spanning all 5 spins
and all 5 eccentricities, with a retrograde/prograde pair at e0=0.3 (idx10 vs idx14):
    idx4  : a=+0.9, e0=0.1     idx8  : a=+0.5, e0=0.2     idx10 : a=-0.9, e0=0.3
    idx14 : a=+0.9, e0=0.3     idx16 : a=-0.5, e0=0.4     idx22 : a= 0.0, e0=0.5

For each point we run THREE template models, each CV-climbed from a SINGLE start:
    models : 0PA (9 params) | 0PA+PN (11) | 0PA+simple (11)
    start  : from_MAP  (the 0PA-vs-2PA recovered best fit)   -- the reliable seed
=> 6 points x 3 models = 18 CV climbs.

The from_injection start is omitted: in every previous run it stalled (0PA template is
dephased at the injection), so it wastes GPU time.

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN     : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2   (additive 2.5PN pdot/edot)
    simple : del_0_p (idx 7) = dev_1, del_0_e (idx 8) = dev_2   (multiplicative Edot/Ldot)

All runs: dt=10, T=1.0, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.

Run:  python gauss_cv_imri_grid_diverse2.py
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
STARTS = ["from_MAP"]


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
DT, T, CHI2 = 10.0, 1.0, 0.95

# map9 = the 0PA-vs-2PA recovered MAP (9 params); the CV start for every model.
POINTS = [
    dict(name="idx4",
         signal_param={"m1": 1e6, "m2": 1e3, "a": 0.9, "p0": 23.34710828052707, "e0": 0.1, "xI0": 1.0,
                       "dist": 86.18237482063395, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[997197.58403, 1001.565926, 0.89972201636, 23.391815851, 0.10018919767, 1.0471975512, 0.7853981634, 0.32197916142, 5.4757189657]),
    dict(name="idx8",
         signal_param={"m1": 1e6, "m2": 1e3, "a": 0.5, "p0": 23.72498880054052, "e0": 0.2, "xI0": 1.0,
                       "dist": 67.00164144043758, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1000394.4568, 999.82057151, 0.50138666982, 23.715951184, 0.1999386234, 1.0471975512, 0.7853981634, 0.17211451221, 0.067504650802]),
    dict(name="idx10",
         signal_param={"m1": 1e6, "m2": 1e3, "a": -0.9, "p0": 25.363852533835356, "e0": 0.3, "xI0": 1.0,
                       "dist": 30.10036613298705, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1001506.5225, 999.6514329, -0.89268515406, 25.326845441, 0.29987401636, 1.0471975512, 0.7853981634, 0.18173031091, 0.17106457884]),
    dict(name="idx14",
         signal_param={"m1": 1e6, "m2": 1e3, "a": 0.9, "p0": 23.187527395714866, "e0": 0.3, "xI0": 1.0,
                       "dist": 92.62356158607001, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[999986.0976, 999.88249927, 0.89987810448, 23.188870736, 0.30000977424, 1.0471975512, 0.7853981634, 0.33357781435, 0.15692584885]),
    dict(name="idx16",
         signal_param={"m1": 1e6, "m2": 1e3, "a": -0.5, "p0": 24.685530204456544, "e0": 0.4, "xI0": 1.0,
                       "dist": 39.70108872519907, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1001292.9423, 999.70161787, -0.49440485147, 24.655026616, 0.39991023234, 1.0471975512, 0.7853981634, 0.20174406872, 0.21775504098]),
    dict(name="idx22",
         signal_param={"m1": 1e6, "m2": 1e3, "a": 0.0, "p0": 23.800463873264377, "e0": 0.5, "xI0": 1.0,
                       "dist": 55.22296699286325, "qS": 1.0471975511965976,
                       "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
                       "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         map9=[1000994.9909, 999.77106515, 0.0035938406719, 23.778725385, 0.49993586465, 1.0471975512, 0.7853981634, 0.21489486193, 0.25162688065]),
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
                         "results_imri_grid_diverse2.json")


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
    return {"system": "IMRI_grid_diverse2", "branch": "hybrid",
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
    print("IMRI diverse grid set 2: 0PA / PN / simple, from MAP")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'model':>7} {'start':>15} "
          f"{'ov@start':>11} {'ov@CV':>11} {'chi2':>11}")
    for r in records:
        print(f"{r['point']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {r['model']:>7} {r['start']:>15} "
              f"{r['ov_start']:>11.6f} {r['ov_final']:>11.6f} {r['chi2']:>11.3e}")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
