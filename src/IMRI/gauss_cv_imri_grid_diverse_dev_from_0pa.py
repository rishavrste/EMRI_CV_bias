"""IMRI diverse grid points: 0PA + PN and 0PA + simple, SEEDED FROM THE 0PA BEST FIT.

Six IMRI (m1=1e6, m2=1e3) grid points (a, e0): idx0 (-0.9, 0.1), idx6 (-0.5, 0.2),
idx12 (0.0, 0.3), idx18 (+0.5, 0.4), idx20 (-0.9, 0.5), idx24 (+0.9, 0.5).

For each point we run BOTH deviation models, each started from that point's 0PA best fit
(9 params, from gauss_cv_imri_grid_diverse.py's from_MAP 0PA run) with dev_1 = dev_2 = 0.
Starting exactly at the 0PA optimum, the CV climb can only move uphill, so every result is
guaranteed >= the 0PA overlap, and it can unlock deviation gains the MAP seed misses.

    6 points x 2 models (PN, simple) = 12 CV climbs.

Hybrid-branch SuperKludgeFlux wiring:
    PN     : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2   (additive 2.5PN pdot/edot)
    simple : del_0_p (idx 7) = dev_1, del_0_e (idx 8) = dev_2   (multiplicative Edot/Ldot)

All runs: dt=10, T=1.0, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.

Run:  python gauss_cv_imri_grid_diverse_dev_from_0pa.py
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

PARAMS_11 = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0", "dev_1", "dev_2"]

MODELS = {
    "PN":     dict(tail=pn_tail,     apa=pn_apa),
    "simple": dict(tail=simple_tail, apa=simple_apa),
}


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


def _sig(a, p0, e0, dist):
    return {"m1": 1e6, "m2": 1e3, "a": a, "p0": p0, "e0": e0, "xI0": 1.0, "dist": dist,
            "qS": 1.0471975511965976, "phiS": 0.7853981633974483, "qK": 0.6283185307179586,
            "phiK": 0.5235987755982988, "Phi_phi0": 0.1, "Phi_theta0": 0.2, "Phi_r0": 0.3,
            "dev_1": 0.0, "dev_2": 0.0}


SIG = {
    "idx0":  _sig(-0.9, 25.49153734341529,  0.1, 27.02843444472985),
    "idx6":  _sig(-0.5, 24.925119118416994, 0.2, 34.986873947895496),
    "idx12": _sig(0.0,  24.210107793538548, 0.3, 49.51878055304779),
    "idx18": _sig(0.5,  23.459873246715226, 0.4, 71.50129336825191),
    "idx20": _sig(-0.9, 24.986950771522526, 0.5, 36.572827590723264),
    "idx24": _sig(0.9,  22.748688134890156, 0.5, 103.74029763727171),
}

# start_0pa = the 0PA best fit (9 params) from the IMRI diverse from_MAP 0PA run.
POINTS = [
    dict(name="idx0", a=-0.9, e0=0.1, ov_0pa=0.999199298429977, signal_param=SIG["idx0"],
         start_0pa=[1000814.8621593703, 999.7711386238577, -0.894444714849143, 25.46506188289737, 0.09987966061512686, 1.0435687749053484, 0.7872293523762621, 0.1873009144477341, -0.14258831192165902]),
    dict(name="idx6", a=-0.5, e0=0.2, ov_0pa=0.9990490961125784, signal_param=SIG["idx6"],
         start_0pa=[1001008.3558462914, 999.7072397298243, -0.4950379141986784, 24.89845972430005, 0.19987311856273443, 1.0516427099942374, 0.781512149255341, 0.1655263732370968, 0.05950552433286592]),
    dict(name="idx12", a=0.0, e0=0.3, ov_0pa=0.9989064907375035, signal_param=SIG["idx12"],
         start_0pa=[1000777.1085280953, 999.7486214253148, 0.00320554163295309, 24.190917901592726, 0.2999264464911814, 1.0495412396803343, 0.7816524373169244, 0.21285470032110831, 0.13443817210546125]),
    dict(name="idx18", a=0.5, e0=0.4, ov_0pa=0.9976515531712495, signal_param=SIG["idx18"],
         start_0pa=[1000328.249117074, 999.8773589721072, 0.5013280655283877, 23.452038701913107, 0.39996653413257727, 1.0436909056227845, 0.7857830847616458, 0.4058317911241096, 0.13350631068575847]),
    dict(name="idx20", a=-0.9, e0=0.5, ov_0pa=0.9988368379592104, signal_param=SIG["idx20"],
         start_0pa=[1001738.7905116025, 999.7241232470424, -0.8921518399208169, 24.94751244377835, 0.4999056591450554, 1.049762293768864, 0.776983661400623, 0.1903168034022235, 0.2515865077609545]),
    dict(name="idx24", a=0.9, e0=0.5, ov_0pa=0.9978522679856837, signal_param=SIG["idx24"],
         start_0pa=[999949.8255638336, 999.9519089495001, 0.8999427794194502, 22.750809301165035, 0.49998428684065715, 1.0468902260380994, 0.7920844085119978, 0.4818952298903054, 0.2016525213655288]),
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
        p.update(dict(zip(PARAMS_11, vec)))
        tail = [CHI2, evolve_1pa, False, False, deviation_on]
        if deviation_on:
            tail = tail + dev_tail(vec[9], vec[10])
        return [p[n] for n in param_names_14] + tail

    def make0(vec):                                    # 0PA + deviation template
        return xp.array(wresp(*resp_args(vec, False, True)))[:nchannels, :]

    true_inf = np.array([sp[n] for n in PARAMS_11])
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
        wp.update(dict(zip(PARAMS_11, vec)))
        apa = {"chi2": CHI2, "evolve_1PA": False, "evolve_primary": False,
               "evolve_2PA": False, "deviation_included": True}
        apa.update(dev_apa(vec[9], vec[10]))
        F = sef(wave_params={n: wp[n] for n in param_names_14}, param_names=PARAMS_11,
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


# --- driver ----------------------------------------------------------------
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results_imri_grid_diverse_dev_from_0pa.json")


def _jsonable(records):
    pts, order = {}, []
    for r in records:
        if r["point"] not in pts:
            pts[r["point"]] = {"name": r["point"], "a": r["a"], "e0": r["e0"],
                               "snr": r["snr"], "ov_0pa": r["ov_0pa"], "runs": {}}
            order.append(r["point"])
        pts[r["point"]]["runs"][r["model"]] = {
            "ov_start": r["ov_start"], "ov_final": r["ov_final"], "chi2": r["chi2"],
            "beats_0pa": bool(r["ov_final"] >= r["ov_0pa"] - 1e-9),
            "start_params": r["start_params"], "params": r["params"]}
    return {"system": "IMRI_grid_diverse", "branch": "hybrid",
            "model": "0PA + deviation (seeded from 0PA best fit)", "models": list(MODELS),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "points": [pts[n] for n in order]}


def save_json(records):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(records), f, indent=2)


def main():
    records = []
    for case in POINTS:
        start = list(case["start_0pa"]) + [0.0, 0.0]
        for mname, mcfg in MODELS.items():
            ctx = build_context(case, mcfg)
            print("=" * 70)
            print(f"[{case['name']} | model={mname} | from 0PA best fit]  SNR={ctx['snr']:.2f}  "
                  f"(0PA ref = {case['ov_0pa']:.6f})")
            d = cv_from(ctx, f"{case['name']}/{mname}", start)
            records.append(dict(
                point=case["name"], a=case["a"], e0=case["e0"],
                model=mname, snr=ctx["snr"], ov_0pa=case["ov_0pa"],
                ov_start=d["ov_start"], ov_final=d["ov_final"], chi2=d["chi2"],
                start_params=[float(x) for x in d["start"]],
                params=[float(x) for x in d["params"]]))
            save_json(records)

    print("\n" + "=" * 70)
    print("IMRI diverse grid: 0PA + PN / simple, seeded from 0PA best fit")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'model':>7} {'ov@0PAfit':>11} {'ov@CV':>11} "
          f"{'chi2':>11} {'ov@0PA':>11} {'>=0PA?':>7}")
    for r in records:
        flag = "YES" if r["ov_final"] >= r["ov_0pa"] - 1e-9 else "no"
        print(f"{r['point']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {r['model']:>7} {r['ov_start']:>11.6f} "
              f"{r['ov_final']:>11.6f} {r['chi2']:>11.3e} {r['ov_0pa']:>11.6f} {flag:>7}")

    print("\noptimized 11-param points  [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, dev_1, dev_2]:")
    for r in records:
        print(f"[{r['point']} {r['model']}] ov={r['ov_final']:.6f} = "
              f"[{', '.join(f'{v:.8e}' for v in r['params'])}]")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
