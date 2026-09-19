"""EMRI a x e0 grid -- the remaining un-optimised cells: 0PA / 0PA+PN / 0PA+simple.

Covers the 11 grid cells that had no CV run at all:
    idx1  (-0.5, 0.1)  idx2  ( 0.0, 0.1)  idx7  ( 0.0, 0.2)  idx8  ( 0.5, 0.2)
    idx10 (-0.9, 0.3)  idx15 (-0.9, 0.4)  idx16 (-0.5, 0.4)  idx19 ( 0.9, 0.4)
    idx21 (-0.5, 0.5)  idx22 ( 0.0, 0.5)  idx23 ( 0.5, 0.5)
Already covered elsewhere: idx0,3,4,5,6,9,11,12,13,14,17,18,20,24.

Models and starts (per point, 5 CV climbs):
    0PA    (9 params)  : from_MAP                                    -> 1 climb
    PN     (11 params) : from_MAP  AND  from_0PA (this run's 0PA)    -> 2 climbs
    simple (11 params) : from_MAP  AND  from_0PA (this run's 0PA)    -> 2 climbs
Both deviation seeds are climbed and BOTH are reported; the JSON also carries a `best` block
naming the winner per model.  Which seed wins is point-dependent (established across the EMRI
and IMRI grids), and the from_0PA seed additionally guarantees a deviation can never end
below 0PA -- the artifact that made some earlier deviations look "worse than 0PA".
=> 11 points x 5 = 55 CV climbs; split across two jobs with EMRI_POINTS / EMRI_TAG.

from_injection is NOT run: it stalled at every point in every previous grid run (the 0PA
template is dephased at the 1PA injection over 2.5 yr), so it only wastes GPU time.

Signal parameters and the MAP come straight from the SK_files arrays -- nothing hardcoded:
    signal : data/signal/signal_parameter_array_EMRI.npy
    MAP    : data/recovered/mle/recovered_parameter_array_EMRI_0pa.npy   (0PA-vs-2PA recovered)
both indexed by grid cell, columns [0,1,2,3,4,7,8,11,13] -> the 9 inferred params.

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN     : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2          (additive 2.5PN pdot/edot)
    simple : del_0_p (idx 7) = dev_1, del_0_e (idx 8) = dev_2  (multiplicative Edot/Ldot)

All runs: dt=10, T=2.5, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.

This script also prints the per-channel (A, E, T) SNR of every injection.  The IMRI grid was
found to alias badly into the null T channel at dt=10; the EMRI grid has never been checked,
so the T% column here is the diagnostic.  T% should be ~0; a large T% means dt=10 is too
coarse for that point and it needs re-running at dt=5.

Run:  python gauss_cv_emri_grid_rest.py
      EMRI_POINTS=1,7,10,16,19,22  EMRI_TAG=a  python gauss_cv_emri_grid_rest.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
SK = "/scratch/e1583490/SK_files/data"
SIGNAL_ARRAY = f"{SK}/signal/signal_parameter_array_EMRI.npy"
MAP_ARRAY = f"{SK}/recovered/mle/recovered_parameter_array_EMRI_0pa.npy"
COLS9 = [0, 1, 2, 3, 4, 7, 8, 11, 13]      # -> [m1,m2,a,p0,e0,qS,phiS,Phi_phi0,Phi_r0]

# the 11 grid cells with no CV run yet
POINT_IDXS = [1, 2, 7, 8, 10, 15, 16, 19, 21, 22, 23]
_env = os.environ.get("EMRI_POINTS", "").strip()
if _env:
    POINT_IDXS = [int(x) for x in _env.replace(",", " ").split()]
_TAG = os.environ.get("EMRI_TAG", "").strip()


def build_points():
    sig, mp = np.load(SIGNAL_ARRAY), np.load(MAP_ARRAY)
    pts = []
    for i in POINT_IDXS:
        r = sig[i]
        pts.append(dict(
            name=f"idx{i}", idx=i,
            signal_param={"m1": float(r[0]), "m2": float(r[1]), "a": float(r[2]),
                          "p0": float(r[3]), "e0": float(r[4]), "xI0": float(r[5]),
                          "dist": float(r[6]), "qS": float(r[7]), "phiS": float(r[8]),
                          "qK": float(r[9]), "phiK": float(r[10]), "Phi_phi0": float(r[11]),
                          "Phi_theta0": float(r[12]), "Phi_r0": float(r[13]),
                          "dev_1": 0.0, "dev_2": 0.0},
            map9=[float(mp[i][c]) for c in COLS9]))
    return pts


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

    per_ch = [float(np.sqrt(max(_to_float(inner_product(
        s[i:i + 1], s[i:i + 1], PSD=PSD[i:i + 1], dt=DT, freq_mask=fmask, use_gpu=use_gpu)), 0.0)))
        for i in range(nchannels)]

    return dict(make0=make0, ov=ov, chi2r=chi2r, ip=ip, s=s,
                fisher_derivs=fisher_derivs, snr=float(np.sqrt(ip(s, s))),
                snr_channels=per_ch)


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
        print(f"    CV{tag} it {it:>3} lam={lam:.1e} ov={ovc:.7f} chi2={c0:.3e} rel={rel:.1e}",
              flush=True)
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
    print(f"    NM  nfev={res.nfev} ov {ov(theta0):.7f} -> {ov(cand):.7f} (accepted={ok})",
          flush=True)
    return cand if ok else theta0


def cv_from(ctx, tag, start):
    theta = np.array(start, dtype=float)
    print(f"    start overlap = {ctx['ov'](theta):.7f}", flush=True)
    theta, ov1, sigma = lm_climb(ctx, theta, tag=f" {tag}#1")
    if ov1 < OVERLAP_TARGET:
        theta = nm_refine(ctx, theta, sigma, NM_MAXITER)
        theta, ov1, sigma = lm_climb(ctx, theta, tag=f" {tag}#2")
    return dict(ov_start=float(ctx["ov"](start)), ov_final=float(ctx["ov"](theta)),
                chi2=float(ctx["chi2r"](theta)), params=theta.copy(),
                start=np.array(start, dtype=float))


# --- driver ----------------------------------------------------------------
JSON_PATH = os.path.join(HERE, f"results_emri_grid_rest{('_' + _TAG) if _TAG else ''}.json")


def _jsonable(records):
    pts, order = {}, []
    for r in records:
        if r["point"] not in pts:
            pts[r["point"]] = {"name": r["point"], "a": r["a"], "e0": r["e0"],
                               "snr": r["snr"], "snr_channels": r["snr_channels"],
                               "T_power_percent": r["T_power_percent"],
                               "map9": r["map9"], "runs": {}}
            order.append(r["point"])
        pts[r["point"]]["runs"].setdefault(r["model"], {})[r["start"]] = {
            "ov_start": r["ov_start"], "ov_final": r["ov_final"], "chi2": r["chi2"],
            "start_params": r["start_params"], "params": r["params"]}
    for pt in pts.values():
        pt["best"] = {m: dict(start=max(sd, key=lambda k: sd[k]["ov_final"]),
                              **sd[max(sd, key=lambda k: sd[k]["ov_final"])])
                      for m, sd in pt["runs"].items()}
    return {"system": "EMRI_grid_rest", "branch": "hybrid", "dt": DT, "T": T,
            "chi2_secondary": CHI2, "models": list(MODELS),
            "note": "remaining EMRI grid cells; 0PA from_MAP, deviations from_MAP and from_0PA",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "points": [pts[n] for n in order]}


def save_json(records):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(records), f, indent=2)


def main():
    points = build_points()
    print(f"[info] EMRI grid, dt={DT}, T={T}; points {[p['name'] for p in points]}; "
          f"{len(points)} x (1 + 2 + 2) = {len(points) * 5} CV climbs; "
          f"out -> {os.path.basename(JSON_PATH)}\n", flush=True)

    records = []
    for case in points:
        name = case["name"]
        zero_pa = None                      # this run's 0PA best fit, the second dev seed
        for mname, mcfg in MODELS.items():
            print("=" * 78, flush=True)
            ctx = build_context(case, mcfg)
            A, E, Tc = ctx["snr_channels"]
            tpc = 100.0 * (Tc / ctx["snr"]) ** 2 if ctx["snr"] > 0 else float("nan")
            print(f"[{name} (a={case['signal_param']['a']:+.1f}, e0={case['signal_param']['e0']:.1f}) "
                  f"| {mname} | dt={DT}]  SNR={ctx['snr']:.3f} "
                  f"(A={A:.3f} E={E:.3f} T={Tc:.3f}; T%={tpc:.2f})", flush=True)
            if tpc > 5.0:
                print(f"    [WARN] T channel carries {tpc:.1f}% of the SNR -- possible aliasing at "
                      f"dt={DT}; this point may need re-running at dt=5.", flush=True)

            if not mcfg["deviation"]:
                starts = [("from_MAP", np.array(case["map9"], dtype=float))]
            else:
                starts = [("from_MAP", np.array(list(case["map9"]) + [0.0, 0.0], dtype=float))]
                if zero_pa is not None:
                    starts.append(("from_0PA", np.concatenate([zero_pa, [0.0, 0.0]])))

            best, best_start = None, None
            for sname, start in starts:
                print(f"    --- start {sname} ---", flush=True)
                d = cv_from(ctx, f"{name}/{mname}/{sname}", start)
                print(f"    RESULT {name}/{mname}/{sname}: ov {d['ov_start']:.7f} -> "
                      f"{d['ov_final']:.7f}, chi2={d['chi2']:.4e}", flush=True)
                records.append(dict(
                    point=name, a=case["signal_param"]["a"], e0=case["signal_param"]["e0"],
                    model=mname, start=sname, snr=ctx["snr"], snr_channels=ctx["snr_channels"],
                    T_power_percent=tpc, map9=[float(x) for x in case["map9"]],
                    ov_start=d["ov_start"], ov_final=d["ov_final"], chi2=d["chi2"],
                    start_params=[float(x) for x in d["start"]],
                    params=[float(x) for x in d["params"]]))
                save_json(records)
                if best is None or d["ov_final"] > best["ov_final"]:
                    best, best_start = d, sname

            if mname == "0PA":
                zero_pa = np.array(best["params"], dtype=float)
            print(f"    BEST  {name}/{mname}: {best_start} ov={best['ov_final']:.7f}", flush=True)

    print("\n" + "=" * 78)
    print(f"EMRI grid -- remaining cells, dt={DT}, T={T}")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'model':>7} {'start':>10} {'SNR':>8} {'T%':>6} "
          f"{'ov@start':>11} {'ov@CV':>11} {'chi2':>11}")
    for r in records:
        print(f"{r['point']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {r['model']:>7} {r['start']:>10} "
              f"{r['snr']:>8.3f} {r['T_power_percent']:>6.2f} {r['ov_start']:>11.7f} "
              f"{r['ov_final']:>11.7f} {r['chi2']:>11.3e}")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
