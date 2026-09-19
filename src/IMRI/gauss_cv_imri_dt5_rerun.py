"""IMRI CV re-run at dt=5 for the grid points whose T channel was aliased at dt=10.

WHY
---
At dt=10 the Nyquist frequency is 0.05 Hz.  Prograde IMRIs (m1=1e6, m2=1e3) carry power
above it, which aliases and dumps spurious power into the otherwise-null T channel.  The
dt=10 per-channel run showed T carrying 29-97% of the total SNR at EVERY grid point
(worst: idx14 96.5%, idx19 97.5%), while the dt=5 ad-hoc case and the short-window tails
cases showed T ~ 0.00%.  Re-evaluating at dt=5 (`snr_overlap_dt5_imri.py`) collapsed T% to
~0 everywhere and pulled every grid SNR back to 20.5-32.8 -- aliasing confirmed.

The dt=10 best-fit points were therefore optimised against a partly-aliased signal.
Re-evaluated at dt=5 the retrograde fits barely moved (dOverlap ~ 1e-4) but the prograde
ones degraded materially (idx4 0.99585 -> 0.99094, idx24 0.99785 -> 0.99466, and idx14's
PN fell BELOW its own 0PA).  This script re-runs the full LM-damped Cutler-Vallisneri
climb at dt=5, seeded from those dt=10 points, so the fits are re-optimised against the
un-aliased signal.

WHAT IT RUNS
------------
Points: the 12 grid points that have dt=10 best fits, ordered by dt=10 T-contamination
(worst first, so a walltime cut still delivers the cases that matter most):
    idx14 (96.5%) idx4 (77.0%) idx24 (76.3%) idx18 (63.6%) idx8 (57.4%) idx12 (49.5%)
    idx6  (39.5%) idx16 (37.4%) idx22 (37.2%) idx10 (36.5%) idx0 (33.7%) idx20 (29.4%)
(pt4 / pt20 / adhoc_A are excluded: their T% is already 0.00 -- pt4/pt20 use T=0.25 and
adhoc_A already ran at dt=5.)

Models: 0PA (9 params) | 0PA+PN (11) | 0PA+simple (11).

Seeding:
    0PA          <- the dt=10 0PA best fit                      (1 climb)
    PN / simple  <- BOTH of these seeds are climbed separately  (2 climbs each):
                      'from_dt10_<model>' : that model's dt=10 deviation best fit
                      'from_0PA_dt5'      : THIS run's dt=5 0PA result + dev=(0,0)
                    and we keep whichever ends higher.  Running both matters because the
                    better seed is point-dependent (established on the EMRI grid), and the
                    0PA-dt5 seed guarantees the deviation can never end below 0PA -- the
                    trap that put idx14's dt=10 PN under its own 0PA.
=> 12 points x (1 + 2 + 2) = 60 CV climbs.

Because 60 dt=5 climbs may not fit one 72 h walltime, the point list can be split across
parallel jobs with the DT5_POINTS environment variable, e.g.
    DT5_POINTS=14,4,24,18,8,12  python gauss_cv_imri_dt5_rerun.py
Each job writes its own results_imri_dt5_rerun[_<tag>].json (set DT5_TAG to name it).

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN     : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2          (additive 2.5PN pdot/edot)
    simple : del_0_p (idx 7) = dev_1, del_0_e (idx 8) = dev_2  (multiplicative Edot/Ldot)

All runs: dt=5, T=1.0, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.
Results are written incrementally, so a walltime kill still leaves a valid JSON.

Run:  python gauss_cv_imri_dt5_rerun.py
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
DT, T, CHI2 = 5.0, 1.0, 0.95          # <-- dt=5 is the point of this script

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_ARRAY = "/scratch/e1583490/SK_files/data/signal/signal_parameter_array_IMRI.npy"

# grid index -> dt=10 T-channel power fraction (%), from snr_channels_imri.log.
# Ordered worst-contaminated first.
POINT_IDXS = [14, 4, 24, 18, 8, 12, 6, 16, 22, 10, 0, 20]
T_PERCENT_DT10 = {14: 96.51, 4: 76.97, 24: 76.33, 18: 63.56, 8: 57.42, 12: 49.53,
                  6: 39.52, 16: 37.38, 22: 37.22, 10: 36.52, 0: 33.70, 20: 29.40}

# optional split across parallel jobs:  DT5_POINTS=14,4,24  DT5_TAG=a  python ...
_env = os.environ.get("DT5_POINTS", "").strip()
if _env:
    POINT_IDXS = [int(x) for x in _env.replace(",", " ").split()]
_TAG = os.environ.get("DT5_TAG", "").strip()


# --- dt=10 best-fit points (the CV seeds) ----------------------------------
def _load(fn):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        print(f"[WARN] missing {fn}")
        return {}
    d = json.load(open(p))
    return {c["name"]: c for c in d.get("points", d.get("cases", []))}


def dt10_best_fits():
    """-> {'idx4': {'0PA': [...], 'PN': [...], 'simple': [...]}, ...} best available seed."""
    out = {}
    dv = _load("results_imri_grid_diverse.json")
    dvf = _load("results_imri_grid_diverse_dev_from_0pa.json")
    dv2 = _load("results_imri_grid_diverse2.json")
    for n, p in dv.items():
        R = p["runs"]
        F = dvf.get(n, {}).get("runs", {})
        e = {"0PA": R["0PA"]["from_MAP"]["params"]}
        for m in ("PN", "simple"):
            cand = [R[m]["from_MAP"]] + ([F[m]] if m in F else [])
            e[m] = max(cand, key=lambda r: r["ov_final"])["params"]
        out[n] = e
    for n, p in dv2.items():
        R = p["runs"]
        out[n] = {m: R[m]["from_MAP"]["params"] for m in ("0PA", "PN", "simple")}
    return out


def build_points():
    sig = np.load(SIGNAL_ARRAY)
    pts = []
    for i in POINT_IDXS:
        r = sig[i]
        pts.append(dict(
            name=f"idx{i}", idx=i, T_percent_dt10=T_PERCENT_DT10[i],
            signal_param={"m1": 1e6, "m2": 1e3, "a": float(r[2]), "p0": float(r[3]),
                          "e0": float(r[4]), "xI0": 1.0, "dist": float(r[6]),
                          "qS": float(r[7]), "phiS": float(r[8]), "qK": float(r[9]),
                          "phiK": float(r[10]), "Phi_phi0": float(r[11]),
                          "Phi_theta0": float(r[12]), "Phi_r0": float(r[13]),
                          "dev_1": 0.0, "dev_2": 0.0}))
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

    # per-channel SNR of the dt=5 injection, for the record
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
JSON_PATH = os.path.join(HERE, f"results_imri_dt5_rerun{('_' + _TAG) if _TAG else ''}.json")


def _jsonable(records):
    pts, order = {}, []
    for r in records:
        if r["point"] not in pts:
            pts[r["point"]] = {"name": r["point"], "a": r["a"], "e0": r["e0"],
                               "snr": r["snr"], "snr_channels": r["snr_channels"],
                               "T_percent_dt10": r["T_percent_dt10"], "runs": {}}
            order.append(r["point"])
        pts[r["point"]]["runs"].setdefault(r["model"], {})[r["seed"]] = {
            "ov_start": r["ov_start"], "ov_final": r["ov_final"],
            "chi2": r["chi2"], "ov_dt10_point_at_dt5": r["ov_dt10_point_at_dt5"],
            "start_params": r["start_params"], "params": r["params"]}
    for pt in pts.values():                       # convenience: best seed per model
        pt["best"] = {m: dict(seed=max(sd, key=lambda k: sd[k]["ov_final"]),
                              **sd[max(sd, key=lambda k: sd[k]["ov_final"])])
                      for m, sd in pt["runs"].items()}
    return {"system": "IMRI_grid_dt5_rerun", "branch": "hybrid", "dt": DT, "T": T,
            "chi2_secondary": CHI2, "models": list(MODELS),
            "note": "CV re-run at dt=5 (un-aliased) seeded from the dt=10 best fits",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "points": [pts[n] for n in order]}


def save_json(records):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(records), f, indent=2)


def main():
    seeds = dt10_best_fits()
    points = build_points()
    print(f"[info] dt={DT}, T={T}; points {[p['name'] for p in points]}; "
          f"{len(points)} x (1 + 2 + 2) = {len(points) * 5} CV climbs; "
          f"seeds loaded for {len(seeds)} points; out -> {os.path.basename(JSON_PATH)}\n",
          flush=True)

    records = []
    for case in points:
        name = case["name"]
        if name not in seeds:
            print(f"[SKIP] {name}: no dt=10 best fit available")
            continue
        zero_pa_dt5 = None                     # this run's dt=5 0PA result, for dev seeding
        for mname, mcfg in MODELS.items():
            print("=" * 78, flush=True)
            ctx = build_context(case, mcfg)
            A, E, Tc = ctx["snr_channels"]
            print(f"[{name} (a={case['signal_param']['a']:+.1f}, e0={case['signal_param']['e0']:.1f}) "
                  f"| {mname} | dt={DT}]  SNR={ctx['snr']:.3f} "
                  f"(A={A:.3f} E={E:.3f} T={Tc:.3f}; T% at dt=10 was {case['T_percent_dt10']:.1f})",
                  flush=True)

            dt10_pt = np.array(seeds[name][mname], dtype=float)
            ov_dt10 = float(ctx["ov"](dt10_pt))

            # ---- seeds: 0PA gets one; each deviation gets BOTH, best result wins ----
            if not mcfg["deviation"]:
                starts = [("from_dt10_0PA", dt10_pt)]
            else:
                starts = [("from_dt10_" + mname, dt10_pt)]
                if zero_pa_dt5 is not None:
                    starts.append(("from_0PA_dt5",
                                   np.concatenate([zero_pa_dt5, [0.0, 0.0]])))
                else:
                    print("      [WARN] no dt=5 0PA result yet -- dt10 seed only", flush=True)

            best = None
            for seed_name, start in starts:
                print(f"    --- seed {seed_name} ---", flush=True)
                d = cv_from(ctx, f"{name}/{mname}/{seed_name}", start)
                print(f"    RESULT {name}/{mname}/{seed_name}: "
                      f"ov {d['ov_start']:.7f} -> {d['ov_final']:.7f}, chi2={d['chi2']:.4e}",
                      flush=True)
                records.append(dict(
                    point=name, a=case["signal_param"]["a"], e0=case["signal_param"]["e0"],
                    T_percent_dt10=case["T_percent_dt10"], model=mname, seed=seed_name,
                    snr=ctx["snr"], snr_channels=ctx["snr_channels"],
                    ov_dt10_point_at_dt5=ov_dt10,
                    ov_start=d["ov_start"], ov_final=d["ov_final"], chi2=d["chi2"],
                    start_params=[float(x) for x in d["start"]],
                    params=[float(x) for x in d["params"]]))
                save_json(records)
                if best is None or d["ov_final"] > best["ov_final"]:
                    best = d
                    best_seed = seed_name

            if mname == "0PA":
                zero_pa_dt5 = np.array(best["params"], dtype=float)
            print(f"    BEST  {name}/{mname}: {best_seed} ov={best['ov_final']:.7f} "
                  f"(dt=10 point scored {ov_dt10:.7f} at dt=5)", flush=True)

    print("\n" + "=" * 78)
    print(f"IMRI CV re-run at dt={DT} (un-aliased), seeded from the dt=10 best fits")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'model':>7} {'seed':>15} {'SNR':>8} "
          f"{'ov@dt10pt':>11} {'ov@start':>11} {'ov@CV':>11} {'chi2':>11}")
    for r in records:
        print(f"{r['point']:6} {r['a']:>5.1f} {r['e0']:>4.1f} {r['model']:>7} {r['seed']:>15} "
              f"{r['snr']:>8.3f} {r['ov_dt10_point_at_dt5']:>11.7f} {r['ov_start']:>11.7f} "
              f"{r['ov_final']:>11.7f} {r['chi2']:>11.3e}")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
