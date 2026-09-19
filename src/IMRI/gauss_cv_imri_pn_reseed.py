"""IMRI a x e0 grid -- targeted PN re-climb of the cells whose PN climb STALLED at dev = 0.

WHY
---
Across all 25 dt=5 grid cells, the PN deviation parameter C_p clusters tightly by spin -- but
only in the cells where PN actually bought overlap.  Counting cells with PN mismatch gain > 2x:

    a=-0.9 : +9.73  +9.72  +9.33  +9.50         (idx0, idx10, idx15, idx20)        18-26x
    a=-0.5 : +10.06 +10.24 +10.25 +10.69 +10.59 (idx1, idx6, idx11, idx16, idx21)  19-28x
    a= 0.0 : +13.08                             (idx12)                             3.4x
    a=+0.5 : -13.40 -15.36                      (idx8, idx13)                     2.1-2.4x
    a=+0.9 : -6.84  -7.29  -6.90                (idx14, idx19, idx24)             2.8-3.2x

The spread inside each retrograde row is under 5%: to that accuracy C_p is a function of spin
alone.  That makes one specific failure mode diagnosable.  If a cell bought no overlap AND its
C_p came out at essentially zero, the climb never left the flat dev = 0 ridge -- the Fisher
gradient <dh/dC_p | r> vanishes there, so LM has nothing to push it off -- even though its row
demonstrably has a solution of order +9.5 / +10.4 / +13.1 / -14.4 / -7.0.  That is a stall, and
re-seeding C_p away from zero is exactly the right fix for it.

Cells that ended at some OTHER non-zero C_p are a different matter: they did leave the ridge and
converged somewhere.  This script does not touch them -- nothing here is being called wrong, and
no claim is made about which C_p a cell "should" have.  Only cells stuck at zero are re-run.

WHAT IT RUNS
------------
Selection is DERIVED at runtime from results_combined.txt, not hardcoded, so it stays honest if
the underlying fits change.  A cell is re-climbed iff BOTH:
    (a) its current PN mismatch gain over 0PA is <= GAIN_THRESHOLD (2.0), i.e. PN bought
        nothing, AND
    (b) its C_p is negligible on its own row's scale: |C_p| < CP_ZERO_TOL * |row_mean|
        (CP_ZERO_TOL = 0.3), i.e. it is sitting on the dev = 0 ridge.
Condition (b) is what restricts this to stalls.  It excludes idx18 and idx23 (gain < 2x but C_p
already in the a=+0.5 cluster) and equally excludes idx3, idx4 and idx9, which ended at large
non-zero C_p -- those converged somewhere and are simply left as they are.

Seed: that cell's own converged dt=5 0PA best fit (9 params) + dev = (row_mean_C_p, 0.0).
Only C_p is being lifted off zero; the 9 astrophysical parameters are already converged, and
C_e is seeded at 0 because the row means of C_e are all ~0 (-0.08, +0.26, +0.03) and carry no
usable signal.

Model: PN only (11 params).  `simple` is NOT re-run: its gains never exceed 2.7x anywhere on the
grid and its del values show no spin clustering, so there is no row mean to seed from.

=> 5 cells x 1 climb.  At ~20 min/climb (measured: 25 climbs in 8h12m) this is under 2 h.

CAVEAT on the a=0.0 row: its row mean rests on a SINGLE working cell (idx12, +13.08), so the
seed for idx2/idx7/idx17/idx22 is far less well determined than for idx5, whose row mean comes
from four cells that agree to 5%.  The script prints a [NOTE] for those.  Set IMRI_POINTS to
drop them if the GPU time is better spent elsewhere.

Each re-climb is compared against the INCUMBENT best PN fit for that cell and only counts as an
improvement if it ends higher; the JSON records both so nothing is silently overwritten.  A cell
that is handed a row-scale C_p and still falls back to zero is positive evidence that the 2.5PN
direction genuinely does not help there.

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2          (additive 2.5PN pdot/edot)

All runs: dt=5, T=1.0, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.
Results are written incrementally, so a walltime kill still leaves a valid JSON.

Run:  python gauss_cv_imri_pn_reseed.py
      IMRI_POINTS=5  python gauss_cv_imri_pn_reseed.py     # only the well-determined row
"""

import json
import os
import re
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
    "PN": dict(params=PARAMS_11, deviation=True, tail=pn_tail, apa=pn_apa),
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
DT, T, CHI2 = 5.0, 1.0, 0.95

HERE = os.path.dirname(os.path.abspath(__file__))
SK = "/scratch/e1583490/SK_files/data"
SIGNAL_ARRAY = f"{SK}/signal/signal_parameter_array_IMRI.npy"
MAP_ARRAY = f"{SK}/recovered/mle/recovered_parameter_array_IMRI_0pa.npy"
COLS9 = [0, 1, 2, 3, 4, 7, 8, 11, 13]      # -> [m1,m2,a,p0,e0,qS,phiS,Phi_phi0,Phi_r0]

COMBINED = os.path.join(HERE, "results_combined.txt")
GAIN_THRESHOLD = 2.0    # "PN bought nothing" -- mismatch gain over 0PA at or below this
CP_ZERO_TOL = 0.3       # "stalled on the dev=0 ridge" -- |C_p| < CP_ZERO_TOL * |row_mean|

_TAG = os.environ.get("IMRI_TAG", "").strip()


def parse_combined():
    """-> {idx: {a, e0, snr, '0PA': (ov, x_bf), 'PN': (ov, x_bf)}} for the dt=5 grid.

    Reads only what precedes SECTION 2, i.e. SECTION 1 (the dt=10-reseeded cells) plus
    SECTION 1b (the MAP-seeded cells) -- together the complete 25-cell dt=5 reference.
    """
    lines = open(COMBINED).read().split("\n")
    end = next(j for j, l in enumerate(lines) if l.startswith("# SECTION 2 -"))
    cells, cur, pend = {}, None, None
    for l in lines[:end]:
        m = re.match(r"# grid_idx(\d+)\s+a=([-+][\d.]+)\s+e0=([\d.]+)\s+SNR=([\d.]+)", l)
        if m:
            cur = int(m.group(1))
            cells[cur] = dict(a=float(m.group(2)), e0=float(m.group(3)), snr=float(m.group(4)))
            pend = None
            continue
        m = re.match(r"## (0PA|PN|simple)\s.*overlap = ([\d.]+)", l)
        if m and cur is not None:
            pend = m.group(1)
            cells[cur].setdefault(pend + "_ov", float(m.group(2)))
            continue
        if l.startswith("x_bf") and cur is not None and pend is not None:
            v = [float(x) for x in l.split("[")[1].split("]")[0].split(",")]
            cells[cur].setdefault(pend + "_x", v)
            pend = None
    if len(cells) != 25:
        raise RuntimeError(f"expected 25 grid cells in {COMBINED}, parsed {len(cells)}: "
                           f"{sorted(cells)}")
    for i, c in cells.items():
        for k in ("0PA_ov", "0PA_x", "PN_ov", "PN_x"):
            if k not in c:
                raise RuntimeError(f"grid_idx{i} is missing {k} in {COMBINED}")
    return cells


def select_points(cells):
    """Row means over the cells where PN WORKED, then the cells that look stalled."""
    for c in cells.values():
        c["gain"] = (1.0 - c["0PA_ov"]) / max(1.0 - c["PN_ov"], 1e-16)
        c["C_p"] = c["PN_x"][9]

    rows = {}
    for a in sorted({c["a"] for c in cells.values()}):
        w = [c["C_p"] for c in cells.values() if c["a"] == a and c["gain"] > GAIN_THRESHOLD]
        rows[a] = (float(np.mean(w)), len(w)) if w else (None, 0)

    print("[select] C_p row means over cells with PN gain > "
          f"{GAIN_THRESHOLD}x:", flush=True)
    for a, (mu, n) in rows.items():
        note = "" if n >= 2 else ("   <-- SINGLE cell, seed is speculative" if n == 1
                                  else "   <-- no working cell, row skipped")
        print(f"    a={a:+.1f}: mean C_p = {'n/a' if mu is None else f'{mu:+8.3f}'} "
              f"from {n} cell(s){note}", flush=True)

    chosen = []
    for i in sorted(cells):
        c = cells[i]
        mu, n = rows[c["a"]]
        if mu is None or c["gain"] > GAIN_THRESHOLD:
            continue
        if abs(c["C_p"]) >= CP_ZERO_TOL * abs(mu):
            print(f"    skip idx{i} (a={c['a']:+.1f}): gain {c['gain']:.1f}x but C_p="
                  f"{c['C_p']:+.2f} is not at zero on its row's scale ({mu:+.2f}) -- it left "
                  f"the ridge and converged, so it is not a stall", flush=True)
            continue
        c["seed_C_p"], c["row_n"] = mu, n
        chosen.append(i)
    return chosen, rows


_CELLS = parse_combined()
POINT_IDXS, _ROWS = select_points(_CELLS)
_env = os.environ.get("IMRI_POINTS", "").strip()
if _env:
    POINT_IDXS = [int(x) for x in _env.replace(",", " ").split()]
    for i in POINT_IDXS:                      # env override still needs a row mean to seed from
        mu, n = _ROWS[_CELLS[i]["a"]]
        if mu is None:
            raise RuntimeError(f"idx{i}: no cell in its a={_CELLS[i]['a']:+.1f} row has PN gain "
                               f"> {GAIN_THRESHOLD}x, so there is no row mean to seed C_p from")
        _CELLS[i]["seed_C_p"], _CELLS[i]["row_n"] = mu, n


def build_points():
    sig = np.load(SIGNAL_ARRAY)
    pts = []
    for i in POINT_IDXS:
        r, c = sig[i], _CELLS[i]
        pts.append(dict(
            name=f"idx{i}", idx=i,
            signal_param={"m1": float(r[0]), "m2": float(r[1]), "a": float(r[2]),
                          "p0": float(r[3]), "e0": float(r[4]), "xI0": float(r[5]),
                          "dist": float(r[6]), "qS": float(r[7]), "phiS": float(r[8]),
                          "qK": float(r[9]), "phiK": float(r[10]), "Phi_phi0": float(r[11]),
                          "Phi_theta0": float(r[12]), "Phi_r0": float(r[13]),
                          "dev_1": 0.0, "dev_2": 0.0},
            zero_pa=list(c["0PA_x"]), zero_pa_ov=c["0PA_ov"],
            incumbent_ov=c["PN_ov"], incumbent_x=list(c["PN_x"]),
            incumbent_gain=c["gain"], incumbent_C_p=c["C_p"],
            seed_C_p=c["seed_C_p"], row_n=c["row_n"]))
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
JSON_PATH = os.path.join(HERE, f"results_imri_pn_reseed{('_' + _TAG) if _TAG else ''}.json")


def _jsonable(records):
    return {"system": "IMRI_grid_PN_reseed", "branch": "hybrid", "dt": DT, "T": T,
            "chi2_secondary": CHI2, "models": list(MODELS),
            "note": ("targeted PN re-climb of grid cells that STALLED on the dev=0 ridge: "
                     f"PN gain <= {GAIN_THRESHOLD}x AND |C_p| < {CP_ZERO_TOL:.0%} of their spin "
                     "row's mean.  Seeded from that cell's dt=5 0PA best fit + "
                     "(row_mean_C_p, 0.0).  Cells that converged to some other non-zero C_p are "
                     "NOT touched.  Incumbent fit is recorded alongside, never overwritten."),
            "gain_threshold": GAIN_THRESHOLD, "cp_zero_tol": CP_ZERO_TOL,
            "row_mean_C_p": {f"{a:+.1f}": mu for a, (mu, n) in _ROWS.items()},
            "row_n_working": {f"{a:+.1f}": n for a, (mu, n) in _ROWS.items()},
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "points": records}


def save_json(records):
    with open(JSON_PATH, "w") as f:
        json.dump(_jsonable(records), f, indent=2)


def main():
    points = build_points()
    print(f"\n[info] IMRI PN re-seed of stalled cells, dt={DT}, T={T}; {len(points)} cell(s), 1 climb each "
          f"-> {os.path.basename(JSON_PATH)}", flush=True)
    print(f"{'point':>6} {'a':>5} {'e0':>4} | {'C_p now':>9} {'gain now':>8} | "
          f"{'C_p seed':>9} {'row n':>5}", flush=True)
    for c in points:
        print(f"{c['name']:>6} {c['signal_param']['a']:>5.1f} {c['signal_param']['e0']:>4.1f} | "
              f"{c['incumbent_C_p']:>9.3f} {c['incumbent_gain']:>7.1f}x | "
              f"{c['seed_C_p']:>9.3f} {c['row_n']:>5}", flush=True)
    print(flush=True)

    mcfg = MODELS["PN"]
    records = []
    for case in points:
        name = case["name"]
        print("=" * 78, flush=True)
        ctx = build_context(case, mcfg)
        A, E, Tc = ctx["snr_channels"]
        tpc = 100.0 * (Tc / ctx["snr"]) ** 2 if ctx["snr"] > 0 else float("nan")
        print(f"[{name} (a={case['signal_param']['a']:+.1f}, e0={case['signal_param']['e0']:.1f}) "
              f"| PN re-seed | dt={DT}]  SNR={ctx['snr']:.3f} "
              f"(A={A:.3f} E={E:.3f} T={Tc:.3f}; T%={tpc:.2f})", flush=True)
        print(f"    incumbent PN: ov={case['incumbent_ov']:.10f}  C_p={case['incumbent_C_p']:+.4f}"
              f"  gain={case['incumbent_gain']:.1f}x     0PA ov={case['zero_pa_ov']:.10f}",
              flush=True)
        print(f"    seeding C_p = {case['seed_C_p']:+.4f} (mean of {case['row_n']} working "
              f"cell(s) at a={case['signal_param']['a']:+.1f}), C_e = 0, from the 0PA best fit",
              flush=True)
        if case["row_n"] < 2:
            print("    [NOTE] that row mean rests on a SINGLE working cell -- speculative seed.",
                  flush=True)

        start = np.array(list(case["zero_pa"]) + [case["seed_C_p"], 0.0], dtype=float)
        d = cv_from(ctx, f"{name}/PN/from_rowmean", start)
        m0 = 1.0 - case["zero_pa_ov"]
        gain = m0 / max(1.0 - d["ov_final"], 1e-16)
        better = d["ov_final"] > case["incumbent_ov"]
        verdict = ("IMPROVED -- the old climb had stalled" if better else
                   "no better than the incumbent")
        print(f"    RESULT {name}: ov {d['ov_start']:.10f} -> {d['ov_final']:.10f}  "
              f"chi2={d['chi2']:.4e}  C_p={d['params'][9]:+.4f} C_e={d['params'][10]:+.4f}  "
              f"gain={gain:.1f}x  [{verdict}]", flush=True)

        records.append(dict(
            point=name, idx=case["idx"], a=case["signal_param"]["a"], e0=case["signal_param"]["e0"],
            snr=ctx["snr"], snr_channels=ctx["snr_channels"], T_power_percent=tpc,
            zero_pa_ov=case["zero_pa_ov"], zero_pa_params=[float(x) for x in case["zero_pa"]],
            incumbent_ov=case["incumbent_ov"], incumbent_C_p=case["incumbent_C_p"],
            incumbent_gain=case["incumbent_gain"],
            incumbent_params=[float(x) for x in case["incumbent_x"]],
            seed_C_p=case["seed_C_p"], row_n_working=case["row_n"],
            ov_start=d["ov_start"], ov_final=d["ov_final"], chi2=d["chi2"], gain=gain,
            C_p=float(d["params"][9]), C_e=float(d["params"][10]),
            start_params=[float(x) for x in d["start"]],
            params=[float(x) for x in d["params"]],
            improved=bool(better)))
        save_json(records)

    print("\n" + "=" * 78)
    print(f"IMRI PN re-seed, dt={DT}, T={T} -- did the stalled cells actually have a solution?")
    print(f"{'point':>6} {'a':>5} {'e0':>4} | {'0PA ov':>12} | {'PN ov (was)':>12} "
          f"{'C_p (was)':>9} {'gain':>7} | {'PN ov (new)':>12} {'C_p (new)':>9} {'gain':>7} | "
          f"{'verdict':>9}")
    for r in records:
        print(f"{r['point']:>6} {r['a']:>5.1f} {r['e0']:>4.1f} | {r['zero_pa_ov']:>12.9f} | "
              f"{r['incumbent_ov']:>12.9f} {r['incumbent_C_p']:>9.3f} "
              f"{r['incumbent_gain']:>6.1f}x | {r['ov_final']:>12.9f} {r['C_p']:>9.3f} "
              f"{r['gain']:>6.1f}x | {'IMPROVED' if r['improved'] else '--':>9}")
    n_imp = sum(r["improved"] for r in records)
    print(f"\n{n_imp}/{len(records)} cells improved on their incumbent PN fit.")
    print("A cell that did NOT improve is evidence the 2.5PN direction genuinely cannot help "
          "there,\nsince it was handed the row-mean C_p and still fell back.")

    save_json(records)
    print(f"\n[saved] all results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
