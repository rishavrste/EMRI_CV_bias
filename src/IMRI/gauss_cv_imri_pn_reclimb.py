"""IMRI grid -- targeted PN re-climb of the cells whose C_p looks stuck off its spin row.

WHY
---
Across the completed 25-cell grid (results_combined.txt SECTIONS 1 + 1b) the fitted 2.5PN
coefficient C_p clusters tightly by spin, and only in cells where PN actually bought overlap:

    a=-0.9 -> +9.33..+9.73   a=-0.5 -> +10.06..+10.69   a=+0.0 -> +13.08 (one cell)
    a=+0.5 -> -13.40,-15.36  a=+0.9 -> -6.84..-7.29

The spread inside each retrograde row is under 5%, so C_p is essentially a function of spin
alone.  Ten cells bought <=2x from PN, and eight of those ALSO sit far off their row's C_p --
e.g. idx5 at C_p=0.16 while its four a=-0.9 row-mates all found ~+9.5 at 18-26x, or idx3 at
+28.8 when the a=+0.5 row sits at -14.  That pattern is what a stalled LM climb looks like:
seeded at dev=(0,0) the gradient <dh/dC_p|r> is ~0 on the flat ridge, so C_p never leaves the
origin (or wanders to the wrong branch) even though a much better minimum exists nearby.

This script tests that hypothesis directly: re-climb PN once per suspect cell, seeded at the
cell's own converged 0PA best fit with C_p placed at its ROW MEAN instead of 0.

    - If the climb lands near the row value with a large overlap gain, the old result was a
      convergence failure and the grid needs correcting.
    - If it falls back to the incumbent (or worse), the cell is genuinely deviation-free and
      the 2.5PN model really cannot help there -- which is the interesting physics result,
      because idx3/idx4/idx9 are the three MOST biased cells on the whole grid.

Either outcome settles the open question.  The incumbent is never discarded: both the
incumbent and the re-climb are recorded and `improved` says which won.

SELECTION (derived at runtime from results_combined.txt, not hardcoded)
-----------------------------------------------------------------------
    row_mean(a) = mean C_p over cells in that spin row with PN mismatch gain > 2x
    selected    = cells with gain <= 2x AND |C_p - row_mean| > 0.3*|row_mean|
This picks idx2, idx3, idx4, idx5, idx7, idx9, idx17, idx22 and correctly SKIPS idx18 (1.6x)
and idx23 (1.9x), whose C_p already sits inside the a=+0.5 cluster -- their small gains are
real, not a stall.

CAVEAT: the a=+0.0 row mean (+13.08) comes from a SINGLE cell, idx12.  The four a=0 cells
(idx2, idx7, idx17, idx22) are therefore the speculative part of this run.  On the EMRI grid
a=0 was uniformly deviation-free, but idx12 shows that is not true for IMRI, which is why
they are included.  Trim them with IMRI_PN_POINTS if you only want the confident cases.

Only PN is re-run.  `simple` never exceeded 2.7x anywhere on the grid and its del values show
no spin clustering, so there is no row mean to seed from.

Hybrid-branch SuperKludgeFlux deviation wiring:
    PN : C_p (idx 5) = dev_1, C_e (idx 6) = dev_2   (additive 2.5PN pdot/edot)
C_e is seeded at 0.0: the row means for C_e are all ~0 (-0.08, +0.26, +0.03) and it carries
far less of the fit than C_p.

All runs: dt=5, T=1.0, nchannels=3, no noise, chi2 (secondary spin) = 0.95.
Requires SuperKludge_r on the 'hybrid' branch.
Results are written incrementally, so a walltime kill still leaves a valid JSON.
8 cells x 1 climb ~ 20 min each ~ 3 h total; one job is enough.

Run:  python gauss_cv_imri_pn_reclimb.py
      IMRI_PN_POINTS=3,4,5,9  python gauss_cv_imri_pn_reclimb.py   # confident cells only
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


# --- deviation wiring (hybrid branch) --------------------------------------
def pn_tail(d1, d2):     return [d1, d2, 0.0, 0.0]                                  # C_p(5), C_e(6)
def pn_apa(d1, d2):      return {"dev_1": d1, "dev_2": d2, "del_0_p": 0.0, "del_0_e": 0.0}

PARAMS_9 = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
PARAMS_11 = PARAMS_9 + ["dev_1", "dev_2"]

MODELS = {"PN": dict(params=PARAMS_11, deviation=True, tail=pn_tail, apa=pn_apa)}

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
GAIN_CUT = 2.0      # a cell "worked" if PN reduced the mismatch by more than this
OFF_ROW = 0.3       # ...and is "off its row" if |C_p - row_mean| exceeds this fraction of it
_TAG = os.environ.get("IMRI_TAG", "").strip()



# --- selection: parse the completed grid and find the off-row cells --------
def parse_grid():
    """results_combined.txt SECTIONS 1 + 1b -> {idx: {a, e0, 0PA_ov, 0PA_p, PN_ov, PN_p, gain}}."""
    lines = open(COMBINED).read().split("\n")
    end = next(j for j, l in enumerate(lines) if l.startswith("# SECTION 2 -"))
    cells, cur, pend = {}, None, None
    for l in lines[:end]:
        m = re.match(r"# grid_idx(\d+)\s+a=([-+][\d.]+)\s+e0=([\d.]+)", l)
        if m:
            cur = int(m.group(1))
            cells[cur] = dict(a=float(m.group(2)), e0=float(m.group(3)))
            continue
        m = re.match(r"## (0PA|PN|simple)\s.*overlap = ([\d.]+)", l)
        if m:
            pend = m.group(1)
            cells[cur].setdefault(pend + "_ov", float(m.group(2)))
            continue
        if l.startswith("x_bf") and cur is not None and pend:
            v = [float(x) for x in l.split("[")[1].split("]")[0].split(",")]
            cells[cur].setdefault(pend + "_p", v)
            pend = None
    if len(cells) != 25:
        raise RuntimeError(f"expected 25 grid cells in {COMBINED}, parsed {len(cells)}")
    for c in cells.values():
        missing = {"0PA_ov", "0PA_p", "PN_ov", "PN_p"} - set(c)
        if missing:
            raise RuntimeError(f"cell missing {missing} in {COMBINED}")
        c["gain"] = (1.0 - c["0PA_ov"]) / max(1.0 - c["PN_ov"], 1e-16)
    return cells


def row_means(cells):
    """mean C_p per spin row, over the cells where PN actually worked."""
    rows = {}
    for i, c in cells.items():
        if c["gain"] > GAIN_CUT:
            rows.setdefault(c["a"], []).append((i, c["PN_p"][9]))
    return {a: (sum(v for _, v in L) / len(L), [i for i, _ in L]) for a, L in rows.items()}


def select(cells, means):
    out = []
    for i in sorted(cells):
        c = cells[i]
        if c["gain"] > GAIN_CUT or c["a"] not in means:
            continue
        mu = means[c["a"]][0]
        if abs(c["PN_p"][9] - mu) > OFF_ROW * abs(mu):
            out.append(i)
    return out


def build_points():
    cells = parse_grid()
    means = row_means(cells)
    idxs = select(cells, means)
    _env = os.environ.get("IMRI_PN_POINTS", "").strip()
    if _env:
        idxs = [int(x) for x in _env.replace(",", " ").split()]

    print("[selection] row means of C_p (cells with PN gain > "
          f"{GAIN_CUT:g}x):", flush=True)
    for a in sorted(means):
        mu, src_cells = means[a]
        flag = "   <-- SINGLE CELL, speculative" if len(src_cells) == 1 else ""
        print(f"    a={a:+.1f}  mean C_p = {mu:+8.3f}  from {len(src_cells)} cell(s) "
              f"{src_cells}{flag}", flush=True)
    print(f"[selection] re-climbing {len(idxs)} cells: {idxs}", flush=True)

    sig = np.load(SIGNAL_ARRAY)
    pts = []
    for i in idxs:
        r, c = sig[i], cells[i]
        mu = row_means(cells)[c["a"]][0]
        print(f"    idx{i:<3} a={c['a']:+.1f} e0={c['e0']:.1f}  C_p now {c['PN_p'][9]:+9.3f} "
              f"-> seed {mu:+8.3f}   incumbent PN ov {c['PN_ov']:.7f} ({c['gain']:.1f}x)",
              flush=True)
        pts.append(dict(
            name=f"idx{i}", idx=i, row_mean_cp=float(mu),
            incumbent=dict(ov=c["PN_ov"], params=list(c["PN_p"]), gain=c["gain"]),
            zero_pa=dict(ov=c["0PA_ov"], params=list(c["0PA_p"])),
            signal_param={"m1": float(r[0]), "m2": float(r[1]), "a": float(r[2]),
                          "p0": float(r[3]), "e0": float(r[4]), "xI0": float(r[5]),
                          "dist": float(r[6]), "qS": float(r[7]), "phiS": float(r[8]),
                          "qK": float(r[9]), "phiK": float(r[10]), "Phi_phi0": float(r[11]),
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
JSON_PATH = os.path.join(HERE, f"results_imri_pn_reclimb{('_' + _TAG) if _TAG else ''}.json")


def save_json(points):
    with open(JSON_PATH, "w") as f:
        json.dump({"system": "IMRI_grid_PN_reclimb", "branch": "hybrid", "dt": DT, "T": T,
                   "chi2_secondary": CHI2, "gain_cut": GAIN_CUT, "off_row": OFF_ROW,
                   "note": "PN re-climb of off-row cells, seeded at 0PA best fit with "
                           "C_p = row mean; incumbent kept for comparison",
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "points": points}, f, indent=2)


def main():
    points = build_points()
    print(f"\n[info] IMRI PN re-climb, dt={DT}, T={T}; {len(points)} cells x 1 climb; "
          f"out -> {os.path.basename(JSON_PATH)}\n", flush=True)

    mcfg, out = MODELS["PN"], []
    for case in points:
        name = case["name"]
        print("=" * 78, flush=True)
        ctx = build_context(case, mcfg)
        A, E, Tc = ctx["snr_channels"]
        tpc = 100.0 * (Tc / ctx["snr"]) ** 2 if ctx["snr"] > 0 else float("nan")
        print(f"[{name} (a={case['signal_param']['a']:+.1f}, e0={case['signal_param']['e0']:.1f}) "
              f"| PN re-climb | dt={DT}]  SNR={ctx['snr']:.3f} "
              f"(A={A:.3f} E={E:.3f} T={Tc:.3f}; T%={tpc:.2f})", flush=True)

        seed = np.array(list(case["zero_pa"]["params"]) + [case["row_mean_cp"], 0.0], dtype=float)
        inc = case["incumbent"]
        print(f"    seed = 0PA best fit + C_p={case['row_mean_cp']:+.3f}, C_e=0", flush=True)
        d = cv_from(ctx, f"{name}/PN/from_rowmean", seed)

        # score the incumbent in THIS context so the comparison is apples-to-apples
        inc_ov = float(ctx["ov"](np.array(inc["params"], dtype=float)))
        m0 = 1.0 - case["zero_pa"]["ov"]
        new_gain = m0 / max(1.0 - d["ov_final"], 1e-16)
        improved = d["ov_final"] > inc_ov
        verdict = ("STALL CONFIRMED -- old result was a convergence failure" if improved
                   else "no improvement -- the cell really is deviation-free at 2.5PN")
        print(f"    RESULT {name}: re-climb ov {d['ov_start']:.7f} -> {d['ov_final']:.7f} "
              f"({new_gain:.1f}x), incumbent {inc_ov:.7f} ({inc['gain']:.1f}x)", flush=True)
        print(f"    C_p {case['row_mean_cp']:+.3f} (seed) -> {d['params'][9]:+.3f}, "
              f"C_e -> {d['params'][10]:+.3f}", flush=True)
        print(f"    VERDICT {name}: {verdict}", flush=True)

        case.update(reclimb=dict(ov_start=d["ov_start"], ov_final=d["ov_final"], chi2=d["chi2"],
                                 gain=new_gain, params=[float(x) for x in d["params"]],
                                 seed=[float(x) for x in seed]),
                    incumbent_ov_rescored=inc_ov, improved=bool(improved), verdict=verdict)
        out.append(case)
        save_json(out)

    print("\n" + "=" * 78)
    print(f"IMRI PN re-climb from row-mean C_p, dt={DT}, T={T}")
    print(f"{'point':6} {'a':>5} {'e0':>4} {'C_p seed':>9} {'C_p out':>9} "
          f"{'inc ov':>11} {'new ov':>11} {'inc x':>7} {'new x':>7}  verdict")
    for c in out:
        r = c["reclimb"]
        print(f"{c['name']:6} {c['signal_param']['a']:>5.1f} {c['signal_param']['e0']:>4.1f} "
              f"{c['row_mean_cp']:>9.3f} {r['params'][9]:>9.3f} "
              f"{c['incumbent_ov_rescored']:>11.7f} {r['ov_final']:>11.7f} "
              f"{c['incumbent']['gain']:>6.1f}x {r['gain']:>6.1f}x  "
              f"{'IMPROVED' if c['improved'] else 'no change'}")
    n = sum(c["improved"] for c in out)
    print(f"\n{n}/{len(out)} cells improved.")
    save_json(out)
    print(f"[saved] -> {JSON_PATH}")


if __name__ == "__main__":
    main()
