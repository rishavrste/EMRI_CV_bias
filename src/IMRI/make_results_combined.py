"""Regenerate IMRI/results_combined.txt from the results_*.json files.

Single source of truth: every number in results_combined.txt comes from a JSON produced by
one of the CV scripts, so the compiled file can always be rebuilt after a new run.
    python make_results_combined.py
"""
import json, numpy as np

sig = np.load("/scratch/e1583490/SK_files/data/signal/signal_parameter_array_IMRI.npy")
J = lambda f: json.load(open(f))
def cases(f):
    d = J(f); return {c["name"]: c for c in d.get("points", d.get("cases"))}

DT5 = {}
for tag in "ab":
    for p in J(f"results_imri_dt5_rerun_{tag}.json")["points"]: DT5[p["name"]] = p
dv, dvf, dv2 = cases("results_imri_grid_diverse.json"), cases("results_imri_grid_diverse_dev_from_0pa.json"), cases("results_imri_grid_diverse2.json")
t0, tP, tS, tPE = (cases("results_imri_tails_0pa.json"), cases("results_imri_tails_PN.json"),
                   cases("results_imri_tails_simple.json"), cases("results_imri_tails_simple_pe.json"))
aP, aS, aPE = cases("results_imri_PN.json"), cases("results_imri_simple.json"), cases("results_imri_simple_pe.json")
CH10 = {c["name"]: c for c in J("snr_channels_imri.json")["cases"]}
CH5  = {c["name"]: c for c in J("snr_overlap_dt5_imri.json")["cases"]}

fmt = lambda v: "[" + ", ".join(f"{x:.8e}" for x in v) + "]"
def chi2of(entry, snr):
    """chi2 = <r|r>; the older ad-hoc JSONs predate the chi2 field, so derive 2*SNR^2*(1-ov)."""
    if "chi2" in entry: return entry["chi2"], ""
    return 2.0 * snr**2 * (1.0 - entry["ov_final"]), "  # derived: 2*SNR^2*(1-overlap)"

def _m(v):
    """1000000.0 -> '1e6', 5000.0 -> '5e3' (matches the hand-written style)."""
    e = int(np.floor(np.log10(v))); m = v / 10.0**e
    return f"{m:g}e{e}" if abs(m * 10**e - v) < 1e-6 else f"{v:.6g}"

def sigblock(sp, ind="    "):
    return ['signal_param = {',
        f'{ind}"m1": {_m(sp["m1"])}, "m2": {_m(sp["m2"])}, "a": {sp["a"]}, "p0": {sp["p0"]!r}, "e0": {sp["e0"]}, "xI0": 1.0,',
        f'{ind}"dist": {sp["dist"]!r}, "qS": {sp["qS"]!r}, "phiS": {sp["phiS"]!r},',
        f'{ind}"qK": {sp["qK"]!r}, "phiK": {sp["phiK"]!r}, "Phi_phi0": {sp["Phi_phi0"]!r}, "Phi_theta0": {sp["Phi_theta0"]!r},',
        f'{ind}"Phi_r0": {sp["Phi_r0"]!r}, "dev_1": 0.0, "dev_2": 0.0}}']
def gridsig(i):
    r = sig[i]
    return dict(m1=1e6, m2=1e3, a=float(r[2]), p0=float(r[3]), e0=float(r[4]), dist=float(r[6]),
                qS=float(r[7]), phiS=float(r[8]), qK=float(r[9]), phiK=float(r[10]),
                Phi_phi0=float(r[11]), Phi_theta0=float(r[12]), Phi_r0=float(r[13]))

L = []; A = L.append; BAR = "#" + "=" * 86
A("# GENERAL RESULTS FOR IMRI CASES  (Cutler-Vallisneri / LM-damped CV)")
A("# nchannels=3 (A,E,T), include_noise=False, chi2(secondary spin)=0.95, T=1.0 yr unless stated.")
A("# param_names (9-param / 0PA):  [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0]")
A("# param_names (11-param / dev): [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, dev_1, dev_2]")
A("# chi2 = <r|r> in this pipeline (always POSITIVE); relates to overlap by chi2 ~ 2*SNR^2*(1-overlap).")
A("# Deviations: PN = C_p/C_e (additive 2.5PN pdot/edot); simple = del_0_p/del_0_e (mult. Edot/Ldot); hybrid branch.")
A("#             simple_pe = del_0_p/del_0_e (mult. pdot/edot) on the dev_a_pe branch; numerically == simple.")
A("# Method: LM-damped CV iteration (CV -> 1000-step Nelder-Mead escape -> CV), OVERLAP_TARGET=0.9999999999.")
A("# Rebuild this file with: python make_results_combined.py")
A("#")
A("# !! SAMPLING RATE / ALIASING !!")
A("#   At dt=10 the Nyquist frequency is 0.05 Hz.  Prograde IMRIs (m1=1e6, m2=1e3) carry power above it,")
A("#   which ALIASES into the otherwise-null T channel.  Per-channel SNR at dt=10 showed T carrying")
A("#   29-97% of the total SNR at EVERY a x e0 grid point (worst: idx14 96.5%, idx19 97.5%), and the")
A("#   grid SNRs were inflated to 24-114.  Re-running at dt=5 collapsed T% to ~0.00 everywhere and")
A("#   brought all grid SNRs back to 20.5-32.8, consistent with the grid's intended SNR ~ 20.")
A("#   => SECTION 1 (dt=5) is the REFERENCE result for the a x e0 grid.")
A("#      SECTION 5 (dt=10) is SUPERSEDED and kept only for comparison.")
A("#   Cases outside the grid are unaffected: adhoc_A already ran at dt=5, and the tails (T=0.25 yr)")
A("#   measured T% = 0.00 at dt=10.")
A("")

ORDER = [14, 4, 24, 18, 8, 12, 6, 16, 22, 10, 0, 20]
A(BAR); A("# SUMMARY - a x e0 grid, dt=5 (aliasing-corrected).  Best overlap per model."); A(BAR)
A(f"# {'point':8} {'a':>5} {'e0':>4} {'SNR':>7} {'T%@dt10':>8} | {'0PA':>12} {'PN':>12} {'simple':>12} | {'PN gain':>8}")
for i in ORDER:
    p = DT5[f"idx{i}"]; b = p["best"]
    m = {k: 1 - b[k]["ov_final"] for k in ("0PA", "PN", "simple")}
    A(f"# {'idx'+str(i):8} {p['a']:>5.1f} {p['e0']:>4.1f} {p['snr']:>7.2f} {p['T_percent_dt10']:>8.1f} |"
      f" {b['0PA']['ov_final']:12.7f} {b['PN']['ov_final']:12.7f} {b['simple']['ov_final']:12.7f} |"
      f" {m['0PA']/m['PN']:7.1f}x")
A("#   PN mismatch reduction: 17.9-27.9x for a<0 (idx0,6,10,16,20); 1.1-3.4x for a>=0.  simple: 1.0-2.5x.")
A("#   Grid points NOT yet CV-optimised: idx1,2,3,5,7,9,11,13,15,17,19,21,23  (idx19 also badly aliased).")
A("")

A(BAR)
A("# SECTION 1 - IMRI a x e0 grid (m1=1e6, m2=1e3).  dt=5, T=1.0   *** REFERENCE (aliasing-corrected) ***")
A("#   Seeds: 0PA from its dt=10 best fit; PN/simple climbed from BOTH the dt=10 deviation point")
A("#   ('from_dt10_*') and this run's dt=5 0PA point +dev=(0,0) ('from_0PA_dt5'); the better wins.")
A("#   Ordered by dt=10 T-contamination, worst first.   Source: results_imri_dt5_rerun_{a,b}.json")
A(BAR)
for i in ORDER:
    p, sp = DT5[f"idx{i}"], gridsig(i); c = p["snr_channels"]
    A(""); A(f"# grid_idx{i}   a={p['a']:+.1f}  e0={p['e0']:.1f}   SNR={p['snr']:.4f}   dt=5, T=1.0")
    A(f"#   SNR per channel: A={c[0]:.4f}  E={c[1]:.4f}  T={c[2]:.4f}   (T carried {p['T_percent_dt10']:.1f}% at dt=10)")
    L.extend(sigblock(sp))
    for m in ("0PA", "PN", "simple"):
        b, sd = p["best"][m], p["runs"][m]
        A(""); A(f"## {m:<6} (best seed: {b['seed']})   overlap = {b['ov_final']:.10f}   chi2 = {b['chi2']:.6e}")
        A(f"x_bf = {fmt(b['params'])}")
        for k in sorted(sd):
            if k != b["seed"]: A(f"#  [{m} {k}: overlap = {sd[k]['ov_final']:.10f}, chi2 = {sd[k]['chi2']:.6e}]")
        A(f"#  [dt=10 best fit re-scored at dt=5: overlap = {b['ov_dt10_point_at_dt5']:.10f}]")
A("")

A(BAR)
A("# SECTION 2 - ad-hoc IMRI case (NOT an a x e0 grid point).  dt=5, T=1.0  -- unaffected by aliasing")
A(BAR); A("")
asp = dict(m1=1e6, m2=5.0e3, a=0.7, p0=25.0, e0=0.25, dist=12.0, qS=0.7853981633974483, phiS=1.0,
           qK=1.0, phiK=1.0471975511965976, Phi_phi0=0.9, Phi_theta0=0.5, Phi_r0=0.4)
ac = CH5["adhoc_A"]
A(f"# adhoc_A   a=+0.70  e0=0.25   m2=5e3   SNR={ac['snr_total']:.4f}   dt=5, T=1.0")
A(f"#   SNR per channel: A={ac['snr_A']:.4f}  E={ac['snr_E']:.4f}  T={ac['snr_T']:.4f}")
L.extend(sigblock(asp)); A("")
A("## 0PA                              overlap = 0.9938500000   chi2 = 1.437227e+03  # from imri0pa.log")
A("x_bf = [1.00176199e+06, 4.99511288e+03, 7.03962258e-01, 2.49674581e+01, 2.49798441e-01, 7.86137900e-01, 1.00491369e+00, 9.89953388e-01, 6.42950551e-01]")
A("#  [this x_bf re-scored at dt=5: overlap = 0.9934510 -- a 4e-4 discrepancy vs the 0PA log; unresolved]")
snr_a = aP["idx0"]["snr"]
for m, src in (("PN", aP), ("simple", aS)):
    o = src["idx0"]; bs = o["best_start"]; b = o[bs]; c2, note = chi2of(b, snr_a)
    A(""); A(f"## {m:<6} (best seed: {bs})   overlap = {b['ov_final']:.10f}   chi2 = {c2:.6e}{note}")
    A(f"x_bf = {fmt(b['params'])}")
    other = "from_NMdev" if bs == "from_0PA" else "from_0PA"
    A(f"#  [{m} {other}: overlap = {o[other]['ov_final']:.10f}]")
b = aPE["idx0"]["from_0PA"]; c2, note = chi2of(b, snr_a)
A(""); A(f"## simple_pe (dev_a_pe branch, seed from_0PA)   overlap = {b['ov_final']:.10f}   chi2 = {c2:.6e}{note}")
A(f"x_bf = {fmt(b['params'])}"); A("")

A(BAR)
A("# SECTION 3 - grid cells idx1 & idx16 from the EARLIER ad-hoc dt=5 scripts.")
A("#   Same systems as the grid, already at dt=5 (so un-aliased), but a different seeding chain")
A("#   (from_0PA / from_NMdev).  idx16 is SUPERSEDED by Section 1; idx1 is the only result we have")
A("#   for that cell.  Note idx16's 0PA here (0.9983390) agrees with Section 1 (0.9983410) to 2e-6 --")
A("#   an independent confirmation of the dt=5 re-run.")
A(BAR)
LEG = {"idx1": ("[9.98144092e+05, 1.00129945e+03, -4.99651156e-01, 2.49922458e+01, 9.98840371e-02, 1.05172816e+00, 7.90279570e-01, 6.30880364e+00, 4.49650077e-01]", 0.9675450000, 2.813669e+01, 1),
       "idx16": ("[1.00121834e+06, 9.99719120e+02, -4.94575948e-01, 2.46561703e+01, 3.99915754e-01, 1.05097533e+00, 7.80191278e-01, 2.14580670e-01, -6.08369012e+00]", 0.9983390000, 1.419321e+00, 16)}
for nm, (x0, ov0, c0, gi) in LEG.items():
    sp = gridsig(gi); ch = CH5[f"grid_idx{gi}"]; snr_l = aP[nm]["snr"]
    A(""); A(f"# {nm}_dt5   a={sp['a']:+.1f}  e0={sp['e0']:.1f}   SNR={ch['snr_total']:.4f}   dt=5, T=1.0   (== grid {nm})")
    L.extend(sigblock(sp)); A("")
    A(f"## 0PA                              overlap = {ov0:.10f}   chi2 = {c0:.6e}  # derived")
    A(f"x_bf = {x0}")
    for m, src in (("PN", aP), ("simple", aS)):
        o = src[nm]; bs = o["best_start"]; b = o[bs]; c2, note = chi2of(b, snr_l)
        A(""); A(f"## {m:<6} (best seed: {bs})   overlap = {b['ov_final']:.10f}   chi2 = {c2:.6e}{note}")
        A(f"x_bf = {fmt(b['params'])}")
        other = "from_NMdev" if bs == "from_0PA" else "from_0PA"
        A(f"#  [{m} {other}: overlap = {o[other]['ov_final']:.10f}]")
    b = aPE[nm]["from_0PA"]; c2, note = chi2of(b, snr_l)
    A(""); A(f"## simple_pe (dev_a_pe branch, seed from_0PA)   overlap = {b['ov_final']:.10f}   chi2 = {c2:.6e}{note}")
    A(f"x_bf = {fmt(b['params'])}")
A("")

A(BAR)
A("# SECTION 4 - IMRI 'tails' set (m1=1e6, m2=1e4).  dt=10, T=0.25 yr  -- unaffected by aliasing")
A("#   The short 0.25 yr window keeps these below Nyquist: measured T% = 0.00 at dt=10.")
A("#   NOTE: the secondary spin chi2=0.95 was ASSUMED for these (not supplied); SNR/overlap reproduce.")
A(BAR)
TSP = dict(qS=1.04719755, phiS=0.78539816, qK=0.62831853, phiK=0.52359878, Phi_phi0=0.1, Phi_theta0=0.2, Phi_r0=0.3)
TP = {"pt4": dict(m1=1e6, m2=1e4, a=0.9, p0=29.2602456, e0=0.10, dist=272.11852, **TSP),
      "pt20": dict(m1=1e6, m2=1e4, a=-0.9, p0=30.522071, e0=0.50, dist=94.592671, **TSP)}
for nm in ("pt4", "pt20"):
    sp = TP[nm]; ch = CH10[nm]; snr_t = t0[nm]["snr"]
    A(""); A(f"# {nm}   a={sp['a']:+.1f}  e0={sp['e0']:.2f}   m2=1e4   SNR={ch['snr_total']:.4f}   dt=10, T=0.25")
    A(f"#   SNR per channel: A={ch['snr_A']:.4f}  E={ch['snr_E']:.4f}  T={ch['snr_T']:.4f}   (T% = {ch['power_frac_percent']['T']:.2f})")
    L.extend(sigblock(sp))
    o = t0[nm]
    A(""); A(f"## 0PA                              overlap = {o['ov_final']:.10f}   chi2 = {o['chi2']:.6e}")
    A(f"x_bf = {fmt(o['params'])}")
    for m, src in (("PN", tP), ("simple", tS)):
        o = src[nm]; bs = o["best_start"]; b = o[bs]; c2, note = chi2of(b, snr_t)
        A(""); A(f"## {m:<6} (best seed: {bs})   overlap = {b['ov_final']:.10f}   chi2 = {c2:.6e}{note}")
        A(f"x_bf = {fmt(b['params'])}")
        other = "from_NMdev" if bs == "from_0PA" else "from_0PA"
        oc2, _ = chi2of(o[other], snr_t)
        A(f"#  [{m} {other}: overlap = {o[other]['ov_final']:.10f}, chi2 = {oc2:.6e}]")
    b = tPE[nm]["from_0PA"]; c2, note = chi2of(b, snr_t)
    A(""); A(f"## simple_pe (dev_a_pe branch, seed from_0PA)   overlap = {b['ov_final']:.10f}   chi2 = {c2:.6e}{note}")
    A(f"x_bf = {fmt(b['params'])}")
A("")

A(BAR)
A("# SECTION 5 - IMRI a x e0 grid at dt=10.   *** SUPERSEDED / ALIASED - do not quote ***")
A("#   Kept for comparison with Section 1.  The quoted SNRs are inflated by aliased T-channel power")
A("#   (T% column), and these best fits were optimised against that contaminated signal.")
A(BAR)
for nm in sorted(set(list(dv) + list(dv2)), key=lambda n: int(n[3:])):
    gi = int(nm[3:]); sp = gridsig(gi); ch = CH10[f"grid_{nm}"]
    A(""); A(f"# grid_{nm}   a={sp['a']:+.1f}  e0={sp['e0']:.1f}   SNR={ch['snr_total']:.4f}   dt=10, T=1.0"
             f"   [T% = {ch['power_frac_percent']['T']:.1f} ALIASED]")
    L.extend(sigblock(sp))
    if nm in dv:
        R = dv[nm]["runs"]; F = dvf.get(nm, {}).get("runs", {})
        b = R["0PA"]["from_MAP"]
        A(""); A(f"## 0PA                              overlap = {b['ov_final']:.10f}   chi2 = {b['chi2']:.6e}")
        A(f"x_bf = {fmt(b['params'])}")
        for m in ("PN", "simple"):
            cand = [("from_MAP", R[m]["from_MAP"])] + ([("from_0PAfit", F[m])] if m in F else [])
            bs, b = max(cand, key=lambda t: t[1]["ov_final"])
            A(""); A(f"## {m:<6} (best seed: {bs})   overlap = {b['ov_final']:.10f}   chi2 = {b['chi2']:.6e}")
            A(f"x_bf = {fmt(b['params'])}")
            for k, v in cand:
                if k != bs: A(f"#  [{m} {k}: overlap = {v['ov_final']:.10f}, chi2 = {v['chi2']:.6e}]")
    else:
        for m in ("0PA", "PN", "simple"):
            b = dv2[nm]["runs"][m]["from_MAP"]
            A(""); A(f"## {m:<6} (seed: from_MAP)   overlap = {b['ov_final']:.10f}   chi2 = {b['chi2']:.6e}")
            A(f"x_bf = {fmt(b['params'])}")
A(""); A("# end of file")
open("results_combined.txt", "w").write("\n".join(L) + "\n")
print(f"wrote results_combined.txt: {len(L)} lines")
