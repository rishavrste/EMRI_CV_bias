"""Summary of every run in environemen_2_eccen: environmental (disk) signal fitted with
0PA vacuum GR and with 0PA + 2.5PN flux deviation, for each parameter set tried.

Reads only the results JSONs -- nothing is re-fitted.  The one thing computed here is the
dephasing for set 1 (result_shub), whose jobs predate the dephasing stage; it is a CPU
trajectory integration of seconds and is cached to result_shub/dephasing_a0.0_e0.04.json.

Writes into summary/:
  SUMMARY.md             every table (runs, optimiser stages, overlap, bias, sigma, dephasing)
  summary.json           all sets, both models, full records
  bias_comparison.json   per set and parameter: bias, sigma, bias/sigma for 0PA vs 0PA+PN
  runs/<set>_<model>.json  one file per optimisation (10 files)
Re-run it, never hand-edit any of them.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "run_4"))
from dephasing import trajectory, dephasing                                  # noqa: E402

PHASES = ("Phi_phi0", "Phi_r0")
MODELS = (("vacuum", "0PA"), ("pn", "0PA+PN"))
SET1_DEPH = os.path.join(HERE, "result_shub", "dephasing_a0.0_e0.04.json")
OUT = os.path.join(HERE, "summary")
PARAMS = ("m1", "m2", "p0", "e0", "Phi_phi0", "Phi_r0")
DEVS = ("dev_1", "dev_2")
FILE_TAG = {"vacuum": "0PA", "pn": "0PA_PN"}


def load(*parts):
    return json.load(open(os.path.join(HERE, *parts)))


def wrap(x):
    """A phase offset folded into (-pi, pi]."""
    return float(np.pi - (np.pi - x) % (2 * np.pi))


def wrapped(names, theta, inj):
    """Best fit with the phases moved by 2 pi k to lie within pi of the injection."""
    return [i + wrap(t - i) if n in PHASES else t for n, t, i in zip(names, theta, inj)]


def record(names, inj, theta, sigma, ov_seed, ov, chi2, deph, method, stages):
    return dict(names=names, theta_inj=inj, theta=wrapped(names, theta, inj), sigma=sigma,
                overlap_seed=ov_seed, overlap=ov, chi2=chi2, dephasing=deph, method=method,
                stages=stages)


def de_stages(r):
    """Seed -> DE -> LM history of a de_run.py / de_env_vs_*.py results JSON."""
    de, lm = r["de"], r["lm"]
    return dict(
        seed=dict(overlap=r["overlap_seed"], chi2=r.get("chi2_seed")),
        de=dict(overlap=de["overlap"], chi2=de["chi2"], nit=de["nit"], nfev=de["nfev"],
                message=de.get("message"), seconds=de.get("seconds"),
                params=wrapped(r["names"], de["params"], r["theta_inj"])),
        lm=dict(overlap=lm["overlap"], chi2=lm["chi2"], iterations=len(lm["history"]) - 1,
                nm_used=lm.get("nm_used"), seconds=lm.get("seconds")),
        search_box=dict(zip(r["names"], r.get("bounds", []))), sigma_inj=r.get("sigma_inj"),
        when=r.get("when"), total_seconds=r.get("seconds"))


def notebook_stages(r):
    return dict(
        seed=dict(overlap=r["overlap_seed"], chi2=r["chi2_seed"]),
        lm1=dict(overlap=r["overlap_lm1"], iterations=r["lm1_iterations"],
                 stalled=r["lm1_stalled"]),
        nm=dict(overlap=r["overlap_nm"], nfev=r["nm_nfev"]),
        final=dict(overlap=r["overlap_final"], chi2=r["chi2_final"]))


# ---- set 1 (top level / result_shub): dephasing computed here ----------------------------
def pn_tail(p):
    # SuperKludgeFlux: chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included,
    # C_p (dev_1), C_e (dev_2), del_0_p, del_0_e
    return [0.0, False, False, False, True, p.get("dev_1", 0.0), p.get("dev_2", 0.0), 0.0, 0.0]


def dephasing_point(inj, disk, template, names, theta, T):
    from few.trajectory.ode.flux import KerrEccEqAccFlux, SuperKludgeFlux
    params = dict(inj)
    params.update(dict(zip(names, theta)))
    flux, tail = ((KerrEccEqAccFlux, []) if template == "vacuum"
                  else (SuperKludgeFlux, pn_tail(params)))
    sig = trajectory(KerrEccEqAccFlux, inj, disk, T)
    tmpl = trajectory(flux, params, tail, T)
    return dephasing(sig, tmpl, (inj["Phi_phi0"], inj["Phi_r0"]),
                     (params["Phi_phi0"], params["Phi_r0"]))


def set1_dephasing(runs):
    if os.path.exists(SET1_DEPH):
        return json.load(open(SET1_DEPH))
    out = {}
    for template, r in runs.items():
        cfg = r["config"]
        disk = [cfg["disk_geometric"]["Sigma0"] if "disk_geometric" in cfg else None,
                cfg["disk"]["h0"], cfg["disk"]["Sigma_p"]]
        out[template] = {pt: dephasing_point(cfg["injection"], disk, template, r["_names"], th,
                                             r["T"])
                         for pt, th in (("injection", r["theta_inj"]),
                                        ("best fit", r["_theta"]))}
    json.dump(out, open(SET1_DEPH, "w"), indent=1)
    return out


def set1():
    runs = {"vacuum": load("results_de_env_vs_vacuum.json"),
            "pn": load("results_de_env_vs_pn.json")}
    runs["vacuum"]["overlap_seed"] = runs["vacuum"]["overlap_inj"]
    runs["vacuum"]["config"]["disk_geometric"] = runs["pn"]["config"]["disk_geometric"]
    for t, r in runs.items():
        r["_names"] = r.get("names") or r["config"]["infer"]
        r["_theta"] = wrapped(r["_names"], r["lm"]["params"], r["theta_inj"])
    deph = set1_dephasing(runs)
    cfg = runs["pn"]["config"]
    return dict(
        tag="set 1", folder="result_shub (+ top-level de_env_vs_*.py)",
        injection=cfg["injection"], disk=cfg["disk"], T=runs["pn"]["T"], snr=runs["pn"]["snr"],
        method="DE +-5 sigma, then LM",
        models={t: record(r["_names"], r["theta_inj"], r["lm"]["params"], r["lm"]["sigma"],
                          r["overlap_seed"], r["lm"]["overlap"], r["lm"]["chi2"], deph[t],
                          "DE+LM", de_stages(dict(r, names=r["_names"])))
                for t, r in runs.items()})


# ---- sets 2, 3, 5 (new_runs, new_run_3, run_5): DE jobs with dephasing in the JSON ------------------
def de_set(tag, folder):
    runs = {"vacuum": load(folder, "results_de_0pa.json"),
            "pn": load(folder, "results_de_0pa_pn.json")}
    cfg = runs["pn"]["config"]
    models = {}
    for t, r in runs.items():
        d = r["dephasing"]
        models[t] = record(r["names"], r["theta_inj"], r["lm"]["params"], r["lm"]["sigma"],
                           r["overlap_seed"], r["lm"]["overlap"], r["lm"]["chi2"],
                           {"injection": d["injection"], "best fit": d["DE+LM"],
                            "DE only": d["DE"]}, "DE+LM", de_stages(r))
    return dict(tag=tag, folder=folder, injection=cfg["injection"], disk=cfg["disk"],
                T=runs["pn"]["T"], snr=runs["pn"]["snr"], method="DE +-5 sigma, then LM",
                models=models)


# ---- set 4, superseded (run_4): the notebook fit at fmax = 0.1 Hz, dominated by the TDI null --
def notebook_set():
    nb = load("run_4", "results_notebook.json")
    labels = {"vacuum": "0PA (vacuum GR)", "pn": "0PA + 2.5PN deviation"}
    models = {}
    for t, label in labels.items():
        r = nb["results"][label]
        models[t] = record(r["names"], r["theta_inj"], r["theta_raw"], r["sigma"],
                           r["overlap_seed"], r["overlap_final"], r["chi2_final"],
                           r["dephasing"], "LM/NM from injection (test.ipynb)",
                           notebook_stages(r))
    return dict(tag="set 4", folder="run_4", injection=nb["injection"], disk=nb["disk"],
                T=nb["T"], snr=None, method="LM -> NM -> LM from the injection (test.ipynb)",
                models=models)


# ---- set 4 (run_4_fmax005): LM -> NM -> LM from the old best fits, band capped at 0.05 Hz --
SET4_DIR = "run_4_fmax005"
SET4_DEPH = os.path.join(HERE, SET4_DIR, "dephasing.json")


def lm_stages(r):
    """Seed -> LM1 -> (NM) -> LM2 history of an lm_from_0pa_env_vs_pn.py results JSON."""
    h = r["lm"]["history"]
    cut = next((k for k, e in enumerate(h) if e.get("nm")), len(h))
    lm1, lm2 = h[:cut], h[cut + 1:]
    return dict(
        seed=dict(overlap=r["overlap_seed"], chi2=r["chi2_seed"]),
        lm1=dict(overlap=lm1[-1]["overlap"], iterations=lm1[-1]["it"],
                 stalled=bool(r["lm"]["nm_used"])),
        nm=dict(overlap=lm2[0]["overlap"] if lm2 else lm1[-1]["overlap"], nfev=None),
        final=dict(overlap=r["lm"]["overlap"], chi2=r["lm"]["chi2"],
                   lm2_iterations=lm2[-1]["it"] if lm2 else None),
        seconds=r["seconds"], when=r["when"], fmax=r["config"]["waveform"]["fmax"])


def set4_dephasing(runs):
    if os.path.exists(SET4_DEPH):
        return json.load(open(SET4_DEPH))
    cfg = runs["pn"]["config"]
    disk = [cfg["disk_geometric"]["Sigma0"], cfg["disk"]["h0"], cfg["disk"]["Sigma_p"]]
    out = {t: {pt: dephasing_point(cfg["injection"], disk, t, r["names"], th, r["T"])
               for pt, th in (("injection", r["theta_inj"]), ("best fit", r["lm"]["params"]))}
           for t, r in runs.items()}
    json.dump(out, open(SET4_DEPH, "w"), indent=1)
    return out


def lm_set():
    runs = {t: load(SET4_DIR, f"results_lm_{FILE_TAG[t].lower()}.json") for t in FILE_TAG}
    deph = set4_dephasing(runs)
    cfg = runs["pn"]["config"]
    method = "LM -> NM -> LM from the fmax = 0.1 best fits, band [1e-5, 0.05] Hz"
    models = {t: record(r["names"], r["theta_inj"], r["lm"]["params"], r["lm"]["sigma"],
                        r["overlap_seed"], r["lm"]["overlap"], r["lm"]["chi2"], deph[t],
                        "LM/NM (fmax 0.05)", lm_stages(r))
              for t, r in runs.items()}
    return dict(tag="set 4", folder=SET4_DIR, injection=cfg["injection"], disk=cfg["disk"],
                T=runs["pn"]["T"], snr=runs["pn"]["snr"], method=method, models=models)


# ---- markdown -----------------------------------------------------------------------------
def fmt_set_row(s):
    inj, d = s["injection"], s["disk"]
    snr = f"{s['snr']:.2f}" if s["snr"] else "-"
    return (f"| {s['tag']} | `{s['folder']}` | {inj['p0']:g} | {inj['e0']:g} | "
            f"{d['Sigma0_cgs']:g} | {d['h0']:g} | {s['T']:.6f} | {snr} | {s['method']} |")


def md_sets(sets):
    rows = ["## The runs", "",
            "All sets: m1 = 1e6, m2 = 2e3, a = 0, xI0 = 1, dist = 16.344, Phi_phi0 = Phi_r0 = 1, "
            "Sigma_p = -1.5, dt = 5 s, AET, T = 0.99 x plunge.  "
            "Fitted: m1, m2, p0, e0, Phi_phi0, Phi_r0 (+ C_p, C_e for PN).", "",
            "| set | folder | p0 | e0 | Sigma0 [g/cm^2] | h0 | T [yr] | SNR | method |",
            "|---|---|---|---|---|---|---|---|---|"]
    return rows + [fmt_set_row(s) for s in sets] + [""]


def md_overlap(sets):
    rows = ["## Overlap", "",
            "| set | overlap at injection | 0PA overlap | 0PA mismatch | 0PA chi2 | "
            "PN overlap | PN mismatch | PN chi2 | mismatch 0PA / PN |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in sets:
        v, p = s["models"]["vacuum"], s["models"]["pn"]
        rows.append(f"| {s['tag']} | {v['overlap_seed']:.6f} | {v['overlap']:.12f} | "
                    f"{1 - v['overlap']:.3e} | {v['chi2']:.3e} | {p['overlap']:.12f} | "
                    f"{1 - p['overlap']:.3e} | {p['chi2']:.3e} | "
                    f"{(1 - v['overlap']) / (1 - p['overlap']):.2f}x |")
    return rows + [""]


def bias(m, n):
    j = m["names"].index(n)
    b = m["theta"][j] - m["theta_inj"][j]
    return b, b / m["sigma"][j]


def md_bias(sets):
    params = ("m1", "m2", "p0", "e0", "Phi_phi0", "Phi_r0")
    rows = ["## Bias / sigma  (bias = best fit - injection, sigma = Fisher at the best fit)", "",
            "| set | model | " + " | ".join(params) + " | C_p +- sigma | C_e +- sigma |",
            "|---|---|" + "---|" * (len(params) + 2)]
    for s in sets:
        for t, label in MODELS:
            m = s["models"][t]
            cells = [f"{bias(m, n)[1]:+.3f}" for n in params]
            devs = []
            for n in ("dev_1", "dev_2"):
                if n in m["names"]:
                    j = m["names"].index(n)
                    devs.append(f"{m['theta'][j]:+.3g} +- {m['sigma'][j]:.3g}")
                else:
                    devs.append("-")
            rows.append(f"| {s['tag']} | {label} | " + " | ".join(cells + devs) + " |")
    return rows + [""]


def md_abs_bias(sets):
    rows = ["## Absolute bias of p0 and e0 -- does PN de-bias, or only inflate sigma?", "",
            "| set | e0 bias 0PA | e0 bias PN | abs ratio PN/0PA | sigma(e0) PN/0PA | "
            "p0 bias 0PA | p0 bias PN | abs ratio PN/0PA |",
            "|---|---|---|---|---|---|---|---|"]
    for s in sets:
        v, p = s["models"]["vacuum"], s["models"]["pn"]
        ev, ep = bias(v, "e0")[0], bias(p, "e0")[0]
        pv, pp = bias(v, "p0")[0], bias(p, "p0")[0]
        sig = p["sigma"][p["names"].index("e0")] / v["sigma"][v["names"].index("e0")]
        rows.append(f"| {s['tag']} | {ev:+.3e} | {ep:+.3e} | {abs(ep / ev):.2f} | {sig:.2f} | "
                    f"{pv:+.3e} | {pp:+.3e} | {abs(pp / pv):.2f} |")
    return rows + [""]


def md_dephasing(sets):
    rows = ["## Dephasing, template - signal, at the end of the common span [rad]", "",
            "The injection row uses plain vacuum (dev = 0), so it is the disk's own dephasing.  "
            "orbital = accumulated from zero; total = orbital + initial-phase offset.", "",
            "| set | point | dPhi_phi orbital | dPhi_phi total | dPhi_r orbital | dPhi_r total |",
            "|---|---|---|---|---|---|"]
    for s in sets:
        points = [("injection", s["models"]["vacuum"]["dephasing"]["injection"])]
        points += [(f"{label} fit", s["models"][t]["dephasing"]["best fit"]) for t, label in MODELS]
        for name, d in points:
            rows.append(f"| {s['tag']} | {name} | {d['Phi_phi']['orbital_final']:+.4e} | "
                        f"{d['Phi_phi']['total_final']:+.4e} | {d['Phi_r']['orbital_final']:+.4e} | "
                        f"{d['Phi_r']['total_final']:+.4e} |")
    return rows + [""]


def md_vectors(sets):
    rows = ["## Best-fit vectors (phases wrapped to within pi of the injection)", "",
            "Order: m1, m2, p0, e0, Phi_phi0, Phi_r0 (, C_p, C_e).", ""]
    for s in sets:
        for t, label in MODELS:
            vec = ", ".join(f"{x:.10e}" for x in s["models"][t]["theta"])
            rows.append(f"- {s['tag']} {label}: `[{vec}]`")
    return rows + [""]


def md_stages(sets):
    rows = ["## Optimiser stages", "",
            "DE sets: overlap after the DE (+-5 sigma box) and after the LM polish.  Set 4: "
            "LM -> NM -> LM from the old (fmax = 0.1 Hz) best fits, band capped at 0.05 Hz.", "",
            "| set | model | seed overlap | stage 1 overlap | stage 1 detail | final overlap | "
            "final detail |", "|---|---|---|---|---|---|---|"]
    for s in sets:
        for t, label in MODELS:
            st = s["models"][t]["stages"]
            if "de" in st:
                d, l = st["de"], st["lm"]
                one = f"DE: nit {d['nit']}, nfev {d['nfev']}, {d['message']}"
                nm = "-" if l["nm_used"] is None else l["nm_used"]
                fin = f"LM: {l['iterations']} it, NM used {nm}"
                ov1, ovf = d["overlap"], l["overlap"]
            else:
                nfev = st["nm"]["nfev"]
                one = (f"LM1: {st['lm1']['iterations']} it, stalled {st['lm1']['stalled']}"
                       + (f"; NM: nfev {nfev}" if nfev is not None else "; NM escape"))
                fin = "final LM"
                ov1, ovf = st["nm"]["overlap"], st["final"]["overlap"]
            rows.append(f"| {s['tag']} | {label} | {st['seed']['overlap']:.6f} | {ov1:.12f} | "
                        f"{one} | {ovf:.12f} | {fin} |")
    return rows + [""]


def md_bias_sigma(sets):
    rows = ["## Absolute bias and Fisher sigma, every parameter", "",
            "Each cell: bias (sigma).", "",
            "| set | model | " + " | ".join(PARAMS) + " |", "|---|---|" + "---|" * len(PARAMS)]
    for s in sets:
        for t, label in MODELS:
            m = s["models"][t]
            cells = [f"{bias(m, n)[0]:+.3e} ({m['sigma'][m['names'].index(n)]:.3e})"
                     for n in PARAMS]
            rows.append(f"| {s['tag']} | {label} | " + " | ".join(cells) + " |")
    return rows + [""]


def md_notes():
    return ["## Caveats", "",
            "- Set 4 is fitted over [1e-5, 0.05] Hz; sets 1, 2, 3 and 5 over [1e-5, 0.1] Hz "
            "(see the TDI-null section below). All use dt = 5 s.",
            "- Set 4's dephasing is computed here (cached in `run_4_fmax005/dephasing.json`).",
            "- Every 0PA+PN DE (sets 1, 2, 3, 5) stopped at its 1100-iteration cap, so the LM "
            "stage did the real work there -- most in set 5 (DE ov 0.98882, phases far off; LM "
            "stopped when no damped step lowered chi2, not on the relative-tolerance test) and "
            "set 1 (DE ov 0.98277, 134 LM iterations).",
            "- Set 1 predates the dephasing stage; its dephasing is recomputed here (cached in "
            "`result_shub/dephasing_a0.0_e0.04.json`).", ""] + md_tdi_null()


# dt = 5 s, band [1e-5, 0.1] Hz; from summary/snr_channels_dt.ipynb, section 6
TDI_NULL_ROWS = (  # set, SNR < 0.1 Hz, SNR < 0.05 Hz, share of SNR^2 within 2 mHz of the null
    ("set 1", 85.3661, 84.2756, 2.538e-02),
    ("set 2", 82.3148, 81.8243, 1.188e-02),
    ("set 3", 82.1978, 82.1336, 1.562e-03),
    ("set 4", 351.3474, 81.7193, 9.459e-01),
    ("set 5", 85.7017, 85.6076, 2.197e-03),
)


def md_tdi_null():
    lines = ["## Set 4: TDI-null artifact and the fmax = 0.05 Hz re-fit", "",
             "Every set here uses dt = 5 s and the band [1e-5, 0.1] Hz. The first-generation TDI "
             "A/E PSD has a null at c/(2L) = 0.05996 Hz (L = 2.5e9 m), which lies inside that "
             "band. There the PSD falls to about 1e-51, so any numerical residual of the response "
             "is divided by an almost-zero noise level and turns into spurious SNR. The source "
             "itself has no power above about 0.01 Hz.", "",
             "Measured per set (`summary/snr_channels_dt.ipynb`, section 6):", "",
             "| set | SNR, f < 0.1 Hz | SNR, f < 0.05 Hz | share of SNR^2 within 2 mHz of the null |",
             "|---|---|---|---|"]
    lines += [f"| {s} | {a:.4f} | {b:.4f} | {sh:.3e} |" for s, a, b, sh in TDI_NULL_ROWS]
    lines += ["",
              "- **Only set 4 is affected.** About 95% of its SNR^2 sits at the null: its SNR is "
              "351 with the band up to 0.1 Hz but 81.7 below 0.05 Hz, in line with the other "
              "sets. The vacuum waveform with the same parameters shows the same excess, so the "
              "disk does not cause it.",
              "- Sets 1, 2, 3 and 5 carry only 0.16-2.5% of their SNR^2 near the null, and "
              "capping the band at 0.05 Hz changes their SNR by 1.3% at most. Their fits stay at "
              "fmax = 0.1 Hz. (The notebook's automatic verdict uses a strict 1e-3 threshold and "
              "flags them anyway; on these numbers they need no redo.)",
              "- **The old set-4 fit (`run_4`, test.ipynb, fmax = 0.1 Hz) is superseded.** It "
              "was optimised against a signal dominated by the artifact (in-band SNR 351).",
              "- **Set 4 in this summary is the re-fit** in `run_4_fmax005/`: fmax = 0.05 Hz, "
              "dt = 5 s, LM -> Nelder-Mead escape -> LM, seeded at the old 0PA and 0PA+PN best "
              "fits. SNR 81.72. The escape gained only ~6e-11 in overlap and the final LM found "
              "no lower chi2, so both fits are converged. Its mismatches (0PA 2.88e-6, PN "
              "5.43e-7) are larger than the old ones (3.18e-7, 9.51e-8): the artifact power, "
              "matched almost equally by both templates, had diluted the true mismatch.", ""]
    return lines


# ---- json outputs -------------------------------------------------------------------------
def param_comparison(s):
    v, p = s["models"]["vacuum"], s["models"]["pn"]
    out = {}
    for n in PARAMS:
        (bv, rv), (bp, rp) = bias(v, n), bias(p, n)
        sv, sp = v["sigma"][v["names"].index(n)], p["sigma"][p["names"].index(n)]
        out[n] = dict(injection=v["theta_inj"][v["names"].index(n)],
                      bias_0PA=bv, sigma_0PA=sv, bias_over_sigma_0PA=rv,
                      bias_PN=bp, sigma_PN=sp, bias_over_sigma_PN=rp,
                      abs_bias_ratio_PN_over_0PA=abs(bp / bv), sigma_ratio_PN_over_0PA=sp / sv)
    devs = {n: dict(best_fit=p["theta"][p["names"].index(n)],
                    sigma=p["sigma"][p["names"].index(n)]) for n in DEVS}
    return dict(tag=s["tag"], folder=s["folder"],
                mismatch_0PA=1 - v["overlap"], mismatch_PN=1 - p["overlap"],
                mismatch_ratio_0PA_over_PN=(1 - v["overlap"]) / (1 - p["overlap"]),
                params=out, pn_deviation=devs)


def run_record(s, t):
    m = s["models"][t]
    return dict(set=s["tag"], model=dict(MODELS)[t], folder=s["folder"],
                injection=s["injection"], disk=s["disk"], T=s["T"], snr=s["snr"],
                mismatch=1 - m["overlap"],
                bias={n: bias(m, n)[0] for n in m["names"] if n not in DEVS},
                bias_over_sigma={n: bias(m, n)[1] for n in m["names"] if n not in DEVS}, **m)


def write_json(sets):
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    json.dump(sets, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    json.dump([param_comparison(s) for s in sets],
              open(os.path.join(OUT, "bias_comparison.json"), "w"), indent=1)
    for s in sets:
        for t in ("vacuum", "pn"):
            name = f"{s['tag'].replace(' ', '')}_{FILE_TAG[t]}.json"
            json.dump(run_record(s, t), open(os.path.join(OUT, "runs", name), "w"), indent=1)


def main():
    sets = [set1(), de_set("set 2", "new_runs"), de_set("set 3", "new_run_3"), lm_set(),
            de_set("set 5", "run_5")]
    lines = ["# environemen_2_eccen: environmental signal vs 0PA and 0PA + PN, all runs", "",
             "Generated by `make_summary.py` from the results JSONs -- do not hand-edit.  "
             "Per-set detail: `result_shub/*.txt`, `new_runs/RESULTS.md`, "
             "`new_run_3/RESULTS.md`, `run_4_fmax005/lm_*.log`, `run_5/RESULTS.md`.", ""]
    for section in (md_sets, md_stages, md_overlap, md_bias, md_abs_bias, md_bias_sigma,
                    md_dephasing, md_vectors):
        lines += section(sets)
    lines += md_notes()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    write_json(sets)
    print(f"wrote {OUT}/: SUMMARY.md, summary.json, bias_comparison.json, runs/*.json")


if __name__ == "__main__":
    main()
