"""Write RESULTS.md (set 5) from the two run JSONs.  Re-run it whenever a job finishes;
a run that has not finished yet is shown as pending.  Never hand-edit RESULTS.md."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("0PA (vacuum GR)", "results_de_0pa.json", "config_de_0pa.json"),
        ("0PA + 2.5PN deviation", "results_de_0pa_pn.json", "config_de_0pa_pn.json")]
PHASES = ("Phi_phi0", "Phi_r0")


def load(name):
    path = os.path.join(HERE, name)
    return json.load(open(path)) if os.path.exists(path) else None


def wrap(x):
    """A phase offset folded into (-pi, pi]."""
    return float(np.pi - (np.pi - x) % (2 * np.pi))


def setup_section(cfg):
    inj, d, w = cfg["injection"], cfg["disk"], cfg["waveform"]
    return [
        "## Setup (identical for both runs, = test.ipynb cells 2-4)", "",
        "| | |", "|---|---|",
        f"| injection | m1 = {inj['m1']:g}, m2 = {inj['m2']:g}, a = {inj['a']:g}, "
        f"p0 = {inj['p0']:g}, e0 = {inj['e0']:g}, xI0 = {inj['xI0']:g}, dist = {inj['dist']:g}, "
        f"Phi_phi0 = {inj['Phi_phi0']:g}, Phi_r0 = {inj['Phi_r0']:g} |",
        f"| disk (signal only) | Sigma0 = {d['Sigma0_cgs']:g} g/cm^2, h0 = {d['h0']:g}, "
        f"Sigma_p = {d['Sigma_p']:g} |",
        f"| waveform | T = {w['T_safety']:g} x plunge, dt = {w['dt']} s, {w['nchannels']} channels, "
        f"band [{w['fmin']:g}, {w['fmax']:g}] Hz |",
        f"| fitted | {', '.join(cfg['infer'])} (+ dev_1 = C_p, dev_2 = C_e for PN); "
        "everything else pinned at the injection |",
        f"| DE box | +-{cfg['bounds']['sigma_range']:g} Fisher sigma at the injection; "
        f"{', '.join(cfg['bounds']['full_period'])} span [inj - pi, inj + pi]; dev capped at +-100 |",
        f"| DE | maxiter {cfg['de']['maxiter']}, popsize {cfg['de']['popsize']}, "
        f"seed {cfg['de']['seed']}, x0 = injection; then LM (lam0 = {cfg['lm']['lambda0']:g}) |",
        ""]


def overlap_section(results):
    rows = ["## Overlap", "",
            "| model | stage | overlap | mismatch | chi2 |", "|---|---|---|---|---|"]
    for label, r in results:
        if r is None or "de" not in r:
            rows.append(f"| {label} | pending | | | |")
            continue
        stages = [("injection", r["overlap_seed"], r["chi2_seed"]),
                  ("DE", r["de"]["overlap"], r["de"]["chi2"])]
        if "lm" in r:
            stages.append(("DE+LM", r["lm"]["overlap"], r["lm"]["chi2"]))
        for name, ov, chi2 in stages:
            rows.append(f"| {label} | {name} | {ov:.12f} | {1 - ov:.6e} | {chi2:.6e} |")
    return rows + [""]


def bias_section(label, r):
    if r is None or "lm" not in r:
        return [f"### {label}: pending", ""]
    names, inj = r["names"], r["theta_inj"]
    bf, sig, de = r["lm"]["params"], r["lm"]["sigma"], r["de"]["params"]
    rows = [f"### {label}", "",
            f"SNR = {r['snr']:.6f}, T = {r['T']:.6f} yr, "
            f"mismatch reduction (inj -> fit) = "
            f"{(1 - r['overlap_seed']) / max(1 - r['lm']['overlap'], 1e-300):.2f}x", "",
            "| param | injected | DE | DE+LM | bias | sigma | bias/sigma |",
            "|---|---|---|---|---|---|---|"]
    for j, n in enumerate(names):
        b = bf[j] - inj[j]
        note = ""
        if n in PHASES and abs(wrap(b) - b) > 1e-12:
            b, note = wrap(b), " (wrapped)"
        rows.append(f"| {n} | {inj[j]:.10g} | {de[j]:.10g} | {bf[j]:.10g} | "
                    f"{b:+.6e}{note} | {sig[j]:.5e} | {b / sig[j]:+.3f} |")
    rows += ["", "best-fit vector `[" + ", ".join(f"{v:.10e}" for v in bf) + "]`", ""]
    return rows


def dephasing_section(results):
    rows = ["## Dephasing against the environmental signal", "",
            "Delta Phi = (template) - (signal), at the end of the common time span.",
            "*orbital* = phase the inspiral accumulates from zero; *total* = orbital + the",
            "initial-phase offset, i.e. what the waveform actually carries.", "",
            "| model | point | dPhi_phi orbital [rad] | dPhi_phi total [rad] | "
            "dPhi_r orbital [rad] | dPhi_r total [rad] |",
            "|---|---|---|---|---|---|"]
    for label, r in results:
        if r is None or "dephasing" not in r:
            rows.append(f"| {label} | pending | | | | |")
            continue
        for point, d in r["dephasing"].items():
            rows.append(f"| {label} | {point} | {d['Phi_phi']['orbital_final']:+.6e} | "
                        f"{d['Phi_phi']['total_final']:+.6e} | "
                        f"{d['Phi_r']['orbital_final']:+.6e} | {d['Phi_r']['total_final']:+.6e} |")
    return rows + ["", "At the injection the template is plain vacuum (dev = 0), so the "
                   "'injection' row is the disk's own dephasing, vacuum - disk.", ""]


def main():
    cfg = json.load(open(os.path.join(HERE, RUNS[0][2])))
    results = [(label, load(res)) for label, res, _ in RUNS]
    lines = ["# Environmental signal vs 0PA and 0PA + PN: DE (+-5 sigma) from the injection (set 5: e0 = 0.2, Sigma0 = 5.25e5, h0 = 0.025)", "",
             "Generated by `make_results_md.py` from the results JSONs -- do not hand-edit.", ""]
    lines += setup_section(cfg) + overlap_section(results)
    lines += ["## Best fit and bias (bias = DE+LM - injection; sigma = Fisher at the fit)", ""]
    for label, r in results:
        lines += bias_section(label, r)
    lines += dephasing_section(results)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
