"""Collect the LM-from-injection results that ../test.ipynb computed, for the set
e0 = 0.0125, Sigma0 = 5.25e5 g/cm^2, h0 = 0.025 (vacuum GR in cell 17, 0PA + PN in cell 23).

Every number is parsed from the notebook's own printed output -- nothing is re-fitted here.
On top of it this script
  * wraps the phase biases (Phi_phi0, Phi_r0) into (-pi, pi]: the 0PA flux does not depend
    on the phases and every mode carries exp(i(m Phi_phi + n Phi_r)) with integer m, n, so
    Phi0 and Phi0 + 2 pi k give the identical waveform;
  * integrates the trajectories (CPU, seconds) for Delta Phi_phi, Delta Phi_r at the injection
    and at each best fit, template - signal, orbital and total.
Writes results_notebook.json and RESULTS.md.  Re-run it, never hand-edit either file.
"""

import json
import os
import re
import sys

import nbformat
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "test.ipynb")
sys.path.insert(0, HERE)
from dephasing import trajectory, dephasing                                  # noqa: E402
from few.trajectory.ode.flux import KerrEccEqAccFlux, SuperKludgeFlux        # noqa: E402
from few.utils.constants import MTSUN_SI, YRSID_SI                           # noqa: E402

# ---- the set, = test.ipynb cells 2 and 4 --------------------------------------------------
INJECTION = dict(m1=1e6, m2=2e3, a=0.0, p0=24.35, e0=0.0125, xI0=1.0, dist=16.344,
                 qS=np.pi / 4, phiS=np.pi / 3, qK=np.pi / 6, phiK=np.pi / 8,
                 Phi_phi0=1.0, Phi_theta0=0.0, Phi_r0=1.0)
SIGMA0_CGS, H0, SIGMA_P = 5.25e5, 0.025, -1.5
T_SAFETY, DT, NCHANNELS = 0.99, 5, 3
PHASES = ("Phi_phi0", "Phi_r0")

# cell index, label, template family
CELLS = [(17, "0PA (vacuum GR)", "vacuum"), (23, "0PA + 2.5PN deviation", "pn")]


def sigma0_geometric(sigma_cgs, m1):
    """g/cm^2 -> geometric units, exactly as test.ipynb cell 2 does it."""
    r_g_cm = m1 * MTSUN_SI * 2.99792458e10
    return sigma_cgs * r_g_cm ** 2 / (m1 * 1.98892e33)


DISK = [sigma0_geometric(SIGMA0_CGS, INJECTION["m1"]), H0, SIGMA_P]


def wrap(x):
    """A phase offset folded into (-pi, pi]."""
    return float(np.pi - (np.pi - x) % (2 * np.pi))


# ---- parsing the notebook output ----------------------------------------------------------
def cell_text(nb, i):
    return "".join(o.get("text", "") for o in nb.cells[i].outputs if o.output_type == "stream")


def parse_report(text):
    """The report_climb() block: overlaps, chi2, the per-parameter table, the best-fit vector."""
    num = r"([-+]?\d\.\d+e[-+]\d+|[-+]?\d+\.\d+)"
    grab = lambda pat: float(re.search(pat, text).group(1))
    rows = re.findall(r"^\s*(\w+)\s+" + r"\s+".join([num] * 5) + r"\s*$", text, re.M)
    rows = [r for r in rows if r[0] not in ("param",)]
    nm = re.search(r"\]\s+NM\s+->\s+" + num, text)
    return dict(
        names=[r[0] for r in rows],
        theta_inj=[float(r[1]) for r in rows],
        theta_raw=[float(v) for v in
                   re.search(r"best-fit vector \[(.*?)\]", text).group(1).split(",")],
        sigma=[float(r[4]) for r in rows],
        overlap_seed=grab(r"overlap\s+seed ->\s+" + num),
        overlap_lm1=grab(r"LM#1 ->\s+" + num),
        overlap_nm=(float(nm.group(1)) if nm else None),
        overlap_final=grab(("LM#2" if "LM#2 ->" in text else "LM#1") + r" ->\s+" + num),
        chi2_seed=grab(r"chi2\s+" + num + r" ->"),
        chi2_final=grab(r"chi2\s+[-+.\de]+ ->\s+" + num),
        lm1_iterations=len(re.findall(r"CV#1 it", text)),
        lm1_stalled="CV#1 STALLED" in text,
        nm_nfev=int(re.search(r"nfev=(\d+)", text).group(1)) if "nfev=" in text else None)


def wrapped_fit(r):
    """Best fit with the phase parameters moved by 2 pi k to lie within pi of the injection."""
    theta = list(r["theta_raw"])
    for j, n in enumerate(r["names"]):
        if n in PHASES:
            theta[j] = r["theta_inj"][j] + wrap(theta[j] - r["theta_inj"][j])
    return theta


# ---- dephasing ----------------------------------------------------------------------------
def extra_args(template, theta, names):
    if template == "vacuum":
        return []
    p = dict(zip(names, theta))
    # SuperKludgeFlux slots: chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included,
    # C_p (dev_1), C_e (dev_2), del_0_p, del_0_e  -- the notebook's pn_tail()
    return [0.0, False, False, False, True, p.get("dev_1", 0.0), p.get("dev_2", 0.0), 0.0, 0.0]


def dephasing_point(template, names, theta, T):
    params = dict(INJECTION)
    params.update({n: v for n, v in zip(names, theta) if n in params})
    flux = KerrEccEqAccFlux if template == "vacuum" else SuperKludgeFlux
    sig = trajectory(KerrEccEqAccFlux, INJECTION, DISK, T)
    tmpl = trajectory(flux, params, extra_args(template, theta, names), T)
    return dephasing(sig, tmpl, (INJECTION["Phi_phi0"], INJECTION["Phi_r0"]),
                     (params["Phi_phi0"], params["Phi_r0"]))


def observation_time():
    t = trajectory(KerrEccEqAccFlux, INJECTION, DISK, 10.0)[0]
    return T_SAFETY * float(t[-1]) / YRSID_SI, float(t[-1]) / YRSID_SI


# ---- markdown -----------------------------------------------------------------------------
def md_setup(T, t_plunge):
    inj = INJECTION
    return ["## Setup (= test.ipynb cells 2-4, 11, 20)", "", "| | |", "|---|---|",
            f"| injection | m1 = {inj['m1']:g}, m2 = {inj['m2']:g}, a = {inj['a']:g}, "
            f"p0 = {inj['p0']:g}, e0 = {inj['e0']:g}, xI0 = 1, dist = {inj['dist']:g}, "
            "Phi_phi0 = 1, Phi_theta0 = 0, Phi_r0 = 1 |",
            f"| disk (signal only) | Sigma0 = {SIGMA0_CGS:g} g/cm^2 (= {DISK[0]:.4e} geometric), "
            f"h0 = {H0:g}, Sigma_p = {SIGMA_P:g} |",
            f"| waveform | T = {T_SAFETY:g} x t_plunge = {T:.6f} yr (t_plunge = {t_plunge:.6f} yr), "
            f"dt = {DT} s, {NCHANNELS} channels, band [1e-05, 0.1] Hz |",
            "| fitted | m1, m2, p0, e0, Phi_phi0, Phi_r0 (+ dev_1 = C_p, dev_2 = C_e for PN); "
            "a, xI0, dist, sky and orientation angles pinned |",
            "| method | LM from the injection (lam0 = 0.1, Nielsen updates) -> NM escape "
            "(1000 steps, 2 sigma simplex) if stalled -> LM again. No DE. |",
            "| PN template | SuperKludge 0PA, chi2 = 0 (assumed), deviation_included = True |", ""]


def md_overlap(results):
    rows = ["## Overlap", "", "| model | stage | overlap | mismatch |", "|---|---|---|---|"]
    for label, r in results:
        stages = [("injection", r["overlap_seed"]), ("LM#1", r["overlap_lm1"])]
        if r["overlap_nm"] is not None:
            stages += [("NM", r["overlap_nm"]), ("LM#2 (final)", r["overlap_final"])]
        for s, ov in stages:
            rows.append(f"| {label} | {s} | {ov:.12f} | {1 - ov:.6e} |")
    rows += ["", "| model | chi2 injection | chi2 final |", "|---|---|---|"]
    rows += [f"| {label} | {r['chi2_seed']:.6e} | {r['chi2_final']:.6e} |" for label, r in results]
    v, p = results[0][1], results[1][1]
    rows += ["", f"PN mismatch reduction over vacuum: "
             f"{(1 - v['overlap_final']) / (1 - p['overlap_final']):.2f}x", ""]
    return rows


def md_bias(label, r):
    rows = [f"### {label}", "",
            "| param | injected | best fit (as climbed) | best fit (phases wrapped) | bias | sigma | bias/sigma |",
            "|---|---|---|---|---|---|---|"]
    for j, n in enumerate(r["names"]):
        b = r["theta"][j] - r["theta_inj"][j]
        rows.append(f"| {n} | {r['theta_inj'][j]:.10g} | {r['theta_raw'][j]:.10g} | "
                    f"{r['theta'][j]:.10g} | {b:+.6e} | {r['sigma'][j]:.5e} | "
                    f"{b / r['sigma'][j]:+.3f} |")
    rows += ["", "best-fit vector, wrapped `[" + ", ".join(f"{x:.10e}" for x in r["theta"]) + "]`",
             "", "best-fit vector, as climbed `[" + ", ".join(f"{x:.10e}" for x in r["theta_raw"])
             + "]`", ""]
    return rows


def md_dephasing(results):
    rows = ["## Dephasing against the environmental signal", "",
            "Delta Phi = (template) - (signal) at the end of the common time span. *orbital* = "
            "phase the inspiral accumulates from zero; *total* = orbital + the initial-phase "
            "offset (wrapped), i.e. what the waveform actually carries.", "",
            "| model | point | dPhi_phi orbital | dPhi_phi total | dPhi_r orbital | dPhi_r total |",
            "|---|---|---|---|---|---|"]
    for label, r in results:
        for point, d in r["dephasing"].items():
            rows.append(f"| {label} | {point} | {d['Phi_phi']['orbital_final']:+.6e} | "
                        f"{d['Phi_phi']['total_final']:+.6e} | {d['Phi_r']['orbital_final']:+.6e} | "
                        f"{d['Phi_r']['total_final']:+.6e} |")
    return rows + ["", "All in rad. The 'injection' rows are the disk's own dephasing "
                   "(vacuum - disk; dev = 0 for PN).", ""]


def main():
    nb = nbformat.read(NOTEBOOK, 4)
    T, t_plunge = observation_time()
    results = []
    for cell, label, template in CELLS:
        r = parse_report(cell_text(nb, cell))
        r["theta"] = wrapped_fit(r)
        r["template"] = template
        r["dephasing"] = {
            "injection": dephasing_point(template, r["names"], r["theta_inj"], T),
            "best fit": dephasing_point(template, r["names"], r["theta"], T)}
        results.append((label, r))
        print(f"[{label}] ov {r['overlap_final']:.12f}, wrapped theta {r['theta']}")

    out = dict(source="test.ipynb cells 17 (vacuum) and 23 (PN)", T=T, t_plunge=t_plunge,
               injection=INJECTION, disk=dict(Sigma0_cgs=SIGMA0_CGS, Sigma0=DISK[0], h0=H0,
                                              Sigma_p=SIGMA_P),
               results={label: r for label, r in results})
    json.dump(out, open(os.path.join(HERE, "results_notebook.json"), "w"), indent=2)

    lines = ["# Environmental signal vs 0PA and 0PA + PN: LM from the injection "
             "(notebook set: e0 = 0.0125, Sigma0 = 5.25e5, h0 = 0.025)", "",
             "Generated by `collect_notebook_results.py` from the test.ipynb outputs -- "
             "do not hand-edit.", ""]
    lines += md_setup(T, t_plunge) + md_overlap(results)
    lines += ["## Best fit and bias (bias = wrapped best fit - injection; sigma = Fisher at the "
              "final LM point)", ""]
    for label, r in results:
        lines += md_bias(label, r)
    lines += md_dephasing(results)
    open(os.path.join(HERE, "RESULTS.md"), "w").write("\n".join(lines))
    print("wrote results_notebook.json, RESULTS.md")


if __name__ == "__main__":
    main()
