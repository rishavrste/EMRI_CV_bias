"""All optimised EMRI grid points in one file: full source parameters and the full best fit of each template.

Reads results/best_fits.json (fisher/best_fits.py) for the fits and the SK_files signal array for the
injection, because theta_inj holds only the free parameters.  Parameters held in the fit (xI0, dist, qK,
phiK, Phi_theta0, chi2_spin) keep their injected value in each best fit, so a best-fit block is a
complete waveform parameter set.

Run outside the container, without importing config, which needs the waveform stack:
    python make_optimized_points.py     -> results/optimized_points.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BEST_FITS = HERE / "results" / "best_fits.json"
OUT = HERE / "results" / "optimized_points.json"
SIGNAL_ARRAY = "/scratch/e1583490/SK_files/data/signal/signal_parameter_array_EMRI.npy"

ROW_NAMES = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS", "qK", "phiK",
             "Phi_phi0", "Phi_theta0", "Phi_r0", "dt", "T", "chi2_spin"]
OBS_NAMES = ("dt", "T")
TEMPLATE_KEYS = {"0pa": "0PA", "pn": "0PA+PN"}

SETUP = dict(
    grid="SK_files EMRI grid", signal="1PA, (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False)",
    templates={"0PA": "(False, False, False)", "0PA+PN": "0PA plus deviation_included, C_p and C_e free"},
    tdi="2nd generation, A and E", setup="rishav, DC bin dropped", f_min_hz=1e-5,
    superkludge_branch="hybrid", signal_snr=20.0,
    held_in_fit=["xI0", "dist", "qK", "phiK", "Phi_theta0", "chi2_spin"],
    units=dict(m1="Msun", m2="Msun", dist="Gpc", dt="s", T="yr", angles="rad"),
    selection="0PA: seed-old ladder fit. 0PA+PN: highest overlap over seeds old, row, ramp, rampfine, best.",
    fit_chi2="<r|r> at the best fit, not the secondary spin chi2_spin")


def source_of(row):
    named = dict(zip(ROW_NAMES, (float(v) for v in row)))
    return {k: v for k, v in named.items() if k not in OBS_NAMES}, {k: named[k] for k in OBS_NAMES}


def best_fit_block(fit, source):
    params = dict(source, **fit["theta_final"])
    ev = fit["final_eval"]
    return dict(params=params, free_params=fit["params"], overlap=ev["overlap"], mismatch=1.0 - ev["overlap"],
                chi2=ev["chi2"], lnL=ev["lnL"], rho_s=ev["rho_s"], rho_h=ev["rho_h"], case=fit["case"],
                seed=fit["case"].split("_")[3], stop_reason=fit["info"]["stop_reason"],
                n_iter=fit["info"]["n_iter"])


def check_source(fit, source):
    """The fits must have been run against this injection."""
    for name, value in fit["theta_inj"].items():
        if not np.isclose(value, source.get(name, 0.0), rtol=1e-12, atol=0.0):   # C_p, C_e injected at 0
            raise ValueError(f"idx{fit['point']} {name}: {value} in the fit, {source[name]} in the grid")


def point_record(idx, row, fits):
    source, obs = source_of(row)
    record = dict(idx=idx, label=f"idx{idx}", **obs, source=source, best_fit={})
    for template, key in TEMPLATE_KEYS.items():
        fit = fits[template][str(idx)]
        check_source(fit, source)
        record["best_fit"][key] = best_fit_block(fit, source)
    return record


def main():
    fits = json.loads(BEST_FITS.read_text())
    rows = np.load(SIGNAL_ARRAY)
    points = [point_record(idx, row, fits) for idx, row in enumerate(rows)]
    OUT.write_text(json.dumps(dict(setup=SETUP, points=points), indent=1))
    print(f"{len(points)} points -> {OUT}")


if __name__ == "__main__":
    main()
