"""The sources that go into run_1/results, gathered from the two runs that produced them.

Nineteen of the twenty-three population sources were fitted by the population run itself
(`results_env_cv.json`).  The four it failed on were followed up in `harder_environmental/`;
8, 13 and 14 were brought to a usable fit there and enter from their own results files, one
model arm at a time.  The parameter list is the same in both runs -- the eight vacuum
parameters, plus C_p for the PN model, e0 free -- so those sources drop straight into the
population's tables with no special casing downstream.

Nothing here touches a waveform.  Both the Fisher job and the plotting read the same list,
so a source cannot appear in one and not the other, and the fit vector the Fisher is taken
at is by construction the fit vector the bias is measured from.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HARDER = os.path.join(ROOT, "harder_environmental")

POP_JSON = os.path.join(HERE, "population.json")
CV_JSON = os.path.join(HERE, "results_env_cv.json")

MODELS = ("0PA", "0PA+PN")

# Bias on an angle is taken after unwrapping to (-pi, pi], so a fit on the far side of 2 pi
# is not counted as a large error.  Same convention as export_run1_harder.py.
ANGLES = {"qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"}

# The converged follow-up fits.  These are the `_ecc` runs: a seeded LM climb with e0 free,
# which is the population run's parameter list exactly.  The earlier harder runs held e0 at
# its injected value and so are not interchangeable with these.
# An entry may name one model or both: a source takes each arm from wherever that arm's best
# overlap was actually found, and the two need not be the same run.  Source 14 is the case --
# its 0PA follow-up reached 0.774 against the population run's 0.005, while its PN follow-up
# reached only 0.780 against the population run's 0.921, so its arms come from opposite runs.
HARDER_FITS = {
    8: {"0PA": "results_cv_de_cv_0pa_s8_ecc.json",
        "0PA+PN": "results_cv_de_cv_pn_s8_ecc.json"},
    13: {"0PA": "results_cv_de_cv_0pa_s13_ecc.json",
         "0PA+PN": "results_cv_de_cv_pn_s13_ecc.json"},
    14: {"0PA": "results_cv_de_cv_0pa_s14_ecc.json"},
}

# Left out, and why.  Carried here rather than in a comment so the markdown can state it.
DROPPED = {
    16: "no PN search converged: four attempts, three with DE collapsing to an overlap of "
        "~1e-3, so its numbers say where the optimiser stopped rather than what the "
        "template absorbs",
}

# Sources that are in the analysis but whose climb did not converge.  They are plotted and
# tabulated -- hollow, and named -- rather than dropped, because a failure at high p0 is
# itself the result; they are held out of the population medians, where a bias measured in
# the wrong basin would sit in the denominator of a ratio and mean nothing.
#
# The test is not a fixed overlap threshold.  Source 8's 0PA fit stops at 0.928 because a
# vacuum template genuinely cannot do better against that environmental term, which is a
# converged answer; source 14's stops at 0.005 because the optimiser never found the signal.
NOT_CONVERGED = {}


def unwrap(diff):
    return (diff + np.pi) % (2 * np.pi) - np.pi


def signed_bias_of(names, truth, fit):
    """Bias per parameter, angles unwrapped first, sign kept.

    The n-d bias is a quadratic form `b^T M b`, whose cross terms need the sign; the
    per-parameter panels want the magnitude.  Both read the unwrap convention from here so
    the two cannot drift apart.
    """
    return {n: float(unwrap(f - t) if n in ANGLES else f - t)
            for n, t, f in zip(names, truth, fit)}


def bias_of(names, truth, fit):
    """Absolute bias per parameter, angles unwrapped first."""
    return {n: abs(v) for n, v in signed_bias_of(names, truth, fit).items()}


def _fit(params, vec, ov, chi2, origin):
    """One model's answer for one source, in the single shape everything downstream reads."""
    return dict(params=list(params), theta=[float(v) for v in vec],
                ov_final=float(ov), chi2=float(chi2), origin=origin)


def _population_fits(rec, params_0pa, params_pn):
    out = {}
    for model, key, names in (("0PA", "fit_0pa", params_0pa),
                              ("0PA+PN", "fit_pn", params_pn)):
        f = rec[key]
        out[model] = _fit(names, f["params"], f["ov_final"], f["chi2"],
                          "results_env_cv.json")
    return out


def _harder_fits(idx):
    """The follow-up fits for one source, or {} if it has none.  One entry per model named."""
    out = {}
    for model, fname in HARDER_FITS.get(idx, {}).items():
        path = os.path.join(HARDER, fname)
        with open(path) as fh:
            d = json.load(fh)
        if d["point"]["idx"] != idx:
            raise ValueError(f"{fname} is source {d['point']['idx']}, expected {idx}")
        s3 = d["stage3_cv"]
        out[model] = _fit(d["params"], s3["params"], s3["ov_final"], s3["chi2"], fname)
    return out


def load_sources():
    """Every source that enters run_1/results, in index order.

    Each entry carries the population record (so a Fisher context can be rebuilt from it),
    the injected truth for both models, and one fit per model.  Sources in DROPPED are absent.
    """
    with open(CV_JSON) as fh:
        cv = json.load(fh)
    with open(POP_JSON) as fh:
        pop = json.load(fh)

    by_idx = {s["idx"]: s for s in pop["sources"]}
    params_0pa, params_pn = cv["params_0pa"], cv["params_pn"]

    out = []
    for rec in cv["sources"]:
        idx = rec["idx"]
        if idx in DROPPED:
            continue
        # Population first, then whichever arms the follow-ups improved on, so a source can
        # mix the two runs arm by arm.
        fits = _population_fits(rec, params_0pa, params_pn)
        fits.update(_harder_fits(idx))
        out.append(dict(idx=idx, source=by_idx[idx],
                        converged=idx not in NOT_CONVERGED,
                        not_converged_why=NOT_CONVERGED.get(idx),
                        truth={"0PA": rec["truth_0pa"], "0PA+PN": rec["truth_pn"]},
                        params={"0PA": params_0pa, "0PA+PN": params_pn},
                        fits=fits))
    return out


def main():
    """A listing, so the selection can be checked without running anything expensive."""
    src = load_sources()
    print(f"{len(src)} sources; dropped {sorted(DROPPED)}; "
          f"not converged {sorted(NOT_CONVERGED)}")
    print(f"{'idx':>4} {'p0':>7} {'origin':>34} "
          f"{'ov 0PA':>14} {'ov 0PA+PN':>14}")
    for s in src:
        f0, fp = s["fits"]["0PA"], s["fits"]["0PA+PN"]
        origin = "/".join(sorted({("pop" if f["origin"] == "results_env_cv.json"
                                   else "harder") for f in (f0, fp)}))
        mark = " " if s["converged"] else "*"
        print(f"{s['idx']:>3}{mark} {s['source']['p0']:>7.3f} {origin:>34} "
              f"{f0['ov_final']:>14.10f} {fp['ov_final']:>14.10f}")


if __name__ == "__main__":
    main()
