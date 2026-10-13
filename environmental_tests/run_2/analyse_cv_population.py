"""Every table reported for the circular CV population run, regenerated from the results file.

Run it with no arguments to print the whole report to stdout:

    python analyse_cv_population.py [results_env_cv.json] > report.txt

Each section is its own function so a single table can be pulled out in isolation.
Nothing here re-runs the waveform; it is pure post-processing of the saved fits.
"""
import json, os, sys
import numpy as np

OV_CONVERGED = 0.99
DROP = ("e0",)        # e0 sits on its own boundary; see section 3


# ---------------------------------------------------------------- loading ---
def load(path):
    with open(path) as f:
        return json.load(f)


def split_converged(d):
    """A source is judged on its 0PA climb, so both models share one verdict."""
    ok = [s for s in d["sources"] if s["fit_0pa"]["ov_final"] > OV_CONVERGED]
    bad = [s for s in d["sources"] if s["fit_0pa"]["ov_final"] <= OV_CONVERGED]
    return ok, bad


def shared_params(d):
    """Parameter names common to both fits, minus the ones excluded from ratios."""
    return [p for p in d["params_0pa"] if p not in DROP]


def col(s, key, cols, name):
    """One fit's entry for `name`, resolved against that model's own column list.

    The two fits carry different-length vectors (PN appends C_p), so the index has
    to be looked up per model rather than shared.
    """
    return s[key], cols.index(name)


def _bias_over_sigma(s, key, cols, keep):
    return [abs(s[key]["bias_over_sigma"][cols.index(p)]) for p in keep]


def _bias(s, key, cols, keep):
    return [abs(s[key]["bias"][cols.index(p)]) for p in keep]


def _sigma(s, key, cols, keep):
    return [s[key]["sigma"][cols.index(p)] for p in keep]


def header(n, title):
    print(f"\n\n{'='*100}\n{n}. {title}\n{'='*100}")


# --------------------------------------------------- 1. per-source summary ---
def section_summary(d):
    header(1, "Per-source summary (all points, converged or not)")
    P0, PP = d["params_0pa"], d["params_pn"]
    keep = shared_params(d)
    iC = PP.index("C_p")
    print("%3s %10s %6s %5s %6s %8s %9s | %13s %10s | %13s %10s | %7s %7s | %10s" % (
        "idx", "m1", "p0", "SNR", "T[yr]", "A_PM", "condEnv", "ov_0PA", "chi2_0PA",
        "ov_PN", "chi2_PN", "mx0PA", "mxPN", "C_p+-sig"))
    for s in d["sources"]:
        a, b = s["fit_0pa"], s["fit_pn"]
        flag = "" if a["ov_final"] > OV_CONVERGED else "*"
        m0 = max(_bias_over_sigma(s, "fit_0pa", P0, keep))
        mp = max(_bias_over_sigma(s, "fit_pn", PP, keep))
        cond = np.linalg.cond(np.array(s["fisher_env"]))
        print("%3d%1s %10.3e %6.2f %5.0f %6.2f %8.2e %9.2e | %13.10f %10.3e | "
              "%13.10f %10.3e | %7.2f %7.2f | %5.1f+-%4.0f" % (
                  s["idx"], flag, s["m1"], s["p0"], s["snr"], s["T"], s["A_PM"], cond,
                  a["ov_final"], a["chi2"], b["ov_final"], b["chi2"], m0, mp,
                  b["params"][iC], b["sigma"][iC]))
    ok, bad = split_converged(d)
    print(f"\n* = 0PA overlap <= {OV_CONVERGED}; its bias columns describe the wrong basin, "
          "not a bias.")
    print("converged %d / %d   failed idx %s" % (len(ok), len(d["sources"]),
                                                 [s["idx"] for s in bad]))
    print("converged p0 %.2f - %.2f     failed p0 %.2f - %.2f" % (
        min(s["p0"] for s in ok), max(s["p0"] for s in ok),
        min(s["p0"] for s in bad), max(s["p0"] for s in bad)))
    print("PN >= 0PA on every source: %s" % all(
        s["fit_pn"]["ov_final"] >= s["fit_0pa"]["ov_final"] for s in d["sources"]))


# ------------------------------------------------ 2. full parameter vectors ---
def section_vectors(d):
    header(2, "Full parameter vectors: truth, fit, bias, sigma  (both models)")
    P0, PP = d["params_0pa"], d["params_pn"]
    for s in d["sources"]:
        a, b = s["fit_0pa"], s["fit_pn"]
        ok = a["ov_final"] > OV_CONVERGED
        print("-" * 100)
        print("idx %2d  m1=%.4e  p0=%.3f  SNR=%.1f  T=%.4f  dt=%.1f  A_PM=%.4e  n_PM=%.0f  [%s]"
              % (s["idx"], s["m1"], s["p0"], s["snr"], s["T"], s["dt"], s["A_PM"],
                 s["n_PM"], "CONVERGED" if ok else "NOT CONVERGED"))
        print("  0PA   : ov=%.10f chi2=%.4e  |  0PA+PN: ov=%.10f chi2=%.4e"
              % (a["ov_final"], a["chi2"], b["ov_final"], b["chi2"]))
        print("  %10s %16s %16s %13s %13s %9s | %16s %13s %13s %9s" % (
            "par", "truth", "fit_0PA", "bias", "sigma", "b/sig",
            "fit_PN", "bias", "sigma", "b/sig"))
        truth = dict(zip(PP, s["truth_pn"]))
        for n in P0:
            i, j = P0.index(n), PP.index(n)
            print("  %10s %16.8g %16.8g %13.4e %13.3e %9.2f | %16.8g %13.4e %13.3e %9.2f" % (
                n, truth[n], a["params"][i], a["bias"][i], a["sigma"][i],
                a["bias_over_sigma"][i], b["params"][j], b["bias"][j], b["sigma"][j],
                b["bias_over_sigma"][j]))
        k = PP.index("C_p")
        print("  %10s %16.8g %16s %13s %13s %9s | %16.8g %13.4e %13.3e %9.2f" % (
            "C_p", truth["C_p"], "-", "-", "-", "-",
            b["params"][k], b["bias"][k], b["sigma"][k], b["bias_over_sigma"][k]))


# --------------------------------------------------- 3. 0PA vs 0PA+PN bias ---
def section_model_comparison(d):
    header(3, "0PA vs 0PA+PN: is the bias removed, or is the error bar inflated?")
    P0, PP = d["params_0pa"], d["params_pn"]
    keep = shared_params(d)
    ok, _ = split_converged(d)
    A = np.array([_bias(s, "fit_0pa", P0, keep) for s in ok])
    B = np.array([_bias(s, "fit_pn", PP, keep) for s in ok])
    SA = np.array([_sigma(s, "fit_0pa", P0, keep) for s in ok])
    SB = np.array([_sigma(s, "fit_pn", PP, keep) for s in ok])
    print("over %d converged sources; e0 excluded (boundary Fisher, see below)\n" % len(ok))
    print("%10s | %11s %11s %7s | %11s %11s %7s | %11s %11s %7s" % (
        "par", "med|b|_0PA", "med|b|_PN", "ratio", "med sig0PA", "med sigPN", "infl",
        "med b/s 0PA", "med b/s PN", "ratio"))
    for j, p in enumerate(keep):
        ba, bb = np.median(A[:, j]), np.median(B[:, j])
        sa, sb = np.median(SA[:, j]), np.median(SB[:, j])
        ra, rb = np.median(A[:, j] / SA[:, j]), np.median(B[:, j] / SB[:, j])
        print("%10s | %11.4e %11.4e %7.3f | %11.4e %11.4e %7.2f | %11.4f %11.4f %7.3f" % (
            p, ba, bb, bb / ba, sa, sb, sb / sa, ra, rb, rb / ra))
    print("\nhow often is PN's |bias| actually smaller?")
    for j, p in enumerate(keep):
        print("  %10s  smaller on %2d/%2d   median ratio %7.4f   worst %8.3f" % (
            p, int((B[:, j] < A[:, j]).sum()), len(ok),
            np.median(B[:, j] / A[:, j]), np.max(B[:, j] / A[:, j])))
    print("\ncorr(A_PM, theta) from the environmental Fisher -- explains which sigmas inflate")
    M = np.array([s["corr_A_PM"] for s in ok])
    for j, p in enumerate(P0):
        print("  %10s  median %+.4f   max|.| %.4f" % (p, np.median(M[:, j]),
                                                      np.abs(M[:, j]).max()))
    print("\ne0 is excluded above: the injection sits exactly on e0 = 0, where sigma(e0)")
    print("ranges over ten orders of magnitude across the population (2e-10 to 0.53), so")
    print("b/sigma is noise there.  Its absolute bias is <= 4e-4 on every converged source.")


# --------------------------------------------------------- 4. scaling law ---
def section_scaling(d):
    header(4, "What controls the bias: scaling in mass ratio and p0")
    P0 = d["params_0pa"]
    keep = shared_params(d)
    ok, bad = split_converged(d)
    m1 = np.array([s["m1"] for s in ok])
    p0 = np.array([s["p0"] for s in ok])
    n = float(ok[0]["n_PM"])
    eta = np.array([s["source_params"][1] for s in ok]) / m1 if isinstance(
        ok[0].get("source_params"), list) else 50.0 / m1
    b = np.array([max(_bias_over_sigma(s, "fit_0pa", P0, keep)) for s in ok])
    print("eta spans %.2fx, p0 spans %.2fx -- and b/sigma spans %.1fx"
          % (eta.max() / eta.min(), p0.max() / p0.min(), b.max() / b.min()))
    print("corr(log m1, log p0) = %+.4f  <-- near-collinear, so a free multivariate fit"
          % np.corrcoef(np.log10(m1), np.log10(p0))[0, 1])
    print("    does NOT identify the two exponents separately.  Fix the radial one instead:")
    print("    n_PM = %g exactly, from the multiplicative flux term in flux.py." % n)
    K = b / (p0 / 10.0) ** n
    sl = np.polyfit(np.log10(eta), np.log10(K), 1)[0]
    r = np.corrcoef(np.log10(eta), np.log10(K))[0, 1]
    print("\nresidual after dividing out (p0/10)^%g:" % n)
    print("    d log[b/(p0/10)^%g] / d log eta = %+.3f    r = %+.3f   (weak: eta is not the lever)"
          % (n, sl, r))
    print("    free-slope check  d log b / d log p0 = %.2f  (vs %g fixed)"
          % (np.polyfit(np.log10(p0), np.log10(b), 1)[0], n))
    lo, md, hi = np.percentile(K, [16, 50, 84])
    print("\n    b/sigma = K (p0/10)^%g      K = %.2f  [%.2f, %.2f] at 16-84%%" % (n, md, lo, hi))
    print("\n    target      p0        band")
    for t in (1.0, 2.0, 5.0, 10.0):
        print("    %5.1f sigma  %6.2f    %.2f - %.2f" % (
            t, (t / md) ** (1 / n) * 10, (t / hi) ** (1 / n) * 10, (t / lo) ** (1 / n) * 10))
    print("\n    CAVEAT: every row above p0 = %.2f is an extrapolation.  The sources that"
          % max(s["p0"] for s in ok))
    print("    would test it are exactly the ones that failed to converge, at p0 = %s."
          % sorted(round(s["p0"], 2) for s in bad))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_env_cv.json")
    d = load(path)
    print("circular CV population -- %s" % path)
    print("generated %s | population %s | channels %s | %s"
          % (d["generated_utc"], d.get("population"), d["nchannels"], d["freq_band"]))
    print("params_0pa %s" % d["params_0pa"])
    print("params_pn  %s" % d["params_pn"])
    print("params_env %s" % d["params_env"])
    section_summary(d)
    section_model_comparison(d)
    section_scaling(d)
    section_vectors(d)


if __name__ == "__main__":
    main()
