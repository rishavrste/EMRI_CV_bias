"""Fits, conditioning and intrinsic critical SNR of the 0PA + REL5_1 template -> results/summary.md

    D^2 = b^T Sigma_S^-1 b, Sigma_S the intrinsic (m1, m2, a, p0, e0) block of Gamma^-1, every other fitted
    parameter (qS, phiS, Phi_phi0, Phi_r0 and the five REL5_1 coefficients) marginalised;
    rho_crit = 20 D_CL / D,  D_CL^2 = chi^2_5 quantile at CL.
    1D: max over the intrinsic parameters of |b_i| / sigma_i (marginalised sigma); the coefficients are excluded.
0PA and 0PA + PN from EMRI_1PA_PN_DEV, each at its own Fisher; this template from fisher_at_best.py here.
Run:  python make_summary.py
"""
import json

import numpy as np
from scipy import stats

import config as C

INTRINSIC = ["m1", "m2", "a", "p0", "e0"]
SNR0 = 20.0
CLS = (0.90, 0.99)
OLD_FISHERS = C.OLD_BEST_FITS.parent / "fisher_at_best"
NEW_FISHERS = C.OUT_ROOT / "fisher_at_best"
PAIRS = (("c_p", "m1"), ("c_p", "m2"), ("c_p", "p0"), ("K", "m1"), ("K", "c_p"), ("c_e", "e0"),
         ("c_Omphi", "c_Omr"))


# --- loading --------------------------------------------------------------------
def load(path):
    return json.loads(path.read_text())


def comparison_fits(point):
    """(label, Fisher JSON) of the earlier templates at a point."""
    return [("0PA", load(OLD_FISHERS / "0pa" / f"fisher_idx{point}.json")),
            ("0PA+PN", load(OLD_FISHERS / "pn" / f"fisher_idx{point}.json"))]


def new_fit(point):
    return load(NEW_FISHERS / f"fisher_idx{point}.json")


# --- Fisher algebra ---------------------------------------------------------------
def jacobi(G):
    s = np.sqrt(np.abs(np.diag(G)))
    return np.asarray(G) / np.outer(s, s), s


def covariance(G):
    """Gamma^-1, inverted in Jacobi-rescaled coordinates."""
    Gs, s = jacobi(np.asarray(G))
    return np.linalg.inv(Gs) / np.outer(s, s)


def intrinsic_block(G, names, bias):
    k = [names.index(n) for n in INTRINSIC]
    return covariance(G)[np.ix_(k, k)], np.array([bias[i] for i in k])


def nd_distance(G, names, bias):
    Sigma, b = intrinsic_block(G, names, bias)
    return float(np.sqrt(b @ np.linalg.solve(Sigma, b)))


def max_1d(G, names, bias):
    """(max |b_i|/sigma_i over the intrinsic parameters, that parameter)."""
    Sigma, b = intrinsic_block(G, names, bias)
    r = np.abs(b) / np.sqrt(np.abs(np.diag(Sigma)))
    return float(r.max()), INTRINSIC[int(r.argmax())]


def rho_crit(D, cl):
    return SNR0 * float(np.sqrt(stats.chi2.ppf(cl, len(INTRINSIC)))) / D


def conditioning(d):
    """Condition number of the Jacobi-rescaled Fisher and the correlations in PAIRS."""
    G, names = np.asarray(d["fisher"]), d["names"]
    cond = float(np.linalg.cond(jacobi(G)[0]))
    Sig = covariance(G)
    sd = np.sqrt(np.abs(np.diag(Sig)))
    corr = {f"{a}-{b}": float(Sig[names.index(a), names.index(b)] / (sd[names.index(a)] * sd[names.index(b)]))
            for a, b in PAIRS}
    return cond, corr


# --- tables -----------------------------------------------------------------------
def fit_rows(point, d):
    th = dict(zip(d["names"], d["theta"]))
    mis = 1 - d["final_eval"]["overlap"]
    gains = [f"{(1 - ref['final_eval']['overlap']) / mis:.3g}x" for _, ref in comparison_fits(point)]
    dev = " | ".join(f"{th[n]:+.6g}" for n in C.DEV_SLOTS)
    tag = " (running)" if d["running"] else ""
    return (f"| idx{point}{tag} | {d['arm']} | {d['final_eval']['overlap']:.10f} | {mis:.3e} | "
            + " | ".join(gains) + f" | {dev} |")


def snr_row(label, d, doubled=False):
    D = nd_distance(d["fisher"], d["names"], d["bias"])
    m1d, which = max_1d(d["fisher"], d["names"], d["bias"])
    cells = " | ".join(f"{rho_crit(D, cl):.3g}" for cl in CLS)
    check = (f"{rho_crit(nd_distance(d['fisher_doubled_steps'], d['names'], d['bias']), CLS[0]):.3g}"
             if doubled else "")
    return (f"| {label} | {1 - d['final_eval']['overlap']:.3e} | {m1d:.3g} ({which}) | {D:.3g} | {cells} "
            f"| {check} |")


def cond_rows(point, d):
    cond, corr = conditioning(d)
    return f"| idx{point} | {cond:.2e} | " + " | ".join(f"{v:+.4f}" for v in corr.values()) + " |"


def full_vectors(point, d):
    th = dict(zip(d["names"], d["theta"]))
    return f"idx{point}: " + ", ".join(f"{n}={v!r}" for n, v in th.items())


def main():
    points = [p for p in C.POINTS if (NEW_FISHERS / f"fisher_idx{p}.json").exists()]
    out = ["# 0PA + REL5_1 deviation, LM from the injection with phase-fitted coefficients", "",
           "X += eta X_0PA (c_X v + K v^4),  v = p^-1/2,  X = pdot, edot, Omega_phi, Omega_r", "",
           "## Fits", "",
           "| point | arm | overlap | 1 - O | gain vs 0PA | gain vs PN | " + " | ".join(C.DEV_SLOTS) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(C.DEV_SLOTS)]
    out += [fit_rows(p, new_fit(p)) for p in points]
    out += ["", "## Conditioning of the 14-parameter Fisher (Jacobi-rescaled)", "",
            "| point | cond | " + " | ".join(f"corr {a}-{b}" for a, b in PAIRS) + " |",
            "|---|---|" + "---|" * len(PAIRS)]
    out += [cond_rows(p, new_fit(p)) for p in points]
    out += ["", "## Critical SNR, intrinsic parameters (m1, m2, a, p0, e0), each at its own Fisher", "",
            "| fit | 1 - O | max 1D bias/sigma at SNR 20 | nD D at SNR 20 | rho_crit 90% | rho_crit 99% "
            "| rho_crit 90%, steps doubled |",
            "|---|---|---|---|---|---|---|"]
    for p in points:
        out += [snr_row(f"idx{p} {lab}", d) for lab, d in comparison_fits(p)]
        out.append(snr_row(f"idx{p} 0PA+REL5_1", new_fit(p), doubled=True))
    out += ["", "## Best-fit vectors", "", "```"] + [full_vectors(p, new_fit(p)) for p in points] + ["```"]
    text = "\n".join(out) + "\n"
    (C.OUT_ROOT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
