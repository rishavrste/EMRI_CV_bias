"""Fits, conditioning and intrinsic nD critical SNR of the PN + frequency template -> results/summary.md

    D^2 = b^T Sigma_S^-1 b, Sigma_S the intrinsic (m1, m2, a, p0, e0) block of Gamma^-1, every other fitted
    parameter marginalised;  rho_crit = 20 D_CL / D,  D_CL^2 = chi^2_5 quantile at CL.
0PA and 0PA + PN from EMRI_1PA_PN_DEV, power-law PN (best_ladder) from EMRI_1PA_PN_POWER_DEV, PN + environmental
(pn_ladder) from EMRI_1PA_PN_ADD_DEV, each at its own Fisher; this template from fisher_at_best.py here.
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
POWER_FISHERS = C.HERE.parent / "EMRI_1PA_PN_POWER_DEV" / "results" / "fisher_at_best"
ENV_FISHERS = C.HERE.parent / "EMRI_1PA_PN_ADD_DEV" / "results" / "fisher_at_best"
NEW_FISHERS = C.OUT_ROOT / "fisher_at_best"
PAIRS = (("C_p", "C_e"), ("C_p", "m1"), ("C_p", "m2"), ("C_p", "a"), ("C_p", "p0"), ("C_e", "e0"))


# --- loading --------------------------------------------------------------------
def load(path):
    return json.loads(path.read_text())


def comparison_fits(point):
    """(label, Fisher JSON) of the earlier templates at a point."""
    return [("0PA", load(OLD_FISHERS / "0pa" / f"fisher_idx{point}.json")),
            ("0PA+PN", load(OLD_FISHERS / "pn" / f"fisher_idx{point}.json")),
            ("0PA+PN_power", load(POWER_FISHERS / f"fisher_idx{point}_best_ladder.json")),
            ("0PA+PN+env", load(ENV_FISHERS / f"fisher_idx{point}_pn_ladder.json"))]


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


def nd_distance(G, names, bias):
    k = [names.index(n) for n in INTRINSIC]
    Sigma = covariance(G)[np.ix_(k, k)]
    b = np.array([bias[i] for i in k])
    return float(np.sqrt(b @ np.linalg.solve(Sigma, b)))


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
    gains = [f"{(1 - ref['final_eval']['overlap']) / mis:.3g}x" for _, ref in comparison_fits(point)[1:]]
    dev = " | ".join(f"{th[n]:+.6g}" for n in C.DEV_SLOTS)
    tag = " (running)" if d["running"] else ""
    return (f"| idx{point}{tag} | {d['arm']} | {d['final_eval']['overlap']:.10f} | {mis:.3e} | "
            + " | ".join(gains) + f" | {dev} |")


def snr_row(label, d, doubled=False):
    D = nd_distance(d["fisher"], d["names"], d["bias"])
    cells = " | ".join(f"{rho_crit(D, cl):.3g}" for cl in CLS)
    check = (f"{rho_crit(nd_distance(d['fisher_doubled_steps'], d['names'], d['bias']), CLS[0]):.3g}"
             if doubled else "")
    return f"| {label} | {1 - d['final_eval']['overlap']:.3e} | {D:.3g} | {cells} | {check} |"


def cond_rows(point, d):
    cond, corr = conditioning(d)
    return f"| idx{point} | {cond:.2e} | " + " | ".join(f"{v:+.4f}" for v in corr.values()) + " |"


def full_vectors(point, d):
    th = dict(zip(d["names"], d["theta"]))
    return f"idx{point}: " + ", ".join(f"{n}={v!r}" for n, v in th.items())


def main():
    points = [p for p in C.POINTS if (NEW_FISHERS / f"fisher_idx{p}.json").exists()]
    out = ["# 0PA + PN + C_p-driven frequency term, LM from the physical 1PA seed", "",
           "pdot += q C_p PN_p,  edot += q e C_e PN_e,  Omega_phi, Omega_r += -(15/56) q C_p (1-e^2)^1.5 / p^2", "",
           "## Fits", "",
           "| point | arm | overlap | 1 - O | gain vs PN | gain vs PN_power | gain vs PN+env | "
           + " | ".join(C.DEV_SLOTS) + " |",
           "|---|---|---|---|---|---|---|" + "---|" * len(C.DEV_SLOTS)]
    out += [fit_rows(p, new_fit(p)) for p in points]
    out += ["", "## Conditioning of the 11-parameter Fisher (Jacobi-rescaled)", "",
            "| point | cond | " + " | ".join(f"corr {a}-{b}" for a, b in PAIRS) + " |",
            "|---|---|" + "---|" * len(PAIRS)]
    out += [cond_rows(p, new_fit(p)) for p in points]
    out += ["", "## nD critical SNR, intrinsic parameters (m1, m2, a, p0, e0), each at its own Fisher", "",
            "| fit | 1 - O | D at SNR 20 | rho_crit 90% | rho_crit 99% | rho_crit 90%, steps doubled |",
            "|---|---|---|---|---|---|"]
    for p in points:
        out += [snr_row(f"idx{p} {lab}", d) for lab, d in comparison_fits(p)]
        out.append(snr_row(f"idx{p} 0PA+PN+freq", new_fit(p), doubled=True))
    out += ["", "## Best-fit vectors", "", "```"] + [full_vectors(p, new_fit(p)) for p in points] + ["```"]
    text = "\n".join(out) + "\n"
    (C.OUT_ROOT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
