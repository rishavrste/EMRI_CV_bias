"""nD critical SNR over the intrinsic parameters m1, m2, a, p0, e0 -> results/critical_snr_intrinsic.md

    D^2 = b^T Sigma_S^-1 b, Sigma_S the intrinsic block of Gamma^-1, every other fitted parameter
    marginalised;  rho_crit = 20 D_CL / D,  D_CL^2 = chi^2_5 quantile at CL.
0PA and 0PA + PN from EMRI_1PA_PN_DEV/results/fisher_at_best, power-law PN from fisher_at_best.py here.
Run:  python critical_snr_intrinsic.py
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
SHOWN_POINTS = ("0", "4")
SHOWN_ARMS = ("best_ladder",)


def covariance(G):
    """Gamma^-1, inverted in Jacobi-rescaled coordinates."""
    s = np.sqrt(np.abs(np.diag(G)))
    return np.linalg.inv(G / np.outer(s, s)) / np.outer(s, s)


def nd_distance(G, names, bias):
    k = [names.index(n) for n in INTRINSIC]
    Sigma = covariance(np.array(G))[np.ix_(k, k)]
    b = np.array([bias[i] for i in k])
    return float(np.sqrt(b @ np.linalg.solve(Sigma, b)))


def rho_crit(D, cl):
    return SNR0 * float(np.sqrt(stats.chi2.ppf(cl, len(INTRINSIC)))) / D


def row(label, overlap, D, D_doubled=None):
    cells = [f"{rho_crit(D, cl):.3g}" for cl in CLS]
    check = f"{rho_crit(D_doubled, CLS[0]):.3g}" if D_doubled is not None else ""
    return f"| {label} | {1 - overlap:.3e} | {D:.3g} | {cells[0]} | {cells[1]} | {check} |"


def old_row(template, point):
    d = json.loads((OLD_FISHERS / template / f"fisher_idx{point}.json").read_text())
    label = {"0pa": "0PA", "pn": "0PA + PN"}[template]
    return row(f"idx{point} {label}", d["final_eval"]["overlap"], nd_distance(d["fisher"], d["names"], d["bias"]))


def new_row(point, arm):
    d = json.loads((NEW_FISHERS / f"fisher_idx{point}_{arm}.json").read_text())
    D = nd_distance(d["fisher"], d["names"], d["bias"])
    D2 = nd_distance(d["fisher_doubled_steps"], d["names"], d["bias"])
    tag = " (running)" if d["running"] else ""
    return row(f"idx{point} power, {arm}{tag}", d["final_eval"]["overlap"], D, D2)


def main():
    out = ["# nD critical SNR, intrinsic parameters (m1, m2, a, p0, e0)", "",
           "| fit | 1 - O | D at SNR 20 | rho_crit 90% | rho_crit 99% | rho_crit 90%, steps doubled |",
           "|---|---|---|---|---|---|"]
    for p in SHOWN_POINTS:
        out += [old_row("0pa", p), old_row("pn", p)] + [new_row(p, a) for a in SHOWN_ARMS]
    text = "\n".join(out) + "\n"
    (C.OUT_ROOT / "critical_snr_intrinsic.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
