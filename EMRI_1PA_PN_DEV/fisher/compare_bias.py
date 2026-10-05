"""1D and nD bias of the 0PA and 0PA + PN best fits -> results/fisher_at_best/bias_comparison_snr{S}.txt
(as in IMRI_TAIL_dev/compare_bias.py).

Reads results/fisher_at_best/{0pa,pn}/fisher_idx{i}.json (fisher_at_best.py; Gamma at SNR 20).
    1D : bias_i / sigma_i, sigma_i = sqrt((Gamma^-1)_ii)
    nD : D^2 = b^T Sigma_S^-1 b over a parameter set S, Sigma_S the S block of Gamma^-1 (the other
         fitted parameters marginalised); D^2 ~ chi^2 with |S| dof, quoted also as the
         two-sided Gaussian-equivalent significance.
Both scale with SNR: 1D x SNR/20, D^2 x (SNR/20)^2 (--snr).

Run (from fisher/):  python compare_bias.py [--snr 20]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # EMRI_1PA_PN_DEV: config
import config as C                                  # noqa: E402

PHYS = C.PHYS
ALL = C.fit_params("pn")
SNR0 = 20.0                                         # SNR of the stored Fishers
POINTS = [str(i) for i in range(25)]
SHORT = {"m1": "m1", "m2": "m2", "a": "a", "p0": "p0", "e0": "e0", "qS": "qS", "phiS": "phiS",
         "Phi_phi0": "Phph", "Phi_r0": "Phr", "C_p": "C_p", "C_e": "C_e"}


def load(template, point, scale):
    path = C.FISHER_DIR / template / f"fisher_{C.point_label(point)}.json"
    d = json.loads(path.read_text())
    G = np.array(d["fisher"]) * scale ** 2
    return dict(d, G=G, Sigma=covariance(G), b=dict(zip(d["names"], d["bias"])),
                cond_hat=jacobi_cond(G))


def covariance(G):
    """Gamma^-1, inverted in Jacobi-rescaled coordinates."""
    s = np.sqrt(np.abs(np.diag(G)))
    return np.linalg.inv(G / np.outer(s, s)) / np.outer(s, s)


def jacobi_cond(G):
    """Condition number of Gamma after Jacobi rescaling (unit-free; the raw one is dominated by
    m1 ~ 1e6 vs e0 ~ 0.1)."""
    s = np.sqrt(np.abs(np.diag(G)))
    return float(np.linalg.cond(G / np.outer(s, s)))


def sigma(rec, name):
    i = rec["names"].index(name)
    var = rec["Sigma"][i, i]
    return float(np.sqrt(var)) if var > 0 else float("nan")     # var <= 0: singular Gamma


def nd_bias(rec, subset):
    """D^2 over subset, the remaining fitted parameters marginalised."""
    k = [rec["names"].index(n) for n in subset]
    S = rec["Sigma"][np.ix_(k, k)]
    b = np.array([rec["b"][n] for n in subset])
    return float(b @ np.linalg.solve(S, b))


def gauss_sigma(D2, dof):
    """Two-sided Gaussian-equivalent significance of D^2 for a chi^2 with dof."""
    if not np.isfinite(D2):
        return float("nan")
    logp = stats.chi2.logsf(D2, dof)
    if logp > -700:
        return float(stats.norm.isf(np.exp(logp) / 2))
    if np.isfinite(logp):
        return float(np.sqrt(-2 * logp))                   # tail asymptote
    return float(np.sqrt(D2))                              # logsf underflow: D itself


def fmt_nd(D2, dof):
    return f"{np.sqrt(D2):8.2f} ({gauss_sigma(D2, dof):5.1f}s)"


# --- tables ----------------------------------------------------------------
def oned_table(title, rows):
    """rows: (idx, a, e0, {name: value}) -> one line per point."""
    names = list(rows[0][3])
    head = f"{'idx':>3} {'a':>4} {'e0':>3} | " + " ".join(f"{SHORT[n]:>7}" for n in names)
    out = [title, head, "-" * len(head)]
    for idx, a, e0, v in rows:
        out.append(f"{idx:>3} {a:+.1f} {e0:.1f} | " + " ".join(f"{v[n]:+7.2f}" for n in names))
    return out + [""]


def build_tables(scale):
    zero, pn, pn_in_0pa, nd = [], [], [], []
    for point in POINTS:
        r1, r2 = load("0pa", point, scale), load("pn", point, scale)
        key = (point, r2["a_inj"], r2["e0_inj"])
        zero.append((*key, {n: r1["b"][n] / sigma(r1, n) for n in PHYS}))
        pn.append((*key, {n: r2["b"][n] / sigma(r2, n) for n in ALL}))
        pn_in_0pa.append((*key, {n: r2["b"][n] / sigma(r1, n) for n in PHYS}))
        nd.append((*key, r1, r2))
    return zero, pn, pn_in_0pa, nd


def nd_table(nd, snr):
    head = (f"{'idx':>3} {'a':>4} {'e0':>3} | {'1-O 0PA':>9} {'1-O PN':>9} | "
            f"{'D 0PA (9)':>17} | {'D PN, all (11)':>17} {'D PN, phys (9)':>17} "
            f"{'D PN, C only(2)':>17} | {'cond^ 0PA':>9} {'cond^ PN':>9}")
    out = [f"nD bias at SNR {snr:g}: D = sqrt(b^T Sigma_S^-1 b), (dof); in brackets the Gaussian "
           "equivalent. 'phys' = the 9 physical parameters with C_p, C_e marginalised;",
           "'C only' = how far the best-fit (C_p, C_e) sits from 0, i.e. the significance of the "
           "spurious deviation. cond^ = condition number of the Jacobi-rescaled Gamma.", head, "-" * len(head)]
    for idx, a, e0, r1, r2 in nd:
        o1, o2 = r1["stored_eval"]["overlap"], r2["stored_eval"]["overlap"]
        out.append(f"{idx:>3} {a:+.1f} {e0:.1f} | {1 - o1:9.2e} {1 - o2:9.2e} | "
                   f"{fmt_nd(nd_bias(r1, PHYS), len(PHYS))} | {fmt_nd(nd_bias(r2, ALL), len(ALL))} "
                   f"{fmt_nd(nd_bias(r2, PHYS), len(PHYS))} {fmt_nd(nd_bias(r2, C.DEV_PARAMS), 2)} | "
                   f"{r1['cond_hat']:9.1e} {r2['cond_hat']:9.1e}")
    return out + [""]


def eval_check(nd):
    """Re-evaluated overlap at each best fit vs the stored one (same setup, so they should agree)."""
    out = ["overlap re-evaluated in fisher_at_best.py minus stored:"]
    for idx, _, _, r1, r2 in nd:
        d1 = r1["final_eval"]["overlap"] - r1["stored_eval"]["overlap"]
        d2 = r2["final_eval"]["overlap"] - r2["stored_eval"]["overlap"]
        out.append(f"  idx{idx:<2s} 0PA {d1:+.2e}   PN {d2:+.2e}   rho_s {r1['final_eval']['rho_s']:.4f}")
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=SNR0)
    args = ap.parse_args()
    zero, pn, pn_in_0pa, nd = build_tables(args.snr / SNR0)
    lines = [f"EMRI grid, 1PA signal, 2nd-generation TDI AE. 1D bias = (best fit - injection) / sigma "
             f"at SNR {args.snr:g}.", ""]
    lines += oned_table(f"0PA template, own sigma ({len(PHYS)} parameters)", zero)
    lines += oned_table(f"0PA + PN template, own sigma ({len(ALL)} parameters)", pn)
    lines += oned_table("0PA + PN best fit, in units of the 0PA sigma", pn_in_0pa)
    lines += nd_table(nd, args.snr) + eval_check(nd)
    text = "\n".join(lines)
    (C.FISHER_DIR / f"bias_comparison_snr{args.snr:g}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
