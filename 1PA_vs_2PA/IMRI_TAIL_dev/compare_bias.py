"""1D and nD bias of the 1PA-only and 1PA + deviation best fits -> results/bias_comparison.txt.

Reads results/fisher_at_best/{1pa,1pa_dev}/fisher_idx{i}.json (fisher_at_best.py; Gamma at SNR 20).
    1D : bias_i / sigma_i, sigma_i = sqrt((Gamma^-1)_ii)
    nD : D^2 = b^T Sigma_S^-1 b over a parameter set S, Sigma_S the S block of Gamma^-1 (the other
         fitted parameters marginalised); D^2 ~ chi^2 with |S| dof, quoted also as the
         two-sided Gaussian-equivalent significance.
Both scale with SNR: 1D x SNR/20, D^2 x (SNR/20)^2 (--snr).

Run:  python compare_bias.py [--snr 20]
"""
import argparse
import json

import numpy as np
from scipy import stats

import config as C

PHYS = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0", "chi2"]
SHORT = {"m1": "m1", "m2": "m2", "a": "a", "p0": "p0", "e0": "e0", "Phi_phi0": "Phph",
         "Phi_r0": "Phr", "chi2": "chi2", "C_p": "C_p", "C_e": "C_e"}
FISHER_DIR = C.OUT_ROOT / "fisher_at_best"


def load(template, idx, scale):
    d = json.loads((FISHER_DIR / template / f"fisher_idx{idx}.json").read_text())
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
    one, dev, dev_in_1pa, nd = [], [], [], []
    for idx in range(25):
        r1, r2 = load("1pa", idx, scale), load("1pa_dev", idx, scale)
        key = (idx, r2["a_inj"], r2["e0_inj"])
        one.append((*key, {n: r1["b"][n] / sigma(r1, n) for n in PHYS}))
        dev.append((*key, {n: r2["b"][n] / sigma(r2, n) for n in C.PARAMS}))
        dev_in_1pa.append((*key, {n: r2["b"][n] / sigma(r1, n) for n in PHYS}))
        nd.append((*key, r1, r2))
    return one, dev, dev_in_1pa, nd


def nd_table(nd, snr):
    head = (f"{'idx':>3} {'a':>4} {'e0':>3} | {'1-O 1PA':>9} {'1-O dev':>9} | "
            f"{'D 1PA (8)':>17} | {'D dev, all (10)':>17} {'D dev, phys (8)':>17} "
            f"{'D dev, C only(2)':>17} | {'cond^ 1PA':>9} {'cond^ dev':>9}")
    out = [f"nD bias at SNR {snr:g}: D = sqrt(b^T Sigma_S^-1 b), (dof); in brackets the Gaussian "
           "equivalent. 'phys' = the 8 physical parameters with C_p, C_e marginalised;",
           "'C only' = how far the best-fit (C_p, C_e) sits from 0, i.e. the significance of the "
           "spurious deviation. cond^ = condition number of the Jacobi-rescaled Gamma.", head, "-" * len(head)]
    for idx, a, e0, r1, r2 in nd:
        o1, o2 = r1["stored_eval"]["overlap"], r2["stored_eval"]["overlap"]
        out.append(f"{idx:>3} {a:+.1f} {e0:.1f} | {1 - o1:9.2e} {1 - o2:9.2e} | "
                   f"{fmt_nd(nd_bias(r1, PHYS), 8)} | {fmt_nd(nd_bias(r2, C.PARAMS), 10)} "
                   f"{fmt_nd(nd_bias(r2, PHYS), 8)} {fmt_nd(nd_bias(r2, C.DEV_PARAMS), 2)} | "
                   f"{r1['cond_hat']:9.1e} {r2['cond_hat']:9.1e}")
    return out + [""]


def eval_check(nd):
    """Re-evaluated overlap at each best fit vs the stored one (same setup, so they should agree)."""
    out = ["overlap re-evaluated in fisher_at_best.py minus stored:"]
    for idx, _, _, r1, r2 in nd:
        d1 = r1["final_eval"]["overlap"] - r1["stored_eval"]["overlap"]
        d2 = r2["final_eval"]["overlap"] - r2["stored_eval"]["overlap"]
        out.append(f"  idx{idx:<2d} 1PA {d1:+.2e}   dev {d2:+.2e}   rho_s {r1['final_eval']['rho_s']:.4f}")
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=20.0)
    args = ap.parse_args()
    scale = args.snr / 20.0
    one, dev, dev_in_1pa, nd = build_tables(scale)
    lines = [f"IMRI_TAIL, 2PA signal. 1D bias = (best fit - injection) / sigma at SNR {args.snr:g}.", ""]
    lines += oned_table("1PA-only template, own sigma (8 parameters)", one)
    lines += oned_table("1PA + deviation template, own sigma (10 parameters)", dev)
    lines += oned_table("1PA + deviation best fit, in units of the 1PA-only sigma", dev_in_1pa)
    lines += nd_table(nd, args.snr) + eval_check(nd)
    text = "\n".join(lines)
    (C.OUT_ROOT / f"bias_comparison_snr{args.snr:g}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
