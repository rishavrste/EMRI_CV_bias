"""Step 3 of the environmental EMRI study: the correlation-vs-bias scatter.

Reads results_env_cv.json (step 2) and plots, for every (EMRI, vacuum parameter) pair,

    x = corr(A_PM, theta_i)     from the 11 x 11 environmental Fisher at the injected truth
    y = (best_fit - injected) / sigma_i    with sigma from the recovery model's own Fisher
                                           recomputed at its own best fit

15 sources x 9 vacuum parameters = 135 points per recovery model, coloured by parameter.
The 0PA and 0PA+PN models are shown side by side and then overlaid, so the reduction in
bias won by the PN deviation is visible at fixed correlation.

Only the 9 vacuum parameters are plotted: C_p and C_e have no injected value to be biased
away from, and A_PM has no counterpart in the recovery models.

Runs on CPU, no GPU needed.

Run:  python plot_env_cv.py [run_dir]
      python plot_env_cv.py run_1/a.json run_1/b.json   # merge several
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_report import RESULTS, run_dir, split

HERE = os.path.dirname(os.path.abspath(__file__))

PARAMS_9 = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
LABELS_9 = [r"$m_1$", r"$m_2$", r"$a$", r"$p_0$", r"$e_0$",
            r"$\theta_S$", r"$\phi_S$", r"$\Phi_{\varphi 0}$", r"$\Phi_{r 0}$"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:len(PARAMS_9)]

MODELS = [("fit_0pa", "0PA", "o"), ("fit_pn", "0PA + PN", "^")]


def load(paths):
    """Merge one or more step-2 JSONs, keeping one record per source idx.

    Sources whose climbs failed are dropped here rather than plotted: a source that never
    reached a maximum has a bias/sigma that says nothing about the model, and a single such
    point (source 4 reached -540 sigma) sets the y-scale for the whole figure.  The criteria
    live in make_report.split so the plots and the report always show the same subset.
    """
    by_idx = {}
    for p in paths:
        with open(p) as f:
            for r in json.load(f)["sources"]:
                by_idx[r["idx"]] = r
    kept, dropped = split([by_idx[i] for i in sorted(by_idx)])
    for r, why in dropped:
        print(f"[DROP] source {r['idx']}: {why}")
    return kept


def collect(records, key):
    """(x, y, param index, source index, converged flag) per (EMRI, vacuum parameter) pair.

    `converged` is step 2's `converged_above_0pa`: False means the PN climb ended below its
    own 0PA even after the re-seed, so that source's PN row is an optimiser failure and not a
    statement about the model.  Those points are drawn hollow with a red edge.
    """
    x, y, ip_, isrc, okf = [], [], [], [], []
    for r in records:
        ok = bool(r[key].get("converged_above_0pa", True))
        for i in range(len(PARAMS_9)):
            x.append(r["corr_A_PM"][i])
            y.append(r[key]["bias_over_sigma"][i])
            ip_.append(i)
            isrc.append(r["idx"])
            okf.append(ok)
    return np.array(x), np.array(y), np.array(ip_), np.array(isrc), np.array(okf)


def decorate(ax, title):
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    for s in (1.0, -1.0):
        ax.axhline(s, color="0.6", lw=0.8, ls="--")
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$\mathrm{corr}(A_{\rm PM},\, \theta_i)$")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)


def main(paths, out_dir):
    records = load(paths)
    print(f"[INFO] {len(records)} sources from {len(paths)} file(s): "
          f"{[r['idx'] for r in records]}")

    seed0 = [r["idx"] for r in records if r["fit_pn"].get("seed") == "from_0PA"]
    bad = [r["idx"] for r in records if not r["fit_pn"].get("converged_above_0pa", True)]
    stops = sorted({r[k].get("stop_reason", "?") for r in records for k in ("fit_0pa", "fit_pn")})
    if seed0:
        print(f"[INFO] the 0PA seed won the PN climb on sources {seed0}")
    print(f"[INFO] climb stop reasons present: {', '.join(stops)}")
    # The two PN seeds disagreeing is the direct measure of how converged a source is; a large
    # spread means the winner is an accident of where the climb started, not a maximum.
    spread = [(r["idx"], r["fit_pn"].get("seed_spread", 0.0)) for r in records]
    loose = [i for i, sp in spread if sp > 1e-6]
    if loose:
        print(f"[WARN] PN seed spread > 1e-6 on sources {loose} -- the two seeds land in "
              f"different places, so these PN rows are not converged")
    if bad:
        print(f"[WARN] PN still below 0PA on sources {bad} -- drawn hollow with a red edge, "
              f"these rows are optimiser failures, not model statements")

    data = {key: collect(records, key) for key, _, _ in MODELS}

    # symmetric y limits, driven by the larger of the two models
    ymax = max(np.max(np.abs(y)) for x, y, _, _, _ in data.values())
    ylim = (-1.15 * ymax, 1.15 * ymax)

    # --- panels, one per recovery model ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (key, name, marker) in zip(axes, MODELS):
        x, y, ip_, _, okf = data[key]
        for i, (p, lab) in enumerate(zip(PARAMS_9, LABELS_9)):
            m = (ip_ == i) & okf
            ax.scatter(x[m], y[m], s=42, marker=marker, color=COLORS[i],
                       edgecolor="k", lw=0.4, alpha=0.85, label=lab)
            b = (ip_ == i) & ~okf
            if b.any():
                ax.scatter(x[b], y[b], s=52, marker=marker, facecolor="none",
                           edgecolor="crimson", lw=1.2)
        decorate(ax, f"{name}   (max $|b/\\sigma|$ = {np.max(np.abs(y)):.2f})")
        ax.set_ylim(*ylim)
    axes[0].set_ylabel(r"bias / $\sigma$  $=(\theta_{\rm bf}-\theta_{\rm inj})/\sigma$")
    axes[1].legend(ncol=2, fontsize=9, framealpha=0.9, loc="best", title="parameter")
    fig.suptitle(f"Environmental CV bias vs $A_{{\\rm PM}}$ correlation, "
                 f"{len(records)} EMRIs at SNR 100", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_corr_bias_panels.png"), dpi=150,
                bbox_inches="tight")
    print("[PLOT] written scatter_corr_bias_panels.png")

    # --- the two models overlaid on one axes -------------------------------
    fig2, ax = plt.subplots(figsize=(8, 6))
    for key, name, marker in MODELS:
        x, y, ip_, _, okf = data[key]
        for i in range(len(PARAMS_9)):
            m = (ip_ == i) & okf
            ax.scatter(x[m], y[m], s=42, marker=marker, color=COLORS[i],
                       edgecolor="k", lw=0.4, alpha=0.8)
            b = (ip_ == i) & ~okf
            if b.any():
                ax.scatter(x[b], y[b], s=52, marker=marker, facecolor="none",
                           edgecolor="crimson", lw=1.2)
    decorate(ax, f"{len(records)} EMRIs, {len(records) * len(PARAMS_9)} points per model")
    ax.set_ylabel(r"bias / $\sigma$")
    ax.set_ylim(*ylim)
    handles = [plt.Line2D([], [], ls="", marker="s", color=COLORS[i], mec="k", mew=0.4,
                          label=LABELS_9[i]) for i in range(len(PARAMS_9))]
    handles += [plt.Line2D([], [], ls="", marker=mk, color="0.5", mec="k", mew=0.4, label=nm)
                for _, nm, mk in MODELS]
    ax.legend(handles=handles, ncol=2, fontsize=9, framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "scatter_corr_bias.png"), dpi=150, bbox_inches="tight")
    print("[PLOT] written scatter_corr_bias.png")

    # --- the numbers behind the figures ------------------------------------
    print(f"\n{'param':>10} | {'mean corr':>10} | {'median |b/s| 0PA':>17} | "
          f"{'median |b/s| PN':>16} | {'> 1 sigma, 0PA':>14} | {'> 1 sigma, PN':>13}")
    print("-" * 96)
    for i, p in enumerate(PARAMS_9):
        x0, y0, ip0, _, _ = data["fit_0pa"]
        _, y1, ip1, _, _ = data["fit_pn"]
        m0, m1_ = ip0 == i, ip1 == i
        print(f"{p:>10} | {np.mean(x0[m0]):>+10.4f} | {np.median(np.abs(y0[m0])):>17.4f} | "
              f"{np.median(np.abs(y1[m1_])):>16.4f} | "
              f"{np.sum(np.abs(y0[m0]) > 1):>7d} / {np.sum(m0):<4d} | "
              f"{np.sum(np.abs(y1[m1_]) > 1):>6d} / {np.sum(m1_):<4d}")

    for key, name, _ in MODELS:
        x, y, _, _, _ = data[key]
        frac = 100.0 * np.mean(np.abs(y) > 1.0)
        rho = np.corrcoef(np.abs(x), np.abs(y))[0, 1]
        print(f"\n[{name:>9}] {frac:.1f}% of the {len(y)} points are biased by more than "
              f"1 sigma; corr(|corr(A_PM)|, |bias/sigma|) = {rho:+.4f}")


if __name__ == "__main__":
    # A bare argument names a run directory; explicit .json paths still work and merge,
    # in which case the figures land beside the first of them.
    argv = sys.argv[1:]
    if argv and all(a.endswith(".json") for a in argv):
        main(argv, os.path.dirname(os.path.abspath(argv[0])))
    else:
        d = run_dir(argv)
        main([os.path.join(d, RESULTS)], d)
