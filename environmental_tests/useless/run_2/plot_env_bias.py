"""Bias plots for the environmental CV population study.

Two figures, both restricted to the sources that actually converged (the same
selection make_report.py applies, imported from it so the two cannot drift):

  bias_vs_corr.png       small multiples, one panel per vacuum parameter:
                         bias/sigma against corr(A_PM, theta_i) across sources,
                         0PA and 0PA+PN overlaid.
  bias_suppression.png   median |bias/sigma| per parameter, 0PA vs 0PA+PN,
                         ordered by how strongly that parameter correlates with A_PM.

Faceting by parameter rather than colouring by it keeps the figure to two colours
(one per recovery model); nine categorical hues would not separate reliably in a
scatter, where every pair of series can end up adjacent.

Runs on CPU, no GPU needed.

Run:  python plot_env_bias.py [run_dir]
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_report import RESULTS, drop_reason, run_dir

HERE = os.path.dirname(os.path.abspath(__file__))

LABELS = {"m1": r"$m_1$", "m2": r"$m_2$", "a": r"$a$", "p0": r"$p_0$", "e0": r"$e_0$",
          "qS": r"$\theta_S$", "phiS": r"$\phi_S$",
          "Phi_phi0": r"$\Phi_{\varphi 0}$", "Phi_r0": r"$\Phi_{r 0}$"}

# Slots 1 and 2 of the reference categorical palette, which validates for scatter forms.
C_0PA, C_PN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d6d5d0"

MODELS = [("fit_0pa", "0PA (vacuum)", C_0PA, "o"),
          ("fit_pn", "0PA + PN", C_PN, "^")]


def load_kept(path):
    with open(path) as f:
        data = json.load(f)
    # Keep every source; drop_reason only labels the unconverged ones.
    kept = data["sources"]
    dropped = [(r, drop_reason(r)) for r in kept if drop_reason(r) is not None]
    return data, kept, dropped


def series(kept, key, i):
    """corr(A_PM, theta_i) and bias/sigma for parameter i across the kept sources."""
    x = np.array([r["corr_A_PM"][i] for r in kept])
    y = np.array([r[key]["bias_over_sigma"][i] for r in kept])
    return x, y


def style_axes(ax):
    ax.grid(alpha=0.5, lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def figure_scatter(kept, names, out):
    """One panel per parameter: bias/sigma vs correlation with A_PM."""
    ylim = 1.18 * max(abs(r[k]["bias_over_sigma"][i])
                      for r in kept for k, *_ in MODELS for i in range(len(names)))

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharey=True)
    for i, (name, ax) in enumerate(zip(names, axes.ravel())):
        ax.axhspan(-1, 1, color=GRID, alpha=0.35, lw=0)     # the +-1 sigma band
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        for key, label, color, marker in MODELS:
            x, y = series(kept, key, i)
            ax.scatter(x, y, s=46, marker=marker, color=color, edgecolor="white",
                       lw=0.8, alpha=0.9, label=label, zorder=3)
        mean_corr = np.mean([r["corr_A_PM"][i] for r in kept])
        ax.set_title(f"{LABELS[name]}   corr $= {mean_corr:+.2f}$",
                     fontsize=11, color=INK, pad=6)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-ylim, ylim)
        style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\mathrm{corr}(A_{\rm PM},\, \theta_i)$", fontsize=9, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"bias $/\ \sigma$", fontsize=9, color=MUTED)

    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor=GRID)
    fig.suptitle("Environmental systematic bias per vacuum parameter",
                 fontsize=14, color=INK, y=0.995)
    fig.text(0.5, 0.963, f"{len(kept)} converged EMRIs at SNR 100; "
             f"shaded band is $\\pm 1\\sigma$; panel title gives the mean correlation",
             ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.952))
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[PLOT] written {os.path.basename(out)}")


def figure_suppression(kept, names, out):
    """Median |bias/sigma| before and after the PN degrees of freedom, ordered by |corr|."""
    corr = np.array([np.mean([r["corr_A_PM"][i] for r in kept]) for i in range(len(names))])
    med = {key: np.array([np.median([abs(r[key]["bias_over_sigma"][i]) for r in kept])
                          for i in range(len(names))]) for key, *_ in MODELS}

    order = np.argsort(np.abs(corr))          # weakest correlation at the bottom
    ypos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvline(1.0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(1.05, len(order) - 0.35, r"$1\sigma$", fontsize=9, color=MUTED, va="center")

    for row, i in enumerate(order):
        ax.plot([med["fit_pn"][i], med["fit_0pa"][i]], [row, row],
                color=GRID, lw=2.5, solid_capstyle="round", zorder=2)
    for key, label, color, marker in MODELS:
        ax.scatter(med[key][order], ypos, s=90, marker=marker, color=color,
                   edgecolor="white", lw=1.0, label=label, zorder=3)

    for row, i in enumerate(order):
        ax.annotate(f"{med['fit_0pa'][i] / med['fit_pn'][i]:.0f}x",
                    (med["fit_0pa"][i], row), xytext=(9, 0), textcoords="offset points",
                    fontsize=9, color=MUTED, va="center")

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{LABELS[names[i]]}   {corr[i]:+.2f}" for i in order], fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel(r"median $|$bias$|\ /\ \sigma$  over the converged sources",
                  fontsize=10, color=MUTED)
    ax.set_ylim(-0.7, len(order) - 0.1)
    ax.set_xlim(5e-4, 4.0)
    style_axes(ax)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=10, framealpha=0, edgecolor="none")
    ax.set_title("The PN deviation absorbs the bias in proportion to how strongly\n"
                 "the parameter correlates with $A_{\\rm PM}$",
                 fontsize=13, color=INK, loc="left", pad=12)
    fig.text(0.5, -0.03, "row labels carry the mean corr$(A_{\\rm PM}, \\theta_i)$; "
             "annotations are the suppression factor",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[PLOT] written {os.path.basename(out)}")


def main(argv=()):
    d = run_dir(argv)
    data, kept, dropped = load_kept(os.path.join(d, RESULTS))
    names = data["params_9"]
    print(f"[INFO] {len(kept)} converged sources {[r['idx'] for r in kept]}; "
          f"flagged unconverged {[r['idx'] for r, _ in dropped]}")
    for r, why in dropped:
        print(f"       source {r['idx']}: {why}")
    figure_scatter(kept, names, os.path.join(d, "bias_vs_corr.png"))
    figure_suppression(kept, names, os.path.join(d, "bias_suppression.png"))


if __name__ == "__main__":
    main(sys.argv[1:])
