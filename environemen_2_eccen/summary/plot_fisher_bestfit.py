"""Plots from fisher_bestfit.json (the Fisher / overlap re-score of every best fit).

  plots/overlap.png              1 - overlap of 0PA and 0PA+PN, per set
  plots/overlap_improvement.png  mismatch 0PA / mismatch PN, per set
  plots/abs_bias_<param>.png     |theta_bf - theta_inj| per parameter, Fisher sigma marked
  plots/abs_bias_all.png         the same, every parameter on one page
  plots/violin_e0.png            split violin of the e0 bias: left 0PA, right 0PA+PN

The violin is the Fisher Gaussian of each fit, e0 ~ N(e0_bf, sigma^2), shifted to the bias
e0 - e0_inj (not normalised), so each half is centred on the bias and as tall as its sigma.

CPU only, seconds.  Run:  python plot_fisher_bestfit.py [fisher_bestfit.json [plots_dir]]
e.g.  python plot_fisher_bestfit.py fisher_bestfit_snr200.json plots_snr200
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402
import numpy as np                                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")

# reference categorical palette, slots 1 and 2 (validated pair)
COLOR = {"vacuum": "#2a78d6", "pn": "#eb6834"}
LABEL = {"vacuum": "0PA", "pn": "0PA + PN"}
MARKER = {"vacuum": "o", "pn": "s"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PARAMS = ("m1", "m2", "p0", "e0", "Phi_phi0", "Phi_r0")
DEVS = ("dev_1", "dev_2")
TEX = {"m1": "m1", "m2": "m2", "p0": "p0", "e0": "e0", "Phi_phi0": "Phi_phi0",
       "Phi_r0": "Phi_r0", "dev_1": "C_p (dev_1)", "dev_2": "C_e (dev_2)"}
OFFSET = {"vacuum": -0.12, "pn": +0.12}          # side-by-side markers within a set


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def load(path):
    return json.load(open(path))


def set_ticks(ax, sets, with_e0=True):
    x = np.arange(len(sets))
    ax.set_xticks(x, [f"{s['tag']}\ne0 = {s['injection']['e0']:g}" if with_e0 else s["tag"]
                      for s in sets])
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    return x


def param(m, name, key):
    return m[key][m["names"].index(name)]


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, name), bbox_inches="tight")
    plt.close(fig)


# ---- overlap ------------------------------------------------------------------------------
def plot_overlap(sets):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = set_ticks(ax, sets)
    for i, s in enumerate(sets):
        ys = [s["models"][t]["mismatch"] for t in ("vacuum", "pn")]
        ax.plot([x[i] + OFFSET["vacuum"], x[i] + OFFSET["pn"]], ys, color=GRID, lw=2, zorder=1)
    for t in ("vacuum", "pn"):
        ax.scatter(x + OFFSET[t], [s["models"][t]["mismatch"] for s in sets], s=55,
                   marker=MARKER[t], color=COLOR[t], edgecolor="white", linewidth=1.5,
                   zorder=3, label=LABEL[t])
    ax.set_yscale("log")
    ax.set_ylabel("mismatch  1 - overlap")
    ax.set_title("Mismatch at the best fit, environmental signal", loc="left")
    ax.legend(loc="lower left")
    save(fig, "overlap.png")


def plot_improvement(sets):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = set_ticks(ax, sets)
    ratio = [s["models"]["vacuum"]["mismatch"] / s["models"]["pn"]["mismatch"] for s in sets]
    bars = ax.bar(x, ratio, width=0.5, color=COLOR["pn"])
    for b, r in zip(bars, ratio):
        ax.annotate(f"{r:.1f}x", (b.get_x() + b.get_width() / 2, r), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_ylabel("mismatch 0PA / mismatch (0PA + PN)")
    ax.set_title("Overlap improvement from the PN deviation", loc="left")
    save(fig, "overlap_improvement.png")


# ---- absolute bias ------------------------------------------------------------------------
def draw_abs_bias(ax, sets, name, models=("vacuum", "pn")):
    """|bias| as filled markers; the Fisher sigma of the same fit as a hollow marker."""
    x = set_ticks(ax, sets)
    for t in models:
        # deviations have no injected value to be biased from: plot |best fit| itself
        key = "theta" if name in DEVS else "bias"
        b = np.abs([param(s["models"][t], name, key) for s in sets])
        sig = [param(s["models"][t], name, "sigma") for s in sets]
        xo = x + (OFFSET[t] if len(models) > 1 else 0)
        ax.scatter(xo, b, s=45, marker=MARKER[t], color=COLOR[t], edgecolor="white",
                   linewidth=1.2, zorder=3, label=f"{LABEL[t]} |bias|")
        ax.scatter(xo, sig, s=45, marker="_", color=COLOR[t], linewidth=2, zorder=2,
                   label=f"{LABEL[t]} sigma")
        for xi, bi, si in zip(xo, b, sig):
            ax.plot([xi, xi], sorted((bi, si)), color=COLOR[t], lw=0.8, alpha=0.4, zorder=1)
    ax.set_yscale("log")
    ax.set_title(f"|{TEX[name]}|" if name in DEVS else f"|bias| of {TEX[name]}", loc="left")


def plot_abs_bias(sets):
    for name in PARAMS + DEVS:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        draw_abs_bias(ax, sets, name, ("pn",) if name in DEVS else ("vacuum", "pn"))
        ax.set_ylabel("best fit (PN only)" if name in DEVS else "|best fit - injection|")
        ax.legend(loc="best", fontsize=8, ncol=2)
        save(fig, f"abs_bias_{name}.png")

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, name in zip(axes.flat, PARAMS + DEVS):
        draw_abs_bias(ax, sets, name, ("pn",) if name in DEVS else ("vacuum", "pn"))
        ax.tick_params(axis="x", labelsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Filled: |bias| (C_p, C_e: |best fit|).  Bar: Fisher sigma at the best fit.",
                 y=-0.01, color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, "abs_bias_all.png")


# ---- e0 violin ----------------------------------------------------------------------------
def half_violin(ax, x, mu, sd, side, color, width=0.38):
    """One half of a violin: the Gaussian N(mu, sd^2), every half scaled to the same peak
    width (seaborn's density_norm="width"), so the spread reads off the height alone."""
    y = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
    dens = np.exp(-0.5 * ((y - mu) / sd) ** 2)
    w = width * dens
    sgn = -1 if side == "left" else +1
    ax.fill_betweenx(y, x, x + sgn * w, color=color, alpha=0.55, lw=0)
    ax.plot(x + sgn * w, y, color=color, lw=1.2)
    ax.plot([x, x + sgn * w.max()], [mu, mu], color=color, lw=2)


def plot_violin(sets):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = set_ticks(ax, sets, with_e0=False)
    ax.axhline(0, color=MUTED, lw=1)
    for i, s in enumerate(sets):
        for t, side in (("vacuum", "left"), ("pn", "right")):
            m = s["models"][t]
            half_violin(ax, x[i], param(m, "e0", "bias"), param(m, "e0", "sigma"),
                        side, COLOR[t])
    for t in ("vacuum", "pn"):
        ax.fill_between([], [], color=COLOR[t], alpha=0.55, label=LABEL[t])
    ax.set_ylabel("bias")
    ax.legend(loc="upper left")
    save(fig, "violin_e0.png")


def main(path, plots_dir=PLOTS):
    global PLOTS
    PLOTS = plots_dir
    style()
    sets = load(path)
    plot_overlap(sets)
    plot_improvement(sets)
    plot_abs_bias(sets)
    plot_violin(sets)
    print(f"wrote {PLOTS}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fisher_bestfit.json"),
         os.path.join(HERE, sys.argv[2]) if len(sys.argv) > 2 else PLOTS)
