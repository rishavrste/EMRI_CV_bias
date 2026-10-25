"""Plots from fisher_bestfit_2g.json (2nd-gen TDI, A and E, SNR 200 re-score of every best fit).
Copy of summary/plot_fisher_bestfit.py with the PN model labelled "0PA with PN deviation" and the
e0 violin ordered by ascending injected e0.

  plots/overlap.png              1 - overlap of 0PA and 0PA with PN deviation, per set
  plots/overlap_improvement.png  mismatch 0PA / mismatch PN, per set
  plots/abs_bias_<param>.png     |theta_bf - theta_inj| per parameter, Fisher sigma marked
  plots/abs_bias_all.png         the same, every parameter on one page
  plots/violin_e0.png            split violin of the e0 bias: left 0PA, right 0PA with PN
                                 deviation, sets in ascending e0, each labelled
                                 by its e0, Sigma0 [g/cm^2] and h0
  plots/violin_e0_panels.png     the same, the N_ZOOM highest-e0 sets in a panel of their own
  plots/violin_e0_twin.png       the same, the N_ZOOM highest-e0 sets hatched on a right y axis

The violin is the Fisher Gaussian of each fit, e0 ~ N(e0_bf, sigma^2), shifted to the bias
e0 - e0_inj (not normalised), so each half is centred on the bias and as tall as its sigma.

CPU only, seconds.  Run:  python plot_fisher_bestfit.py   -> plots/
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
LABEL = {"vacuum": "0PA", "pn": "0PA with PN deviation"}
MARKER = {"vacuum": "o", "pn": "s"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PARAMS = ("m1", "m2", "p0", "e0", "Phi_phi0", "Phi_r0")
DEVS = ("dev_1", "dev_2")
TEX = {"m1": "m1", "m2": "m2", "p0": "p0", "e0": "e0", "Phi_phi0": "Phi_phi0",
       "Phi_r0": "Phi_r0", "dev_1": "C_p (dev_1)", "dev_2": "C_e (dev_2)"}
OFFSET = {"vacuum": -0.12, "pn": +0.12}          # side-by-side markers within a set
N_ZOOM = 2                                       # highest-e0 sets given their own y axis
ZOOM_HATCH = "///"


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
    ax.set_ylabel("mismatch 0PA / mismatch (0PA with PN deviation)")
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
def half_violin(ax, x, mu, sd, side, color, hatch=None, width=0.38):
    """One half of a violin: the Gaussian N(mu, sd^2), every half scaled to the same peak
    width (seaborn's density_norm="width"), so the spread reads off the height alone."""
    y = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
    dens = np.exp(-0.5 * ((y - mu) / sd) ** 2)
    w = width * dens
    sgn = -1 if side == "left" else +1
    ax.fill_betweenx(y, x, x + sgn * w, facecolor=color, alpha=0.55, lw=0, hatch=hatch,
                     edgecolor="white")
    ax.plot(x + sgn * w, y, color=color, lw=1.2)
    ax.plot([x, x + sgn * w.max()], [mu, mu], color=color, lw=2)


def source_label(s):
    """The three injection values that vary between sets: e0, Sigma0 and h0."""
    mant, exp = f"{s['disk']['Sigma0_cgs']:.3g}".split("e+")
    return (f"e0 = {s['injection']['e0']:g}\n\u03a30 = {mant}e{int(exp)}\n"
            f"h0 = {s['disk']['h0']:g}")


def draw_violins(ax, sets, x, hatch=None):
    """Both halves of every set at positions x: left 0PA, right 0PA with PN deviation."""
    for xi, s in zip(x, sets):
        for t, side in (("vacuum", "left"), ("pn", "right")):
            m = s["models"][t]
            half_violin(ax, xi, param(m, "e0", "bias"), param(m, "e0", "sigma"),
                        side, COLOR[t], hatch)
    ax.axhline(0, color=MUTED, lw=1)


def symmetric_ylim(ax):
    """y range symmetric about zero, so the zero lines of a twin pair coincide."""
    lim = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-lim, lim)


def violin_legend(ax, hatch=None, **kw):
    handles = [plt.matplotlib.patches.Patch(facecolor=COLOR[t], alpha=0.55, label=LABEL[t])
               for t in ("vacuum", "pn")]
    if hatch:
        handles.append(plt.matplotlib.patches.Patch(facecolor=MUTED, alpha=0.55, hatch=hatch,
                                                    edgecolor="white", label="right axis"))
    ax.legend(handles=handles, **(kw or dict(loc="upper right")))


def violin_ticks(ax, sets, x):
    ax.set_xticks(x, [source_label(s) for s in sets])
    ax.grid(axis="x", visible=False)


def plot_violin_panels(sets):
    """The N_ZOOM highest-e0 sets in their own panel, each panel with its own y range."""
    wide, zoom = sets[:-N_ZOOM], sets[-N_ZOOM:]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), width_ratios=(len(wide), len(zoom)))
    for ax, group in zip(axes, (wide, zoom)):
        x = np.arange(len(group))
        draw_violins(ax, group, x)
        violin_ticks(ax, group, x)
        ax.set_xlim(-0.6, len(group) - 0.4)
    axes[0].set_ylabel("bias")
    violin_legend(axes[0])
    fig.tight_layout()
    save(fig, "violin_e0_panels.png")


def plot_violin_twin(sets):
    """One axis; the N_ZOOM highest-e0 sets hatched and read off a second y axis on the right."""
    wide, zoom = sets[:-N_ZOOM], sets[-N_ZOOM:]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(sets))
    draw_violins(ax, wide, x[:len(wide)])
    right = ax.twinx()
    right.spines["right"].set_visible(True)
    right.grid(False)
    draw_violins(right, zoom, x[len(wide):], hatch=ZOOM_HATCH)
    violin_ticks(ax, sets, x)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.axvline(len(wide) - 0.5, color=MUTED, lw=1, ls=":")
    symmetric_ylim(ax)
    symmetric_ylim(right)
    ax.set_ylabel("bias (left " + str(len(wide)) + " sets)")
    right.set_ylabel("bias (hatched, right " + str(N_ZOOM) + " sets)")
    violin_legend(ax, ZOOM_HATCH, loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    save(fig, "violin_e0_twin.png")


def plot_violin(sets):
    sets = sorted(sets, key=lambda s: s["injection"]["e0"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(sets))
    draw_violins(ax, sets, x)
    violin_ticks(ax, sets, x)
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.set_ylabel("bias")
    violin_legend(ax)
    save(fig, "violin_e0.png")
    plot_violin_panels(sets)
    plot_violin_twin(sets)


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
    main(os.path.join(HERE, "fisher_bestfit_2g.json"))
