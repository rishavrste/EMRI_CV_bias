"""1D bias as split violins, one violin per grid point (as EMRI_1PA_PN_DEV/plot_codes/plot_bias_violin_params.py).

Each violin is the Fisher distribution of x - x_inj at SNR 20: left half 0PA, N(b_0PA, sigma_0PA^2); right half
0PA + REL5_1, N(b_REL5, sigma_REL5^2) with the five coefficients marginalised (the coefficients themselves:
right half only, offset from 0 = their best-fit value). b = best fit - injection (phases wrapped), sigma from
the Fisher at that best fit. Each half is cut at +-N_SIGMA sigma; the bar is b.
    -> plots/bias_1D/violin_grid.png              all parameters
       plots/bias_1D/per_param/violin_{param}.png one file per parameter
       plots/bias_1D/zoom/violin_{param}.png      physical parameters, y range set by the 0PA halves and both
                                                  biases, so the bias bars stay visible under the wide REL5_1 sigma

Run:  python plot_violins.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import COLOR, GRID, LABEL, SURFACE, TEXT, TEXT_2, save
from data import A_VALUES, DEV, E_VALUES, PHYS, TAGS, load_points

PARAMS = PHYS + DEV
HALVES = (("0pa", -1), ("rel5", +1))
N_SIGMA = 4.0
HALF_WIDTH = 0.42
GROUP_GAP = 1.0


def ordered(pts):
    """Points grouped by spin, e0 ascending within a group, with their x positions."""
    xs, out = [], []
    for g, a in enumerate(A_VALUES):
        for k, e in enumerate(E_VALUES):
            out.append(next(p for p in pts if p["a"] == a and p["e0"] == e))
            xs.append(g * (len(E_VALUES) + GROUP_GAP) + k)
    return out, np.array(xs, dtype=float)


def half_violin(ax, x, b, s, color, side):
    if not (np.isfinite(b) and np.isfinite(s) and s > 0):
        return
    y = np.linspace(b - N_SIGMA * s, b + N_SIGMA * s, 200)
    w = HALF_WIDTH * np.exp(-0.5 * ((y - b) / s) ** 2)
    ax.fill_betweenx(y, x, x + side * w, facecolor=color, alpha=0.35, edgecolor=color, lw=0.8)
    ax.plot([x, x + side * HALF_WIDTH], [b, b], color=color, lw=1.8, solid_capstyle="butt")


def param_axes(ax, pts, xs, name):
    for p, x in zip(pts, xs):
        for tag, side in HALVES:
            f = p["fits"][tag]
            if name in f["bias"]:
                half_violin(ax, x, f["bias"][name], f["sigma"][name], COLOR[tag], side)
    ax.axhline(0, color=TEXT_2, lw=0.8, zorder=0)
    ax.set_xticks(xs, [f"{p['e0']:.1f}" for p in pts], fontsize=7)
    for g, a in enumerate(A_VALUES):
        mid = xs[g * len(E_VALUES):(g + 1) * len(E_VALUES)].mean()
        ax.text(mid, -0.13, f"a = {a:+.1f}" if a else "a = 0", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=TEXT)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    ax.set_xlabel("e0", color=TEXT_2, fontsize=8, labelpad=1)
    ylabel = f"{LABEL[name]}_best" if name in DEV else f"{LABEL[name]}_best − {LABEL[name]}_inj"
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_title(f"{LABEL[name]}, SNR 20", color=TEXT, fontsize=12, loc="left")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def zoom_limit(pts, name):
    """Half-range covering every 0PA half (b +- N_SIGMA sigma) and every REL5_1 bias bar."""
    f0 = [p["fits"]["0pa"] for p in pts]
    reach = [abs(f["bias"][name]) + N_SIGMA * f["sigma"][name] for f in f0]
    reach += [abs(p["fits"]["rel5"]["bias"][name]) for p in pts]
    return 1.1 * max(reach)


def legend(where):
    handles = [Patch(facecolor=COLOR[t], edgecolor=COLOR[t], alpha=0.5,
                     label=f"{TAGS[t]} ({'left' if side < 0 else 'right'})") for t, side in HALVES]
    where.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=2)


def single_figure(pts, xs, name):
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE, layout="constrained")
    param_axes(ax, pts, xs, name)
    legend(ax)
    return fig


def zoom_figure(pts, xs, name):
    fig = single_figure(pts, xs, name)
    ax = fig.axes[0]
    lim = zoom_limit(pts, name)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"{LABEL[name]}, SNR 20 (zoomed to the 0PA spread; REL5_1 halves extend beyond)",
                 color=TEXT, fontsize=12, loc="left")
    return fig


def grid_figure(pts, xs):
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(13, 3.6 * len(PARAMS)), facecolor=SURFACE,
                             layout="constrained")
    for ax, name in zip(axes, PARAMS):
        param_axes(ax, pts, xs, name)
    legend(axes[0])
    return fig


def main():
    pts, xs = ordered(load_points())
    save(grid_figure(pts, xs), "bias_1D", "violin_grid.png", dpi=110)
    for name in PARAMS:
        fig = single_figure(pts, xs, name)
        save(fig, "bias_1D", "per_param", f"violin_{name}.png", dpi=150)
        plt.close(fig)
    for name in PHYS:
        fig = zoom_figure(pts, xs, name)
        save(fig, "bias_1D", "zoom", f"violin_{name}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
