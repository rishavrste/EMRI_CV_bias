"""1D bias as split violins, one figure per parameter, one violin per grid point. Each violin is
the Fisher distribution of x - x_inj: left half 0PA, N(b_0PA, sigma_0PA^2); right half
0PA + PN, N(b_pn, sigma_pn^2) with C_p, C_e marginalised (C_p, C_e themselves: right half
only). b = best fit - injection (phases wrapped), sigma from the Fisher matrix at that best fit
(fisher_at_best.py) at the chosen SNR. Each half is scaled to the same maximum width and cut at
+-N_SIGMA sigma; the bar is b.
    -> plots/bias_1D/snr{SNR}/violin_grid.png                  all parameters
       plots/bias_1D/snr{SNR}/per_param/violin_{param}.png    one file per parameter

Run:  python plot_bias_violin_params.py [--snr 20]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from bias_data import ALL, load_bias
from common import A_VALUES, E_VALUES, LABEL, SURFACE, TEXT, TEXT_2, out_file

HALVES = (("0pa", "0PA", "#2a78d6", -1), ("pn", "0PA + PN", "#eb6834", +1))
N_SIGMA = 4.0           # each half drawn over b +- N_SIGMA sigma
HALF_WIDTH = 0.42       # maximum half-width of a violin, in units of the point spacing
GROUP_GAP = 1.0         # extra space between spin groups


def ordered(pts):
    """Points grouped by spin, e0 ascending within a group, with their x positions."""
    xs, out = [], []
    for g, a in enumerate(A_VALUES):
        for k, e in enumerate(E_VALUES):
            out.append(next(p for p in pts if p["a"] == a and p["e0"] == e))
            xs.append(g * (len(E_VALUES) + GROUP_GAP) + k)
    return out, np.array(xs, dtype=float)


def half_violin(ax, x, b, s, color, side):
    """Gaussian N(b, s^2) as one half of a violin at x (side -1 left, +1 right), bar at b."""
    if not (np.isfinite(b) and np.isfinite(s) and s > 0):
        return
    y = np.linspace(b - N_SIGMA * s, b + N_SIGMA * s, 200)
    w = HALF_WIDTH * np.exp(-0.5 * ((y - b) / s) ** 2)
    ax.fill_betweenx(y, x, x + side * w, facecolor=color, alpha=0.35, edgecolor=color, lw=0.8)
    ax.plot([x, x + side * HALF_WIDTH], [b, b], color=color, lw=1.8, solid_capstyle="butt")


def param_axes(ax, pts, name, snr, xs):
    for p, x in zip(pts, xs):
        for tag, _, color, side in HALVES:
            if name in p[f"b_{tag}"]:
                half_violin(ax, x, p[f"b_{tag}"][name], p[f"s_{tag}"][name], color, side)
    ax.axhline(0, color=TEXT_2, lw=0.8, zorder=0)
    ax.set_xticks(xs, [f"{p['e0']:.1f}" for p in pts], fontsize=7)
    for g, a in enumerate(A_VALUES):
        mid = xs[g * len(E_VALUES):(g + 1) * len(E_VALUES)].mean()
        ax.text(mid, -0.13, f"a = {a:+.1f}" if a else "a = 0", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=TEXT)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    ax.set_xlabel("e0", color=TEXT_2, fontsize=8, labelpad=1)
    ax.set_ylabel(f"{LABEL[name]}_best − {LABEL[name]}_inj", color=TEXT_2)
    ax.set_title(f"{LABEL[name]}, SNR {snr:g}", color=TEXT, fontsize=12, loc="left")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2)
    ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend(where):
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=f"{lab} (left)" if side < 0
                     else f"{lab} (right)") for _, lab, c, side in HALVES]
    where.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=2)


def single_figure(pts, xs, name, snr):
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
    param_axes(ax, pts, name, snr, xs)
    legend(ax)
    fig.tight_layout()
    return fig


def grid_figure(pts, xs, snr):
    fig, axes = plt.subplots(len(ALL), 1, figsize=(13, 3.6 * len(ALL)), facecolor=SURFACE)
    for ax, name in zip(axes, ALL):
        param_axes(ax, pts, name, snr, xs)
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=20.0)
    args = ap.parse_args()
    pts, xs = ordered(load_bias(args.snr))
    snr_dir = f"snr{args.snr:g}"
    out = out_file("bias_1D", snr_dir, "violin_grid.png")
    grid_figure(pts, xs, args.snr).savefig(out, dpi=130, facecolor=SURFACE)
    print(f"-> {out}")
    for name in ALL:
        fig = single_figure(pts, xs, name, args.snr)
        out = out_file("bias_1D", snr_dir, "per_param", f"violin_{name}.png")
        fig.savefig(out, dpi=160, facecolor=SURFACE)
        plt.close(fig)
    print(f"-> {out.parent}/violin_*.png ({len(ALL)} files)")


if __name__ == "__main__":
    main()
