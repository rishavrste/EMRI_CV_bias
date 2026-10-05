"""Worst and best r-factor per grid point.

r = |theta_PN - theta_inj| / |theta_0PA - theta_inj| for each of common.R_PARAMS: how far the
0PA + PN best fit sits from the injection relative to the 0PA best fit (r > 1: the
deviation pushed that parameter further off). Worst = max over the parameters, best = min; each
cell names the parameter. The ratio is unit-free, so it needs no Fisher sigma.
    -> plots/fit/r_factor.png
With --no-phases, Phi_phi0 and Phi_r0 are left out -> plots/fit/r_factor_nophases.png

Run:  python plot_r_factor.py [--no-phases]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common import (DIV, FLAG_NOTE, LABEL, R_PARAMS, SURFACE, TEXT_2, C, grid, ink_on,
                    load_points, mark_flags, out_file, style_axes)

LOG_LIM = 1.7           # colour range: r from 10^-1.7 (0.02) to 10^1.7 (50)


def extreme(p, fn, params):
    name = fn(params, key=p["r"].get)
    return name, p["r"][name]


def cell_label(p, fn, params):
    name, r = extreme(p, fn, params)
    return f"{LABEL[name]}\n{r:.2f}"


def panel(ax, fig, pts, fn, params, title):
    logr = grid(pts, lambda p: np.log10(extreme(p, fn, params)[1]))
    labels = grid(pts, lambda p: cell_label(p, fn, params))
    norm = TwoSlopeNorm(0.0, -LOG_LIM, LOG_LIM)
    im = ax.imshow(np.clip(logr.astype(float), -LOG_LIM, LOG_LIM), origin="lower", cmap=DIV,
                   norm=norm, aspect="auto")
    for (y, x), text in np.ndenumerate(labels):
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
                color=ink_on(DIV(norm(np.clip(logr[y, x], -LOG_LIM, LOG_LIM)))))
    style_axes(ax, title)
    ax.set_facecolor(SURFACE)
    mark_flags(ax, pts)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    cb.set_ticks(ticks, labels=[f"{10 ** t:.3g}" for t in ticks])
    cb.set_label("r  (blue: closer to injection than 0PA, red: further)", color=TEXT_2)
    cb.outline.set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-phases", action="store_true", help="leave out Phi_phi0 and Phi_r0")
    args = ap.parse_args()
    params = [n for n in R_PARAMS if not (args.no_phases and n in C.PHASES)]
    out = out_file("fit", "r_factor_nophases.png" if args.no_phases else "r_factor.png")
    pts = load_points()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=SURFACE)
    panel(axes[0], fig, pts, max, params, "Worst r-factor (parameter pushed furthest off)")
    panel(axes[1], fig, pts, min, params, "Best r-factor (parameter brought closest in)")
    fig.suptitle("r = |θ_PN − θ_inj| / |θ_0PA − θ_inj|, over " + ", ".join(LABEL[n] for n in params),
                 x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01, FLAG_NOTE, fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
