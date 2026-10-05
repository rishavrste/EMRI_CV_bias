"""Grid table of the 0PA + PN best fits: overlap (with the gain over 0PA), C_p, C_e,
on a (spin a) x (e0) grid.
    -> plots/fit/fit_table.png      with gain, flag marks and notes
       plots/fit/fit_table_clean.png      clean: overlap, C_p, C_e only

Run:  python plot_fit_table.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from common import (DIV, FLAG_NOTE, TITLE, SEQ, SURFACE, TEXT_2, grid, ink_on, load_points,
                    mark_flags, out_file, style_axes)



def heat(ax, fig, values, labels, cmap, norm, title, cbar_label):
    im = ax.imshow(values.astype(float), origin="lower", cmap=cmap, norm=norm, aspect="auto")
    for (y, x), text in np.ndenumerate(labels):
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                color=ink_on(cmap(norm(values[y, x]))))
    style_axes(ax, title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, color=TEXT_2)
    cb.outline.set_visible(False)


def overlap_panel(ax, fig, pts, clean):
    mism = grid(pts, lambda p: np.log10(1 - p["O_pn"]))
    if clean:
        labels = grid(pts, lambda p: f"{p['O_pn']:.10f}")
        title = "Overlap"
    else:
        labels = grid(pts, lambda p: f"{p['O_pn']:.10f}\ngain {p['gain']:.2f}")
        title = "Overlap, 0PA + PN (gain over 0PA)"
    lo, hi = np.floor(np.nanmin(mism.astype(float))), np.ceil(np.nanmax(mism.astype(float)))
    heat(ax, fig, mism, labels, SEQ.reversed(), Normalize(lo, hi), title, "log10(1 − overlap)")


def coeff_panel(ax, fig, pts, name):
    v = grid(pts, name)
    lim = max(abs(float(x)) for x in v.ravel())
    labels = grid(pts, lambda p: f"{p[name]:+.3f}" if abs(p[name]) >= 1e-3 else f"{p[name]:+.1e}")
    heat(ax, fig, v, labels, DIV, TwoSlopeNorm(0.0, -lim, lim), f"{name} at the best fit", name)


def figure(pts, clean):
    """clean: overlap, C_p, C_e only, without gain, flag marks, heading or notes."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor=SURFACE)
    overlap_panel(axes[0], fig, pts, clean)
    coeff_panel(axes[1], fig, pts, "C_p")
    coeff_panel(axes[2], fig, pts, "C_e")
    for ax in axes:
        ax.set_facecolor(SURFACE)
        if not clean:
            mark_flags(ax, pts)
    if clean:
        fig.tight_layout()
        return fig
    fig.suptitle(TITLE,
                 x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01, FLAG_NOTE, fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def main():
    pts = load_points()
    for out, clean in ((out_file("fit", "fit_table.png"), False), (out_file("fit", "fit_table_clean.png"), True)):
        figure(pts, clean).savefig(out, dpi=170, facecolor=SURFACE)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
