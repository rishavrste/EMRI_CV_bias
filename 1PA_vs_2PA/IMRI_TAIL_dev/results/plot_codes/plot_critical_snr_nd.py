"""nD critical SNR only, 1PA and 1PA + deviation side by side on one shared colour scale, no flag
marks or footnotes; CL and phase choice only in the file name. plot_critical_snr.py has the
full 1D/nD/ratio figure. Same definition:
rho_crit = SNR0 D_CL / D, D over the physical parameters (C_p, C_e marginalised for 1PA + dev).
    -> 1PA_vs_2PA/results/plots/snr_crit_nD_{with,no}_phases_{CL}CL.png

Run:  python plot_critical_snr_nd.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import SEQ, SURFACE, TEXT_2, grid, ink_on, style_axes
from plot_critical_snr import CLS, SNR0, VARIANTS, critical
from bias_data import load_bias

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "plots"
TEMPLATES = (("1pa", "1PA"), ("dev", "1PA + deviation"))


def panel(ax, values, norm, title):
    im = ax.imshow(values.astype(float), origin="lower", cmap=SEQ, norm=norm, aspect="auto")
    for (y, x), v in np.ndenumerate(values):
        ax.text(x, y, f"{v:.3g}", ha="center", va="center", fontsize=9, color=ink_on(SEQ(norm(v))))
    style_axes(ax, title)
    ax.set_facecolor(SURFACE)
    return im


def figure(pts, crits, cl, variant):
    maps = {tag: grid(pts, lambda p, k=f"nD_{tag}": crits[p["idx"]][k]).astype(float)
            for tag, _ in TEMPLATES}
    norm = LogNorm(min(m.min() for m in maps.values()), max(m.max() for m in maps.values()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)
    for ax, (tag, name) in zip(axes, TEMPLATES):
        im = panel(ax, maps[tag], norm, name)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("ρ_crit", color=TEXT_2)
    cb.outline.set_visible(False)
    fig.suptitle("nD critical SNR", x=0.01, ha="left", fontsize=13)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pts = load_bias(SNR0)
    for variant, (folder, _) in VARIANTS.items():
        for cl in CLS:
            crits = {p["idx"]: critical(p, cl, variant) for p in pts}
            out = OUT_DIR / f"snr_crit_nD_{folder}_{cl * 100:.0f}CL.png"
            figure(pts, crits, cl, variant).savefig(out, dpi=160, facecolor=SURFACE,
                                                    bbox_inches="tight")
            plt.close("all")
            print(f"-> {out}")


if __name__ == "__main__":
    main()
