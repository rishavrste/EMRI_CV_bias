"""Shared styling and output paths for the 0PA / 0PA + PN / 0PA + REL5_1 grid plots
(palette and grid layout as EMRI_1PA_PN_DEV/plot_codes/common.py)."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from data import A_VALUES, E_VALUES, HERE

PLOT_DIR = HERE.parent / "plots"
TITLE = "EMRI: 1PA signal, 0PA templates (T = 2.5 yr, 2nd-generation TDI AE, SNR 20)"
LABEL = {"m1": "m1", "m2": "m2", "a": "a", "p0": "p0", "e0": "e0", "qS": "qS", "phiS": "φS",
         "Phi_phi0": "Φφ0", "Phi_r0": "Φr0", "c_p": "c_p", "c_e": "c_e", "c_Omphi": "c_Ωφ",
         "c_Omr": "c_Ωr", "K": "K", "C_p": "C_p", "C_e": "C_e"}
COLOR = {"0pa": "#2a78d6", "pn": "#eb6834", "rel5": "#1b9e77"}   # validated categorical order
TEXT, TEXT_2, SURFACE, MID, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#f0efec", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", MID,
                                                         "#f2a9a8", "#e34948", "#a32a29"])


def out_file(*parts):
    path = PLOT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ink_on(rgba):
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.5 else TEXT


def style_axes(ax, title):
    ax.set_xticks(range(len(A_VALUES)), [f"{a:+.1f}" if a else "0" for a in A_VALUES])
    ax.set_yticks(range(len(E_VALUES)), [f"{e:.1f}" for e in E_VALUES])
    ax.set_xlabel("a (spin)", color=TEXT_2)
    ax.set_ylabel("e0", color=TEXT_2)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.tick_params(colors=TEXT_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(A_VALUES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(E_VALUES)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_facecolor(SURFACE)


def heat(ax, fig, values, labels, cmap, norm, title, cbar_label, fontsize=8):
    """Grid heat map with a text label per cell and its own colour bar."""
    im = ax.imshow(values, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    for (y, x), text in np.ndenumerate(labels):
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=ink_on(cmap(norm(values[y, x]))))
    style_axes(ax, title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, color=TEXT_2)
    cb.outline.set_visible(False)
    return im


def save(fig, *parts, dpi=150):
    out = out_file(*parts)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"-> {out}")
