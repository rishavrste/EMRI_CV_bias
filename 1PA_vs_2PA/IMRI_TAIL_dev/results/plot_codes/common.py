"""Shared data and styling for the IMRI_TAIL 1PA vs 1PA + deviation grid plots.

    load_points()  -> one dict per grid point: overlaps, gain, C_p, C_e, offsets, r-factors, flags
    grid(points, key) -> 5 x 5 array, rows e0 = 0.1 .. 0.5, columns a = -0.9 .. 0.9
Best fits: SK_files best_fit_IMRI_TAIL_1pa.json and results/best_fit_IMRI_TAIL_1pa_dev.json
(best_fit_dev.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                  # IMRI_TAIL_dev, for config
import config as C                                       # noqa: E402

PLOT_DIR = HERE.parent / "plot"
# Output layout under PLOT_DIR:
#   fit/                    fit table, r-factor maps, trend check
#   bias_1D/snr{S}/         violins (all parameters, grid), 1D table; per_param/ one violin each
#   bias_nD/                nD tables
#   critical_snr/           critical SNR maps and table
#   tables/                 copy-ready markdown of the 1D and nD bias


def out_file(*parts):
    """PLOT_DIR/parts..., with its directory created."""
    path = PLOT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
A_VALUES = [-0.9, -0.5, 0.0, 0.5, 0.9]
E_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
# Offsets compared in the r-factor. chi2 is left out: it is held fixed in several fits, where its
# offset is exactly 0 or the 1PA value.
R_PARAMS = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0"]
LABEL = {"m1": "m1", "m2": "m2", "a": "a", "p0": "p0", "e0": "e0", "Phi_phi0": "Φφ0", "Phi_r0": "Φr0"}

MARGINAL_GAIN = 1.5     # deviation fit barely above 1PA-only: gain below this
SUSPECT_FACTOR = 10.0   # 1PA-only mismatch this many times its spin row's median: likely secondary max

# --- styling (reference palette: blue sequential, blue <-> red diverging, gray midpoint) -----------
TEXT, TEXT_2, SURFACE, MID = "#0b0b0b", "#52514e", "#fcfcfb", "#f0efec"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
DIV = LinearSegmentedColormap.from_list("div_blue_red",
                                        ["#104281", "#3987e5", "#9ec5f4", MID,
                                         "#f2a9a8", "#e34948", "#a32a29"])   # MID at the centre


def wrap_phase(x):
    return float(np.pi - np.mod(np.pi - x, 2.0 * np.pi))


def offset(theta, inj, name):
    d = theta[name] - inj[name]
    return wrap_phase(d) if name in C.PHASES else d


def point_record(i, b1, b2):
    inj = b1["theta_inj"]
    o1, o2 = b1["final_eval"]["overlap"], b2["final_eval"]["overlap"]
    d1 = {n: offset(b1["theta_final"], inj, n) for n in R_PARAMS}
    d2 = {n: offset(b2["theta_final"], inj, n) for n in R_PARAMS}
    r = {n: abs(d2[n]) / abs(d1[n]) for n in R_PARAMS}
    return dict(idx=i, a=b2["a_inj"], e0=b2["e0_inj"], O_1pa=o1, O_dev=o2,
                gain=(1 - o1) / (1 - o2), C_p=b2["theta_final"]["C_p"],
                C_e=b2["theta_final"]["C_e"], d_1pa=d1, d_dev=d2, r=r)


def add_flags(points):
    """marginal: gain < MARGINAL_GAIN. suspect: 1PA mismatch > SUSPECT_FACTOR x its row median."""
    for a in A_VALUES:
        row = [p for p in points if p["a"] == a]
        med = np.median([1 - p["O_1pa"] for p in row])
        for p in row:
            p["marginal"] = p["gain"] < MARGINAL_GAIN
            p["suspect"] = (1 - p["O_1pa"]) > SUSPECT_FACTOR * med
    return points


def load_points():
    b1 = json.loads(C.BEST_1PA.read_text())
    b2 = json.loads(C.BEST_DEV.read_text())
    return add_flags([point_record(i, b1[str(i)], b2[str(i)]) for i in range(25)])


def grid(points, key):
    g = np.full((len(E_VALUES), len(A_VALUES)), np.nan, dtype=object)
    for p in points:
        g[E_VALUES.index(p["e0"]), A_VALUES.index(p["a"])] = key(p) if callable(key) else p[key]
    return g


def ink_on(rgba):
    """Text colour readable on a cell of colour rgba."""
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


def mark_flags(ax, points):
    """Outline marginal cells (dashed) and suspect-1PA cells (dotted), with a corner tag."""
    for p in points:
        x, y = A_VALUES.index(p["a"]), E_VALUES.index(p["e0"])
        tags = ("M" if p["marginal"] else "") + ("S" if p["suspect"] else "")
        if tags:
            ax.add_patch(Rectangle(
                (x - 0.47, y - 0.47), 0.94, 0.94, fill=False, linewidth=1.6,
                linestyle="--" if p["marginal"] else ":", edgecolor=TEXT))
            ax.text(x + 0.42, y + 0.40, tags, ha="right", va="top", fontsize=7.5,
                    fontweight="bold", color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))


FLAG_NOTE = (f"M = marginal deviation fit (gain < {MARGINAL_GAIN:g}, dashed);  "
             f"S = suspect 1PA-only reference (mismatch > {SUSPECT_FACTOR:g}x its spin row's "
             "median, likely a secondary maximum, dotted)")
