"""Shared data and styling for the IMRI 0PA vs 0PA + PN grid plots (as in EMRI_1PA_PN_DEV, without
the flag marks: user, 2026-10-07).

    load_points()  -> one dict per grid point: overlaps, gain, C_p, C_e, offsets, r-factors
    grid(points, key) -> 5 x 5 array, rows e0 = 0.1 .. 0.5, columns a = -0.9 .. 0.9
Best fits: results/best_fits.json (fisher/best_fits.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # IMRI_1PA_PN_DEV, for config
sys.path.insert(0, str(HERE.parent / "fisher"))          # compare_bias
import config as C                                       # noqa: E402

PLOT_DIR = HERE.parent / "plots"
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
TEMPLATES = {"0pa": "0PA", "pn": "0PA + PN"}
# Offsets compared in the r-factor: every fitted physical parameter, sky included.
R_PARAMS = list(C.PHYS)
LABEL = {"m1": "m1", "m2": "m2", "a": "a", "p0": "p0", "e0": "e0", "qS": "qS", "phiS": "φS",
         "Phi_phi0": "Φφ0", "Phi_r0": "Φr0", "C_p": "C_p", "C_e": "C_e"}
TITLE = "IMRI: 1PA signal, 0PA + PN template (T = 1 yr, 2nd-generation TDI AE)"


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


def point_record(point, b1, b2):
    inj = b1["theta_inj"]
    o1, o2 = b1["final_eval"]["overlap"], b2["final_eval"]["overlap"]
    d1 = {n: offset(b1["theta_final"], inj, n) for n in R_PARAMS}
    d2 = {n: offset(b2["theta_final"], inj, n) for n in R_PARAMS}
    r = {n: abs(d2[n]) / abs(d1[n]) for n in R_PARAMS}
    return dict(idx=int(point), point=point, a=b2["a_inj"], e0=b2["e0_inj"], O_0pa=o1, O_pn=o2,
                gain=(1 - o1) / (1 - o2), C_p=b2["theta_final"]["C_p"],
                C_e=b2["theta_final"]["C_e"], d_0pa=d1, d_pn=d2, r=r,
                pn_seed=b2["case"].split("_")[3])


def load_points():
    best = json.loads(C.BEST_FITS.read_text())
    return [point_record(p, best["0pa"][p], best["pn"][p]) for p in C.POINTS]


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

