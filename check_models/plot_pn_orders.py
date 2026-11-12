"""Plot the importance of each SuperKludge 1PA PN order over the whole EMRI and IMRI grid (from pn_orders.py).

One figure per grid, 2 x 3 panels, x = grid point ordered by spin then e0 (spin rows separated by gaps):
    top    : rms(order-i term)/rms(full 1PA) for pdot, edot, Omega_phi   (Omega_r is close to Omega_phi; in the json)
    bottom : Omega_r fraction, then end dephasing against 0PA (Phi_phi, Phi_r) of order i alone, with the full 1PA
-> plots/pn_orders_{grid}.png
Run:  python plot_pn_orders.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ORDER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")      # order 0, 1, 2, 3 (fixed categorical order)
ORDER_LABELS = ("leading 1PA term", "+0.5PN term", "+1PN term", "+1.5PN term")
POWERS = {"pdot": "v^7..v^10", "edot": "v^9..v^12", "Omega_phi": "v^4..v^7", "Omega_r": "v^4..v^7"}
FULL_COLOR = "#2b2b2b"
SPINS = (-0.9, -0.5, 0.0, 0.5, 0.9)
N_E0 = 5
GAP = 1.5


def load(grid):
    return [json.loads((HERE / "results" / f"pn_orders_{grid}_idx{i}.json").read_text()) for i in range(25)]


def x_positions():
    """idx = e0_index*5 + a_index -> x, grouped by spin (5 e0 values per group)."""
    x = np.empty(25)
    for idx in range(25):
        e_i, a_i = divmod(idx, 5)
        x[idx] = a_i * (N_E0 + GAP) + e_i
    return x


def spin_axis(ax):
    centres = [a_i * (N_E0 + GAP) + (N_E0 - 1) / 2 for a_i in range(5)]
    ax.set_xticks(centres, [f"a={a:+.1f}" for a in SPINS])
    for a_i in range(1, 5):
        ax.axvline(a_i * (N_E0 + GAP) - (GAP + 1) / 2, color="0.85", lw=0.8)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def plot_series(ax, x, ys, order):
    """One line per spin group, so the lines do not join across spins."""
    for a_i in range(5):
        sel = np.argsort(x)[a_i * N_E0:(a_i + 1) * N_E0]
        ax.plot(x[sel], ys[sel], color=ORDER_COLORS[order] if order is not None else FULL_COLOR,
                lw=2 if order is not None else 2.5, marker="o", ms=4,
                label=(ORDER_LABELS[order] if order is not None else "full 1PA") if a_i == 0 else None)


def rate_panel(ax, data, x, channel):
    for i in range(4):
        plot_series(ax, x, np.array([d["rate_fraction"][channel][i] for d in data]), i)
    ax.axhline(1, color="0.6", lw=0.8, ls="--")
    ax.set_title(f"{channel}1PA PN terms ({POWERS[channel]}): rms(term) / rms(1PA)")
    ax.set_ylim(bottom=0)


def dephase_panel(ax, data, x, phase):
    for i in range(4):
        plot_series(ax, x, np.array([d["dephasing"][f"single_{i}"][phase] for d in data]), i)
    plot_series(ax, x, np.array([d["dephasing"]["upto_3"][phase] for d in data]), None)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title(f"end {phase[1:]} (rad): 0PA + one 1PA PN term vs 0PA")


def figure(grid):
    data, x = load(grid), x_positions()
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), constrained_layout=True)
    for ax, ch in zip(axes[0], ("pdot", "edot", "Omega_phi")):
        rate_panel(ax, data, x, ch)
    rate_panel(axes[1, 0], data, x, "Omega_r")
    dephase_panel(axes[1, 1], data, x, "dPhi_phi")
    dephase_panel(axes[1, 2], data, x, "dPhi_r")
    for ax in axes.flat:
        spin_axis(ax)
        ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85, edgecolor="0.85")
    fig.suptitle(f"{grid} grid: weight of each PN term of the SuperKludge 1PA, as written in SKequatorialfluxes.py "
                 f"(within each spin group, e0 = 0.1 ... 0.5 left to right)", fontsize=12)
    out = HERE / "plots" / f"pn_orders_{grid}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    for grid in ("EMRI", "IMRI"):
        figure(grid)
