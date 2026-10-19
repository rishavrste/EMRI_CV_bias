"""Fit quality of the combined population, 0PA against 0PA+PN, from the fits alone.

    python plot_overlap.py

overlap.png      1 - overlap against p0, both models, both runs
improvement.png  mismatch improvement (1 - ov_0PA) / (1 - ov_PN) against p0

Reads results_inputs only; no Fisher, nothing recomputed.  p0 is the x axis because the
environmental term enters as (p/10)^8.  One population: colour + marker = model, every marker filled.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import results_inputs as RI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")

C_0PA, C_PN, C_GAIN = "#1f6feb", "#d1650f", "#6f42c1"   # as run_1/results
M_0PA, M_PN = "o", "^"
MODELS = [("0PA", C_0PA, M_0PA), ("0PA+PN", C_PN, M_PN)]
LABEL_ABOVE = 1e-6           # name only the sources the fit struggled with
GAIN_LABEL_BELOW = 3.0       # improvement plot: name only the outliers below the main band


def mismatch(s, model):
    return max(1.0 - s["fits"][model]["ov_final"], 1e-16)


def _scatter(ax, x, y, colour, marker, label):
    ax.scatter(x, y, marker=marker, s=34, color=colour, edgecolors="white", linewidths=0.5,
               label=label, zorder=3)


def _style(ax, ylabel):
    ax.set_yscale("log")
    ax.set_xlabel("p0")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _label(ax, s, y, dy=-12):
    ax.annotate(f"p0={s['p0']:.2f}", (s["p0"], y),
                textcoords="offset points", xytext=(0, dy), ha="center", fontsize=6.5,
                color="0.35")


def overlap_panel(srcs, path):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for model, colour, marker in MODELS:
        _scatter(ax, [s["p0"] for s in srcs], [mismatch(s, model) for s in srcs],
                 colour, marker, model)
    for s in srcs:
        if mismatch(s, "0PA") > LABEL_ABOVE:
            _label(ax, s, mismatch(s, "0PA"))
    _style(ax, "1 - overlap")
    ax.set_title("Fit quality against p0, both recovery models\n"
                 f"{len(srcs)} sources, e0 fixed, 2nd-gen TDI", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def improvement_panel(srcs, path):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    gain = {s["name"]: mismatch(s, "0PA") / mismatch(s, "0PA+PN") for s in srcs}
    _scatter(ax, [s["p0"] for s in srcs], [gain[s["name"]] for s in srcs], C_GAIN, "o", None)
    for s in srcs:
        if gain[s["name"]] < GAIN_LABEL_BELOW:
            _label(ax, s, gain[s["name"]], dy=7)
    ax.axhline(1.0, color="0.45", lw=1.2, ls="--", zorder=1)
    med = float(np.median(list(gain.values())))
    ax.axhline(med, color=C_GAIN, lw=1, ls=":", zorder=1)
    ax.text(0.99, 0.97, f"median {med:.3g}x  ({len(srcs)} sources)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=C_GAIN)
    _style(ax, "mismatch improvement,  (1 - ov 0PA) / (1 - ov 0PA+PN)")
    ax.set_yticks([1, 2, 3, 5, 7, 10])
    ax.set_yticklabels(["1", "2", "3", "5", "7", "10"])
    ax.minorticks_off()
    ax.set_title("What the PN deviation buys in fit quality\n"
                 "above 1 the PN template fits better",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def print_table(srcs):
    print(f"{'source':>24} {'p0':>7} {'1-ov 0PA':>10} {'1-ov PN':>10} {'gain':>8} {'C_p':>9}")
    for s in sorted(srcs, key=lambda s: s["p0"]):
        a, b = mismatch(s, "0PA"), mismatch(s, "0PA+PN")
        print(f"{s['name']:>24} {s['p0']:>7.3f} {a:>10.3e} {b:>10.3e} {a / b:>8.3f} "
              f"{s['fits']['0PA+PN']['theta'][-1]:>9.3f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    srcs = RI.kept_in_plots(RI.load_sources(), lambda s: (s["fits"]["0PA"]["ov_final"],
                                                          s["fits"]["0PA+PN"]["ov_final"]))
    overlap_panel(srcs, os.path.join(OUT, "overlap.png"))
    improvement_panel(srcs, os.path.join(OUT, "improvement.png"))
    print_table(srcs)


if __name__ == "__main__":
    main()
