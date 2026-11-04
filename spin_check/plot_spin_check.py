"""Spin-check plots against a, 0PA vs 0PA + PN (the numbers of critical_snr.py):
    fits.png          1 - overlap, and the PN fit's C_p, C_e
    critical_snr.png  1D and nD rho_crit at 90 % (solid) and 99 % (dashed) CL, with phases (top) and
                      phases left out (bottom); the 1D markers name the worst parameter at 90 %
Spins in EXCLUDE are left out. The spins sit at evenly spaced positions (not to scale).
    -> plots/

Run (CPU):  python plot_spin_check.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import spin_config as S
from critical_snr import CLS, VARIANTS, critical, load

sys.path.insert(0, str(S.C.HERE / "plot_codes"))
from common import LABEL, SURFACE, TEXT, TEXT_2                 # noqa: E402  EMRI_1PA_PN_DEV styling

EXCLUDE = {0.1: "a = +0.1 left out: its 0PA fit is a secondary maximum (overlap 0.565), re-runs pending"}
COLOUR = {"0pa": "#2a78d6", "pn": "#eb6834"}                    # as plot_bias_violin_params.py
NAME = {"0pa": "0PA", "pn": "0PA + PN"}
LINE = {0.90: "-", 0.99: "--"}
SNR = 20.0
OUT = S.HERE / "plots"


def records():
    recs = {}
    for spin in sorted(S.SPINS):
        r0, rp = load(spin, "0pa"), load(spin, "pn")
        if spin not in EXCLUDE and r0 is not None and rp is not None:
            recs[spin] = {"0pa": r0, "pn": rp}
    return recs


def style(ax, spins, title, ylabel):
    ax.set_xticks(range(len(spins)), [f"{a:+g}" if a else "0" for a in spins])
    ax.set_xlabel("a (spin, evenly spaced, not to scale)", color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(color="#e4e3df", lw=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)


def series(ax, spins, values, template, ls="-", label=None):
    ax.plot(range(len(spins)), values, ls, color=COLOUR[template], lw=2, marker="o", ms=7,
            mec=SURFACE, mew=2, label=label, zorder=3)


def footer(fig):
    fig.text(0.01, 0.01, "Injection: idx22 (e0 = 0.5) with a varied; p0 from the 1PA plunge at 2.5 yr, "
             "dist for SNR 20.  " + "  ".join(EXCLUDE.values()), fontsize=8.5, color=TEXT_2)


def fits_figure(recs):
    spins = list(recs)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=SURFACE)
    for t in ("0pa", "pn"):
        series(axes[0], spins, [1 - recs[a][t]["eval"]["overlap"] for a in spins], t, label=NAME[t])
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, labelcolor=TEXT)
    style(axes[0], spins, "Mismatch at the best fit", "1 - overlap")
    for ax, n in zip(axes[1:], ("C_p", "C_e")):
        series(ax, spins, [recs[a]["pn"]["b"][n] for a in spins], "pn")
        style(ax, spins, f"{n} of the 0PA + PN fit", n)
    fig.suptitle("Spin check: fits at T = 2.5 yr", x=0.01, ha="left", fontsize=13)
    footer(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def critical_figure(recs):
    spins = list(recs)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=SURFACE, sharey="col")
    for row, (vname, subset) in enumerate(VARIANTS.items()):
        crit = {cl: {a: {t: critical(recs[a][t], subset, cl) for t in recs[a]} for a in spins}
                for cl in CLS}
        for col, (k, kind) in enumerate(((0, "1D"), (2, "nD"))):
            ax = axes[row, col]
            for t in ("0pa", "pn"):
                for cl in CLS:
                    series(ax, spins, [crit[cl][a][t][k] for a in spins], t, LINE[cl],
                           f"{NAME[t]}, {cl * 100:.0f}% CL")
                if kind == "1D":                                 # worst parameter, 90 % CL only
                    for a in spins:
                        v, worst = crit[0.90][a][t][0], crit[0.90][a][t][1]
                        ax.annotate(LABEL[worst], (spins.index(a), v), xytext=(0, 9), textcoords="offset points",
                                    ha="center", fontsize=8, color=TEXT_2)
            ax.axhline(SNR, color=TEXT_2, lw=1, ls=":", zorder=1)
            ax.annotate(f"SNR {SNR:g} (injection)", (0, SNR), xytext=(2, 3),
                        textcoords="offset points", fontsize=8, color=TEXT_2)
            ax.set_yscale("log")
            what = "worst parameter, own σ" if kind == "1D" else "physical set, C_p and C_e marginalised"
            style(ax, spins, f"{kind} critical SNR, {vname} ({what})", "ρ_crit")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=TEXT, fontsize=10, ncol=4,
               loc="upper left", bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Spin check: critical SNR, 0PA vs 0PA + PN (bias undetectable below ρ_crit)",
                 x=0.01, ha="left", fontsize=13)
    footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return fig


def main():
    recs = records()
    OUT.mkdir(exist_ok=True)
    for name, make in (("fits.png", fits_figure), ("critical_snr.png", critical_figure)):
        make(recs).savefig(OUT / name, dpi=160, facecolor=SURFACE)
        plt.close("all")
        print(f"-> {OUT / name}")


if __name__ == "__main__":
    main()
