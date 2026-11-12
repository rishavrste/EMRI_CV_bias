"""What the extra degrees of freedom do: 0PA (+0), 0PA + PN (+2: C_p, C_e), 0PA + REL5_1 (+5).

    extra_dof/sigma_bias.png   per intrinsic parameter (rows), on the (a, e0) grid:
                               sigma_tmpl / sigma_0PA (marginal, own Fisher)  and  |b_0PA| / |b_tmpl|,
                               for PN and REL5_1 -- rho_crit rises if the bias falls OR sigma grows
    extra_dof/conditioning.png Jacobi-rescaled condition number of each Fisher, and the REL5_1 correlations
                               of c_p with m1, m2, p0 and of c_Omphi with c_Omr
    extra_dof/summary.png      medians over the grid of the sigma and bias ratios, per parameter and template
-> EMRI_REL5_LM/plots/

Run:  python plot_extra_dof.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from common import COLOR, DIV, GRID, LABEL, SEQ, SURFACE, TEXT, TEXT_2, heat, save
from data import MS, PHYS, TAGS, fisher_json, grid, load_points

INTRINSIC = PHYS[:5]
NDEV = {"0pa": 0, "pn": 2, "rel5": 5}
CORR = (("c_p", "m1"), ("c_p", "m2"), ("c_p", "p0"), ("c_Omphi", "c_Omr"))


def sigma_ratio(p, t, n):
    return p["fits"][t]["sigma"][n] / p["fits"]["0pa"]["sigma"][n]


def bias_ratio(p, t, n):
    return abs(p["fits"]["0pa"]["bias"][n]) / abs(p["fits"][t]["bias"][n])


def log_panel(ax, fig, g, title, label):
    lim = max(np.abs(np.log10(g)).max(), 0.5)
    heat(ax, fig, np.log10(g), np.vectorize(lambda v: f"{v:.2g}")(g), DIV, TwoSlopeNorm(0.0, -lim, lim),
         title, label, fontsize=7)


def sigma_bias_figure(pts):
    cols = [("pn", sigma_ratio, "σ / σ_0PA"), ("rel5", sigma_ratio, "σ / σ_0PA"),
            ("pn", bias_ratio, "|b_0PA| / |b|"), ("rel5", bias_ratio, "|b_0PA| / |b|")]
    fig, axes = plt.subplots(len(INTRINSIC), 4, figsize=(23, 4.3 * len(INTRINSIC)), layout="constrained")
    for row, n in zip(axes, INTRINSIC):
        for ax, (t, fn, what) in zip(row, cols):
            log_panel(ax, fig, grid(pts, lambda p: fn(p, t, n)),
                      f"{LABEL[n]}: {what}, {TAGS[t]} (+{NDEV[t]})", "log10 ratio")
    fig.suptitle("Extra coefficients: marginal σ grows (red, left) vs |bias| falls (red, right); "
                 "ρ_crit grows with both", x=0.01, ha="left")
    return fig


def condition(t, point):
    return float(np.linalg.cond(MS.jacobi(np.asarray(fisher_json(t, point)["fisher"]))[0]))


def correlation(point, a, b):
    d = fisher_json("rel5", point)
    S, names = MS.covariance(np.asarray(d["fisher"])), d["names"]
    i, j = names.index(a), names.index(b)
    return S[i, j] / np.sqrt(abs(S[i, i] * S[j, j]))


def conditioning_figure(pts):
    fig, axes = plt.subplots(2, 4, figsize=(23, 9.5), layout="constrained")
    conds = {t: grid(pts, lambda p, t=t: condition(t, str(p["idx"]))) for t in TAGS}
    norm = LogNorm(min(c.min() for c in conds.values()), max(c.max() for c in conds.values()))
    for ax, (t, name) in zip(axes[0], TAGS.items()):
        heat(ax, fig, conds[t], np.vectorize(lambda v: f"{v:.1e}")(conds[t]), SEQ, norm,
             f"cond(Γ), {name} (+{NDEV[t]})", "condition number", fontsize=7)
    axes[0, 3].axis("off")
    for ax, (a, b) in zip(axes[1], CORR):
        g = grid(pts, lambda p: correlation(str(p["idx"]), a, b))
        heat(ax, fig, g, np.vectorize(lambda v: f"{v:+.3f}")(g), DIV, Normalize(-1, 1),
             f"corr({LABEL[a]}, {LABEL[b]}), REL5_1", "correlation", fontsize=7)
    fig.suptitle("Conditioning of the Fisher (Jacobi-rescaled) and the REL5_1 degeneracies", x=0.01, ha="left")
    return fig


def summary_figure(pts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    x = np.arange(len(INTRINSIC))
    for ax, (fn, what) in zip(axes, ((sigma_ratio, "median σ / σ_0PA"), (bias_ratio, "median |b_0PA| / |b|"))):
        for k, t in enumerate(("pn", "rel5")):
            med = [np.median([fn(p, t, n) for p in pts]) for n in INTRINSIC]
            ax.bar(x + (k - 0.5) * 0.38, med, 0.36, color=COLOR[t], label=f"{TAGS[t]} (+{NDEV[t]})")
            for xi, v in zip(x + (k - 0.5) * 0.38, med):
                ax.text(xi, v * 1.15, f"{v:.2g}", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_yscale("log")
        ax.axhline(1, color=TEXT_2, lw=0.8)
        ax.set_xticks(x, [LABEL[n] for n in INTRINSIC])
        ax.set_title(what + " over the 25 points", loc="left", color=TEXT, fontsize=11)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    return fig


def main():
    pts = load_points()
    save(sigma_bias_figure(pts), "extra_dof", "sigma_bias.png", dpi=120)
    save(conditioning_figure(pts), "extra_dof", "conditioning.png")
    save(summary_figure(pts), "extra_dof", "summary.png")


if __name__ == "__main__":
    main()
