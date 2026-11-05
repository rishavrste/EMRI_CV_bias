"""Sky localization: 2-d marginal (qS, phiS) contours of every source, 0PA and 0PA+PN_dev.

The injected sky position is the same for all sources, so every contour should sit around it.
Each contour is the Gaussian from that model's own Fisher covariance, marginalized over the
other parameters (C_p included for PN) and centred on that model's best fit.

  plots/sky_contours.png        all sources together, one panel per model, coloured by p0
  plots/sky_contours_grid.png   one panel per source, both models, 90% and 99%
  plots/sky_contours_most_biased.png   the sources 0PA places furthest from the truth, one
                                       panel each, against 0PA+PN_dev
  plots/sky_offsets_normalised.png     every best fit in units of its own sky uncertainty, so
                                       the credible contours become fixed circles
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

import plot_results as PR

SKY = ("qS", "phiS")
LEVEL = 0.99                             # the level drawn in the overlay
GRID_LEVELS = ((0.99, "-"), (0.90, "--"))
GRID_COLUMNS = 8
N_MOST_BIASED = 4                        # D falls from 1.96 to 1.27 after the fourth source
LABELS = {"0PA": "0PA", "0PA+PN": "0PA+PN_dev"}
P0_MAP = LinearSegmentedColormap.from_list("p0", plt.cm.Blues(np.linspace(0.35, 1.0, 64)))


def sky_block(model_rec):
    """Best fit, truth and marginal covariance of the sky angles for one model."""
    i = [model_rec["params"].index(n) for n in SKY]
    return (np.array(model_rec["fit"])[i], np.array(model_rec["truth"])[i],
            np.array(model_rec["cov"])[np.ix_(i, i)])


def ellipse(centre, cov, p):
    """Boundary of the p credible region of a 2-d Gaussian, as (2, n) points."""
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    return centre[:, None] + PR.threshold_nd(p, 2) * np.linalg.cholesky(cov) @ np.array(
        [np.cos(t), np.sin(t)])


def truth_of(recs):
    return sky_block(recs[0]["models"]["0PA"])[1]


def sky_limits(recs, pad=1.08):
    """Symmetric limits about the truth that contain every contour of both models."""
    truth = truth_of(recs)
    reach = np.zeros(2)
    for r in recs:
        for model, _, _ in PR.MODELS:
            fit, _, cov = sky_block(r["models"][model])
            reach = np.maximum(reach, np.abs(ellipse(fit, cov, LEVEL) - truth[:, None]).max(axis=1))
    return [(t - pad * d, t + pad * d) for t, d in zip(truth, reach)]


def mark_truth(ax, truth, size=110):
    ax.axvline(truth[0], color="0.55", lw=0.8, ls=":", zorder=1)
    ax.axhline(truth[1], color="0.55", lw=0.8, ls=":", zorder=1)
    ax.scatter(*truth, marker="*", s=size, color="black", edgecolors="white", linewidths=0.5,
               zorder=5, label="injected")


def corner_label(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="0.25")


def overlay_panel(recs, path):
    limits = sky_limits(recs)
    norm = Normalize(min(r["p0"] for r in recs), max(r["p0"] for r in recs))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (model, _, _) in zip(axes, PR.MODELS):
        for r in recs:                   # sorted by p0, so the most biased sources land on top
            fit, _, cov = sky_block(r["models"][model])
            ax.plot(*ellipse(fit, cov, LEVEL), color=P0_MAP(norm(r["p0"])), lw=1.0, zorder=3)
        mark_truth(ax, truth_of(recs))
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_xlabel("qS", fontsize=13)
        corner_label(ax, f"{LABELS[model]}, {LEVEL:.0%} contours")
        ax.grid(True, color="0.88", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("phiS", fontsize=13)
    axes[0].legend(fontsize=11, frameon=False, loc="upper right")
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=P0_MAP), ax=axes, pad=0.015)
    bar.set_label("p0", fontsize=13)
    PR.save(fig, path)


def source_panel(ax, rec, limits):
    for model, colour, _ in PR.MODELS:
        fit, _, cov = sky_block(rec["models"][model])
        for p, style in GRID_LEVELS:
            ax.plot(*ellipse(fit, cov, p), color=colour, lw=0.9, ls=style, zorder=3)
    mark_truth(ax, truth_of([rec]), size=45)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.text(0.04, 0.95, f"p0 = {rec['p0']:.2f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=8)
    ax.tick_params(labelsize=7)


def grid_legend(ax):
    """The spare cell carries the key, so no source panel loses space to it."""
    ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=1.4, label=LABELS[m]) for m, c, _ in PR.MODELS]
    handles += [plt.Line2D([], [], color="0.3", lw=1.2, ls=s, label=f"{p:.0%}")
                for p, s in GRID_LEVELS]
    handles.append(plt.Line2D([], [], color="black", marker="*", ls="", ms=9, label="injected"))
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="center")


def grid_panel(recs, path):
    limits = sky_limits(recs)
    rows = -(-(len(recs) + 1) // GRID_COLUMNS)
    fig, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(2.0 * GRID_COLUMNS, 2.1 * rows),
                             sharex=True, sharey=True, layout="constrained")
    cells = axes.ravel()
    for ax, rec in zip(cells, recs):
        source_panel(ax, rec, limits)
    grid_legend(cells[len(recs)])
    for ax in cells[len(recs) + 1:]:
        ax.axis("off")
    fig.supxlabel("qS", fontsize=13)
    fig.supylabel("phiS", fontsize=13)
    PR.save(fig, path)


def most_biased(recs):
    """The sources whose 0PA best fit sits furthest from the truth, worst first."""
    ranked = sorted(recs, key=lambda r: sky_offset(r["models"]["0PA"]), reverse=True)
    return ranked[:N_MOST_BIASED]


def pair_panel(ax, rec, limits):
    """One source: both models' filled contours and best fits around the truth."""
    for model, colour, marker in PR.MODELS:
        fit, _, cov = sky_block(rec["models"][model])
        ax.fill(*ellipse(fit, cov, LEVEL), facecolor=colour, alpha=0.15, zorder=2)
        ax.plot(*ellipse(fit, cov, LEVEL), color=colour, lw=1.6, zorder=3,
                label=f"{LABELS[model]}, {LEVEL:.0%}")
        ax.scatter(*fit, marker=marker, s=26, color=colour, edgecolors="white", linewidths=0.5,
                   zorder=4)
    mark_truth(ax, truth_of([rec]))
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    corner_label(ax, f"p0 = {rec['p0']:.2f}")


def most_biased_panel(recs, path):
    top = most_biased(recs)
    limits = sky_limits(top)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), sharex=True, sharey=True,
                             layout="constrained")
    for ax, rec in zip(axes.ravel(), top):
        pair_panel(ax, rec, limits)
    fig.supxlabel("qS", fontsize=13)
    fig.supylabel("phiS", fontsize=13)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, fontsize=11, frameon=False)
    PR.save(fig, path)


def normalised_offset(model_rec):
    """Best fit minus truth, whitened by the sky covariance: its length is sky_offset."""
    fit, truth, cov = sky_block(model_rec)
    return np.linalg.solve(np.linalg.cholesky(cov), fit - truth)


def draw_circles(ax):
    t = np.linspace(0.0, 2.0 * np.pi, 300)
    for p, style in GRID_LEVELS:
        r = PR.threshold_nd(p, 2)
        ax.plot(r * np.cos(t), r * np.sin(t), color="0.45", lw=1.1, ls=style, zorder=2)
        ax.text(0.0, r, f"{p:.0%}", ha="center", va="center", fontsize=8.5, color="0.25", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


def draw_links(ax, top):
    """Join each of the worst sources' two points, and name it beside its 0PA point."""
    for rec in top:
        a, b = (normalised_offset(rec["models"][m]) for m in ("0PA", "0PA+PN"))
        ax.plot(*zip(a, b), color="0.6", lw=0.8, zorder=2)
        away = 9.0 * a / np.linalg.norm(a)
        ax.annotate(f"p0 = {rec['p0']:.2f}", a, xytext=away, textcoords="offset points",
                    ha="left" if a[0] > 0 else "right", va="center", fontsize=8.5, color="0.25")


def normalised_panel(recs, path):
    fig, ax = plt.subplots(figsize=(6.6, 6.2), layout="constrained")
    draw_circles(ax)
    draw_links(ax, most_biased(recs))
    for model, colour, marker in PR.MODELS:
        pts = np.array([normalised_offset(r["models"][model]) for r in recs])
        ax.scatter(*pts.T, marker=marker, s=34, color=colour, edgecolors="white", linewidths=0.5,
                   zorder=3, label=LABELS[model])
    ax.scatter(0.0, 0.0, marker="*", s=90, color="black", edgecolors="white", linewidths=0.5,
               zorder=5, label="injected")
    reach = 1.2 * max(np.linalg.norm(normalised_offset(r["models"]["0PA"])) for r in recs)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_xlabel("qS offset / sigma", fontsize=13)
    ax.set_ylabel("phiS offset / sigma, decorrelated from qS", fontsize=13)
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    PR.save(fig, path)


def sky_area(model_rec, p):
    """Area of the p credible region in square degrees, with the sin(qS) solid-angle factor."""
    fit, _, cov = sky_block(model_rec)
    return (np.pi * PR.threshold_nd(p, 2) ** 2 * np.sin(fit[0]) * np.sqrt(np.linalg.det(cov))
            * np.degrees(1.0) ** 2)


def sky_offset(model_rec):
    """Distance of the truth from the best fit, in units of the 2-d marginal Gaussian."""
    fit, truth, cov = sky_block(model_rec)
    return float(np.sqrt((fit - truth) @ np.linalg.solve(cov, fit - truth)))


def print_summary(recs):
    print(f"{'model':>12} {'median area':>12} {'range':>17} {'median D':>9} {'max D':>6} "
          f"{'truth outside 90%':>18} {'99%':>4}   (areas at {LEVEL:.0%}, deg^2)")
    for model, _, _ in PR.MODELS:
        area = np.array([sky_area(r["models"][model], LEVEL) for r in recs])
        d = np.array([sky_offset(r["models"][model]) for r in recs])
        print(f"{LABELS[model]:>12} {np.median(area):>12.3f} {area.min():>8.3f} - "
              f"{area.max():>6.3f} {np.median(d):>9.3f} {d.max():>6.2f} "
              f"{int(np.sum(d > PR.threshold_nd(0.90, 2))):>18d} "
              f"{int(np.sum(d > PR.threshold_nd(0.99, 2))):>4d}")
    for r in recs:
        d = {m: sky_offset(r["models"][m]) for m, _, _ in PR.MODELS}
        if max(d.values()) > PR.threshold_nd(0.90, 2):
            print(f"  {r['name']:>24} p0 {r['p0']:6.2f}  D 0PA {d['0PA']:.2f}, "
                  f"0PA+PN_dev {d['0PA+PN']:.2f}")


def main():
    recs = PR.load_records()
    overlay_panel(recs, PR.out_path("", "sky_contours.png"))
    grid_panel(recs, PR.out_path("", "sky_contours_grid.png"))
    most_biased_panel(recs, PR.out_path("", "sky_contours_most_biased.png"))
    normalised_panel(recs, PR.out_path("", "sky_offsets_normalised.png"))
    print_summary(recs)


if __name__ == "__main__":
    main()
