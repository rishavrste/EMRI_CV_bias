"""Fit maps on the (a, e0) grid.

    fit/mismatch.png      1 - O of 0PA, 0PA + PN, 0PA + REL5_1 (shared log scale), and the REL5_1 gain
                          (1 - O_ref) / (1 - O_REL5) over 0PA and over PN
    fit/coefficients.png  the five REL5_1 coefficients at the best fit (diverging around 0)
    fit/r_factor.png      r = |b_REL5| / |b_ref| per physical parameter: worst and best over the nine,
                          and how many of the nine (and of the five intrinsic) have r < 1, ref = 0PA and PN
-> EMRI_REL5_LM/plots/

Run:  python plot_fit_maps.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm, TwoSlopeNorm

from common import DIV, LABEL, SEQ, TITLE, heat, save
from data import DEV, PHYS, TAGS, grid, load_points

INTRINSIC = PHYS[:5]
R_LOG_LIM = 2.0


def theta(p, name):
    """REL5_1 best-fit value of a coefficient (= bias, the injection has none)."""
    return p["fits"]["rel5"]["bias"][name]


def fmt(v):
    return f"{v:.2g}"


def mismatch_figure(pts):
    maps = {t: grid(pts, lambda p, t=t: p["fits"][t]["mismatch"]) for t in TAGS}
    norm = LogNorm(min(m.min() for m in maps.values()), max(m.max() for m in maps.values()))
    fig, axes = plt.subplots(1, 5, figsize=(29, 5.2), layout="constrained")
    for ax, (t, name) in zip(axes, TAGS.items()):
        heat(ax, fig, maps[t], np.vectorize(fmt)(maps[t]), SEQ, norm, f"1 − O, {name}", "1 − O")
    for ax, ref in zip(axes[3:], ("0pa", "pn")):
        g = maps[ref] / maps["rel5"]
        lim = np.abs(np.log10(g)).max()
        heat(ax, fig, np.log10(g), np.vectorize(lambda v: f"{v:.3g}×")(g), DIV,
             TwoSlopeNorm(0.0, -lim, lim), f"REL5_1 gain over {TAGS[ref]}",
             "log10 (1 − O_ref) / (1 − O_REL5)")
    fig.suptitle(TITLE, x=0.01, ha="left")
    return fig


def coefficient_figure(pts):
    fig, axes = plt.subplots(1, len(DEV), figsize=(5.8 * len(DEV), 5.2), layout="constrained")
    for ax, name in zip(axes, DEV):
        g = grid(pts, lambda p, n=name: theta(p, n))
        lim = np.abs(g).max()
        heat(ax, fig, g, np.vectorize(lambda v: f"{v:+.3g}")(g), DIV,
             SymLogNorm(1.0, vmin=-lim, vmax=lim), f"{LABEL[name]} (REL5_1 best fit)", LABEL[name])
    fig.suptitle("REL5_1:  X += η X_0PA (c_X v + K v⁴),  v = p^-1/2", x=0.01, ha="left")
    return fig


def r_values(p, ref):
    b5, b0 = p["fits"]["rel5"]["bias"], p["fits"][ref]["bias"]
    return {n: abs(b5[n]) / abs(b0[n]) for n in PHYS}


def r_figure(pts):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), layout="constrained")
    norm = TwoSlopeNorm(0.0, -R_LOG_LIM, R_LOG_LIM)
    for row, ref in zip(axes, ("0pa", "pn")):
        for ax, (fn, word) in zip(row[:2], ((max, "worst"), (min, "best"))):
            names = np.full((5, 5), "", dtype=object)
            r = grid(pts, lambda p: fn(r_values(p, ref).values()))
            for p in pts:
                rv = r_values(p, ref)
                names[[0.1, 0.2, 0.3, 0.4, 0.5].index(p["e0"]),
                      [-0.9, -0.5, 0.0, 0.5, 0.9].index(p["a"])] = LABEL[fn(rv, key=rv.get)]
            labels = np.vectorize(lambda v, n: f"{n}\n{v:.2g}")(r, names)
            heat(ax, fig, np.clip(np.log10(r), -R_LOG_LIM, R_LOG_LIM), labels, DIV, norm,
                 f"{word} r vs {TAGS[ref]}", "log10 r  (r < 1: REL5_1 closer)")
        n9 = grid(pts, lambda p: sum(v < 1 for v in r_values(p, ref).values()))
        n5 = grid(pts, lambda p: sum(r_values(p, ref)[n] < 1 for n in INTRINSIC))
        labels = np.vectorize(lambda a, b: f"{int(b)}/5 int.\n{int(a)}/9 phys.")(n9, n5)
        heat(row[2], fig, n9, labels, SEQ, Normalize(0, 9),
             f"parameters with smaller |bias| than {TAGS[ref]}", "count of 9")
    fig.suptitle("r = |θ_REL5 − θ_inj| / |θ_ref − θ_inj|  (phases wrapped)", x=0.01, ha="left")
    return fig


def main():
    pts = load_points()
    save(mismatch_figure(pts), "fit", "mismatch.png")
    save(coefficient_figure(pts), "fit", "coefficients.png")
    save(r_figure(pts), "fit", "r_factor.png")


if __name__ == "__main__":
    main()
