"""Per-parameter r-factor and sigma growth of 0PA + REL5_1 against 0PA, every grid point and physical parameter.

    r_i     = |theta_0PA,i - theta_inj,i| / |theta_REL5,i - theta_inj,i|   (phases wrapped)
              r > 1: REL5_1 sits closer to the injection in that parameter
    s_i     = sigma_REL5,i / sigma_0PA,i   marginal Fisher sigma at each template's own best fit, SNR 20
              (REL5_1: the five coefficients marginalised); s > 1: REL5_1 measures the parameter worse
    -> plots/tables/r_sigma.md       both tables, copy-ready, with median and count rows
       plots/fit/r_table.png         r as a point x parameter table
       plots/extra_dof/sigma_table.png  s as a point x parameter table

Run:  python r_sigma_table.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from common import DIV, LABEL, SEQ, SURFACE, TEXT, TEXT_2, ink_on, out_file, save
from data import PHYS, load_points


def r_factor(p, n):
    return abs(p["fits"]["0pa"]["bias"][n]) / abs(p["fits"]["rel5"]["bias"][n])


def sigma_growth(p, n):
    return p["fits"]["rel5"]["sigma"][n] / p["fits"]["0pa"]["sigma"][n]


def matrix(pts, fn):
    return np.array([[fn(p, n) for n in PHYS] for p in pts])


def row_label(p):
    return f"idx{p['idx']}  a={p['a']:+.1f} e0={p['e0']:.1f}"


def table_figure(pts, M, cmap, norm, title, cbar):
    fig, ax = plt.subplots(figsize=(12, 13), layout="constrained", facecolor=SURFACE)
    im = ax.imshow(np.log10(M), cmap=cmap, norm=norm, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=7.5,
                color=ink_on(cmap(norm(np.log10(v)))))
    ax.set_xticks(range(len(PHYS)), [LABEL[n] for n in PHYS])
    ax.set_yticks(range(len(pts)), [row_label(p) for p in pts], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(colors=TEXT_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(PHYS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pts)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar, color=TEXT_2)
    cb.outline.set_visible(False)
    return fig


def markdown(pts, M, title, count_label, count_fn):
    lines = [f"### {title}", "", "| point | a | e0 | " + " | ".join(PHYS) + " |",
             "|---|---|---|" + "---|" * len(PHYS)]
    lines += [f"| idx{p['idx']} | {p['a']:+.1f} | {p['e0']:.1f} | " + " | ".join(f"{v:.3g}" for v in row) + " |"
              for p, row in zip(pts, M)]
    lines.append("| median | | | " + " | ".join(f"{v:.3g}" for v in np.median(M, axis=0)) + " |")
    lines.append(f"| {count_label} | | | " + " | ".join(str(int(count_fn(M[:, j]))) for j in range(len(PHYS))) + " |")
    return lines


def main():
    pts = load_points()
    R, S = matrix(pts, r_factor), matrix(pts, sigma_growth)
    lim = np.abs(np.log10(R)).max()
    save(table_figure(pts, R, DIV.reversed(), TwoSlopeNorm(0.0, -lim, lim),
                      "r = |θ_0PA − θ_inj| / |θ_REL5 − θ_inj|   (r > 1, blue: REL5_1 closer to the injection)",
                      "log10 r"), "fit", "r_table.png")
    save(table_figure(pts, S, SEQ, Normalize(0, np.log10(S.max())),
                      "σ_REL5 / σ_0PA  (marginal Fisher σ at SNR 20, coefficients marginalised)",
                      "log10 σ ratio"), "extra_dof", "sigma_table.png")
    text = "\n".join(
        markdown(pts, R, "r = |θ_0PA − θ_inj| / |θ_REL5 − θ_inj|  (r > 1: REL5_1 closer)", "r > 1 (of 25)",
                 lambda c: (c > 1).sum())
        + [""] + markdown(pts, S, "σ_REL5 / σ_0PA  (marginal, SNR 20)", "σ ratio > 1 (of 25)",
                          lambda c: (c > 1).sum())) + "\n"
    out = out_file("tables", "r_sigma.md")
    out.write_text(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
