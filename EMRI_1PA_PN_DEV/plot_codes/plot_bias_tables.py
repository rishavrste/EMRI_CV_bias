"""1D and nD bias tables, as figures and as markdown.

    bias_1D/snr{SNR}/table_1D.png : bias / sigma per point (rows) and parameter (columns): 0PA,
                                    0PA + PN (own sigma), 0PA + PN in 0PA sigma
    bias_nD/{with,no}_phases/table_nD_snr{SNR}.png : nD bias on the (a, e0) grid: 0PA, 0PA + PN
                                    over the physical parameters (C_p, C_e marginalised), and over
                                    (C_p, C_e); no_phases: Phi_phi0, Phi_r0 marginalised too
    tables/bias_tables_snr{SNR}.md : the same numbers, copy-ready
all under plots/.

Run:  python plot_bias_tables.py [--snr 20]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from bias_data import ALL, ND_SETS, PHYS, load_bias
from common import (DIV, FLAG_NOTE, flag_tag, LABEL, SEQ, SURFACE, TEXT, TEXT_2, grid, ink_on,
                    mark_flags, out_file, style_axes)

CASES_1D = [("z_0pa", "0PA, own σ", PHYS), ("z_pn", "0PA + PN, own σ", ALL),
            ("z_pn_0pa", "0PA + PN, in 0PA σ", PHYS)]
VARIANTS = {"": ("with_phases", "with phases"), "_np": ("no_phases", "phases marginalised")}


def cases_nd(variant):
    n = {k: len(v) for k, v in ND_SETS[variant].items()}
    return [("0pa", f"0PA, {n['0pa']} parameters"),
            ("pn_phys", f"0PA + PN, {n['pn_phys']} physical (C marginalised)"),
            ("pn_C", "0PA + PN, (C_p, C_e) only")]
Z_LIM = 5.0             # colour range of the 1D table, |bias/sigma|


def row_label(p):
    return f"idx{p['idx']}  a={p['a']:+.1f} e0={p['e0']:.1f} {flag_tag(p)}"


# --- 1D --------------------------------------------------------------------
def table_1d(ax, pts, key, title, names):
    z = np.array([[p[key][n] for n in names] for p in pts])
    norm = TwoSlopeNorm(0.0, -Z_LIM, Z_LIM)
    ax.imshow(np.clip(z, -Z_LIM, Z_LIM), cmap=DIV, norm=norm, aspect="auto")
    for (i, j), v in np.ndenumerate(z):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                color=ink_on(DIV(norm(np.clip(v, -Z_LIM, Z_LIM)))))
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names])
    ax.set_yticks(range(len(pts)), [row_label(p) for p in pts], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=22)
    ax.tick_params(colors=TEXT_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def figure_1d(pts, snr):
    fig, axes = plt.subplots(1, 3, figsize=(21, 11), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[len(c[2]) for c in CASES_1D]))
    for ax, (key, title, names) in zip(axes, CASES_1D):
        table_1d(ax, pts, key, title, names)
    fig.suptitle(f"1D bias, bias / σ at SNR {snr:g} (colour clipped at ±{Z_LIM:g})", x=0.01,
                 ha="left", fontsize=13)
    fig.text(0.01, 0.005, FLAG_NOTE, fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


# --- nD --------------------------------------------------------------------
def nd_label(p, key):
    return f"D {p[f'D_{key}']:.2f}\n{p[f'n_{key}']:.1f}σ"


def panel_nd(ax, fig, pts, key, title, vmax):
    D = grid(pts, f"D_{key}")
    labels = grid(pts, lambda p: nd_label(p, key))
    norm = Normalize(0.0, vmax)
    im = ax.imshow(D.astype(float), origin="lower", cmap=SEQ, norm=norm, aspect="auto")
    for (y, x), text in np.ndenumerate(labels):
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=ink_on(SEQ(norm(D[y, x]))))
    style_axes(ax, title)
    ax.set_facecolor(SURFACE)
    mark_flags(ax, pts)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("D = sqrt(bᵀ Σ⁻¹ b)", color=TEXT_2)
    cb.outline.set_visible(False)


def figure_nd(pts, snr, variant):
    cases = cases_nd(variant)
    vmax = max(p[f"D_{k}{variant}"] for p in pts for k, _ in cases)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, cases):
        panel_nd(ax, fig, pts, key + variant, title, vmax)
    fig.suptitle(f"nD bias at SNR {snr:g}, {VARIANTS[variant][1]}: D and its Gaussian-equivalent "
                 "significance", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01, FLAG_NOTE, fontsize=8.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


# --- markdown --------------------------------------------------------------
def md_1d(pts, key, title, names):
    out = [f"### {title}", "", "| idx | a | e0 | flag | " + " | ".join(LABEL[n] for n in names) + " |",
           "|" + "---|" * (4 + len(names))]
    out += [f"| {p['idx']} | {p['a']:+.1f} | {p['e0']:.1f} | {flag_tag(p)} | "
            + " | ".join(f"{p[key][n]:+.3f}" for n in names) + " |" for p in pts]
    return out + [""]


def md_nd(pts, variant):
    n = {k: len(v) for k, v in ND_SETS[variant].items()}
    keys = [k + variant for k in ("0pa", "pn_all", "pn_phys", "pn_C")]
    out = [f"### nD bias, {VARIANTS[variant][1]}", "",
           "D = sqrt(b^T Sigma_S^-1 b) (Gaussian-equivalent significance); cond^ = "
           "Jacobi-rescaled condition number of Gamma.", "",
           f"| idx | a | e0 | flag | gain | D 0PA ({n['0pa']}) | D PN all ({n['pn_all']}) "
           f"| D PN phys ({n['pn_phys']}) | D PN C (2) | cond^ 0PA | cond^ PN |", "|" + "---|" * 11]
    for p in pts:
        cells = " | ".join(f"{p[f'D_{k}']:.2f} ({p[f'n_{k}']:.1f}σ)" for k in keys)
        out.append(f"| {p['idx']} | {p['a']:+.1f} | {p['e0']:.1f} | {flag_tag(p)} | {p['gain']:.2f} "
                   f"| {cells} | {p['cond_0pa']:.1e} | {p['cond_pn']:.1e} |")
    return out + [""]


def markdown(pts, snr):
    lines = [f"# EMRI 0PA vs 0PA + PN: bias at SNR {snr:g}", "",
             "Generated by plot_codes/plot_bias_tables.py. bias = best fit - injection "
             "(phases wrapped); sigma from the Fisher matrix at each best fit (fisher_at_best.py). "
             "Flags: " + FLAG_NOTE, ""]
    for key, title, names in CASES_1D:
        lines += md_1d(pts, key, f"1D bias / sigma: {title}", names)
    for variant in VARIANTS:
        lines += md_nd(pts, variant)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=20.0)
    args = ap.parse_args()
    pts = load_bias(args.snr)
    tag = f"snr{args.snr:g}"
    figs = [(out_file("bias_1D", tag, "table_1D.png"), figure_1d(pts, args.snr))]
    figs += [(out_file("bias_nD", VARIANTS[v][0], f"table_nD_{tag}.png"), figure_nd(pts, args.snr, v))
             for v in VARIANTS]
    for out, fig in figs:
        fig.savefig(out, dpi=160, facecolor=SURFACE)
        print(f"-> {out}")
    out = out_file("tables", f"bias_tables_{tag}.md")
    out.write_text(markdown(pts, args.snr))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
