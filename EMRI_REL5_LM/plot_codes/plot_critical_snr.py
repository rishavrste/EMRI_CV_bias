"""Critical SNR on the (a, e0) grid: 0PA, 0PA + PN, 0PA + REL5_1, each at its own Fisher (SNR0 = 20).

    nD : rho_crit = SNR0 D_CL / D,  D^2 = b^T Sigma_S^-1 b over the intrinsic (m1, m2, a, p0, e0),
         every other fitted parameter (sky, phases, deviation coefficients) marginalised; D_CL^2 = chi^2_5 at CL
    1D : rho_crit = SNR0 z_CL / max_i |b_i / sigma_i| over the same five, z_CL = Phi^-1(1/2 + CL/2)
Below rho_crit the bias is hidden at that CL. A ratio panel gives rho_crit(REL5_1) / rho_crit(ref).
    -> plots/critical_snr/snr_crit_{1D,nD}_{CL}CL.png

Run:  python plot_critical_snr.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy import stats

from common import DIV, SEQ, TITLE, heat, save
from data import TAGS, grid, load_points

SNR0 = 20.0
CLS = (0.90, 0.99)


def crit_nd(f, cl):
    return f["rho_crit"][cl]


def crit_1d(f, cl):
    return SNR0 * stats.norm.ppf(0.5 + cl / 2) / f["max1d"][0]


def figure(pts, kind, cl):
    fn = crit_nd if kind == "nD" else crit_1d
    maps = {t: grid(pts, lambda p, t=t: fn(p["fits"][t], cl)) for t in TAGS}
    norm = LogNorm(min(m.min() for m in maps.values()), max(m.max() for m in maps.values()))
    fig, axes = plt.subplots(1, 5, figsize=(29, 5.2), layout="constrained")
    for ax, (t, name) in zip(axes, TAGS.items()):
        heat(ax, fig, maps[t], np.vectorize(lambda v: f"{v:.3g}")(maps[t]), SEQ, norm, name, "ρ_crit")
    for ax, ref in zip(axes[3:], ("0pa", "pn")):
        g = maps["rel5"] / maps[ref]
        lim = np.abs(np.log10(g)).max()
        heat(ax, fig, np.log10(g), np.vectorize(lambda v: f"{v:.3g}×")(g), DIV, TwoSlopeNorm(0.0, -lim, lim),
             f"ρ_crit REL5_1 / {TAGS[ref]}", "log10 ratio")
    what = ("D over (m1, m2, a, p0, e0), others marginalised" if kind == "nD"
            else "worst |b/σ| over (m1, m2, a, p0, e0), others marginalised")
    fig.suptitle(f"{TITLE}\n{kind} critical SNR at {cl:.0%} CL: {what}", x=0.01, ha="left")
    return fig


def main():
    pts = load_points()
    for kind in ("nD", "1D"):
        for cl in CLS:
            save(figure(pts, kind, cl), "critical_snr", f"snr_crit_{kind}_{cl * 100:.0f}CL.png")
            plt.close("all")


if __name__ == "__main__":
    main()
