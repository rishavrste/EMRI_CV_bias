"""Corner plot of the PARIS posterior against the Fisher Gaussian at the best fit.

Every axis is (theta - theta_best) / sigma_F, so the Fisher prediction is the same
unit-variance ellipse family in every panel and any PE deviation from it reads directly.

    python plot_pe_corner.py --template 0pa       # or pn
"""
import argparse
from pathlib import Path

import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# same as pe_sampling.PE_DIR, without importing the waveform stack
PE_DIR = Path(__file__).resolve().parents[1] / "results" / "pe"
PE_COLOR, FISHER_COLOR, INJ_COLOR = "#2a78d6", "#eb6834", "#555555"
SIGMA_LEVELS = (1.0, 2.0)
HALF_RANGE = 4.0                       # plotted range in units of sigma_F, about the best fit
LABELS = {"m1": "m_1", "m2": "m_2", "a": "a", "p0": "p_0", "e0": "e_0", "qS": r"\theta_S",
          "phiS": r"\phi_S", "Phi_phi0": r"\Phi_{\phi,0}", "Phi_r0": r"\Phi_{r,0}",
          "C_p": "C_p", "C_e": "C_e"}


def run_dir(template, point, tag):
    return PE_DIR / f"{template}_idx{point}{'_' + tag if tag else ''}"


def load(template, point, tag):
    d = np.load(run_dir(template, point, tag) / "pe_results.npz")
    return {k: d[k] for k in d.files}


def standardise(d):
    """Samples, injection and Fisher correlation in sigma_F units about the best fit."""
    sigma = np.sqrt(np.diag(d["cov_fisher"]))
    x = (d["samples"] - d["theta_best"]) / sigma
    x_inj = (d["theta_inj"] - d["theta_best"]) / sigma
    return x, x_inj, d["cov_fisher"] / np.outer(sigma, sigma)


def ellipse(cov2, k, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return k * np.linalg.cholesky(cov2) @ np.vstack([np.cos(t), np.sin(t)])


def overlay_fisher(fig, rho, ndim):
    axes = np.array(fig.axes).reshape(ndim, ndim)
    u = np.linspace(-HALF_RANGE, HALF_RANGE, 400)
    for i in range(ndim):
        ax = axes[i, i]
        # corner normalises 1D histograms to peak 1 when density is off; match that
        ax.plot(u, np.exp(-0.5 * u ** 2) * ax.get_ylim()[1] / 1.05, color=FISHER_COLOR, lw=1.5, ls="--")
        for j in range(i):
            cov2 = rho[np.ix_([j, i], [j, i])]
            for k in SIGMA_LEVELS:
                ex, ey = ellipse(cov2, k)
                axes[i, j].plot(ex, ey, color=FISHER_COLOR, lw=1.5, ls="--")


def plot(template, point, tag=""):
    d = load(template, point, tag)
    names = [str(n) for n in d["names"]]
    x, x_inj, rho = standardise(d)
    ndim = len(names)
    w = d["weights"] / d["weights"].sum()
    ess = 1.0 / np.sum(w ** 2)

    fig = corner.corner(
        x, weights=w, labels=[rf"$\Delta {LABELS.get(n, n)}/\sigma_F$" for n in names],
        range=[(-HALF_RANGE, HALF_RANGE)] * ndim, bins=30, smooth=1.0, smooth1d=1.0,
        levels=[1 - np.exp(-0.5 * k ** 2) for k in SIGMA_LEVELS], color=PE_COLOR,
        plot_datapoints=False, fill_contours=True, truths=x_inj, truth_color=INJ_COLOR,
        label_kwargs={"fontsize": 13}, hist_kwargs={"lw": 1.5})
    overlay_fisher(fig, rho, ndim)

    handles = [plt.Line2D([], [], color=PE_COLOR, lw=2, label=f"PARIS posterior (ESS {ess:.0f})"),
               plt.Line2D([], [], color=FISHER_COLOR, lw=1.5, ls="--",
                          label="Fisher Gaussian at best fit (1, 2 sigma)"),
               plt.Line2D([], [], color=INJ_COLOR, lw=1.5, label="1PA injection")]
    title = {"0pa": "0PA template vs 1PA signal", "pn": "0PA + PN deviation template vs 1PA signal"}
    fig.legend(handles=handles, loc="upper right", fontsize=16, frameon=False,
               title=f"EMRI idx{point}, SNR 20: {title.get(template, template)}", title_fontsize=17)

    out = run_dir(template, point, tag) / f"corner_{template}_idx{point}{'_' + tag if tag else ''}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", required=True, choices=["0pa", "pn"])
    ap.add_argument("--point", default="24")
    ap.add_argument("--tag", default="", help="output folder suffix, as in pe_sampling.py")
    args = ap.parse_args()
    plot(args.template, args.point, args.tag)


if __name__ == "__main__":
    main()
