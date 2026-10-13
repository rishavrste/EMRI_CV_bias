"""Plot the whole circular CV population: every source, converged or not.

Three panels, each answering one question:
  (a) did the climb find the signal?        1 - overlap against p0
  (b) how big is the systematic bias?       |bias| / sigma, parameter by parameter
  (c) is the deviation itself recovered?    C_p +- sigma, 0PA+PN only

Non-converged sources are drawn hollow rather than dropped, so the failures stay
visible next to the successes instead of silently thinning the population.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_env_cv.json")
OUT = os.path.join(HERE, "cv_population.png")

OV_CONVERGED = 0.99           # what counts as "the climb found the signal"
C_0PA, C_PN = "#1f6feb", "#d1650f"   # one hue per model, held fixed everywhere


def load(path=RESULTS):
    with open(path) as f:
        return json.load(f)


def converged_mask(sources):
    """A source is judged on its 0PA climb, so both models share one verdict."""
    return np.array([s["fit_0pa"]["ov_final"] > OV_CONVERGED for s in sources])


def _scatter_split(ax, x, y, ok, color, label, marker="o"):
    """Filled = converged, hollow = not.  Identity is marker+color, never color alone."""
    ax.scatter(x[ok], y[ok], s=34, marker=marker, color=color, label=label, zorder=3)
    ax.scatter(x[~ok], y[~ok], s=44, marker=marker, facecolors="none",
               edgecolors=color, linewidths=1.4, zorder=3)


def panel_overlap(ax, src, ok):
    """1 - overlap against p0: the convergence cliff, if there is one."""
    p0 = np.array([s["p0"] for s in src])
    for key, c, lab in (("fit_0pa", C_0PA, "0PA"), ("fit_pn", C_PN, "0PA+PN")):
        d = 1.0 - np.array([s[key]["ov_final"] for s in src])
        _scatter_split(ax, p0, np.clip(d, 1e-12, None), ok, c, lab,
                       "o" if key == "fit_0pa" else "s")
    ax.set_yscale("log")
    ax.axhline(1 - OV_CONVERGED, color="0.55", lw=1, ls="--", zorder=1)
    ax.text(ax.get_xlim()[1], 1 - OV_CONVERGED, " converged", va="center",
            color="0.35", fontsize=8)
    ax.set_xlabel(r"$p_0$")
    ax.set_ylabel(r"$1-\mathcal{O}$")
    ax.set_title("(a) climb quality", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def panel_bias(ax, src, ok, params_0pa, params_pn, drop=("e0",)):
    """|bias|/sigma per parameter.

    e0 is dropped by default: the injection sits exactly on the e0 = 0 boundary, where
    the Fisher sigma is unstable (it ranges over ten orders of magnitude across the
    population), so the ratio is meaningless even though the absolute bias is < 5e-4.
    """
    # the two fits carry different-length vectors (PN appends C_p), so each model
    # resolves the shared parameter names against its own column list.
    keep = [p for p in params_0pa if p not in drop]
    x = np.arange(len(keep))
    for key, cols, c, lab, off in (("fit_0pa", params_0pa, C_0PA, "0PA", -0.12),
                                   ("fit_pn", params_pn, C_PN, "0PA+PN", 0.12)):
        idx = [cols.index(p) for p in keep]
        b = np.abs(np.array([[s[key]["bias_over_sigma"][i] for i in idx] for s in src]))
        for j in range(len(keep)):
            _scatter_split(ax, np.full(len(src), x[j] + off), np.clip(b[:, j], 1e-4, None),
                           ok, c, lab if j == 0 else None,
                           "o" if key == "fit_0pa" else "s")
    ax.set_yscale("log")
    ax.axhline(1.0, color="0.55", lw=1, ls="--", zorder=1)
    ax.text(x[-1] + 0.4, 1.0, r" $1\sigma$", va="center", color="0.35", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(keep, rotation=45, ha="right")
    ax.set_ylabel(r"$|\Delta\theta|/\sigma$")
    ax.set_title("(b) systematic bias  (e0 omitted: boundary Fisher)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def panel_deviation(ax, src, ok, params_pn):
    """C_p recovered by the 0PA+PN model.  The injected value is zero by construction."""
    i = params_pn.index("C_p")
    p0 = np.array([s["p0"] for s in src])
    val = np.array([s["fit_pn"]["params"][i] for s in src])
    sig = np.array([s["fit_pn"]["sigma"][i] for s in src])
    ax.errorbar(p0[ok], val[ok], yerr=sig[ok], fmt="s", ms=5, color=C_PN,
                ecolor=C_PN, elinewidth=1.2, capsize=2, label="0PA+PN", zorder=3)
    ax.errorbar(p0[~ok], val[~ok], yerr=sig[~ok], fmt="s", ms=6, mfc="none",
                mec=C_PN, ecolor=C_PN, elinewidth=1.2, capsize=2, zorder=3)
    ax.axhline(0.0, color="0.55", lw=1, ls="--", zorder=1)
    ax.set_xlabel(r"$p_0$")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(r"(c) deviation parameter (injected $C_p=0$)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def main():
    d = load()
    src = d["sources"]
    ok = converged_mask(src)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panel_overlap(axes[0], src, ok)
    panel_bias(axes[1], src, ok, d["params_0pa"], d["params_pn"])
    panel_deviation(axes[2], src, ok, d["params_pn"])
    n_bad = int((~ok).sum())
    fig.suptitle("Cutler-Vallisneri bias, circular EMRI population "
                 f"({len(src)} sources, {n_bad} non-converged shown hollow)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=160)
    print(f"[saved] {OUT}   {len(src)} sources, {n_bad} non-converged")


if __name__ == "__main__":
    main()
