"""Figures for the combined population (run_1_second_gen + run_2), 0PA against 0PA+PN.

    python plot_results.py

Reads the per-source records written by refit_fisher.py (results/<run>_idxNN.json: both
models' fits and the order-8 / Ndelta-24 Fisher at each model's own best fit).  Nothing is
recomputed.  Same figures and conventions as ../run_1/plot_results.py, except:

* e0 is fixed in these fits, so there are no `_with_e0` variants;
* the two runs are one population: x axis is p0 everywhere, colour + marker = model,
  every marker filled;
* two figures run_1 did not have: bias_sigma_ratio.png (does PN lower the bias, or only
  inflate sigma?) and the per-parameter histograms.

Written to plots/ (all seven vacuum parameters) and plots/intrinsic/ (m1, m2, a, p0):
  bias_panels.png            per parameter, |bias| and |bias|/sigma against p0
  max_bias[_p90].png         per source, worst single-parameter |bias|/sigma
  nd_bias[_p90].png          per source, n-d bias, marginalized and conditional
  improvement_3panel.png     0PA / PN on mismatch, max |bias|/sigma and n-d bias
  bias_sigma_ratio.png       |bias|_PN / |bias|_0PA and sigma_PN / sigma_0PA per parameter
  hist_bias.png              per parameter, histogram of log10 |bias|, 0PA against PN
  hist_bias_over_sigma.png   per parameter, histogram of log10 |bias|/sigma
  critical_snr[_p90].png     SNR at which the bias reaches the credible threshold, 1-d and n-d
  conditioning.png           cond(Gamma) raw and rescaled, and the inversion residual
plus overlap.png / improvement.png from plot_overlap.py.

Sources whose PN overlap falls below their 0PA overlap are left out (user, 2026-10-19;
RI.pn_below_0pa): only run_1_second_gen idx20.

Reference lines are credible regions (99%, or 90% in *_p90), never a flat 1 sigma.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import chi2, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import plot_overlap as PO
import results_inputs as RI

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "plots")

MODELS = PO.MODELS                       # (name, colour, marker)
C_BIAS, C_SIGMA = "#6f42c1", "#2a9d8f"   # bias_sigma_ratio: |bias| ratio, sigma ratio

CREDIBLE = {"": 0.99, "_p90": 0.90}      # filename suffix -> level
DEV_PARAMS = {"C_p", "C_e"}
EXTRINSIC = {"qS", "phiS", "Phi_phi0"}
BLOCKS = {                               # subfolder -> (parameters excluded, description)
    "": (DEV_PARAMS, "m1, m2, a, p0, qS, phiS, Phi_phi0 -- C_p held out"),
    "intrinsic": (DEV_PARAMS | EXTRINSIC, "m1, m2, a, p0 only -- sky position, phase and "
                                          "C_p held out"),
}


def threshold_1d(p):
    return float(norm.ppf(0.5 + p / 2.0))


def threshold_nd(p, k):
    return float(np.sqrt(chi2.ppf(p, k)))


def out_path(block, name):
    d = os.path.join(OUT, block) if block else OUT
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save(fig, path):
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


# --- records ---------------------------------------------------------------
def load_records():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, "*_idx*.json")))]
    if not recs:
        raise SystemExit(f"no records in {RESULTS}: run refit_fisher.py first")
    recs = RI.kept_in_plots(recs, lambda r: (r["models"]["0PA"]["ov_final"],
                                             r["models"]["0PA+PN"]["ov_final"]))
    return sorted(recs, key=lambda r: r["p0"])


def vac_params(recs):
    return list(recs[0]["models"]["0PA"]["params"])


def value(model_rec, param, key):
    """|model_rec[key]| for one parameter, nan if the model does not fit it."""
    if param not in model_rec["params"]:
        return np.nan
    v = model_rec[key][model_rec["params"].index(param)]
    return abs(float(v)) if v is not None else np.nan


def kept(model_rec, exclude):
    return [i for i, n in enumerate(model_rec["params"]) if n not in exclude]


def max_bias_over_sigma(model_rec, exclude):
    vals = [model_rec["bias_over_sigma"][i] for i in kept(model_rec, exclude)]
    vals = [abs(float(v)) for v in vals if v is not None and np.isfinite(v)]
    return max(vals) if vals else np.nan


def _scaled(M):
    d = np.sqrt(np.abs(np.diag(M)))
    return (None, None) if np.any(d <= 0) else (d, M / np.outer(d, d))


def _quad(M, b):
    d, Mh = _scaled(M)
    if d is None:
        return np.nan
    q = float((b * d) @ Mh @ (b * d))
    return np.sqrt(q) if q > 0 else np.nan


def _quad_inv(M, b):
    d, Mh = _scaled(M)
    if d is None:
        return np.nan
    q = float((b / d) @ np.linalg.solve(Mh, b / d))
    return np.sqrt(q) if q > 0 else np.nan


def nd_bias(model_rec, exclude):
    """sqrt(b^T (C_vv)^-1 b) marginalized and sqrt(b^T G_vv b) conditional, over the kept block."""
    k = kept(model_rec, exclude)
    b = np.array([model_rec["bias"][i] for i in k], dtype=float)
    G = np.array(model_rec["fisher"], dtype=float)[np.ix_(k, k)]
    C = np.array(model_rec["cov"], dtype=float)[np.ix_(k, k)]
    return dict(marginalized=_quad_inv(C, b), conditional=_quad(G, b))


# --- drawing helpers -------------------------------------------------------
def scatter_p0(ax, recs, y, colour, marker, label, s=30):
    """One filled point per source at its p0."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    ax.scatter(np.array([r["p0"] for r in recs])[ok], y[ok], marker=marker, s=s, color=colour,
               edgecolors="white", linewidths=0.5, label=label, zorder=3)


def models_scatter(ax, recs, value_of, s=30):
    for model, colour, marker in MODELS:
        scatter_p0(ax, recs, [value_of(r["models"][model]) for r in recs],
                   colour, marker, model, s=s)


def log_axes(ax, xlabel="p0"):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def plain_ticks(ax):
    """Log y axis with ticks written as plain numbers (2, 3, 0.5), not 2 x 10^0."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="minor", labelsize=7)


def ref_line(ax, ref, text):
    ax.axhline(ref, color="0.55", lw=1.2, ls="--", zorder=1)
    ax.text(0.99, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.35")


def subtitle(recs):
    return (f"{len(recs)} sources, e0 fixed, 2nd-gen TDI; sigma from an order-"
            f"{recs[0]['der_order']} Fisher over {recs[0]['Ndelta']} steps at each model's "
            f"own best fit")


# --- figures ---------------------------------------------------------------
def bias_panels(recs, path, exclude):
    names = [n for n in vac_params(recs) if n not in exclude] + ["C_p"]
    fig, axes = plt.subplots(len(names), 2, figsize=(10.0, 2.0 * len(names)),
                             sharex=True, squeeze=False)
    for row, param in enumerate(names):
        for col, (key, title) in enumerate((("bias", "|bias|"),
                                            ("bias_over_sigma", "|bias| / sigma"))):
            ax = axes[row][col]
            models_scatter(ax, recs, lambda m, k=key, p=param: value(m, p, k), s=22)
            log_axes(ax, "p0" if row == len(names) - 1 else "")
            if col == 1:
                ax.axhline(threshold_1d(0.99), color="0.55", lw=1.2, ls="--", zorder=1)
            if row == 0:
                ax.set_title(title, fontsize=10)
        axes[row][0].set_ylabel("C_p\n(absorbed)" if param == "C_p" else param, fontsize=10)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left", )
    axes[0][1].text(0.99, 0.04, f"dashed: 99% credible interval ({threshold_1d(0.99):.2f} sigma)",
                    transform=axes[0][1].transAxes, ha="right", fontsize=7, color="0.35")
    fig.suptitle("Systematic bias from an environmental flux deviation: 0PA against 0PA+PN\n"
                 + subtitle(recs), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def max_bias_panel(recs, path, exclude, note, p):
    ref = threshold_1d(p)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    models_scatter(ax, recs, lambda m: max_bias_over_sigma(m, exclude))
    log_axes(ax)
    ref_line(ax, ref, f"{p:.0%} credible interval, one parameter ({ref:.2f} sigma)")
    ax.set_ylabel("max |bias| / sigma  over parameters")
    ax.set_title(f"Worst single-parameter bias per source, 0PA against 0PA+PN\nmax over {note}",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, )
    fig.tight_layout()
    save(fig, path)


def nd_bias_panel(recs, path, exclude, note, p):
    """Per source, the n-d bias over the kept block, marginalized over C_p for the PN model
    (C_e is not a fit parameter here: it multiplies e, which is 0 on these circular orbits)."""
    k = len(kept(recs[0]["models"]["0PA"], exclude))
    ref = threshold_nd(p, k)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    models_scatter(ax, recs, lambda m: nd_bias(m, exclude)["marginalized"])
    log_axes(ax)
    ref_line(ax, ref, f"{p:.0%} region, {k}-d ({ref:.2f} sigma)")
    ax.set_ylabel(f"n-d bias  sqrt(b^T (C_vv)^-1 b)  over {k} parameters")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Normalized n-d bias per source, 0PA against 0PA+PN (PN marginalized over C_p)\n"
                 f"{note}\ndashed: {p:.0%} credible region of a {k}-d Gaussian, "
                 f"sqrt(chi2_ppf({p:g}, {k})) = {ref:.2f}", fontsize=9.5)
    fig.tight_layout()
    save(fig, path)


def critical_snr(model_rec, exclude, p):
    """SNR at which the systematic bias reaches the p credible threshold.

    The bias is SNR-independent and sigma scales as 1 / SNR, so |b|/sigma and the n-d bias
    both grow linearly with SNR from their value at SNR0 = RI.TARGET_SNR.  Same definition as
    Fisher_results/plot_bias.py: critical_snr --
      1-d  SNR0 * z_thr / max |b|/sigma   over the kept parameters (C_p held out)
      n-d  SNR0 * D_thr / D               D marginalized over C_p, D_thr = sqrt(chi2_ppf(p, k))
    """
    k = len(kept(model_rec, exclude))
    zmax = max_bias_over_sigma(model_rec, exclude)
    D = nd_bias(model_rec, exclude)["marginalized"]
    snr0 = RI.TARGET_SNR
    return dict(rho_1d=snr0 * threshold_1d(p) / zmax if zmax > 0 else np.inf,
                rho_nd=snr0 * threshold_nd(p, k) / D if D > 0 else np.inf)


def critical_snr_panel(recs, path, exclude, note, p):
    """Critical SNR against p0, 1-d and n-d.  Below the dashed line the bias is already
    significant at the injected SNR."""
    k = len(kept(recs[0]["models"]["0PA"], exclude))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), sharey=True)
    for ax, key, title in ((axes[0], "rho_1d", f"1-d: worst single parameter, "
                                               f"{p:.0%} interval ({threshold_1d(p):.2f} sigma)"),
                           (axes[1], "rho_nd", f"n-d: {k} parameters, PN marginalized over C_p, "
                                               f"{p:.0%} region ({threshold_nd(p, k):.2f} sigma)")):
        models_scatter(ax, recs, lambda m, kk=key: critical_snr(m, exclude, p)[kk])
        log_axes(ax)
        ax.axhline(RI.TARGET_SNR, color="0.55", lw=1.2, ls="--", zorder=1)
        ax.text(0.99, RI.TARGET_SNR * 1.15, f"injected SNR = {RI.TARGET_SNR:g}", ha="right",
                va="bottom", fontsize=7.5, color="0.35", transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("critical SNR")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Critical SNR at which the systematic bias becomes significant ({p:.0%} CL), "
                 f"0PA against 0PA+PN\n{note}; below the dashed line the bias is already "
                 f"significant at SNR {RI.TARGET_SNR:g}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, path)


def improvement_3panel(recs, path, exclude, note):
    """0PA / PN on three quantities; above 1 the PN template is better."""
    def ov_gain(r):
        return PO.mismatch(_as_src(r), "0PA") / PO.mismatch(_as_src(r), "0PA+PN")

    def ratio(fn):
        def g(r):
            a, b = fn(r["models"]["0PA"]), fn(r["models"]["0PA+PN"])
            return a / b if b > 0 else np.nan
        return g

    panels = (("mismatch  (1 - overlap)", ov_gain, "#6f42c1"),
              ("max single-parameter |bias|/sigma",
               ratio(lambda m: max_bias_over_sigma(m, exclude)), "#1f6feb"),
              ("n-d bias, marginalized", ratio(lambda m: nd_bias(m, exclude)["marginalized"]),
               "#d1650f"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, (title, fn, colour) in zip(axes, panels):
        y = np.array([fn(r) for r in recs], dtype=float)
        scatter_p0(ax, recs, y, colour, "o", None, s=36)
        ax.axhline(1.0, color="0.45", lw=1.2, ls="--", zorder=1)
        med = float(np.nanmedian(y))
        ax.axhline(med, color=colour, lw=1, ls=":", zorder=1)
        ax.text(0.99, 0.97, f"median {med:.3g}x", transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=colour)
        log_axes(ax)
        plain_ticks(ax)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("improvement factor,  0PA / 0PA+PN")
    fig.suptitle("What the PN deviation buys, per source -- above 1 the PN template is better\n"
                 f"bias panels over {note}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, path)


def _as_src(r):
    """A record in the shape plot_overlap.mismatch reads."""
    return {"fits": {m: {"ov_final": r["models"][m]["ov_final"]} for m in RI.MODELS}}


def bias_sigma_ratio(recs, path, exclude):
    """Does PN lower the bias, or only inflate sigma?  Both ratios per parameter, per source."""
    names = [n for n in vac_params(recs) if n not in exclude]
    ncol = min(4, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), squeeze=False)
    for ax, param in zip(axes.flat, names):
        for key, colour, marker, lab in (("bias", C_BIAS, "o", "|bias| PN / 0PA"),
                                         ("sigma", C_SIGMA, "s", "sigma PN / 0PA")):
            y = [value(r["models"]["0PA+PN"], param, key) / value(r["models"]["0PA"], param, key)
                 for r in recs]
            scatter_p0(ax, recs, y, colour, marker, lab, s=22)
            med = float(np.nanmedian(y))
            ax.axhline(med, color=colour, lw=1, ls=":", zorder=1)
        ax.axhline(1.0, color="0.45", lw=1.1, ls="--", zorder=1)
        log_axes(ax)
        ax.set_title(param, fontsize=10)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower right",
               bbox_to_anchor=(0.97, 0.18))
    fig.suptitle("PN against 0PA, per parameter: change in absolute bias and in sigma "
                 "(dotted = median, dashed = no change)\nwhere both are above 1, |bias|/sigma "
                 "falls only because sigma grows faster than the bias", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def _hist_grid(recs, path, exclude, key, xlabel, title, ref=None):
    """Per parameter, log10 |key| of 0PA against PN, common bins, step histograms."""
    names = [n for n in vac_params(recs) if n not in exclude]
    ncol = min(4, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow), squeeze=False)
    for ax, param in zip(axes.flat, names):
        data = {m: np.log10([value(r["models"][m], param, key) for r in recs])
                for m, _, _ in MODELS}
        allv = np.concatenate([v[np.isfinite(v)] for v in data.values()])
        bins = np.linspace(allv.min(), allv.max(), 16)
        for model, colour, _ in MODELS:
            v = data[model][np.isfinite(data[model])]
            ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.25, color=colour, zorder=2)
            ax.hist(v, bins=bins, histtype="step", lw=1.6, color=colour,
                    label=f"{model} (median {10 ** np.median(v):.2g})", zorder=3)
        if ref is not None:
            ax.axvline(np.log10(ref), color="0.55", lw=1.1, ls="--", zorder=1)
        ax.set_title(param, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.grid(True, color="0.9", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes[0][0].set_ylabel("sources")
    fig.suptitle(f"{title}\n{subtitle(recs)}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def hist_bias(recs, path, exclude):
    _hist_grid(recs, path, exclude, "bias", "log10 |bias|",
               "Distribution of the absolute bias per parameter, 0PA against 0PA+PN")


def hist_bias_over_sigma(recs, path, exclude):
    _hist_grid(recs, path, exclude, "bias_over_sigma", "log10 |bias| / sigma",
               "Distribution of |bias| / sigma per parameter, 0PA against 0PA+PN "
               f"(dashed: 99% interval, {threshold_1d(0.99):.2f} sigma)",
               ref=threshold_1d(0.99))


def conditioning_panel(recs, path):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for ax, key, title in ((axes[0], "cond_raw", "condition number of the Fisher, raw"),
                           (axes[1], "cond_scaled", "condition number, Jacobi-rescaled")):
        models_scatter(ax, recs, lambda m, k=key: m[k])
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7, frameon=False, )
    ax = axes[2]
    models_scatter(ax, recs, lambda m: m["inv_residual_ratio"])
    ax.axhline(1.0, color="0.55", lw=1, ls="--", zorder=1)
    ax.text(0.98, 0.95, "float64 limit -- below it the inverse is as\naccurate as the "
                        "arithmetic allows", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="0.35")
    ax.set_title("inversion residual / (cond x eps)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left", )
    for a in axes:
        log_axes(a)
    axes[2].set_ylim(top=3.0)
    fig.suptitle(f"Fisher conditioning, per source and model ({len(recs)} sources)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, path)


def main():
    recs = load_records()
    PO.main()
    for block, (exclude, note) in BLOCKS.items():
        bias_panels(recs, out_path(block, "bias_panels.png"), exclude)
        for lvl, p in CREDIBLE.items():
            max_bias_panel(recs, out_path(block, f"max_bias{lvl}.png"), exclude, note, p)
            nd_bias_panel(recs, out_path(block, f"nd_bias{lvl}.png"), exclude, note, p)
            critical_snr_panel(recs, out_path(block, f"critical_snr{lvl}.png"), exclude, note, p)
        improvement_3panel(recs, out_path(block, "improvement_3panel.png"), exclude, note)
        bias_sigma_ratio(recs, out_path(block, "bias_sigma_ratio.png"), exclude)
        hist_bias(recs, out_path(block, "hist_bias.png"), exclude)
        hist_bias_over_sigma(recs, out_path(block, "hist_bias_over_sigma.png"), exclude)
    conditioning_panel(recs, out_path("", "conditioning.png"))


if __name__ == "__main__":
    main()
