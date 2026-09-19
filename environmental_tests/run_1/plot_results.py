"""Figures for run_1/results: the systematic bias, parameter by parameter, 0PA against 0PA+PN.

    python plot_results.py

Reads the per-source records written by `refit_fisher.py` -- one file per converged source,
each carrying both models' fits and the sigma from the order-8 / Ndelta-24 Fisher taken at
that model's own best fit.  Nothing is recomputed here.

Four figures:
  bias_panels.png   one row per parameter, |bias| on the left and |bias|/sigma on the right
  overlap.png       1 - overlap for both models, so the fit quality sits next to the bias
  max_bias.png      per source, the worst |bias|/sigma any single parameter shows
  nd_bias.png       per source, the n-d bias, marginalized and conditional
  improvement.png   per source, the factor by which PN beats 0PA on all three quantities
  *_with_e0.png     each of the above that has a parameter block, drawn again with e0 in it
  intrinsic/        the same four figures over m1, m2, a, p0 only, in their own folder --
                    sky position and phase held out, same filenames as at the top level
  *_p90.png         the summary figures again at the 90% credible level rather than 99%
  conditioning.png  cond(Gamma) raw and rescaled, and the inversion residual, per source

Reference lines are credible regions, never a flat 1 sigma.  For one parameter that means
the two-sided 99% (or 90%) interval; for the n-d bias it means sqrt(chi2_ppf(p, k)), which
at k = 7 is 4.30 sigma at 99% and 3.47 at 90%.  A line at 1.0 would be the 68% interval in
one dimension and very much less than that in seven.

e0 is the reason for the pairing.  It is injected at its boundary value of zero, so its
Fisher direction is degenerate, sigma(e0) is not a confidence interval, and any total it
enters is dominated by that artefact.  Both versions are drawn rather than one being
argued for.

p0 is the x axis throughout rather than the source index: the environmental term enters the
flux as (p/10)^8, so p0 is the variable the whole effect is organised by, and an index axis
would scatter the trend into noise.  Every point is labelled with its index anyway.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import chi2, norm

import results_inputs as RI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

C_0PA, C_PN = "#1f6feb", "#d1650f"     # one hue per model, as in cv_population.png
M_0PA, M_PN = "o", "^"                 # identity is marker+colour, never colour alone
MODELS = [("0PA", C_0PA, M_0PA), ("0PA+PN", C_PN, M_PN)]

# e0 = 0 sits on its own physical boundary, so its Fisher direction is degenerate and
# sigma(e0) is not a confidence interval.  The absolute bias is still meaningful; the ratio
# is not, and the panel says so rather than quietly plotting a number in the thousands.
UNRELIABLE_SIGMA = {"e0"}

# Below this the sources pile up at the bottom of the overlap panel and their labels overlap.
LABEL_ABOVE = 1e-6

# The credible levels the reference lines are drawn at.  A flat line at 1 sigma is the wrong
# yardstick for both quantities plotted here and is not used anywhere.
#   - For a single parameter, "1 sigma" is the 68.3% interval, which is an odd level to judge
#     a systematic against; 90% and 99% are the levels one would actually quote.
#   - For the n-d bias it is worse than odd.  sqrt(b^T M b) = 1 is NOT the 68% region of a
#     k-dimensional Gaussian: the radius that encloses probability p is sqrt(chi2_ppf(p, k)),
#     which grows with k.  At k = 7 the 90% region sits at 3.47 sigma and the 99% at 4.30,
#     so judging a 7-d bias against 1.0 overstates how biased everything is by a factor of
#     three or four.
CREDIBLE = {"": 0.99, "_p90": 0.90}      # filename suffix -> level


def threshold_1d(p):
    """Half-width of the two-sided p credible interval for one parameter, in sigma."""
    return float(norm.ppf(0.5 + p / 2.0))


def threshold_nd(p, k):
    """Radius of the p credible region of a k-dimensional Gaussian, in sigma."""
    return float(np.sqrt(chi2.ppf(p, k)))

# The deviation amplitudes.  Always held out of a summary figure: they are injected at zero
# and are what the template is there to absorb, so their displacement is not a bias in a
# measured parameter, and 0PA has no such parameter to compare against anyway.
DEV_PARAMS = {"C_p", "C_e"}

# The extrinsic parameters: sky position and the initial orbital phase.  They are the ones
# a systematic tends to hide in, because they are cheap for a fit to move and carry no
# astrophysics.  Held out of the "intrinsic" blocks so the question becomes narrower and
# harder: does the PN freedom fix the parameters we actually want to measure?
EXTRINSIC = {"qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"}

# The parameter blocks every summary figure is drawn over.  Two axes of choice, crossed:
#
#   e0 in or out -- e0 sits on its physical boundary, so its Fisher direction is degenerate
#     and sigma(e0) is not a confidence interval.  Held out, a figure shows the parameters
#     whose sigma means what it says; kept in, it shows what the boundary direction does to
#     the totals.  Both are drawn rather than one being asserted over the other.
#
#   all parameters, or intrinsic only -- m1, m2, a, p0 are the astrophysics.  A template
#     that absorbs an environmental deviation by shifting the sky position has not damaged
#     the measurement in the way that one absorbing it into the masses has.
#
#   suffix -> (parameters excluded, how the figure describes its block)
BLOCKS = {
    "": (DEV_PARAMS | {"e0"},
         "m1, m2, a, p0, qS, phiS, Phi_phi0 -- e0 and the deviation amplitude held out"),
    "_with_e0": (DEV_PARAMS,
                 "the same seven with e0 added back -- sigma(e0) is degenerate at the "
                 "boundary, so this block is dominated by that direction"),
    "_intrinsic": (DEV_PARAMS | EXTRINSIC | {"e0"},
                   "m1, m2, a, p0 only -- the intrinsic parameters, sky position and phase "
                   "held out along with e0"),
    "_intrinsic_with_e0": (DEV_PARAMS | EXTRINSIC,
                           "m1, m2, a, p0, e0 -- the intrinsic parameters with e0 added "
                           "back, sky position and phase held out"),
}


def block_path(suffix, name):
    """Where a figure for one parameter block goes.

    The intrinsic blocks live in `results/intrinsic/`, and drop the now-redundant
    `_intrinsic` from their filenames once they are in that folder -- so the two directories
    hold the same four names and can be read side by side.
    """
    if "intrinsic" in suffix:
        d = os.path.join(OUT, "intrinsic")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name.replace("_intrinsic", ""))
    return os.path.join(OUT, name)


def load_records(path=OUT):
    """Every per-source record in the results folder, in index order."""
    recs = []
    for f in sorted(glob.glob(os.path.join(path, "idx*.json"))):
        with open(f) as fh:
            recs.append(json.load(fh))
    if not recs:
        raise SystemExit(f"no idx*.json in {path}: run refit_fisher.py first")
    return sorted(recs, key=lambda r: r["idx"])


def all_params(recs):
    """Every parameter any model fits, vacuum block first, then the deviation."""
    names = list(recs[0]["models"]["0PA"]["params"])
    for n in recs[0]["models"]["0PA+PN"]["params"]:
        if n not in names:
            names.append(n)
    return names


def converged(recs):
    """The sources whose climb reached a maximum -- the ones a population median is over."""
    return [r for r in recs if r.get("converged", True)]


def series(recs, model, param, key):
    """(p0, value, idx) for one model and one parameter, skipping sources it does not fit."""
    x, y, lab = [], [], []
    for r in recs:
        m = r["models"][model]
        if param not in m["params"]:
            continue
        v = m[key][m["params"].index(param)]
        if v is None or not np.isfinite(v):
            continue
        x.append(r["p0"])
        y.append(abs(float(v)))
        lab.append(r["idx"])
    return np.array(x), np.array(y), lab


def _conv_map(recs):
    """idx -> did its climb converge.  Non-converged points are drawn hollow, not dropped."""
    return {r["idx"]: r.get("converged", True) for r in recs}


def _plot_models(ax, recs, param, key):
    bad = _conv_map(recs)
    for model, colour, marker in MODELS:
        x, y, lab = series(recs, model, param, key)
        if len(x) == 0:
            continue
        ok = np.array([bad.get(i, True) for i in lab])
        ax.scatter(x[ok], y[ok], marker=marker, s=28, color=colour, label=model, zorder=3)
        ax.scatter(x[~ok], y[~ok], marker=marker, s=34, facecolors="none",
                   edgecolors=colour, linewidths=1.3, zorder=3)
    ax.set_yscale("log")
    ax.grid(True, which="major", color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def bias_panels(recs, path, drop=frozenset()):
    """One row per parameter: absolute bias on the left, bias in sigma on the right.

    `drop` removes whole rows.  It exists so the figure can be produced with and without
    e0: the e0 row is the one whose right-hand panel is not a ratio, and reading the other
    seven without it beside them is easier than reading around it.  C_p is never dropped
    here -- it is not a bias, but how much deviation the template took on is the point of
    the comparison, so the row stays and is labelled as such.
    """
    names = [n for n in all_params(recs) if n not in drop]
    fig, axes = plt.subplots(len(names), 2, figsize=(9.5, 2.0 * len(names)),
                             sharex=True, squeeze=False)

    for row, param in enumerate(names):
        for col, (key, title) in enumerate((("bias", "|bias|"),
                                            ("bias_over_sigma", "|bias| / sigma"))):
            ax = axes[row][col]
            _plot_models(ax, recs, param, key)
            if col == 1:
                ax.axhline(threshold_1d(0.99), color="0.55", lw=1.2, ls="--", zorder=1)
                if param in UNRELIABLE_SIGMA:
                    ax.set_facecolor("#f4f1ec")
                    ax.text(0.5, 0.9, "sigma degenerate at the boundary -- not a ratio",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=7, color="0.35")
            if row == 0:
                ax.set_title(title, fontsize=10)
        lab = "C_p\n(absorbed)" if param == "C_p" else param
        axes[row][0].set_ylabel(lab, fontsize=10)

    for ax in axes[-1]:
        ax.set_xlabel("p0")
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    block = "without e0" if "e0" in drop else "e0 included"
    axes[0][1].text(0.99, 0.04, f"dashed: 99% credible interval "
                                f"({threshold_1d(0.99):.2f} sigma)",
                    transform=axes[0][1].transAxes, ha="right", fontsize=7, color="0.35")
    fig.suptitle("Systematic bias from an environmental flux deviation: "
                 f"0PA against 0PA+PN ({block})\n"
                 f"{len(recs)} sources ({len(recs) - len(converged(recs))} hollow: the "
                 f"climb did not converge), sigma from an order-{recs[0]['der_order']} "
                 f"Fisher over {recs[0]['Ndelta']} steps at each model's own best fit",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def overlap_panel(recs, path):
    """1 - overlap for both models.  Log y: the interesting range spans eight decades."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for model, colour, marker in MODELS:
        x = np.array([r["p0"] for r in recs])
        y = np.array([max(1.0 - r["models"][model]["ov_final"], 1e-16) for r in recs])
        ok = np.array([r.get("converged", True) for r in recs])
        ax.scatter(x[ok], y[ok], marker=marker, s=34, color=colour, label=model, zorder=3)
        ax.scatter(x[~ok], y[~ok], marker=marker, s=44, facecolors="none",
                   edgecolors=colour, linewidths=1.4, zorder=3)
    # Labelling all 21 collides at low p0, where the points are piled on each other and all
    # say the same thing.  Only the sources the fit actually struggled with are named.
    for r in recs:
        y0 = max(1.0 - r["models"]["0PA"]["ov_final"], 1e-16)
        if y0 > LABEL_ABOVE:
            ax.annotate(str(r["idx"]), (r["p0"], y0), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7.5, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("p0")
    ax.set_ylabel("1 - overlap")
    ax.set_title("Fit quality against p0, both recovery models\n"
                 "labelled by source index; hollow = the climb did not converge",
                 fontsize=11)
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def kept(model_rec, exclude):
    """Indices of the parameters a summary figure is built on, in the record's order."""
    return [i for i, n in enumerate(model_rec["params"]) if n not in exclude]


def max_bias_over_sigma(model_rec, exclude):
    """The worst |bias|/sigma any single kept parameter shows for this model."""
    vals = [model_rec["bias_over_sigma"][i] for i in kept(model_rec, exclude)]
    vals = [abs(float(v)) for v in vals if v is not None and np.isfinite(v)]
    return max(vals) if vals else np.nan


def _scaled(M):
    """`d, M_hat` with `M_hat = D^-1 M D^-1` -- the same Jacobi rescaling refit_fisher uses.

    A sub-block is rescaled by its own diagonal rather than by the parent matrix's, because
    the block is the matrix the quadratic form is actually taken against.  The form itself
    is invariant: b^T M b = (D b)^T M_hat (D b) and b^T M^-1 b = (D^-1 b)^T M_hat^-1 (D^-1 b),
    so nothing here changes the answer -- it only stops the answer being computed as a
    difference of huge numbers.
    """
    d = np.sqrt(np.abs(np.diag(M)))
    if np.any(d <= 0):
        return None, None
    return d, M / np.outer(d, d)


def _quad(M, b):
    """sqrt(b^T M b), formed in the rescaled basis.  nan if M is indefinite along b."""
    d, Mh = _scaled(M)
    if d is None:
        return np.nan
    q = float((b * d) @ Mh @ (b * d))
    return np.sqrt(q) if q > 0 else np.nan


def _quad_inv(M, b):
    """sqrt(b^T M^-1 b), rescaled and solved rather than inverted."""
    d, Mh = _scaled(M)
    if d is None:
        return np.nan
    q = float((b / d) @ np.linalg.solve(Mh, b / d))
    return np.sqrt(q) if q > 0 else np.nan


def nd_bias(model_rec, exclude):
    """The n-d bias over the kept parameters, marginalized and conditional.

    Both are the Mahalanobis length of the same bias vector; they differ in what is done
    with the parameters held out of it (e0, and C_p where the model has one).

      marginalized  sqrt(b^T (C_vv)^-1 b) -- C is the full covariance, C_vv its kept block,
                    so the held-out parameters have been integrated over and their freedom
                    is still paid for.  This is the SK/CV convention.  With e0 in the block
                    its near-null direction enters (C_vv)^-1 and dominates the total.
      conditional   sqrt(b^T G_vv b) -- G is the Fisher, G_vv its kept block, which is the
                    inverse covariance of the kept parameters with the held-out ones frozen.

    G_vv >= (C_vv)^-1 in the Loewner order, so the conditional value is never the smaller
    of the two, and the gap between them is exactly what absorbing the deviation costs in
    the vacuum parameters.
    """
    k = kept(model_rec, exclude)
    sb = RI.signed_bias_of(model_rec["params"], model_rec["truth"], model_rec["fit"])
    b = np.array([sb[model_rec["params"][i]] for i in k], dtype=float)
    G = np.array(model_rec["fisher"], dtype=float)[np.ix_(k, k)]
    C = np.array(model_rec["cov"], dtype=float)[np.ix_(k, k)]
    return dict(marginalized=_quad_inv(C, b), conditional=_quad(G, b))


def _idx_scatter(ax, recs, value_of, ref=None, ref_label=None, label_suffix=""):
    """One point per source per model, x = source index, with an optional reference line."""
    conv = _conv_map(recs)
    for model, colour, marker in MODELS:
        x = np.array([r["idx"] for r in recs], dtype=float)
        y = np.array([value_of(r["models"][model]) for r in recs], dtype=float)
        ok = np.array([conv.get(r["idx"], True) for r in recs])
        good = np.isfinite(y)
        ax.scatter(x[ok & good], y[ok & good], marker=marker, s=38, color=colour,
                   label=model + label_suffix, zorder=3)
        ax.scatter(x[~ok & good], y[~ok & good], marker=marker, s=48, facecolors="none",
                   edgecolors=colour, linewidths=1.4, zorder=3)
    ax.set_yscale("log")
    if ref is not None:
        ax.axhline(ref, color="0.55", lw=1.2, ls="--", zorder=1)
        if ref_label:
            ax.text(0.99, 0.02, ref_label, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="0.35")
    ax.set_xticks([r["idx"] for r in recs])
    ax.set_xticklabels([str(r["idx"]) for r in recs], fontsize=7.5)
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def max_bias_panel(recs, path, exclude, block_note, p):
    """Per source, the worst |bias|/sigma any one kept parameter shows."""
    ref = threshold_1d(p)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    _idx_scatter(ax, recs, lambda m: max_bias_over_sigma(m, exclude),
                 ref=ref, ref_label=f"{p:.0%} credible interval, one parameter "
                                    f"({ref:.2f} sigma)")
    ax.set_xlabel("source index")
    ax.set_ylabel("max |bias| / sigma  over parameters")
    ax.set_title("Worst single-parameter bias per source, 0PA against 0PA+PN\n"
                 f"max over {block_note}",
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def nd_bias_panel(recs, path, exclude, block_note, p):
    """Per source, the n-d bias over the kept block: marginalized beside conditional."""
    k = len(kept(recs[0]["models"]["0PA"], exclude))
    ref = threshold_nd(p, k)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, key, title in ((axes[0], "marginalized",
                            "marginalized over the held-out parameters"),
                           (axes[1], "conditional",
                            "conditional: the held-out parameters frozen")):
        _idx_scatter(ax, recs, lambda m, kk=key: nd_bias(m, exclude)[kk],
                     ref=ref, ref_label=f"{p:.0%} region, {k}-d ({ref:.2f} sigma)")
        ax.set_xlabel("source index")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(f"n-d bias   sqrt( b^T M b )   over {k} parameters")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("Normalized n-d bias per source, 0PA against 0PA+PN\n"
                 f"{block_note}\n"
                 f"dashed line is the {p:.0%} credible region of a {k}-d Gaussian, "
                 f"sqrt(chi2_ppf({p:g}, {k})) = {ref:.2f} -- not 1 sigma, which in {k} "
                 f"dimensions encloses almost nothing",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def improvement_panel(recs, path, exclude, block_note):
    """What the PN freedom actually buys, per source, on the three headline quantities.

    Every panel is 0PA-over-PN, so a value above 1 means PN improved things and the height
    is the factor.  Log y, one dashed line at 1: no improvement.  Plotting the ratio rather
    than the two values side by side is the point -- the absolute numbers span five decades
    across the population, which hides a consistent factor-of-ten gain completely.
    """
    def ov_gain(r):
        # Residual mismatch, not overlap: 0.9999 -> 0.999999 is a hundredfold improvement
        # and a difference of 1e-4, and only the first of those is meaningful.
        a = max(1.0 - r["models"]["0PA"]["ov_final"], 1e-16)
        b = max(1.0 - r["models"]["0PA+PN"]["ov_final"], 1e-16)
        return a / b

    def max_gain(r):
        a = max_bias_over_sigma(r["models"]["0PA"], exclude)
        b = max_bias_over_sigma(r["models"]["0PA+PN"], exclude)
        return a / b if b > 0 else np.nan

    def nd_gain(r):
        a = nd_bias(r["models"]["0PA"], exclude)["marginalized"]
        b = nd_bias(r["models"]["0PA+PN"], exclude)["marginalized"]
        return a / b if b > 0 else np.nan

    panels = (("mismatch  (1 - overlap)", ov_gain, "#6f42c1"),
              ("max single-parameter bias", max_gain, "#1f6feb"),
              ("n-d bias, marginalized", nd_gain, "#d1650f"))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for ax, (title, fn, colour) in zip(axes, panels):
        x = np.array([r["idx"] for r in recs], dtype=float)
        y = np.array([fn(r) for r in recs], dtype=float)
        ok = np.isfinite(y)
        ax.scatter(x[ok & (y >= 1)], y[ok & (y >= 1)], marker="o", s=42, color=colour,
                   zorder=3, label="PN better")
        ax.scatter(x[ok & (y < 1)], y[ok & (y < 1)], marker="v", s=46, facecolors="none",
                   edgecolors=colour, linewidths=1.5, zorder=3, label="PN worse")
        ax.axhline(1.0, color="0.45", lw=1.2, ls="--", zorder=1)
        med = np.median(y[ok])
        ax.axhline(med, color=colour, lw=1, ls=":", zorder=1)
        ax.text(0.99, 0.97, f"median {med:.3g}x", transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=colour)
        ax.set_yscale("log")
        ax.set_xlabel("source index")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([r["idx"] for r in recs])
        ax.set_xticklabels([str(r["idx"]) for r in recs], fontsize=6.5)
        ax.grid(True, color="0.88", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("improvement factor,  0PA / 0PA+PN")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("What the PN deviation buys, per source\n"
                 f"above 1 the PN template is better; bias panels are over {block_note}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def resid_ratio(model_rec):
    """The inversion residual in units of what float64 can deliver at this conditioning.

    `cond_scaled * eps` is the accuracy limit of the arithmetic, so this ratio is the honest
    diagnostic: order one means the inverse is as good as it can be, and a fixed absolute
    threshold would instead flag that limit itself as a failure.  Computed here rather than
    read, so records written before the field existed still work.
    """
    if model_rec.get("inv_residual_ratio") is not None:
        return float(model_rec["inv_residual_ratio"])
    return float(model_rec["inv_residual"]
                 / (model_rec["cond_scaled"] * np.finfo(float).eps))


def conditioning_panel(recs, path):
    """Is the Fisher well enough conditioned to be inverted?  Three ways of asking.

    Left and centre share a y axis so the effect of the rescaling is the vertical distance
    between the two clouds.  Right is the test that actually decides the question: the
    largest element of `G_hat C_hat - I`.  A condition number says how much error an
    inversion *could* amplify; the residual says how much it *did*.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: raw and rescaled on one log axis, so the drop is a visible vertical gap rather
    # than two panels the reader has to hold in their head.  Filled = as computed, open =
    # after rescaling; colour stays the model, as everywhere else.
    ax = axes[0]
    for model, colour, marker in MODELS:
        x = np.array([r["idx"] for r in recs], dtype=float)
        ax.scatter(x, [r["models"][model]["cond_raw"] for r in recs], marker=marker, s=38,
                   color=colour, label=f"{model}  raw", zorder=3)
        ax.scatter(x, [r["models"][model]["cond_scaled"] for r in recs], marker=marker,
                   s=44, facecolors="none", edgecolors=colour, linewidths=1.4,
                   label=f"{model}  rescaled", zorder=3)
    med_raw = np.median([r["models"][m]["cond_raw"] for r in recs for m in ("0PA", "0PA+PN")])
    med_sca = np.median([r["models"][m]["cond_scaled"] for r in recs
                         for m in ("0PA", "0PA+PN")])
    ax.annotate("", xy=(0.5, med_sca), xytext=(0.5, med_raw),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.2))
    ax.text(0.9, np.sqrt(med_raw * med_sca),
            f"{np.log10(med_raw / med_sca):.0f} orders,\nat the median",
            fontsize=8, color="0.35", va="center")
    ax.set_title("condition number of the Fisher", fontsize=10)

    # Right: the test that decides it.
    ax = axes[1]
    for model, colour, marker in MODELS:
        ax.scatter([r["idx"] for r in recs], [resid_ratio(r["models"][model]) for r in recs],
                   marker=marker, s=38, color=colour, label=model, zorder=3)
    ax.axhline(1.0, color="0.55", lw=1, ls="--", zorder=1)
    ax.text(0.98, 0.93, "float64 limit -- below this the inverse is\n"
                        "as accurate as the arithmetic allows",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="0.35")
    ax.set_title("inversion residual / (cond x eps)", fontsize=10)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("source index")
        ax.set_xticks([r["idx"] for r in recs])
        ax.set_xticklabels([str(r["idx"]) for r in recs], fontsize=6.5)
        ax.grid(True, color="0.88", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[1].set_ylim(top=3.0)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[1].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("Fisher conditioning, per source and model\n"
                 "the rescaling is a congruence, so by Sylvester's law of inertia it leaves "
                 "the signs of the eigenvalues alone -- only the arithmetic improves",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] {path}")


def main():
    recs = load_records()
    bias_panels(recs, os.path.join(OUT, "bias_panels.png"), drop={"e0"})
    bias_panels(recs, os.path.join(OUT, "bias_panels_with_e0.png"))
    overlap_panel(recs, os.path.join(OUT, "overlap.png"))
    # Each summary figure is drawn over both parameter blocks and at both credible levels,
    # so the e0 contribution and the choice of level are both visible rather than asserted.
    for suffix, (exclude, note) in BLOCKS.items():
        for lvl, p in CREDIBLE.items():
            max_bias_panel(recs, block_path(suffix, f"max_bias{suffix}{lvl}.png"),
                           exclude, note, p)
            nd_bias_panel(recs, block_path(suffix, f"nd_bias{suffix}{lvl}.png"),
                          exclude, note, p)
        improvement_panel(recs, block_path(suffix, f"improvement{suffix}.png"),
                          exclude, note)
    conditioning_panel(recs, os.path.join(OUT, "conditioning.png"))


if __name__ == "__main__":
    main()
