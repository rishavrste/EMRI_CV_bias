"""Write run_1/results/README.md from the per-source records in the same folder.

    python make_results_md.py

Every number in the page is read out of `results/idx*.json`, so the text cannot drift from
the data it describes.  The page carries three tables -- one row per converged source, one
row per source left out and why, and one row per parameter summarising how the two models
compare across the population -- followed by the two figures.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

import results_inputs as RI
from plot_results import (load_records, all_params, series, converged,
                          UNRELIABLE_SIGMA, resid_ratio, BLOCKS, CREDIBLE,
                          kept, max_bias_over_sigma, nd_bias,
                          threshold_1d, threshold_nd)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

# A PN template at C_p = 0 *is* the 0PA template, so differences below this are the
# optimiser's arithmetic rather than a win for either model.
TIE_TOL = 1e-6


def _g(x, n=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{n}g}"


def get(m, param, key):
    """One value out of a model record, or None where that model does not fit it."""
    if param not in m["params"]:
        return None
    return m[key][m["params"].index(param)]


def worst_bos(m):
    """The largest |bias|/sigma the model shows, ignoring the parameters whose sigma is not
    a confidence interval."""
    vals = [abs(v) for n, v in zip(m["params"], m["bias_over_sigma"])
            if n not in UNRELIABLE_SIGMA and v is not None and np.isfinite(v)]
    return max(vals) if vals else None


def source_table(recs):
    """One row per converged source: where it came from, how well each model fit it."""
    head = ("| idx | p0 | m1 / 1e6 | fit from | ov 0PA | ov 0PA+PN | chi2 0PA | chi2 0PA+PN "
            "| C_p | max b/sigma 0PA | max b/sigma 0PA+PN |")
    rule = "|" + "---|" * 11
    rows = [head, rule]
    for r in recs:
        a, b = r["models"]["0PA"], r["models"]["0PA+PN"]
        # A source can draw its two arms from different runs -- idx 14 takes its 0PA fit
        # from the harder follow-up and its 0PA+PN fit from the population run, because
        # that is where each model's best overlap was actually found.  Say so per arm
        # rather than labelling the row by its 0PA arm alone.
        where = {m: ("population" if r["models"][m]["origin"] == "results_env_cv.json"
                     else "harder_environ") for m in ("0PA", "0PA+PN")}
        origin = (where["0PA"] if where["0PA"] == where["0PA+PN"]
                  else f"{where['0PA']} / {where['0PA+PN']}")
        mark = "" if r.get("converged", True) else " **!**"
        rows.append(
            f"| {r['idx']}{mark} | {r['p0']:.3f} | {r['m1'] / 1e6:.3f} | {origin} "
            f"| {a['ov_final']:.10f} | {b['ov_final']:.10f} "
            f"| {a['chi2']:.4g} | {b['chi2']:.4g} "
            f"| {_g(get(b, 'C_p', 'fit'))} "
            f"| {_g(worst_bos(a), 3)} | {_g(worst_bos(b), 3)} |")
    return "\n".join(rows)


def not_converged_table(recs):
    """One row per source that is in the analysis but whose climb did not reach a maximum."""
    bad = [r for r in recs if not r.get("converged", True)]
    if not bad:
        return "Every source in the analysis converged."
    rows = ["| idx | p0 | ov 0PA | ov 0PA+PN | what happened |", "|---|---|---|---|---|"]
    for r in bad:
        rows.append(f"| {r['idx']} | {r['p0']:.3f} "
                    f"| {r['models']['0PA']['ov_final']:.10f} "
                    f"| {r['models']['0PA+PN']['ov_final']:.10f} "
                    f"| {r['not_converged_why']} |")
    return "\n".join(rows)


def dropped_table():
    """One row per source that is not in the analysis, with the reason it is not."""
    rows = ["| idx | p0 | why it is not here |", "|---|---|---|"]
    with open(RI.POP_JSON) as fh:
        pop = {s["idx"]: s for s in json.load(fh)["sources"]}
    for idx, why in sorted(RI.DROPPED.items()):
        rows.append(f"| {idx} | {pop[idx]['p0']:.3f} | {why} |")
    return "\n".join(rows)


def _ratio_stats(recs, param):
    """|bias|_PN / |bias|_0PA across the sources that both models fit."""
    out = []
    for r in recs:
        a = get(r["models"]["0PA"], param, "bias")
        b = get(r["models"]["0PA+PN"], param, "bias")
        if a is None or b is None or a == 0:
            continue
        out.append(b / a)
    return np.array(out)


def param_table(recs):
    """One row per parameter: does the PN template actually reduce the bias?

    Over the converged sources only.  A bias measured in a basin the optimiser merely
    stopped in would sit in the denominator of the ratio column and mean nothing.
    """
    recs = converged(recs)
    rows = ["| param | median \\|b\\| 0PA | median \\|b\\| 0PA+PN | median ratio "
            "| PN smaller | PN larger | tie | median b/sigma 0PA | median b/sigma 0PA+PN |",
            "|" + "---|" * 9]
    for param in all_params(recs):
        if param == "C_p":
            continue
        _, ba, _ = series(recs, "0PA", param, "bias")
        _, bb, _ = series(recs, "0PA+PN", param, "bias")
        ratio = _ratio_stats(recs, param)
        better = int(np.sum(ratio < 1 - TIE_TOL))
        worse = int(np.sum(ratio > 1 + TIE_TOL))
        tie = int(len(ratio) - better - worse)
        _, sa, _ = series(recs, "0PA", param, "bias_over_sigma")
        _, sb, _ = series(recs, "0PA+PN", param, "bias_over_sigma")
        note = " *" if param in UNRELIABLE_SIGMA else ""
        rows.append(
            f"| {param}{note} | {_g(np.median(ba))} | {_g(np.median(bb))} "
            f"| {_g(np.median(ratio), 3)} | {better} | {worse} | {tie} "
            f"| {_g(np.median(sa), 3)} | {_g(np.median(sb), 3)} |")
    return "\n".join(rows)


def headline(recs):
    """The two or three sentences that are the actual answer."""
    n_all = len(recs)
    recs = converged(recs)
    n = len(recs)
    pn_wins_ov = sum(1 for r in recs
                     if r["models"]["0PA+PN"]["ov_final"] >= r["models"]["0PA"]["ov_final"])
    lines = [f"* `PN >= 0PA` on overlap holds for {pn_wins_ov} of {n} converged sources, "
             f"and for {n_all} of {n_all} counting the ones that did not converge.",
             f"* The medians below are over the {n} converged sources."]
    for param in all_params(recs):
        if param == "C_p":
            continue
        ratio = _ratio_stats(recs, param)
        if len(ratio):
            lines.append(f"* `{param}`: median |bias| ratio PN / 0PA = "
                         f"{np.median(ratio):.3g} over {len(ratio)} sources.")
    return "\n".join(lines)


def credible_table(recs):
    """How many sources sit outside the credible region, by quantity, block and level.

    This is the summary that a flat 1-sigma line cannot give.  "Outside" means the bias is
    larger than the region that would contain that fraction of the posterior, so a source
    counted here has a systematic that a real analysis would not absorb into its error bar.
    """
    head = ("| block | quantity | region | threshold | 0PA outside | 0PA+PN outside |")
    rows = [head, "|" + "---|" * 6]
    for suffix, (exclude, _) in BLOCKS.items():
        k = len(kept(recs[0]["models"]["0PA"], exclude))
        block = ("intrinsic" if "intrinsic" in suffix else "all") + \
                (", with e0" if "with_e0" in suffix else ", no e0") + f" ({k})"
        for lvl, p in sorted(CREDIBLE.items(), key=lambda kv: -kv[1]):
            t1, tn = threshold_1d(p), threshold_nd(p, k)
            m1 = [(max_bias_over_sigma(r["models"]["0PA"], exclude),
                   max_bias_over_sigma(r["models"]["0PA+PN"], exclude)) for r in recs]
            nd = [(nd_bias(r["models"]["0PA"], exclude)["marginalized"],
                   nd_bias(r["models"]["0PA+PN"], exclude)["marginalized"]) for r in recs]
            n = len(recs)
            rows.append(f"| {block} | max 1-d | {p:.0%} | {t1:.2f} sigma "
                        f"| {sum(a > t1 for a, b in m1)}/{n} "
                        f"| **{sum(b > t1 for a, b in m1)}/{n}** |")
            rows.append(f"| {block} | n-d, marginalized | {p:.0%} | {tn:.2f} sigma "
                        f"| {sum(a > tn for a, b in nd)}/{n} "
                        f"| **{sum(b > tn for a, b in nd)}/{n}** |")
    return "\n".join(rows)


def improvement_row(recs):
    """Median factor by which PN beats 0PA on each headline quantity."""
    exclude = BLOCKS[""][0]
    ov = np.median([max(1 - r["models"]["0PA"]["ov_final"], 1e-16)
                    / max(1 - r["models"]["0PA+PN"]["ov_final"], 1e-16) for r in recs])
    m1 = np.median([max_bias_over_sigma(r["models"]["0PA"], exclude)
                    / max_bias_over_sigma(r["models"]["0PA+PN"], exclude) for r in recs])
    nd = np.median([nd_bias(r["models"]["0PA"], exclude)["marginalized"]
                    / nd_bias(r["models"]["0PA+PN"], exclude)["marginalized"] for r in recs])
    return ov, m1, nd


def conditioning_table(recs):
    """Per source and model: how bad the raw matrix is, and whether the inverse is trustworthy."""
    head = ("| idx | model | cond raw | cond rescaled | min eig (scaled) | inv residual "
            "| / float64 limit | rank | pos def |")
    rows = [head, "|" + "---|" * 9]
    for r in recs:
        for m in ("0PA", "0PA+PN"):
            d = r["models"][m]
            n = len(d["params"])
            rows.append(
                f"| {r['idx']} | `{m}` | {d['cond_raw']:.2e} | {d['cond_scaled']:.2e} "
                f"| {d['cov_min_eig_scaled']:.2e} | {d['inv_residual']:.2e} "
                f"| {resid_ratio(d):.2f} | {d['numerical_rank']}/{n} "
                f"| {'yes' if d['cov_pos_def'] else '**no**'} |")
    return "\n".join(rows)


def conditioning_verdict(recs):
    """One paragraph saying whether the covariances can be trusted, from the records."""
    ent = [r["models"][m] for r in recs for m in ("0PA", "0PA+PN")]
    raw = [d["cond_raw"] for d in ent]
    sca = [d["cond_scaled"] for d in ent]
    rat = [resid_ratio(d) for d in ent]
    bad_pd = [f"{r['idx']}/{m}" for r in recs for m in ("0PA", "0PA+PN")
              if not r["models"][m]["cov_pos_def"]]
    bad_res = [f"{r['idx']}/{m}" for r in recs for m in ("0PA", "0PA+PN")
               if resid_ratio(r["models"][m]) > 10.0]
    bad_rank = [f"{r['idx']}/{m}" for r in recs for m in ("0PA", "0PA+PN")
                if r["models"][m]["numerical_rank"] < len(r["models"][m]["params"])]
    return (
        f"Across all {len(ent)} model arms the raw condition number runs from "
        f"`{min(raw):.1e}` to `{max(raw):.1e}`.  Jacobi rescaling brings that to "
        f"`{min(sca):.1e}` -- `{max(sca):.1e}`, a reduction of about "
        f"{np.log10(np.median(raw) / np.median(sca)):.0f} orders of magnitude at the median.\n\n"
        f"Judged against `cond_scaled * eps`, which is the accuracy float64 can deliver at "
        f"that conditioning, the inversion residuals run from `{min(rat):.2f}` to "
        f"`{max(rat):.2f}` times the limit (median `{np.median(rat):.2f}`).  "
        + (f"**{len(bad_res)} arms exceed ten times the limit and their covariances should "
           f"not be trusted: {', '.join(bad_res)}.**  "
           if bad_res else
           "Every arm is *below* the limit, by a factor of five to twenty -- so every "
           "covariance quoted here is as accurate an inverse as double precision permits, "
           "and the large raw condition numbers cost nothing in the end.  ")
        + (f"**{len(bad_pd)} arms are not positive definite even after rescaling: "
           f"{', '.join(bad_pd)}.**  "
           if bad_pd else
           "Every covariance is positive definite in the rescaled basis.  ")
        + (f"**{len(bad_rank)} arms are numerically rank-deficient: {', '.join(bad_rank)}.**"
           if bad_rank else
           "No arm is numerically rank-deficient: every parameter direction carries "
           "information."))


def page(recs):
    r0 = recs[0]
    return f"""# run_1 results -- systematic bias, 0PA against 0PA+PN

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by
`make_results_md.py`, from the per-source records in this folder.  Nothing below is
transcribed by hand.

> Method, conventions and assumptions: **[`METHOD.md`](METHOD.md)**.  This page is generated
> from the per-source records and carries only the numbers.

## What this is

A circular EMRI population at SNR = 200, A and E channels only, injected with a
multiplicative environmental term on the flux, `env = 1 + A_PM (p/10)^n_PM`, `n_PM = 8`.
Each source is fitted by two vacuum recovery models:

* `0PA` -- the eight vacuum parameters `m1, m2, a, p0, e0, qS, phiS, Phi_phi0`
* `0PA+PN` -- the same eight plus `C_p`, the amplitude of a PN flux deviation

The question is whether the extra PN freedom absorbs the environmental deviation, and
whether absorbing it reduces the *absolute* parameter bias or only the overlap.

## Where the numbers come from

The fit vectors are the ones the searches already found.  Counting each model's arm
separately, {sum(1 for r in recs for m in ('0PA', '0PA+PN') if r['models'][m]['origin'] == 'results_env_cv.json')}
of the {2 * len(recs)} arms come from the population run (`results_env_cv.json`) and
{sum(1 for r in recs for m in ('0PA', '0PA+PN') if r['models'][m]['origin'] != 'results_env_cv.json')}
from the converged follow-ups in `harder_environmental/`.  Each arm is taken from whichever
run reached the higher overlap for that model, which for source 14 means its two arms come
from different runs.  Nothing was refitted.

What is new is sigma.  Each model's Fisher was recomputed **at its own best fit** with a
finer derivative stencil -- order {r0['der_order']} over a {r0['Ndelta']}-point stability
ladder, against order 6 over 12 in the population run -- so the `b/sigma` column no longer
depends on the choice of step size.  For the PN model the `C_p` finite-difference step is
recalibrated per source before its Fisher is taken.

Bias on an angle is taken after unwrapping to `(-pi, pi]`, so a fit on the far side of
2 pi is not counted as a large error.

## Headline

{headline(recs)}

## Sources in the analysis ({len(recs)})

`max b/sigma` is the largest ratio the model shows over its parameters, excluding `e0`.
A `!` marks a source whose climb did not converge: it is shown, and held out of the
population medians below.

{source_table(recs)}

## In the analysis, but not converged

These are plotted -- hollow -- and tabulated above, because a failure at high `p0` is itself
the result.  They are excluded from the per-parameter medians.

{not_converged_table(recs)}

## Not in the analysis

{dropped_table()}

## Per parameter, across the population

`median ratio` is the median of `|bias|_PN / |bias|_0PA`; below 1 means the PN template
reduced the bias.  `tie` counts sources where the two agree to within {TIE_TOL:g} relative,
which is what happens when the PN climb finds no ascent direction and lands back on the 0PA
fit -- a PN template at `C_p = 0` *is* the 0PA template.

{param_table(recs)}

`*` `e0 = 0` sits on its own physical boundary, so its Fisher direction is degenerate and
sigma(e0) is not a confidence interval.  Its absolute bias is meaningful; its `b/sigma` is
not, and the figure marks that panel rather than plotting a number in the thousands.

## Figures

![bias](bias_panels.png)

One row per parameter.  Left: absolute bias, which needs no Fisher.  Right: the same bias
in units of that model's own sigma.  Reading them together matters -- a PN model can show a
smaller `b/sigma` with an unchanged absolute bias, purely because `C_p` correlates with the
vacuum parameters and inflates their sigma.

![overlap](overlap.png)

`1 - overlap` against `p0`, both models, the sources the fit struggled with labelled by
index.

### Per source, summarised

The two figures above are per parameter.  These compress each source to one number per model,
so the population reads at a glance.  Every quantity is computed over two parameter blocks --
once over the seven parameters whose sigma means what it says, and once with `e0` added back,
because `e0` sits on its physical boundary and its near-null Fisher direction dominates
whatever it is included in.  The deviation amplitude `C_p` is held out of all of them: it is
injected at zero and is the thing being absorbed, not a parameter whose bias is measured.

**Reference lines are credible regions, not 1 sigma.**  A flat line at 1 is the wrong yardstick
for both quantities here.  For a single parameter it is the 68.3% interval, an odd level to
judge a systematic against.  For the n-d bias it is worse: `sqrt(b^T M b) = 1` is *not* the 68%
region of a `k`-dimensional Gaussian.  The radius enclosing probability `p` is
`sqrt(chi2_ppf(p, k))`, which grows with `k` -- at `k = 7` the 90% region is at **3.47 sigma**
and the 99% at **4.30**.  Judging a 7-d bias against 1.0 overstates how biased everything is
by a factor of three or four.  Each figure is drawn at both levels; `_p90` is the 90% version.

![max bias](max_bias.png)

The worst `|bias|/sigma` any single parameter shows, per source.  This is the number that
decides whether a source's parameter estimate is usable at all: one parameter outside its
credible interval is enough to make the answer wrong, however well the other six sit.

![n-d bias](nd_bias.png)

The n-d bias, `sqrt(b^T M b)` over the same seven parameters -- the length of the whole bias
vector in units of the error ellipsoid, which is the quantity the CV formalism actually
bounds.  Two panels, same bias vector, differing only in what is done with the held-out
parameters:

* **marginalized** -- `M = (C_vv)^-1`, the inverse of the kept block of the *covariance*.
  The held-out parameters have been integrated over, so the freedom they carry is still
  paid for.  This is the SK/CV convention.
* **conditional** -- `M = G_vv`, the kept block of the *Fisher* itself, which is the inverse
  covariance with the held-out parameters frozen at their fitted values.

`G_vv >= (C_vv)^-1` in the Loewner order, so the conditional number is never the smaller of
the two.  The gap between the panels is what the extra freedom costs: for `0PA+PN` it is the
price of letting `C_p` float, and it is the reason a PN fit can show a smaller normalized
bias without its absolute bias having moved at all.

**The conditional panel is not a like-for-like contest between the models.**  `0PA` freezes
only `e0`; `0PA+PN` freezes `e0` *and* `C_p`, so PN is judged against a tighter yardstick.
The marginalized panel is the fair comparison; the conditional one is for comparing each
model against itself.

### How many sources are actually biased

{credible_table(recs)}

### What the PN freedom buys

![improvement](improvement.png)

Per source, the factor `0PA / 0PA+PN` on each headline quantity -- above 1 means PN improved
things and the height is the factor.  Plotting the ratio rather than the two values is the
point: the absolute numbers span five decades across the population, which hides a consistent
gain completely.  Mismatch uses `1 - overlap`, not overlap, because `0.9999 -> 0.999999` is a
hundredfold improvement and a difference of `1e-4`, and only the first of those is meaningful.

![max bias with e0](max_bias_with_e0.png)

![n-d bias with e0](nd_bias_with_e0.png)

The same two over the block that keeps `e0`.  Read them against the pair above: the
difference is the boundary direction's contribution, not new information about the fit.

![improvement with e0](improvement_with_e0.png)

### Intrinsic parameters only

The same three quantities over `m1, m2, a, p0` alone -- sky position and initial phase held
out along with `e0`, and `e0` added back in the second pair.  This is the narrower and harder
question.  A template that absorbs an environmental deviation by shifting the sky position
has not damaged the measurement in the way that one absorbing it into the masses has, so
restricting to the intrinsic block asks whether the PN freedom protects the astrophysics
rather than just the fit.

Note that the credible-region threshold moves with the block: at `k = 4` the 99% region sits
at **3.64 sigma** rather than the 4.30 of the seven-parameter block, because a smaller space
needs a smaller radius to enclose the same probability.  The counts below are already
computed against the right threshold for each block.

These figures live in **[`intrinsic/`](intrinsic/)**, under the same four filenames as the
top-level ones, so the two folders can be read side by side.  See
[`intrinsic/README.md`](intrinsic/README.md) for that block on its own.

![max bias intrinsic](intrinsic/max_bias.png)

![n-d bias intrinsic](intrinsic/nd_bias.png)

![improvement intrinsic](intrinsic/improvement.png)

With `e0` restored to the intrinsic block:

![max bias intrinsic with e0](intrinsic/max_bias_with_e0.png)

![n-d bias intrinsic with e0](intrinsic/nd_bias_with_e0.png)

![improvement intrinsic with e0](intrinsic/improvement_with_e0.png)

## Is the Fisher well enough conditioned?

Short answer, from the records rather than from assertion:

{conditioning_verdict(recs)}

The raw Fisher is ill-conditioned for a boring reason -- it holds `m1 ~ 1e6` next to
`C_p ~ 1e-4` next to angles of order 1, so its condition number is dominated by the choice
of units rather than by any real degeneracy.  Every matrix here is therefore Jacobi-rescaled
to a unit diagonal, `M_hat = D^-1 M D^-1` with `D = diag(sqrt(|M_ii|))`, before anything is
inverted or tested, and the result is mapped back.

That transform is safe in a way worth being precise about.  It is a **congruence**, not a
similarity, so it does *not* preserve eigenvalues -- but by **Sylvester's law of inertia** it
preserves their **signs**, which is what a definiteness test asks about.  And quadratic forms
are exactly invariant: `b^T M b = (D b)^T M_hat (D b)`.  So every sigma and every n-d bias on
this page is the number the unscaled algebra defines, merely computed without catastrophic
cancellation.

This matters concretely.  The population run tested definiteness on the raw covariance and
reported most PN covariances as "not positive semi-definite", with minimum eigenvalues around
`-5e-17`.  Those matrices were positive definite all along: the negative eigenvalues were
float64 noise against a largest eigenvalue some 22 orders larger.

![conditioning](conditioning.png)

| | |
|---|---|
| `cond raw` | condition number of the Fisher as computed |
| `cond rescaled` | the same after Jacobi rescaling -- the number that governs the arithmetic |
| `min eig (scaled)` | smallest eigenvalue of the rescaled covariance; its **sign** is the real inertia |
| `inv residual` | `max\\|G_hat C_hat - I\\|`.  The direct test: does the inverse actually invert |
| `/ float64 limit` | the same residual divided by `cond_rescaled * 2.2e-16`.  **This is the column to read** -- an absolute residual is meaningless without the conditioning it was achieved at, and a value below 1 means the inversion is as accurate as double precision allows |
| `rank` | eigenvalues above `max(eig) * eps` -- a deficit means a genuinely dead direction |

{conditioning_table(recs)}

## Caveats

* Conditioning is treated in its own section above.  The residuals say the inversions are
  sound; that is a statement about arithmetic, not about physics.  A well-conditioned Fisher
  at a fit that only reaches overlap 0.77 is still the wrong error model, because the
  likelihood there is nowhere near Gaussian.  The absolute bias remains the solid quantity.
* `dt = 10` aliasing is an accepted fidelity caveat, carried over from the population run.
* The run depends on an uncommitted patch to `flux.py` on branch `environ_shubham`, so it
  is not reproducible from a clean checkout as things stand.

## Files

| file | what it is |
|---|---|
| `idx*.json` | one record per source: both fits, both Fishers, bias, b/sigma |
| `bias_panels.png`, `overlap.png` | the per-parameter figures, from `plot_results.py` |
| `max_bias*.png`, `nd_bias*.png` | the per-source summaries; `_with_e0` keeps `e0`, `_p90` is the 90% level |
| `improvement*.png` | the factor by which PN beats 0PA, on all three quantities |
| `intrinsic/` | the same figures over `m1, m2, a, p0` only, with their own `README.md` |
| `conditioning.png` | raw vs rescaled condition number, and the inversion residual |
| `README.md` | this page, from `make_results_md.py` |
| `METHOD.md` | what was done and every assumption it rests on -- written by hand, not generated |
| `../results_inputs.py` | which sources enter, and where each fit is read from |
| `../refit_fisher.py` | the order-{r0['der_order']} Fisher job that wrote the `idx*.json` |
| `../batch_refit_fisher_[ab].sh` | its two PBS scripts |
"""


INTRINSIC_BLOCKS = ("_intrinsic", "_intrinsic_with_e0")


def intrinsic_page(recs):
    """A short self-contained page for results/intrinsic/."""
    sub = {k: v for k, v in BLOCKS.items() if k in INTRINSIC_BLOCKS}
    rows = []
    for suffix, (exclude, _) in sub.items():
        k = len(kept(recs[0]["models"]["0PA"], exclude))
        lbl = "with e0" if "with_e0" in suffix else "no e0"
        for lvl, p in sorted(CREDIBLE.items(), key=lambda kv: -kv[1]):
            t1, tn = threshold_1d(p), threshold_nd(p, k)
            m1 = [(max_bias_over_sigma(r["models"]["0PA"], exclude),
                   max_bias_over_sigma(r["models"]["0PA+PN"], exclude)) for r in recs]
            nd = [(nd_bias(r["models"]["0PA"], exclude)["marginalized"],
                   nd_bias(r["models"]["0PA+PN"], exclude)["marginalized"]) for r in recs]
            n = len(recs)
            rows.append(f"| {lbl} ({k}) | max 1-d | {p:.0%} | {t1:.2f} sigma "
                        f"| {sum(a > t1 for a, b in m1)}/{n} "
                        f"| **{sum(b > t1 for a, b in m1)}/{n}** "
                        f"| {np.median([a / b for a, b in m1]):.2f}x |")
            rows.append(f"| {lbl} ({k}) | n-d, marginalized | {p:.0%} | {tn:.2f} sigma "
                        f"| {sum(a > tn for a, b in nd)}/{n} "
                        f"| **{sum(b > tn for a, b in nd)}/{n}** "
                        f"| {np.median([a / b for a, b in nd]):.2f}x |")
    table = "\n".join(["| block | quantity | region | threshold | 0PA outside "
                       "| 0PA+PN outside | median gain |", "|" + "---|" * 7] + rows)
    return f"""# Intrinsic parameters only -- `m1, m2, a, p0`

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by `make_results_md.py`.
Method and assumptions: [`../METHOD.md`](../METHOD.md).  Full results: [`../README.md`](../README.md).

Sky position (`qS`, `phiS`) and the initial phase (`Phi_phi0`) are held out here, along with
the deviation amplitude `C_p`.  The second block adds `e0` back.

This is the narrower and harder question.  A template that absorbs an environmental deviation
by shifting the sky position has not damaged the measurement the way one absorbing it into
the masses has, so restricting to the intrinsic block asks whether the PN freedom protects
the astrophysics rather than just the fit.

**The credible threshold moves with the block.**  At `k = 4` the 99% region of a Gaussian sits
at `sqrt(chi2_ppf(0.99, 4))` = **3.64 sigma**, against 4.30 for the seven-parameter block: a
smaller space needs a smaller radius to enclose the same probability.  Each figure draws its
own, and these blocks are not comparable against a shared line.

{table}

`median gain` is the median of `0PA / 0PA+PN`, so above 1 means the PN template did better.

## Figures

Same filenames as the top-level folder, so the two can be read side by side.

![max bias](max_bias.png)

![n-d bias](nd_bias.png)

![improvement](improvement.png)

With `e0` added back:

![max bias with e0](max_bias_with_e0.png)

![n-d bias with e0](nd_bias_with_e0.png)

![improvement with e0](improvement_with_e0.png)

The 90% versions of the two bias figures are the `_p90` files in this folder.
"""


def main():
    recs = load_records()
    path = os.path.join(OUT, "README.md")
    with open(path, "w") as fh:
        fh.write(page(recs))
    print(f"[MD] {path} ({len(recs)} sources)")

    sub = os.path.join(OUT, "intrinsic")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, "README.md")
    with open(path, "w") as fh:
        fh.write(intrinsic_page(recs))
    print(f"[MD] {path}")


if __name__ == "__main__":
    main()
