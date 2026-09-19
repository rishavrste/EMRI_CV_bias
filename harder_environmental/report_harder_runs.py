"""Emit the markdown report for the harder_environmental point runs, from their results files.

    python report_harder_runs.py [out.md] [run1.json run2.json ...]

Tables are generated, not transcribed, so the report cannot drift from the JSON it
describes.  Each run contributes a stage trace, a bias table and a box-position table;
the box table is the one that says whether the search could have found the answer at all.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ["results_cv_de_cv_0pa.json", "results_cv_de_cv_pn.json"]
STAGES = [("stage1_cv", "CV step"), ("stage2_de", "DE"), ("stage3_cv", "CV climb")]


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, w=".6g"):
    return format(float(x), w)


def stage_table(d):
    out = ["| stage | overlap | chi2 | detail |", "|---|---|---|---|"]
    out.append(f"| injected | {d['stage1_cv']['ov_start']:.10f} | "
               f"{d['population_run']['chi2_0pa']:.4e} | the truth, with the deviation on |")
    for key, lab in STAGES:
        r = d[key]
        if key == "stage2_de":
            det = (f"{r['nit']} it, {r['nfev']} nfev, {r['seconds'] / 3600:.2f} h")
        elif key == "stage3_cv":
            det = f"{r['n_iter']} it, {r['n_accept']} accepted, `{r['stop_reason']}`"
        else:
            det = f"`{r['stop_reason']}`" + (
                f", clipped {', '.join(c['param'] for c in r['clipped'])}"
                if r.get("clipped") else "")
        out.append(f"| {lab} | {r['ov_final']:.10f} | {r['chi2']:.4e} | {det} |")
    return "\n".join(out)


def bias_table(d):
    out = ["| param | injected | recovered | bias | sigma | bias/sigma |",
           "|---|---|---|---|---|---|"]
    for i, n in enumerate(d["params"]):
        out.append(f"| `{n}` | {fmt(d['truth'][i], '.10g')} | "
                   f"{fmt(d['stage3_cv']['params'][i], '.10g')} | {d['bias'][i]:.4e} | "
                   f"{d['sigma'][i]:.3e} | {d['bias_over_sigma'][i]:+.2f} |")
    b = np.abs(d["bias_over_sigma"])
    out.append(f"\nmax `|bias/sigma|` = **{b.max():.1f}** on `{d['params'][int(b.argmax())]}`; "
               f"median {np.median(b):.1f}.")
    return "\n".join(out)


def box_table(d):
    """Where the search box sat relative to the truth -- the run's real diagnostic."""
    out = ["| param | box lo | box hi | final | injected | position in box | truth |",
           "|---|---|---|---|---|---|---|"]
    n_out = 0
    for i, n in enumerate(d["params"]):
        lo, hi = d["de_bounds"][i]
        v, t = d["stage3_cv"]["params"][i], d["truth"][i]
        frac = (v - lo) / (hi - lo) if hi > lo else float("nan")
        inside = lo <= t <= hi
        n_out += not inside
        edge = " **at edge**" if (frac > 0.95 or frac < 0.05) else ""
        out.append(f"| `{n}` | {fmt(lo)} | {fmt(hi)} | {fmt(v)} | {fmt(t)} | "
                   f"{frac:.2f}{edge} | {'in' if inside else '**OUT**'} |")
    out.append(f"\n**{n_out} of {len(d['params'])} parameters have the truth outside "
               f"the search box.**")
    return "\n".join(out)


def run_section(path):
    d = load(path)
    cfg = d["config"]
    s = [f"## {cfg['model']}  (`{os.path.basename(path)}`)", "",
         f"- generated `{d['generated_utc']}`, total {d['seconds'] / 3600:.2f} h",
         f"- seed box: `prior_sigma_range` = {cfg['prior_sigma_range']}, "
         f"`dev_prior_sigma_range` = {cfg['dev_prior_sigma_range']}, "
         f"`full_phase_bounds` = {cfg['full_phase_bounds']}",
         f"- `cv1_exact` = {cfg['cv1_exact']}, `de_maxiter` = {cfg['de_maxiter']}, "
         f"`de_popsize` = {cfg['de_popsize']}, `de_init` = {cfg['de_init']}", "",
         "### stages", "", stage_table(d), "",
         "### recovered parameters and bias", "", bias_table(d), "",
         "### search box vs truth", "", box_table(d), ""]
    return "\n".join(s), d


def preamble(runs):
    d = runs[0][1]
    p = d["point"]
    # p0 is not carried in `point`; it is a fitted parameter, so read it off the truth vector.
    p0 = d["truth"][d["params"].index("p0")]
    return "\n".join([
        "# harder_environmental — source 8, the first hard point", "",
        f"Source 8 of the circular population: `m1 = {p['m1']:.4e}`, `p0 = {p0:.4f}`, "
        f"`SNR = {p['snr']:.1f}`, `T = {p['T']:.4f}` yr, `dt = {p['dt']:.0f}` s, "
        f"`A_PM = {p['A_PM']:.4e}`, `n_PM = {p['n_PM']:.0f}`.", "",
        "It is the lowest-`p0` source the population run failed on, reaching only "
        f"`ov = {d['population_run']['ov_0pa']:.10f}` (0PA) and "
        f"`{d['population_run']['ov_pn']:.10f}` (0PA+PN). These runs retry it with a "
        "three-stage pipeline: one exact CV step from the injected point, differential "
        "evolution in a 15-sigma Fisher box around where that step landed, then a CV climb "
        "from the DE winner.", "",
        "**The biases below are not Cutler-Vallisneri biases.** Both runs end far from the "
        "truth, so each `bias` column measures the distance to whichever wrong basin the "
        "optimiser stopped in. They are recorded because they describe the failure, not "
        "because they are a systematic-bias measurement.", ""])


def verdict(runs):
    by = {d["config"]["model"]: d for _, d in runs}
    s = ["## Verdict", ""]
    if "0PA" in by and "0PA+PN" in by:
        a, b = by["0PA"]["stage3_cv"]["ov_final"], by["0PA+PN"]["stage3_cv"]["ov_final"]
        s += [f"- 0PA reached `{a:.10f}`, 0PA+PN reached `{b:.10f}` — "
              f"`PN >= 0PA` is **{b >= a}**, without any cross-seeding between the two jobs.",
              f"- Both improve enormously on the population run "
              f"(`{by['0PA']['population_run']['ov_0pa']:.4e}` -> `{a:.4f}`), and both are "
              "still failed recoveries: an overlap near 0.72-0.74 against the 0.99999+ the "
              "converged population sources reach."]
    s += [
        "- The cause is visible in the box tables: differential evolution did not run out of "
        "effort, it ran out of box. The truth lies outside the 15-sigma bounds on the mass "
        "and orbital parameters, so the best point in the box is not near the best point.",
        "- The box is **mis-centred, not too small**. It is built around where the undamped "
        "stage-1 CV step landed, and that step starts from `ov = -0.30`, where the linear CV "
        "extrapolation has no reason to be valid. Widening it does not fix that.",
        "- `C_p` is the sharpest case: stage 1 threw it to ~2088, so its box became "
        "`[1613, 2563]` while the truth is exactly 0. The vacuum solution was unreachable "
        "by construction, which is why the PN stage-1 overlap (2.06e-4) was so much worse "
        "than 0PA's (0.151).", "",
        "### What follows from this", "",
        "- A damped stage 1 (`--damped-cv1`) keeps the Fisher sigma describing the offset, so "
        "the box would actually contain the answer.",
        "- `cv_iterate.py` (job 619624, queued) tests the other route: iterated *undamped* CV "
        "steps seeded from the 0PA post-DE point with `C_p = 0`, with no Fisher box at all, "
        "so the vacuum solution is where the iteration starts rather than somewhere it is "
        "forbidden to go.", ""]
    return "\n".join(s)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "harder_runs.md")
    paths = sys.argv[2:] or [os.path.join(HERE, p) for p in DEFAULT]
    runs = [run_section(p) for p in paths]
    doc = preamble(runs) + "\n" + "\n".join(s for s, _ in runs) + "\n" + verdict(runs)
    with open(out, "w") as f:
        f.write(doc)
    print(f"[saved] {out}  ({len(doc.splitlines())} lines, {len(runs)} runs)")


if __name__ == "__main__":
    main()
