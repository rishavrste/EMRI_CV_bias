"""Build a markdown report of the environmental CV population study.

Every source is included. Sources whose climb did not converge, or whose 0PA+PN climb
never moved off the 0PA answer, are flagged with the reason but kept in every table.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# These live inside the run directory they analyse, so the default target is right here.
# Point them at another round with an explicit path: python make_report.py ../run_2
DEFAULT_RUN = "."
RESULTS = "results_env_cv.json"
REPORT = "results_env_cv.md"
NOTEBOOK = os.path.join(os.pardir, "test.ipynb")   # deep dive, shared across runs

OV_MIN = 0.999      # below this a climb never reached a maximum
PN_GAIN_MIN = 1e-6  # PN must beat 0PA by more than the ~5e-9 evaluation noise floor


# ---------------------------------------------------------------- selection

def pn_gain(rec):
    """Overlap the PN model gains over 0PA on the same source."""
    return rec["fit_pn"]["ov_final"] - rec["fit_0pa"]["ov_final"]


def drop_reason(rec):
    """Why this source is unusable, or None if it is fine."""
    ov0, ovp = rec["fit_0pa"]["ov_final"], rec["fit_pn"]["ov_final"]
    if min(ov0, ovp) < OV_MIN:
        return f"climb did not converge (0PA ov = {ov0:.6f}, PN ov = {ovp:.6f})"
    if pn_gain(rec) < PN_GAIN_MIN:
        return f"PN climb never left the 0PA point (gain = {pn_gain(rec):.2e})"
    return None


def split(sources):
    kept = [r for r in sources if drop_reason(r) is None]
    dropped = [(r, drop_reason(r)) for r in sources if drop_reason(r) is not None]
    return kept, dropped


# ---------------------------------------------------------------- formatting

def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def population_table(kept):
    rows = [[str(r["idx"]), f"{r['m1']:.4e}", f"{r['m2'] if 'm2' in r else r['source_params']['m2']:.0f}",
             f"{r['source_params']['a']:.2f}", f"{r['p0']:.3f}", f"{r['source_params']['e0']:.2f}",
             f"{r['dist']:.3f}", f"{r['T']:.4f}", f"{r['dt']:.0f}",
             f"{r['A_PM']:.4e}", f"{r['snr']:.2f}"] for r in kept]
    return table(["src", "m1 [Msun]", "m2", "a", "p0", "e0", "d [Gpc]", "T [yr]", "dt [s]",
                  "A_PM", "SNR"], rows)


def climb_table(kept):
    rows = []
    for r in kept:
        z, p = r["fit_0pa"], r["fit_pn"]
        rows.append([str(r["idx"]), f"{z['ov_start']:.6f}", f"{z['ov_final']:.10f}",
                     f"{p['ov_final']:.10f}", f"{pn_gain(r):.2e}",
                     f"{1 - z['ov_final']:.2e}", f"{1 - p['ov_final']:.2e}",
                     p["seed"], f"{p['seed_spread']:.2e}",
                     z["stop_reason"], p["stop_reason"], f"{r['seconds']:.0f}"])
    return table(["src", "ov start", "0PA ov", "PN ov", "PN gain", "1-ov 0PA", "1-ov PN",
                  "PN seed", "seed spread", "0PA stop", "PN stop", "s"], rows)


def bias_table(kept, tag, names):
    rows = []
    for r in kept:
        b = r[tag]["bias_over_sigma"][:len(names)]
        rows.append([str(r["idx"])] + [f"{x:+.4f}" for x in b]
                    + [f"{np.max(np.abs(b)):.4f}"])
    return table(["src"] + names + ["max abs(b/s)"], rows)


def summary_table(kept, names):
    rows = []
    for i, n in enumerate(names):
        corr = np.mean([r["corr_A_PM"][i] for r in kept])
        b0 = np.median([abs(r["fit_0pa"]["bias_over_sigma"][i]) for r in kept])
        bp = np.median([abs(r["fit_pn"]["bias_over_sigma"][i]) for r in kept])
        n0 = sum(abs(r["fit_0pa"]["bias_over_sigma"][i]) > 1 for r in kept)
        npn = sum(abs(r["fit_pn"]["bias_over_sigma"][i]) > 1 for r in kept)
        rows.append([n, f"{corr:+.4f}", f"{b0:.4f}", f"{bp:.4f}", f"{b0 / bp:.1f}x",
                     f"{n0}/{len(kept)}", f"{npn}/{len(kept)}"])
    return table(["param", "mean corr(A_PM)", "median abs(b/s) 0PA", "median abs(b/s) PN",
                  "suppression", "gt 1 sigma 0PA", "gt 1 sigma PN"], rows)


def deviation_table(kept):
    rows = []
    for r in kept:
        p = r["fit_pn"]
        rows.append([str(r["idx"]),
                     f"{p['params'][9]:+.6e}", f"{p['sigma'][9]:.4e}", f"{p['bias_over_sigma'][9]:+.4f}",
                     f"{p['params'][10]:+.6e}", f"{p['sigma'][10]:.4e}", f"{p['bias_over_sigma'][10]:+.4f}",
                     f"{r['dev_delta_range']['C_p'][0]:.3e} -> {r['dev_delta_range']['C_p'][-1]:.3e}",
                     f"{r['dev_delta_range']['C_e'][0]:.3e} -> {r['dev_delta_range']['C_e'][-1]:.3e}"])
    return table(["src", "C_p fit", "sigma C_p", "b/s", "C_e fit", "sigma C_e", "b/s",
                  "C_p step range", "C_e step range"], rows)


def per_source_block(rec, names11):
    z, p = rec["fit_0pa"], rec["fit_pn"]
    lines = [f"### Source {rec['idx']}\n",
             f"m1 = {rec['m1']:.6e} Msun, p0 = {rec['p0']:.6f}, "
             f"A_PM = {rec['A_PM']:.6e}, n_PM = {rec['n_PM']:.0f}, "
             f"SNR = {rec['snr']:.4f} (A = {rec['snr_channels'][0]:.4f}, "
             f"E = {rec['snr_channels'][1]:.4f}), T = {rec['T']:.4f} yr, dt = {rec['dt']:.0f} s\n",
             f"0PA: ov {z['ov_start']:.10f} -> {z['ov_final']:.12f}, chi2 = {z['chi2']:.6e}, "
             f"{z['n_iter']} iters, {z['n_accept']} accepted, stop = {z['stop_reason']}\n",
             f"0PA+PN: ov -> {p['ov_final']:.12f}, chi2 = {p['chi2']:.6e}, "
             f"{p['n_iter']} iters, {p['n_accept']} accepted, stop = {p['stop_reason']}, "
             f"seed = {p['seed']} (spread {p['seed_spread']:.3e})\n"]
    rows = []
    for i, n in enumerate(names11):
        t = rec["truth_11"][i]
        row = [n, f"{t:.10g}", f"{z['params'][i]:.10g}" if i < 9 else "-",
               f"{z['bias_over_sigma'][i]:+.4f}" if i < 9 else "-",
               f"{p['params'][i]:.10g}", f"{p['bias_over_sigma'][i]:+.4f}"]
        rows.append(row)
    lines.append(table(["param", "injected", "0PA best fit", "0PA b/s",
                        "0PA+PN best fit", "PN b/s"], rows))
    return "\n".join(lines)


# ---------------------------------------------------------------- notebook

# test.ipynb is the single-source deep dive that this population study grew out of.
# It never wrote its summary to disk, so the numbers are read back out of the stored
# cell outputs rather than from a results file.
NB_SUMMARY_CELL = 35          # the cell that prints the summary dict as JSON
NB_VERBATIM_CELLS = {21: "Marginalization over the environmental parameters",
                     23: "Correlation of the vacuum parameters with A_PM and n_PM"}


def nb_stream(cell):
    """Concatenated stdout of one notebook cell."""
    return "".join("".join(o.get("text", []))
                   for o in cell.get("outputs", [])
                   if o.get("output_type") == "stream")


def nb_load(path):
    with open(path) as f:
        cells = json.load(f)["cells"]
    text = nb_stream(cells[NB_SUMMARY_CELL])
    summary, _ = json.JSONDecoder().raw_decode(text, 0)   # a [DONE] line follows the JSON
    verbatim = {i: "\n".join(l for l in nb_stream(cells[i]).splitlines()
                             if not l.startswith("[PLOT]")).strip()
                for i in NB_VERBATIM_CELLS}
    return summary, verbatim


def nb_config_lines(nb):
    c, sp = nb["config"], nb["source_params"]
    return [f"- Source: m1 = {sp['m1']:.6e} Msun, m2 = {sp['m2']}, a = {sp['a']}, "
            f"p0 = {sp['p0']}, e0 = {sp['e0']}, d = {sp['dist']} Gpc",
            f"- Environment: A_PM = {nb['environmental']['A_PM']:.6e}, "
            f"n_PM = {nb['environmental']['n_PM']:.0f}, A_GC = {nb['environmental']['A_GC']}, "
            f"n_GC = {nb['environmental']['n_GC']}",
            f"- Window: T = {c['T']:.6f} yr ({c['T'] / c['T_plunge']:.2f} x t_plunge = "
            f"{c['T_plunge']:.6f} yr), dt = {c['dt']} s, {c['nchannels']} channels, "
            f"{c['n_bins_kept']:,} bins kept in [{c['f_min']:g}, {c['f_max']:g}] Hz",
            f"- Fisher: Ndelta = {c['Ndelta']}, derivative order = {c['der_order']}",
            f"- SNR: GR = {nb['snr_gr']:.6f}, environmental = {nb['snr_env']:.6f}",
            f"- Injected signal vs a GR template at the truth: overlap = {nb['overlap']:.12f}, "
            f"mismatch = {nb['mismatch']:.6e}, chi2 = {nb['chi2_residual']:.6e}",
            f"- Dephasing over the window: dPhi_phi = {nb['dPhi_phi']:.6e} rad, "
            f"dPhi_r = {nb['dPhi_r']:.6e} rad"]


def nb_sigma_table(nb):
    rows = [[n, f"{g:.6e}", f"{e:.6e}", f"{e / g:.5f}"]
            for n, g, e in zip(nb["params"], nb["sigma_gr"], nb["sigma_env"])]
    return table(["param", "sigma GR", "sigma ENV (coeffs fixed)", "ratio ENV/GR"], rows)


def nb_cv_table(nb):
    cv = nb["cv"]
    z, p = cv["zero_pa"], cv["zero_pa_pn"]
    rows = []
    for i, n in enumerate(cv["params_11"]):
        t = cv["truth_11"][i]
        vac = ([f"{z['x_bf'][i]:.10g}", f"{z['bias'][i]:+.6e}",
                f"{z['bias'][i] / z['sigma'][i]:+.4f}"] if i < 9 else ["-", "-", "-"])
        rows.append([n, f"{t:.10g}"] + vac
                    + [f"{p['x_bf'][i]:.10g}", f"{p['bias'][i]:+.6e}",
                       f"{p['bias'][i] / p['sigma'][i]:+.4f}"])
    return table(["param", "injected", "0PA best fit", "0PA bias", "0PA b/s",
                  "0PA+PN best fit", "0PA+PN bias", "0PA+PN b/s"], rows)


def notebook_section(nb, verbatim):
    cv = nb["cv"]
    z, p = cv["zero_pa"], cv["zero_pa_pn"]
    out = ["\n# Appendix: single-source deep dive (`test.ipynb`)\n",
           "A separate, louder source run to plunge at finer cadence. It is not part of the "
           "population above and is reported on its own terms: the population uses SNR 100 and "
           f"dt = 10 s, this uses SNR {nb['snr_gr']:.0f} and dt = {nb['config']['dt']} s.\n",
           "## Configuration\n"] + nb_config_lines(nb)
    out += ["\n## Fisher errors, environment on vs off (coefficients held fixed)\n",
            nb_sigma_table(nb),
            f"\nFisher condition numbers: GR {nb['cond_fisher_gr']:.4e}, "
            f"environmental {nb['cond_fisher_env']:.4e}. Every ratio is below 1 because with "
            "`A_PM, n_PM` held fixed the environmental model is never charged for the extra "
            "physics it carries; the next section frees them.\n"]
    for i, title in NB_VERBATIM_CELLS.items():
        out += [f"\n## {title}\n", "```", verbatim[i], "```\n"]
    out += [f"\n## Iterated CV recovery\n",
            f"- 0PA: overlap = {z['ov']:.12f}, mismatch = {1 - z['ov']:.6e}, "
            f"chi2 = {z['chi2']:.6e}",
            f"- 0PA+PN: overlap = {p['ov']:.12f}, mismatch = {1 - p['ov']:.6e}, "
            f"chi2 = {p['chi2']:.6e}",
            f"- PN gain over 0PA (mismatch ratio) = {cv['pn_gain']:.4f}x\n",
            nb_cv_table(nb)]
    over0 = sum(abs(z["bias"][i] / z["sigma"][i]) > 1 for i in range(9))
    overp = sum(abs(p["bias"][i] / p["sigma"][i]) > 1 for i in range(9))
    out += [f"\n**{over0}/9 vacuum parameters are biased past 1 sigma under 0PA, "
            f"{overp}/9 under 0PA+PN.**\n"]
    return "\n".join(out)


# ---------------------------------------------------------------- report

def build(data, nb, nb_verbatim):
    S = data["sources"]
    # Round 2 keeps every source: an unconverged climb is reported and flagged, never removed,
    # so the tables and the figures cover the whole population.
    kept = S
    flagged = [(r, drop_reason(r)) for r in S if drop_reason(r) is not None]
    n9, n11 = data["params_9"], data["params_11"]
    corr_all = [abs(r["corr_A_PM"][i]) for r in kept for i in range(9)]
    bias_all = [abs(r["fit_0pa"]["bias_over_sigma"][i]) for r in kept for i in range(9)]
    rho = np.corrcoef(corr_all, bias_all)[0, 1]
    over0 = sum(abs(x) > 1 for r in kept for x in r["fit_0pa"]["bias_over_sigma"][:9])
    overp = sum(abs(x) > 1 for r in kept for x in r["fit_pn"]["bias_over_sigma"][:9])
    npts = 9 * len(kept)

    out = [f"# Environmental CV bias — population results\n",
           f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
           f"from `{RESULTS}` (run of {data['generated_utc']}).\n",
           "## Setup\n",
           f"- Population: `{data['population']}`, generated {data['population_generated']}",
           f"- Channels: {data['nchannels']} (A, E); band "
           f"{data['freq_band'][0]:g} - {data['freq_band'][1]:g} Hz",
           f"- Injection: environmental PM flux deviation, A_GC = n_GC = 0",
           f"- Vacuum model (0PA): {', '.join(n9)}",
           f"- Deviation model (0PA+PN): {', '.join(n11)}",
           f"- Environmental Fisher basis: {', '.join(data['params_env'])}, free = {', '.join(data['env_free'])}\n",
           f"> {data['note']}\n",
           "## Sources included\n",
           f"All {len(S)} sources are included. {len(S) - len(flagged)} of them pass both "
           f"convergence criteria: both climbs reach overlap > {OV_MIN}, and the 0PA+PN climb "
           f"beats 0PA by more than {PN_GAIN_MIN:g} (i.e. it actually moved off the 0PA "
           f"answer). The rest are kept and flagged, not excluded.\n",
           f"Included: {[r['idx'] for r in kept]}\n",
           "Flagged as unconverged (still plotted and tabulated):\n"]
    out += [f"- source {r['idx']}: {why}" for r, why in flagged]
    out += ["\n## Population\n", population_table(kept),
            "\n## Climb diagnostics\n", climb_table(kept),
            "\n## Summary: correlation with A_PM vs recovered bias\n", summary_table(kept, n9),
            f"\nOver the {npts} (source, parameter) points of the kept subset: "
            f"**{over0}/{npts} ({100 * over0 / npts:.1f}%) biased by more than 1 sigma under the "
            f"vacuum 0PA model, {overp}/{npts} ({100 * overp / npts:.1f}%) under 0PA+PN.** "
            f"Rank correlation between |corr(A_PM, param)| and |bias/sigma| under 0PA: "
            f"**{rho:+.4f}**.\n",
            "\n## Bias / sigma, vacuum 0PA model\n", bias_table(kept, "fit_0pa", n9),
            "\n## Bias / sigma, 0PA+PN model\n", bias_table(kept, "fit_pn", n9),
            "\n## Recovered deviation parameters and their calibrated step ranges\n",
            deviation_table(kept),
            "\n## Per-source parameter vectors\n"]
    out += [per_source_block(r, n11) for r in kept]
    out.append(notebook_section(nb, nb_verbatim))
    return "\n".join(out)


def run_dir(argv):
    """The run directory to read and write, absolute."""
    d = argv[0] if argv else DEFAULT_RUN
    return d if os.path.isabs(d) else os.path.join(HERE, d)


def main(argv=()):
    d = run_dir(argv)
    results, report = os.path.join(d, RESULTS), os.path.join(d, REPORT)
    with open(results) as f:
        data = json.load(f)
    nb, nb_verbatim = nb_load(os.path.join(HERE, NOTEBOOK))
    with open(report, "w") as f:
        f.write(build(data, nb, nb_verbatim))
    clean, flagged = split(data["sources"])
    print(f"[report] {report}: all {len(data['sources'])} sources included; "
          f"{len(clean)} converged cleanly {[r['idx'] for r in clean]}, "
          f"{len(flagged)} flagged {[r['idx'] for r, _ in flagged]}")


if __name__ == "__main__":
    main(sys.argv[1:])
