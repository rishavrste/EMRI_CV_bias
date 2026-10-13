"""The final e0-fixed fits of run_1_second_gen and run_2, gathered in one list.

Both runs are circular populations at SNR 200 under 2nd-generation TDI (A, E, dt = 10), fitted
with e0 FIXED at 0 by their `lm_e0fixed.py`: 0PA = 7 params, 0PA+PN = 8 (+ C_p).  Each source
takes each model from the file named below; later entries in OVERRIDES replace earlier ones
(the fine-rung redos of the T = 1 step).

Nothing here touches a waveform.  The Fisher job and the plots read the same list, so the
vector a Fisher is taken at is by construction the one the bias is measured from.

    python results_inputs.py        # listing, to check the selection
"""
import copy
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.abspath(os.path.join(HERE, ".."))

MODELS = ("0PA", "0PA+PN")
KEY = {"0PA": "0pa", "0PA+PN": "pn"}           # model -> block name in the run JSONs

# Bias on an angle is taken after unwrapping to (-pi, pi], as in run_1/results.
ANGLES = {"qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"}

RUNS = {
    "run_1_second_gen": dict(
        population=os.path.join(ENV, "run_1", "population.json"),
        files=("results_lm_e0fixed_converged.json",
               "results_lm_e0fixed_ladder_a.json",
               "results_lm_e0fixed_ladder_b.json"),
        overrides={}),
    "run_2": dict(
        population=os.path.join(ENV, "run_2", "population.json"),
        files=("results_lm_e0fixed_lm.json",
               "results_lm_e0fixed_ladder_a.json",
               "results_lm_e0fixed_ladder_b.json",
               "results_lm_e0fixed_ladder_c.json",
               "results_lm_e0fixed_ladder_d.json",
               "results_lm_e0fixed_ladder_e.json"),
        # (model, idx) -> file: the T = 0.95 -> 1 redos in steps of 0.01
        overrides={("0PA", 2): "results_lm_e0fixed_redo0pa_idx2.json",
                   ("0PA", 11): "results_lm_e0fixed_redo0pa.json",
                   ("0PA", 19): "results_lm_e0fixed_redo0pa.json",
                   ("0PA+PN", 22): "results_lm_e0fixed_redopn.json"}),
}
RUN_LABEL = {"run_1_second_gen": "run_1 (2nd gen)", "run_2": "run_2"}


def unwrap(diff):
    return (diff + np.pi) % (2 * np.pi) - np.pi


def signed_bias_of(names, truth, fit):
    """Bias per parameter, angles unwrapped first, sign kept."""
    return {n: float(unwrap(f - t) if n in ANGLES else f - t)
            for n, t, f in zip(names, truth, fit)}


def _records(run, fname):
    with open(os.path.join(ENV, run, fname)) as fh:
        d = json.load(fh)
    return {r["idx"]: r for r in d["sources"]}


def _fit(block, origin):
    """One model's answer, in the single shape everything downstream reads."""
    return dict(params=list(block["params"]), theta=[float(v) for v in block["theta_final"]],
                ov_final=float(block["ov_final"]), chi2=float(block["chi2"]),
                stop_reason=block.get("stop_reason"), seed=block.get("seed"), origin=origin)


def _source(pop_entry, rec):
    """The population entry with the dist the run actually used (run_1_second_gen rescaled it)."""
    src = copy.deepcopy(pop_entry)
    src["dist"] = float(rec["dist"])
    src["source_params"]["dist"] = float(rec["dist"])
    return src


def _truth(src, names):
    return [float(src["source_params"].get(n, 0.0)) for n in names]


def load_run(run):
    """Every source of one run, in index order."""
    cfg = RUNS[run]
    with open(cfg["population"]) as fh:
        pop = {s["idx"]: s for s in json.load(fh)["sources"]}
    out = {}
    for fname in cfg["files"]:
        for idx, rec in _records(run, fname).items():
            fits = {m: _fit(rec[KEY[m]], f"{run}/{fname}") for m in MODELS}
            out[idx] = dict(rec=rec, fits=fits)
    for (model, idx), fname in cfg["overrides"].items():
        rec = _records(run, fname)[idx]
        out[idx]["fits"][model] = _fit(rec[KEY[model]], f"{run}/{fname}")

    sources = []
    for idx in sorted(out):
        rec, fits = out[idx]["rec"], out[idx]["fits"]
        src = _source(pop[idx], rec)
        sources.append(dict(run=run, idx=idx, name=f"{run}/idx{idx:02d}",
                            p0=float(src["p0"]), m1=float(src["m1"]), T=float(src["T"]),
                            source=src, fits=fits,
                            truth={m: _truth(src, fits[m]["params"]) for m in MODELS}))
    return sources


def load_sources(runs=tuple(RUNS)):
    return [s for run in runs for s in load_run(run)]


def main():
    src = load_sources()
    print(f"{len(src)} sources")
    print(f"{'source':>24} {'p0':>7} {'ov 0PA':>14} {'ov 0PA+PN':>14} {'C_p':>9}  origin")
    for s in src:
        f0, fp = s["fits"]["0PA"], s["fits"]["0PA+PN"]
        redo = [m for m in MODELS if "redo" in s["fits"][m]["origin"]]
        print(f"{s['name']:>24} {s['p0']:>7.3f} {f0['ov_final']:>14.10f} "
              f"{fp['ov_final']:>14.10f} {fp['theta'][-1]:>9.3f}  "
              f"{os.path.basename(f0['origin'])}{'  redo ' + ','.join(redo) if redo else ''}")


if __name__ == "__main__":
    main()
