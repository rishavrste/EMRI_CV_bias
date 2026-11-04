"""The combined population with the GC term removed from the injection.

Same 48 sources as ../combined (grid, dist, T, dt, A_PM, n_PM), with A_GC = n_GC = 0: GC is the
time-varying gravitational constant of arXiv:2312.13028, not an environmental effect (user,
2026-11-04).  e0 stays fixed at 0, so 0PA = 7 params and 0PA + PN = 8.

load_seeds gives what refit_no_gc.py starts from: the ../combined selection of best fits.
load_run / load_sources give its output, in the shape plot_overlap.py and plot_results.py read.

    python results_inputs.py        # listing, to check the seeds
"""
import glob
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
COMBINED = os.path.abspath(os.path.join(HERE, "..", "combined"))


def load_combined(name, fname):
    """A ../combined module by path: its names clash with the ones in this directory."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(COMBINED, fname))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OLD = load_combined("combined_results_inputs", "results_inputs.py")

ENV, MODELS, KEY, ANGLES = _OLD.ENV, _OLD.MODELS, _OLD.KEY, _OLD.ANGLES
RUNS, RUN_LABEL, TARGET_SNR = _OLD.RUNS, _OLD.RUN_LABEL, _OLD.TARGET_SNR
signed_bias_of, pn_below_0pa = _OLD.signed_bias_of, _OLD.pn_below_0pa
kept_in_plots = _OLD.kept_in_plots


def without_gc(src):
    """One ../combined source with the GC term of its injection switched off."""
    source = src["source"]
    gc = dict(A_GC=float(source["A_GC"]), n_GC=float(source["n_GC"]))
    if source["source_params"]["e0"] != 0.0:
        raise ValueError(f"{src['name']}: e0 must stay fixed at 0")
    source.update(A_GC=0.0, n_GC=0.0)
    return dict(src, gc_removed=gc)


def load_seeds(run):
    """Every source of one run, GC off, carrying its previous best fits."""
    return [without_gc(s) for s in _OLD.load_run(run)]


def _source(rec):
    fits = {m: dict(rec["models"][m], theta=rec["models"][m]["fit"]) for m in MODELS}
    return dict(run=rec["run"], idx=rec["idx"], name=rec["name"], p0=rec["p0"], m1=rec["m1"],
                T=rec["T"], fits=fits)


def load_run(run):
    """The refitted sources of one run, in index order."""
    paths = sorted(glob.glob(os.path.join(RESULTS, f"{run}_idx*.json")))
    return [_source(json.load(open(p))) for p in paths]


def load_sources(runs=tuple(RUNS)):
    return [s for run in runs for s in load_run(run)]


def main():
    seeds = [s for run in RUNS for s in load_seeds(run)]
    print(f"{len(seeds)} sources")
    print(f"{'source':>24} {'p0':>7} {'A_GC':>5} {'n_GC':>5} {'e0':>4} {'seed 0PA':>14} "
          f"{'seed PN':>14}")
    for s in seeds:
        src = s["source"]
        print(f"{s['name']:>24} {s['p0']:>7.3f} {src['A_GC']:>5g} {src['n_GC']:>5g} "
              f"{src['source_params']['e0']:>4g} {s['fits']['0PA']['ov_final']:>14.10f} "
              f"{s['fits']['0PA+PN']['ov_final']:>14.10f}")


if __name__ == "__main__":
    main()
