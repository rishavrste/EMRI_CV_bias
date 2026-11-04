"""LM and Fisher of the combined population with A_GC = n_GC = 0 in the injection.

    FISH_RUN=run_2 [FISH_IDXS="0 2 3"] python refit_no_gc.py

Per source and model: one full-T LM from the ../combined best fit, at the settings those fits
used (DER_ORDER 6, NDELTA 12, LM_MAX_ITERS 300, no Nelder-Mead), then the ../combined Fisher
at the new fit (order 8, Ndelta 24, C_p step calibrated at the truth).  e0 stays fixed at 0 and
templates have the environment off, as before.  The waveform is the committed flux.py of
SuperKludge_r environ_shubham; the job stops on anything else.

One JSON per source, results/<run>_idxNN.json, in the ../combined shape plus `lm` and `gc`.
"""
import json
import os
import subprocess
import time

import numpy as np

import results_inputs as RI

RF = RI.load_combined("combined_refit_fisher", "refit_fisher.py")
CP, LME = RF.CP, RF.LME

LM_DER_ORDER, LM_NDELTA = 6, 12        # what the seed fits used; RF sets the Fisher's on import
FLUX = "src/few/trajectory/ode/flux.py"


# --- setup -----------------------------------------------------------------
def check_model():
    """Stop unless few runs the committed flux.py of environ_shubham; returns its commit."""
    LME.check_branch()

    def git(*args):
        return subprocess.run(["git", "-C", LME.SUPERKLUDGE, *args],
                              capture_output=True, text=True)

    if git("diff", "--quiet", "HEAD", "--", FLUX).returncode != 0:
        raise SystemExit(f"{FLUX} differs from the commit on {LME.SUPERKLUDGE_BRANCH}")
    import few
    if not os.path.realpath(few.__file__).startswith(os.path.realpath(LME.SUPERKLUDGE)):
        raise SystemExit(f"few comes from {few.__file__}, not {LME.SUPERKLUDGE}")
    commit = git("log", "-1", "--format=%h", "--", FLUX).stdout.strip()
    print(f"[INFO] flux.py at {commit}, few from {few.__file__}", flush=True)
    return commit


def selected():
    """The sources of FISH_RUN this job handles, optionally narrowed by FISH_IDXS."""
    run = os.environ.get("FISH_RUN", "").strip()
    if run not in RI.RUNS:
        raise SystemExit(f"FISH_RUN must be one of {list(RI.RUNS)}, got {run!r}")
    sources = RI.load_seeds(run)
    sel = os.environ.get("FISH_IDXS", "").strip()
    if not sel:
        return sources
    want = {int(x) for x in sel.replace(",", " ").split()}
    missing = want - {s["idx"] for s in sources}
    if missing:
        raise ValueError(f"FISH_IDXS names sources not in {run}: {sorted(missing)}")
    return [s for s in sources if s["idx"] in want]


def use_settings(der_order, ndelta):
    """build_source hands these to SEF and fisher_derivs reads them at call time."""
    CP.DER_ORDER, CP.NDELTA = der_order, ndelta


# --- per source ------------------------------------------------------------
def gc_effect(S, gc):
    """Overlap of the GC-free signal with the same signal carrying the removed term.  Below 1
    it also shows that the slots switch GC off."""
    tail = S["env_tail"](True)
    tail[-2:] = [gc["A_GC"], gc["n_GC"]]
    with_gc = CP.xp.array(S["waveform_response"](
        *([S["source_params"][n] for n in CP.param_names_14] + tail)))[0:CP.NCHANNELS, :]
    ip, s = S["ip"], S["signal"]
    return dict(gc, overlap_with_gc=ip(s, with_gc) / np.sqrt(ip(s, s) * ip(with_gc, with_gc)),
                snr=float(S["snr"]))


def refit_model(S, src, model):
    """One LM from the previous best fit, as the fit record the Fisher stage reads."""
    seed = src["fits"][model]
    if seed["params"] not in (LME.PARAMS_0PA, LME.PARAMS_PN):
        raise ValueError(f"{src['name']} {model}: unexpected parameters {seed['params']}")
    done = LME.climb(S, src["source"], seed["params"], seed["theta"],
                     f"{src['name']}/{RI.KEY[model]}", f"previous best fit, {seed['origin']}")
    lm = dict(seed_ov_with_gc=seed["ov_final"], seed_theta=seed["theta"],
              ov_start=done["ov_start"], chi2_start=done["chi2_start"],
              n_iter=done["n_iter"], n_accept=done["n_accept"], seconds=done["seconds"])
    return dict(params=seed["params"], theta=done["theta_final"], ov_final=done["ov_final"],
                chi2=done["chi2"], stop_reason=done["stop_reason"], seed="previous best fit",
                origin=seed["origin"]), lm


def refit_source(src):
    """New best fits for both models, sharing one signal, and the GC check."""
    use_settings(LM_DER_ORDER, LM_NDELTA)
    S = CP.build_source(src["source"])
    gc = gc_effect(S, src["gc_removed"])
    print(f"[{src['name']}] snr = {gc['snr']:.6f}, overlap with GC = "
          f"{gc['overlap_with_gc']:.14f}", flush=True)
    fits, lm = {}, {}
    for model in RI.MODELS:
        fits[model], lm[model] = refit_model(S, src, model)
    del S
    LME.free_gpu()
    return fits, lm, gc


def run_source(src, commit):
    t0 = time.time()
    fits, lm, gc = refit_source(src)
    use_settings(RF.DER_ORDER, RF.NDELTA)
    rec = RF.run_source(dict(src, fits=fits))
    rec.update(A_GC=0.0, n_GC=0.0, gc=gc, lm=lm, flux_commit=commit,
               seconds=time.time() - t0)
    return rec


def write_source(rec):
    path = os.path.join(RI.RESULTS, f"{rec['run']}_idx{rec['idx']:02d}.json")
    with open(f"{path}.tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(f"{path}.tmp", path)
    print(f"[SAVE] {path} ({rec['seconds']:.1f} s)", flush=True)


def main():
    commit = check_model()
    os.makedirs(RI.RESULTS, exist_ok=True)
    sources = selected()
    print(f"[INFO] {len(sources)} sources: {[s['name'] for s in sources]}", flush=True)
    t0, failed = time.time(), []
    for src in sources:
        try:
            write_source(run_source(src, commit))
        except Exception as exc:                        # noqa: BLE001  keep the batch going
            print(f"[FAIL] {src['name']}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(src["name"])
    print(f"\n[DONE] {len(sources) - len(failed)}/{len(sources)} sources in "
          f"{time.time() - t0:.1f} s; failed {failed}", flush=True)


if __name__ == "__main__":
    main()
