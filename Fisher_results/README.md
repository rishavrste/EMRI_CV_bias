# Fisher-based bias plots (Mahalanobis distance, critical SNR, ...)

Same analysis as `bias_inference_emri/new_results_EMRI/combined_plot.py` +
`fisher_common.py`, but applied to *every* optimized best-fit case already in
`src/IMRI/results_combined.txt` and `src/EMRI/results_compiled.txt`, instead of
3 hand-picked EMRI benchmark points.

## Pipeline

1. **`parse_results.py`** -- parses both `results_*.txt` files into a flat list
   of case dicts (`system`, `point`, `model`, `dt`, `T`, `chi2_sec`,
   `signal_param` (truth), `x_bf` (optimized point), `param_names`, `overlap`,
   `chi2_val`, `snr`, `branch`). Superseded/duplicate blocks are dropped
   automatically (IMRI Section 5 dt=10 grid, marked `SUPERSEDED` in the file;
   `idx16_dt5`, an exact re-derivation of `grid_idx16`). Run
   `python parse_results.py` on its own to print a summary of what it found
   (97 cases: 52 IMRI + 45 EMRI, as of writing).

2. **`fisher_common2.py`** + **`compute_fishers.py`** (GPU, needs `few`,
   `fastlisaresponse`, `lisatools`, `stableemrifisher`) -- for every case,
   computes the Fisher matrix at the already-optimized `x_bf` point (a single
   `StableEMRIFisher` derivative evaluation, *not* a CV climb -- these points
   are already at the CV optimum) and, once per point, the SNR of the true
   injected signal. Deviation wiring (verified against every script in `src/`,
   including the newest `gauss_cv_imri_dt5_rerun.py` / `gauss_cv_imri_grid_diverse2.py`):
   - **0PA**: `deviation_included=False`, no dev kwargs.
   - **PN** (hybrid branch): `{"dev_1": d1, "dev_2": d2, "del_0_p": 0.0, "del_0_e": 0.0}`
   - **simple** (hybrid branch): `{"C_p": 0.0, "C_e": 0.0, "dev_1": d1, "dev_2": d2}`
   - **simple_pe** (dev_a_pe branch): `{"dev_1": d1, "dev_2": d2}`

   The two branches are mutually exclusive SuperKludge_r checkouts, so this
   must run once per branch:
   ```
   python compute_fishers.py --branch hybrid      # 0PA/PN/simple, 93 cases
   python compute_fishers.py --branch dev_a_pe    # simple_pe only, 4 cases
   ```
   `--dry-run` lists what would be (re)computed without doing it. Output is
   resumable/incremental: `<system>/Fisher_<point>_<model>.npy` and
   `<system>/SNR_<point>.npy` are skipped if already on disk, so a walltime
   kill + requeue just continues. Submit via the matching PBS script:
   `batch_fisher_hybrid.sh` (24h walltime) then, after switching the
   SuperKludge_r branch, `batch_fisher_devapa.sh` (6h).

3. **`plot_bias.py`** (CPU-only: numpy/scipy/matplotlib, no GPU/FEW needed) --
   ports `combined_plot.py`'s methodology (Jacobi-preconditioned Fisher
   inversion, dev_1/dev_2-marginalized Mahalanobis distance, critical-SNR
   rescaling, deviation-benefit ratios -- see that file's docstring for the
   full derivations) to read the case list from `parse_results.py` generically
   instead of 3 hardcoded points. Produces, separately for IMRI and EMRI
   (different masses/params -- not meant to be compared to each other), at
   both 90% and 99% CL:
   - `plots/<system>/corner_full/` -- per-point, per-model pairwise CL
     ellipses + truth marker, plus a models-overlaid version.
   - `plots/<system>/corner_5param/` -- same, restricted to the 5 intrinsic
     params `[m1, m2, a, p0, e0]`, 90% CL only.
   - `plots/<system>/bias_summary/` -- normalized bias (`z = (bf-true)/sigma`)
     across every point/model/parameter.
   - `plots/<system>/mahalanobis/` -- joint (covariance-aware) distance of
     best fit from truth, normalized by its own chi2 threshold.
   - `plots/<system>/snr_crit/` -- critical SNR (1D worst-marginal-parameter
     and nD joint) at which the 0PA/deviation template mismatch becomes
     detectable.
   - `plots/<system>/deviation_benefit/` -- how much PN/simple/simple_pe
     improve on plain 0PA, per point (Mahalanobis + critical-SNR ratios).

   Just run `python plot_bias.py` once the Fisher/SNR `.npy` files exist for
   (at least some of) the cases; it skips and warns about any case still
   missing its Fisher matrix rather than failing.

4. **`make_bias_table.py`** (CPU) -- writes `bias_0PA_vs_1PA.md`: per-parameter
   bias/sigma of the 0PA fit at each point's own SNR, next to its overlap. Reuses
   `plot_bias.load_dataset`, so the numbers match `plot_bias.log`.
   `make_results_table.py` writes `results_table.md` (overlaps and deviation values).
   Both mark IMRI fits with overlap < 0.99 (idx3, idx9, idx1_dt5) as probable secondary
   maxima. Edit the generators, not the `.md` files.

## Notes

- **1st-generation TDI, A+E+T channels** throughout (fits and Fishers); the move to 2nd
  generation, A+E only, is planned -- see `TDI_generation.md`.
- **No 0PA-vs-2PA bias here.** Every Fisher uses a 1PA signal (`fisher_common2.py`).
- **Do not use the Mahalanobis D or the critical-SNR plots.** D exceeds the direct bound
  2 x SNR at every point; the stored best fits carry 9 significant figures and the Fisher
  condition numbers reach ~1e18, so rounding dominates D. Per-parameter bias/sigma is fine.

- IMRI and EMRI are plotted as two separate systems (different `m2`, distances,
  and in general different injected points) -- there's no cross-system plot.
- `results_combined.txt`'s `adhoc_A` (IMRI) has a different secondary-spin
  `chi2=0.0` than the grid default `0.95`; this is parsed from the header line,
  not hardcoded.
- EMRI's `grid_idx4` and `grid_idx14` blocks in `results_compiled.txt` don't
  quote an injected SNR; `compute_fishers.py` computes it directly (SNR of the
  true 1PA-like injection) rather than guessing, so those two points still get
  an exact `SNR_<point>.npy` like everyone else.
