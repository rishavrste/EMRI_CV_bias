# EMRI_CV_bias — agent onboarding

Read this first in a new session. It is the durable state of the project: what the pipeline
does, what has been established, what the traps are, and what is left to do.

## 1. What this project is

**Cutler–Vallisneri (CV) systematic-bias analysis for LISA EMRI/IMRI parameter estimation.**

We inject a **1PA** waveform and fit it with a deliberately imperfect **0PA (adiabatic)**
template, optionally granting the template two extra **flux-deviation** parameters. The offset
between the best-fit point and the injection is the systematic bias. A non-zero best-fit
deviation means the extra freedom absorbed part of the 0PA↔1PA mismatch — i.e. that flux
deviation "mimics" the missing post-adiabatic physics.

Everything is noise-free; the residual is pure systematics.

### The optimiser
The CV bias formula `Δθ = Γ⁻¹⟨∂h|s−h⟩` is exactly one Gauss–Newton step, so we **iterate** it
with Levenberg–Marquardt damping until it converges on the true 0PA maximum-likelihood point:

```
δ = (Γ + λ·diag Γ)⁻¹ g       g_i = ⟨∂_i h | s−h⟩     Γ_ij = ⟨∂_i h | ∂_j h⟩
```
with Nielsen gain-ratio λ updates. Because the climb is local, every run is
**`CV → (if it stalls below the overlap target) one 1000-step Nelder–Mead escape → CV again`**.

Shared controls in every script:
```python
F_MIN=1e-5, NDELTA=12, RECOMPUTE_DELTAS_EVERY=5, OVERLAP_TARGET=0.9999999999,
LAMBDA0=1e-2, LM_MAX_ITERS=150, MAX_INNER=30, REL_TOL=1e-9, NM_MAXITER=1000, NM_STEP=2.0
```
`chi2 = ⟨r|r⟩` is **always positive** here and satisfies `chi2 ≈ 2·SNR²·(1−overlap)`. Note this
is a different absolute normalisation from the user's external Nelder-Mead pipeline — compare
overlaps across pipelines, not chi2 values.

Derivatives come from **StableEMRIFisher** with `deriv_type="stable"` (differentiates FEW
internals). `Fisher[0]` = derivatives, `Fisher[-1]` = the matrix. `add_param_args` are appended
**positionally in dict order** — get the order wrong and you silently fit the wrong model.

## 2. THE TRAP: deviation wiring is branch-dependent

`few` is an **editable** install pointing at `packages_to_install/SuperKludge_r/src` — a
`git checkout` there changes what actually runs, with no reinstall. **Verify the branch after
every switch**, and check the deviation slot indices before trusting any deviation result.

`additional_args` layouts:

| branch | layout | models available |
|---|---|---|
| `hybrid` | `[chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p(5), C_e(6), del_0_p(7), del_0_e(8)]` | **PN** = `C_p`/`C_e` (additive 2.5PN pdot/edot); **simple** = `del_0_p`/`del_0_e` (multiplicative Edot/Ldot) |
| `dev_a_pe` | `[…, deviation_included, del_0_p(5), del_0_e(6)]` | **simple_pe** = multiplicative pdot/edot |

`simple_pe` is numerically identical to hybrid-`simple` (related by a Jacobian) — confirmed on
both EMRI and IMRI. There is **no** "PN_PE" model; `dev_a_pe` has no `C_p`/`C_e` slots.

```bash
cd /home/svu/e1583490/packages_to_install/SuperKludge_r && git branch --show-current
```

## 3. THE OTHER TRAP: dt=10 aliases the IMRI grid

**Discovered and fixed in Sept 2026 — the single most important finding for the IMRI results.**

At `dt=10` the Nyquist frequency is 0.05 Hz. The IMRI grid systems (m1=1e6, **m2=1e3**) carry
power above it, which aliases and dumps spurious power into the otherwise-null **T** channel.

- `IMRI/snr_channels_imri.py` at dt=10: T carried **29–97%** of the SNR at *every* grid point
  (worst: idx14 96.5%, idx19 97.5%), and grid SNRs were inflated to 24–114.
- `IMRI/snr_overlap_dt5_imri.py` at dt=5: T% → **~0.00 everywhere**, SNRs back to 20.5–32.8,
  matching the grid's intended SNR ≈ 20.

**Use dt=5 for IMRI grid systems.** The dt=10 fits were optimised against a partly-fake signal.
Unaffected: `adhoc_A` (already dt=5) and the tails set (T=0.25 yr keeps it below Nyquist,
measured T% = 0.00 at dt=10). **The EMRI runs have not been checked for this** — see §7.

Encouragingly, re-optimising at dt=5 **did not change the physics conclusion** (§5).

## 4. Layout

```
src/
  batch.sh              PBS + Apptainer + venv; edit its `python …` line, then qsub
  batch_dt5_a.sh        the dt=5 IMRI re-run, points idx14,4,24,18,8,12  (DT5_POINTS/DT5_TAG)
  batch_dt5_b.sh        the dt=5 IMRI re-run, points idx6,16,22,10,0,20
  EMRI/  README.md      every EMRI script documented — READ IT
         results_compiled.txt      compiled EMRI results
  IMRI/  README.md      every IMRI script documented — READ IT
         results_combined.txt      compiled IMRI results  ** the deliverable **
         make_results_combined.py  regenerates it from the JSONs — edit this, not the .txt
         results_dt5.txt           dt=5 long-form dump (per-parameter bias included)
```

Every CV script shares one skeleton: `build_context(case, model)` → `{make0, ov, chi2r, ip, s,
fisher_derivs, snr}`; then `lm_climb()`, `nm_refine()`, `cv_from()`; results written
**incrementally** to JSON so a walltime kill still leaves valid output.

`results_combined.txt` is **generated** — never hand-edit it. Change
`IMRI/make_results_combined.py` and re-run it; all numbers trace to a `results_*.json`.

### Job submission
```bash
cd /nfs/home/svu/e1583490/EMRI_CV_bias/src && qsub batch.sh     # GPU, 72 h walltime
qstat -u e1583490                                               # check
```
Logs land in `{EMRI,IMRI}/*.log`; PBS output/error in `src/*.output`, `src/*.error`.
(`module: command not found` in the `.error` file is harmless — it's `.bashrc` inside the
container.) "minimum relative error is greater than 1% for … Fisher may be unstable!" is a
routine SEF delta-stability notice, present in all runs.

## 5. Established results — do not re-derive

### Seeding (the recurring question)
| seed | what it is | verdict |
|---|---|---|
| `from_injection` | the 1PA injected params | **Never works.** The 0PA template is dephased there; it stalls at overlap ~0.13–0.77. Don't spend GPU time on it. |
| `from_MAP` | the 0PA-vs-2PA recovered fit from the SK critical-SNR grid | Reliable — re-phasing is baked in. |
| `from_0PA` | our own CV 0PA best fit, + dev=(0,0) | Guarantees a deviation ends ≥ 0PA, but **stalls at dev≈0** on a flat ridge wherever `⟨∂h/∂dev|r⟩ ≈ 0`. |
| `from_NMdev` | an earlier NM/DE deviation fit | Sometimes the only seed that finds the basin. |

**Which seed wins is point- and system-dependent** — an early over-generalisation ("always seed
from the 0PA best fit") was disproved by the EMRI diverse2 and all IMRI runs. Run both and take
the max. On the dt=5 IMRI re-run the previous-deviation seed won every case that differed by
>1e-5; the 0PA seed only ever contributed sub-1e-5 refinements plus the ≥0PA floor.

### "0PA + deviation came out worse than 0PA"
Impossible at the true optimum — 0PA is nested at dev=0. It is always a **convergence artifact**
on a flat deviation ridge. Fix it by re-seeding from the 0PA best fit, not by changing λ.

### Step size will not fix a stall
Where `⟨∂h/∂dev|r⟩ ≈ 0`, the LM step vanishes for *any* λ, and the inverse-Fisher step is
likewise zero. That is a basin/global problem — which is what the 1000-step NM escape is for.

### Physics
- **PN is the only deviation that matters.** `simple` / `simple_pe` collapse to ~0PA everywhere.
- **IMRI, dt=5, re-optimised** — PN mismatch reduction `(1−ov_0PA)/(1−ov_PN)`:
  **17.9–27.9× for a<0** (idx0, 6, 10, 16, 20) vs **1.1–3.4× for a≥0**. The dt=10 (aliased) run
  gave 17.8–27.5× and 1.1–2.7× — i.e. the retrograde/prograde split is *real*, not an artifact
  of the contamination, even though the contamination ran along the same spin axis.
- **idx4 (a=+0.9, e0=0.1) is the most biased IMRI point** by an order of magnitude: 0PA
  mismatch 9.0e-3, chi2 10.5, and PN barely helps (1.1×).
- **EMRI**: PN reaches ~1.0 at several points and de-biases the parameters back toward the
  injection; `dev_1 ≈ −36.7…−36.9` recurs at idx4/idx9/idx14 (a=+0.9) — a real signature.
  EMRIs are less biased overall than IMRIs.
- Grid indexing: `idx = e0_index*5 + a_index`, a ∈ {−0.9,−0.5,0,0.5,0.9}, e0 ∈ {0.1,…,0.5}.
- **idx14 (EMRI) has an anomalous SNR — the user asked to ignore that point.** (IMRI idx14 is
  fine and is in the results.)

## 6. Infrastructure gotcha

`/scratch` is pathologically slow for **h5py** (>110 s for a single Fisher file) but fast for
`cp`. If a job appears hung on file reads, copy the files to `/hpctmp` local scratch first
(225 files copied in ~1 s) and read them there. Plain `np.load` of the small signal array from
`/scratch` is fine.

## 7. Open items

1. **Check the EMRI runs for the same aliasing.** They use different dt/T; the per-channel SNR
   test (`IMRI/snr_channels_imri.py`) is trivial to point at the EMRI cases and has not been run.
   Nothing downstream should be trusted as final until this is done.
2. **13 IMRI grid cells have no CV fit**: idx1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23.
   (idx1 has only the legacy dt=5 ad-hoc run, in Section 3 of `results_combined.txt`.)
   idx19 was the second-worst aliased point and has never been optimised.
3. **11 EMRI grid cells unrun**: idx1, 2, 7, 8, 10, 15, 16, 19, 21, 22, 23. Offered, not requested.
4. **`adhoc_A` 0PA discrepancy**: the 0PA log reports overlap 0.993850, but re-scoring the same
   `x_bf` at dt=5 gives 0.9934510 — a 4e-4 gap in what should be an identical evaluation.
   Suggests a config difference between `gauss_cv_imri_0pa.py` and the deviation scripts.
5. **IMRI tails secondary spin was never supplied** — `chi2 = 0.95` is an *assumption* (it does
   reproduce the SNR and overlap). Flagged in `results_combined.txt` Section 4.

## 8. Working style the user expects

- **"Ask if in doubt."** Use AskUserQuestion rather than guessing at physics setup.
- **Verify the SuperKludge_r branch after every checkout**, and confirm the deviation wiring.
- **Give results copy-ready**: full parameter vectors, overlap and chi2 together, in the
  terminal. The user pastes them into their own notes.
- **State assumptions prominently** when a parameter wasn't supplied.
- Don't over-claim. Several early generalisations were disproved by later runs; prefer
  "point-dependent, run both" over a rule.

## Update (2026-09-08): EMRI grid completed

`src/EMRI/gauss_cv_emri_grid_rest.py` ran the 11 cells that had no CV fit
(idx 1,2,7,8,10,15,16,19,21,22,23) — 0PA from MAP; PN and simple each climbed from BOTH
MAP and the 0PA best fit. Jobs 606658/606659, exit 0, 4h28m / 3h52m.
`src/EMRI/results_compiled.txt` now holds **all 25 grid cells**.

Established, do not re-derive:

- **The EMRI grid does not alias at dt=10.** T-channel power is 0.00013%–0.0022% of SNR at all
  11 measured cells (worst: idx19, a=+0.9 e0=0.4). m2=10 vs the IMRI m2=1e3 — the orbit evolves
  far more slowly and stays well below the 0.05 Hz Nyquist. No dt=5 re-run is needed. This
  closes the open item that the EMRI grid might share the IMRI aliasing defect.
- **For EMRI the `from_MAP` deviation seed is the load-bearing one** — the exact opposite of
  IMRI, where the previous-deviation seed carried the result. All 8 stalled deviation climbs
  (of 44) were `from_0PA`, none from MAP. Starting at the 0PA minimum leaves LM on a flat
  ridge with `<dh/d(dev)|r> ~ 0` and nothing to push it off dev=0. Most dramatic: idx19 PN,
  from_0PA stalls at the 0PA overlap 0.9960306 while from_MAP reaches 0.9998086 (20.7x).
  Keep both seeds; do not "simplify" to one.
- **idx19 (a=+0.9, e0=0.4) is the most biased EMRI cell** by an order of magnitude:
  0PA mismatch 4.0e-3, chi2 3.16. Analogous to IMRI idx4 (a=+0.9) — prograde high spin.
- **a=0.0 cells (idx2, idx7, idx22) gain nothing from either deviation model** (1.0x) even
  from MAP, with dev driven to ~1e-2. Not a stall; genuinely no direction to help.

## Update (2026-09-09): IMRI grid completed

`src/IMRI/gauss_cv_imri_grid_rest.py` ran the 13 cells that had no CV fit
(idx 1,2,3,5,7,9,11,13,15,17,19,21,23) at **dt=5 from the start** — 0PA from the SK_files MAP;
PN and simple each climbed from BOTH the MAP and the 0PA best fit. Jobs 607037/607038/607039,
exit 0, 8h12m / 8h03m / 5h07m. `src/IMRI/results_combined.txt` now holds **all 25 grid cells**
(SECTION 1 = the 12 dt=10-reseeded cells, SECTION 1b = these 13). Backup:
`results_combined.txt.bak_prerest`.

Established, do not re-derive:

- **dt=5 removes the IMRI aliasing at these cells too.** T% measured directly at dt=5: 0.00 at
  11 of 13, 0.03% (idx9), 0.01% (idx13), 0.64% (idx19). idx19 carried 97.5% at dt=10, so its
  0.64% at dt=5 is the sharpest single confirmation. No `[WARN]` lines in any of the three logs.
- **IMRI PN `C_p` clusters by spin, like EMRI but with different values and a sign flip at a=0.**
  Counting only cells where PN actually bought >2x: a=-0.9 -> +9.3..+9.7; a=-0.5 -> +10.1..+10.7;
  a=+0.5 -> -13.4, -15.4; a=+0.9 -> -6.8, -7.3, -6.9. Spread within each retrograde row is <5%.
  Compare EMRI, where the same rows gave +20 / +16 / +64 / -36.5 — same qualitative structure
  (C_p is essentially a function of spin), different numbers because m2 is 100x larger.
- **PN helps hugely for a<0 (17.9x-27.9x) and barely for a>=0 (1.0x-3.4x).** Unchanged by the
  new cells; `simple` never exceeds 2.7x anywhere.
- **The three most-biased cells are exactly the ones PN cannot fix**: idx9 (a=+0.9, e0=0.2,
  0PA ov 0.97961, PN 1.1x), idx3 (a=+0.5, e0=0.1, 0.98541, 1.0x), idx4 (a=+0.9, e0=0.1,
  0.99101, 1.1x). Whether that is a genuine limit of the 2.5PN deviation or a convergence
  failure is NOT settled — see the next item.
- **Cells that STALLED on the dev=0 ridge, worth a targeted re-climb seeded at the row-mean
  C_p:** a cell that bought no overlap AND ended at C_p ~ 0 never left the flat ridge, where
  <dh/dC_p | r> vanishes and LM has nothing to push it off. Those are idx5 (a=-0.9: C_p=0.16 at
  1.2x, while its four row-mates all found ~+9.5 at 18-26x — the clearest, and the best-determined
  seed) and idx2/7/17/22 (a=0.0, C_p ~ 0 at 1.0-1.2x). The a=0.0 seed is weak: it rests on idx12
  alone (+13.08 at 3.4x), which is nonetheless enough to show a=0 is not uniformly
  deviation-free.
  **Cells that ended at some other non-zero C_p are NOT stalls and are left alone** — idx3
  (+28.8), idx4 (+8.4), idx9 (-22.2), idx18 (-10.85), idx23 (-16.53) all left the ridge and
  converged somewhere. There is no notion of a "wrong" C_p; do not re-run them on the grounds
  that they disagree with their row.
- **Seed winners are near-even on this batch**: PN from_MAP 9 / from_0PA 4, simple 9 / 4, with
  5 and 4 exact ties respectively (max delta 2.2e-03 PN, 7.1e-04 simple). On the earlier dt=5
  re-run from_0PA won almost everywhere. Keep climbing both; neither seed dominates.
- **idx1 cross-check**: SECTION 3's old ad-hoc idx1 fit is now superseded. Its poorly-seeded 0PA
  reached only 0.9675450 vs 0.9995968 from the MAP, and its PN 0.9824551 vs 0.9999812 — yet its
  `simple` landed on 0.9997917 against the new 0.9997918, agreeing to 1e-7 from a completely
  different seed chain.
