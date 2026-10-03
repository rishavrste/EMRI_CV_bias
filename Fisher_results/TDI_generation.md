# TDI generation: these results use 1st-generation TDI (to be redone with 2nd)

Status as of 2026-10-02.

## What used 1st-generation TDI

Everything in the 0PA-vs-1PA pipeline was computed with **1st-generation TDI and all three
channels A, E, T**:

| stage | where | setting |
|---|---|---|
| CV/LM best fits (the `x_bf` and overlaps in `results_table.md`) | `src/EMRI/*.py`, `src/IMRI/*.py` | `tdi="1st generation"`, `tdi_chan="AET"`, PSDs `A1TDISens`, `E1TDISens`, `T1TDISens` |
| Fisher matrices and injected SNRs (`EMRI/`, `IMRI/`) | `fisher_common2.py` | same: `tdi="1st generation"`, `tdi_chan="AET"`, `CHANNELS = [A1TDISens, E1TDISens, T1TDISens]` |

So `results_table.md`, `bias_0PA_vs_1PA.md`, and the plots in `plots/` are all
1st-generation, AET results.

The T channel is included in every inner product. That is how the dt=10 IMRI aliasing showed
up: spurious power in T, see `AGENTS.md` section 3.

## Plan

1. Move the pipeline to **2nd-generation TDI with the A and E channels only**:
   - `tdi="2nd generation"`;
   - PSDs `A2TDISens`, `E2TDISens`;
   - T dropped.

   This matches the 1PA-vs-2PA work.
2. Re-evaluate the overlaps of the existing best fits under 2nd-generation TDI.
3. Re-optimise **only if needed**, for example where an overlap changes appreciably or where the
   fit is flagged as a probable secondary maximum (IMRI idx3, idx9). Then recompute the Fishers
   and regenerate the tables with `make_results_table.py` and `make_bias_table.py`.

Until then, treat the numbers here as 1st-generation results. Don't compare them directly with
the 2nd-generation 1PA-vs-2PA results.

## For comparison: the 1PA-vs-2PA work already uses 2nd-generation TDI, A and E only

- **SK_files `overlap.py`** (the shared evaluation stack, `rishav` setup):
  - `tdi="2nd generation"`.
  - The response is generated as AET, then T is dropped (`chans[:2]`).
  - The PSDs are `A2TDISens` and `E2TDISens`.
- **`1PA_vs_2PA/IMRI_TAIL_dev/model.py`:** the SEF derivatives use `tdi="2nd generation"` and
  `tdi_chan="AE"`, with `A2TDISens` and `E2TDISens`.
- **`1PA_vs_2PA/test_of_LM/model.py`:**
  - The SEF derivatives use `tdi="2nd generation"` and `tdi_chan="AET"`.
  - Only the A and E derivatives are kept (`dH[j][0]`, `dH[j][1]`), so T never enters a fit.
