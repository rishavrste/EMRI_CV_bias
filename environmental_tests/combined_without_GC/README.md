# combined_without_GC -- the combined population with no GC term in the injection

GC in `A_GC (p/M)^n_GC` is the time-varying gravitational constant of arXiv:2312.13028, not an
environmental effect, so A_GC = n_GC = 0 here (user, 2026-11-04). ../combined had (1e-12, 4).
Everything else matches ../combined: same 48 sources, dist, T, dt, A_PM, n_PM, 2nd-gen TDI A + E,
DC dropped, e0 fixed at 0 (0PA 7 params, PN 8), SuperKludge_r `environ_shubham`, flux.py a71f184.

`refit_no_gc.py`: per source and model, one full-T LM from the ../combined best fit (order 6,
Ndelta 12, 300 iterations, no NM), then the ../combined Fisher at the new fit (order 8, Ndelta 24).
Each record also stores `gc.overlap_with_gc`, the overlap of the injection with and without GC.
`plot_results.py`, `plot_overlap.py` are the ../combined scripts, reading results/ here.

Do not switch the SuperKludge_r branch while a job runs: few is an editable install.
