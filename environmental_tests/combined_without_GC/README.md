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

Jobs (2026-11-04, all exit 0, 48/48 sources, none failed): 699824 nogc_r1a, 699825 nogc_r1b
(run_1_second_gen), 699826 nogc_r2a, 699827 nogc_r2b (run_2).

`plot_sky.py`: 2-d marginal (qS, phiS) Fisher contours of every kept source, both models.
`plot_results.py` here has also moved on from the ../combined copy: critical_snr labels carry
the counts of sources below each line.
