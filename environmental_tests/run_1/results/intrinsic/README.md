# Intrinsic parameters only -- `m1, m2, a, p0`

Generated 2026-09-19T15:10:12+00:00 by `make_results_md.py`.
Method and assumptions: [`../METHOD.md`](../METHOD.md).  Full results: [`../README.md`](../README.md).

Sky position (`qS`, `phiS`) and the initial phase (`Phi_phi0`) are held out here, along with
the deviation amplitude `C_p`.  The second block adds `e0` back.

This is the narrower and harder question.  A template that absorbs an environmental deviation
by shifting the sky position has not damaged the measurement the way one absorbing it into
the masses has, so restricting to the intrinsic block asks whether the PN freedom protects
the astrophysics rather than just the fit.

**The credible threshold moves with the block.**  At `k = 4` the 99% region of a Gaussian sits
at `sqrt(chi2_ppf(0.99, 4))` = **3.64 sigma**, against 4.30 for the seven-parameter block: a
smaller space needs a smaller radius to enclose the same probability.  Each figure draws its
own, and these blocks are not comparable against a shared line.

| block | quantity | region | threshold | 0PA outside | 0PA+PN outside | median gain |
|---|---|---|---|---|---|---|
| no e0 (4) | max 1-d | 99% | 2.58 sigma | 3/22 | **1/22** | 6.46x |
| no e0 (4) | n-d, marginalized | 99% | 3.64 sigma | 17/22 | **9/22** | 4.30x |
| no e0 (4) | max 1-d | 90% | 1.64 sigma | 3/22 | **2/22** | 6.46x |
| no e0 (4) | n-d, marginalized | 90% | 2.79 sigma | 19/22 | **9/22** | 4.30x |
| with e0 (5) | max 1-d | 99% | 2.58 sigma | 15/22 | **15/22** | 4.08x |
| with e0 (5) | n-d, marginalized | 99% | 3.88 sigma | 21/22 | **16/22** | 11.19x |
| with e0 (5) | max 1-d | 90% | 1.64 sigma | 16/22 | **15/22** | 4.08x |
| with e0 (5) | n-d, marginalized | 90% | 3.04 sigma | 22/22 | **16/22** | 11.19x |

`median gain` is the median of `0PA / 0PA+PN`, so above 1 means the PN template did better.

## Figures

Same filenames as the top-level folder, so the two can be read side by side.

![max bias](max_bias.png)

![n-d bias](nd_bias.png)

![improvement](improvement.png)

With `e0` added back:

![max bias with e0](max_bias_with_e0.png)

![n-d bias with e0](nd_bias_with_e0.png)

![improvement with e0](improvement_with_e0.png)

The 90% versions of the two bias figures are the `_p90` files in this folder.
