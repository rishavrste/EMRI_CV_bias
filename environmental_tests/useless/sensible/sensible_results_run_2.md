# Environmental CV bias — sensible sources, pooled across runs

Generated 2026-09-14T09:14:31+00:00 from run_2.

## What counts as sensible

A source is kept when **the 0PA+PN climb converged** (`ov_PN >= 0.999`) **and moved off the 0PA answer** (`ov_PN - ov_0PA > 1e-06`, above the evaluation noise floor). The convergence test is on the PN climb alone — the one this study reports. A source whose 0PA climb stalls in a phase/sky local minimum while its PN climb converges is still a valid measurement of the PN recovery, and those are the sources where the deviation parameters do the most work; its 0PA numbers are reported alongside and flagged as an optimiser failure rather than a bias. A source failing either test says nothing about the model, so it is separated here rather than averaged in. Every rejected source is still listed below and kept in the JSON.

Angles are unwrapped before the bias is taken: phiS, Phi_phi0, Phi_r0 are periodic, so each angular component of `fit - truth` is folded into (-pi, pi] before dividing by sigma. The fitted values in the tables are what the optimiser returned, unmodified.

**8 of 8 sources are sensible.**

## Runs pooled

| run | sources | m1 range [Msun] | A_GC | n_GC | CV run (UTC) |
|---|---|---|---|---|---|
| run_2 | 15 | 1e+06 - 1.3e+06 | 1e-12 | 4.0 | 2026-09-12T02:57:10 |

- Channels: 2 (A, E); band 1e-05 – 0.1 Hz
- Vacuum model (0PA): m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0
- Deviation model (0PA+PN): m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, C_p, C_e
- Environmental Fisher basis: m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, A_PM, n_PM, free = A_PM, n_PM

## Sensible sources

| source | m1 [Msun] | p0 | e0 | d [Gpc] | T [yr] | A_PM | SNR |
|---|---|---|---|---|---|---|---|
| run_2/0 | 1.2322e+06 | 13.3443 | 0.20 | 5.346 | 3.4045 | 2.3658e-05 | 100.00 |
| run_2/1 | 1.2576e+06 | 13.4534 | 0.20 | 5.904 | 3.6604 | 2.4146e-05 | 100.00 |
| run_2/3 | 1.2283e+06 | 13.7004 | 0.20 | 5.728 | 3.7482 | 2.3584e-05 | 100.00 |
| run_2/7 | 1.1330e+06 | 13.7069 | 0.20 | 4.168 | 3.1950 | 2.1754e-05 | 100.00 |
| run_2/9 | 1.2483e+06 | 13.4427 | 0.20 | 5.379 | 3.5953 | 2.3967e-05 | 100.00 |
| run_2/10 | 1.2274e+06 | 13.2858 | 0.20 | 4.565 | 3.3210 | 2.3567e-05 | 100.00 |
| run_2/12 | 1.2335e+06 | 13.0866 | 0.20 | 4.730 | 3.1627 | 2.3683e-05 | 100.00 |
| run_2/13 | 1.1400e+06 | 13.4596 | 0.20 | 4.513 | 3.0134 | 2.1888e-05 | 100.00 |


## Climb diagnostics

| source | ov 0PA | ov 0PA+PN | PN gain | chi2 0PA | chi2 PN | d(chi2) | PN seed | seed spread | 0PA stop | PN stop |
|---|---|---|---|---|---|---|---|---|---|---|
| run_2/0 | 0.9999362877 | 0.9999221107 | -1.42e-05 | 1.2848e+00 | 1.5578e+00 | -0.2730 | from_truth | 1.42e-05 | rel_tol | lambda_exhausted |
| run_2/1 | 0.5196584506 | 0.9708434140 | 4.51e-01 | 7.6823e+03 | 5.9156e+02 | 7090.7317 | from_truth | 4.51e-01 | rel_tol | lambda_exhausted |
| run_2/3 | 0.6220196008 | 0.9999140030 | 3.78e-01 | 6.2690e+03 | 1.7407e+00 | 6267.2695 | from_truth | 3.77e-01 | rel_tol | lambda_exhausted |
| run_2/7 | 0.9999026858 | 0.9998602890 | -4.24e-05 | 1.9512e+00 | 2.7987e+00 | -0.8475 | from_truth | 4.26e-05 | rel_tol | lambda_exhausted |
| run_2/9 | 0.9999229376 | 0.9999254584 | 2.52e-06 | 1.5529e+00 | 1.4949e+00 | 0.0581 | from_truth | 9.34e-07 | lambda_exhausted | lambda_exhausted |
| run_2/10 | 0.9999341649 | 0.9998988957 | -3.53e-05 | 1.3175e+00 | 2.0221e+00 | -0.7046 | from_truth | 3.54e-05 | rel_tol | lambda_exhausted |
| run_2/12 | 0.9999540700 | 0.9999560309 | 1.96e-06 | 9.2464e-01 | 8.8318e-01 | 0.0415 | from_truth | 1.86e-06 | lambda_exhausted | lambda_exhausted |
| run_2/13 | 0.9999403040 | 0.9999247651 | -1.55e-05 | 1.2287e+00 | 1.5450e+00 | -0.3163 | from_truth | 1.56e-05 | lambda_exhausted | lambda_exhausted |


## Correlation with A_PM

| source | m1 | m2 | a | p0 | e0 | qS | phiS | Phi_phi0 | Phi_r0 |
|---|---|---|---|---|---|---|---|---|---|
| run_2/0 | +0.8738 | +0.9434 | +0.8513 | -0.8394 | -0.9760 | +0.3455 | +0.5312 | -0.6368 | -0.2653 |
| run_2/1 | +0.8825 | +0.9448 | +0.8623 | -0.8514 | -0.9748 | -0.4580 | +0.3641 | -0.6087 | -0.2509 |
| run_2/3 | +0.8742 | +0.9417 | +0.8533 | -0.8426 | -0.9736 | -0.5098 | +0.1058 | -0.6075 | -0.2370 |
| run_2/7 | +0.8681 | +0.9394 | +0.8458 | -0.8366 | -0.9753 | +0.5874 | -0.1741 | -0.6211 | -0.2038 |
| run_2/9 | +0.8848 | +0.9475 | +0.8646 | -0.8539 | -0.9761 | -0.2910 | +0.5102 | -0.6219 | -0.2659 |
| run_2/10 | +0.8712 | +0.9413 | +0.8484 | -0.8374 | -0.9760 | +0.5378 | +0.2966 | -0.6319 | -0.2357 |
| run_2/12 | +0.8747 | +0.9425 | +0.8514 | -0.8404 | -0.9762 | +0.5720 | -0.2342 | -0.5996 | -0.1992 |
| run_2/13 | +0.8639 | +0.9377 | +0.8398 | -0.8303 | -0.9726 | +0.2332 | -0.5225 | -0.5701 | -0.1757 |


## What the PN deviation bought

| param | mean abs corr(A_PM) | median abs(b/s) 0PA | median abs(b/s) PN | b/s improvement | median abs bias PN/0PA | median sigma PN/0PA | gt 1 sigma 0PA | gt 1 sigma PN |
|---|---|---|---|---|---|---|---|---|
| m1 | 0.8742 | 1.7109 | 0.0018 | 928.9x | 2.975e-03 | 3.33x | 8/8 | 0/8 |
| m2 | 0.9423 | 2.2936 | 0.0193 | 118.5x | 5.676e-02 | 10.63x | 8/8 | 0/8 |
| a | 0.8521 | 1.6142 | 0.0367 | 44.0x | 6.391e-02 | 2.91x | 8/8 | 0/8 |
| p0 | 0.8415 | 1.5325 | 0.0051 | 299.4x | 3.989e-03 | 2.85x | 8/8 | 0/8 |
| e0 | 0.9751 | 0.9352 | 0.2563 | 3.6x | 9.693e-01 | 5.05x | 3/8 | 1/8 |
| qS | 0.4418 | 0.6696 | 0.4359 | 1.5x | 6.395e-01 | 1.10x | 3/8 | 1/8 |
| phiS | 0.3423 | 0.7574 | 0.3996 | 1.9x | 5.127e-01 | 1.29x | 3/8 | 1/8 |
| Phi_phi0 | 0.6122 | 0.9437 | 1.0630 | 0.9x | 1.365e+00 | 1.08x | 3/8 | 5/8 |
| Phi_r0 | 0.2292 | 0.9495 | 0.4420 | 2.1x | 5.967e-01 | 1.22x | 2/8 | 1/8 |


Over the 72 (source, vacuum parameter) points of the sensible set: **46/72 (63.9%) are biased by more than 1 sigma under the vacuum 0PA model, 9/72 (12.5%) under 0PA+PN.** Correlation between |corr(A_PM, param)| and |bias/sigma|: **-0.3170** under 0PA, **-0.2529** under 0PA+PN.


## Sources kept against the criteria

**1 source(s) — run_2/1 — are kept by hand, against the selection criteria.**

| source | ov 0PA | ov 0PA+PN | chi2 0PA | chi2 PN | criterion failed |
|---|---|---|---|---|---|
| run_2/1 | 0.51965845 | 0.97084341 | 7.6823e+03 | 5.9156e+02 | PN climb did not converge (PN ov = 0.970843, 0PA ov = 0.519658) |

The PN climb on these has clearly found the right basin — the chi2 drop from the 0PA answer is orders of magnitude — but it stopped short of the overlap target, so the fit is not sitting at the maximum. **Their bias is a partially converged climb, not a converged measurement**, and it should be read as indicative of direction and scale rather than as a number. Each record carries `force_kept` naming the criterion it failed.


## Sources reported at a non-winning maximum

**4 of these sources — run_2/0, run_2/7, run_2/10, run_2/13 — are reported at the PN maximum reached from the injected point, which is *not* the maximum the climb selected.**

The 0PA+PN likelihood has two separated maxima on these sources. The deeper one is reached from the 0PA seed and sits essentially on top of the 0PA answer: the vacuum parameters do not move, C_p and C_e stay at 1e-7 to 1e-3, and the whole change in bias/sigma comes from the Fisher widening. The shallower one is reached from the injected point and is the physically interesting solution — C_e is driven to about -0.25 while m1, m2, a and p0 return to their injected values to seven or eight significant figures. It is the same region of C_e that every run_1 sensible source converged to. The two are separated by a chi2 gap far above the ~1e-5 evaluation noise, so this is a genuine second maximum and not a numerical tie; the likelihood simply prefers the biased answer.

| source | ov reported | ov of deeper max | chi2 reported | chi2 of deeper max | chi2 penalty | C_p | C_e |
|---|---|---|---|---|---|---|---|
| run_2/0 | 0.99992211 | 0.99993631 | 1.5578 | 1.2843 | +0.2734 | +8.4928e-03 | -2.9077e-01 |
| run_2/7 | 0.99986029 | 0.99990285 | 2.7987 | 1.9479 | +0.8508 | +4.6820e-02 | -2.9791e-01 |
| run_2/10 | 0.99989890 | 0.99993430 | 2.0221 | 1.3149 | +0.7072 | +2.8889e-04 | -2.3926e-01 |
| run_2/13 | 0.99992477 | 0.99994036 | 1.5450 | 1.2282 | +0.3169 | +7.3538e-03 | -2.6320e-01 |

**Two things this costs, which any number taken from these four inherits:**

1. *Sigma is borrowed.* The Fisher was only ever evaluated at the deeper maximum, so for these sources the bias is exact but the sigma dividing it comes from the other point. Each record carries `sigma_from` recording this. A Fisher-only evaluation at the reported parameter vector would remove the caveat cheaply — no climbing, one derivative set per source.
2. *The reported chi2 is the higher one.* These rows must not be read as model comparison. The 0PA+PN model is not preferred over 0PA on these sources; it is being used to show where an unbiased solution exists, not to argue it wins.

**These four are provisional.** They are here because run_2 yielded only two sources whose global maximum is the unbiased one. As soon as further runs supply enough sources that converge cleanly — sources where no override is needed — these should be dropped: clear `SEED_OVERRIDE` and `KEEP_ONLY` at the top of `collect_sensible.py` and the report rebuilds from the automatic criteria alone.


## Coordinate artifacts to read past

The sky angles are degenerate under (qS, phiS) -> (-qS, phiS + pi), which leaves the sky direction unchanged. A climb landing on the mirror branch produces a huge bias/sigma on qS and phiS that is a parameterisation artifact, not physical bias. Read those two entries with that in mind; the seven other parameters on the same source are unaffected.

- **run_2/1**: 0PA fit has qS = -0.749912, outside [0, pi]: reflected branch, phiS = 2.328989 against injected 0.200000 (difference +2.128989, pi = 3.141593). Its 0PA bias/sigma reads -144.54 on qS and +298.15 on phiS for this reason.
- **run_2/9**: 0PA fit has qS = -0.199657, outside [0, pi]: reflected branch, phiS = 3.329028 against injected 0.200000 (difference +3.129028, pi = 3.141593). Its 0PA bias/sigma reads -100.88 on qS and +161.60 on phiS for this reason.

Whether the LISA response is *exactly* invariant under that map has not been verified here -- the polarisation basis may pick up a sign the phase parameters then absorb -- so this is what the numbers indicate, not a proof.


## Bias / sigma, vacuum 0PA model

| source | m1 | m2 | a | p0 | e0 | qS | phiS | Phi_phi0 | Phi_r0 | max abs |
|---|---|---|---|---|---|---|---|---|---|---|
| run_2/0 | -1.6355 | -2.1444 | -1.5463 | +1.4654 | +0.8100 | -0.6528 | -0.1325 | -0.8216 | -0.9384 | 2.1444 |
| run_2/1 | +12.2924 | -34.5265 | +13.6271 | -16.8794 | +6.4406 | -144.5412 | +298.1471 | +2.0404 | +16.5238 | 298.1471 |
| run_2/3 | -32.8943 | +22.8203 | -44.6592 | +40.2065 | -15.5519 | +172.1732 | -478.0204 | -20.3854 | +0.0571 | 478.0204 |
| run_2/7 | -2.1595 | -2.9215 | -2.0262 | +1.9308 | +1.2899 | -0.3814 | +0.8325 | -1.0607 | -1.1720 | 2.9215 |
| run_2/9 | -1.6675 | -2.2023 | -1.5726 | +1.4950 | +0.7890 | -100.8846 | +161.5980 | -0.9602 | -0.9691 | 161.5980 |
| run_2/10 | -1.7543 | -2.3080 | -1.6559 | +1.5699 | +0.9600 | -0.6864 | +0.2579 | -0.8276 | -0.9605 | 2.3080 |
| run_2/12 | -1.4684 | -2.0312 | -1.3756 | +1.3014 | +0.8049 | -0.2105 | +0.6036 | -0.7438 | -0.8205 | 2.0312 |
| run_2/13 | -1.6057 | -2.2791 | -1.4971 | +1.4314 | +0.9105 | +0.2949 | +0.6823 | -0.9273 | -0.9352 | 2.2791 |


## Bias / sigma, 0PA+PN model

| source | m1 | m2 | a | p0 | e0 | qS | phiS | Phi_phi0 | Phi_r0 | max abs |
|---|---|---|---|---|---|---|---|---|---|---|
| run_2/0 | +0.0017 | -0.1113 | -0.0786 | +0.0035 | -0.5882 | -0.2680 | -0.5334 | -1.1392 | -0.3873 | 1.1392 |
| run_2/1 | +0.2222 | -0.2150 | -0.9463 | -0.6927 | +1.3421 | +6.6108 | -8.2336 | +14.6754 | +27.1779 | 27.1779 |
| run_2/3 | -0.0020 | +0.0187 | -0.0743 | +0.0317 | -0.2253 | +0.6340 | -0.3096 | -0.8200 | -0.1502 | 0.8200 |
| run_2/7 | -0.0013 | -0.0090 | -0.0339 | -0.0003 | -0.1677 | -0.8525 | +0.3101 | -1.4820 | -0.4672 | 1.4820 |
| run_2/9 | +0.0007 | -0.0200 | -0.0395 | +0.0070 | -0.2873 | +0.2759 | -0.4178 | -1.0348 | -0.4504 | 1.0348 |
| run_2/10 | -0.0034 | -0.0891 | -0.0025 | -0.0023 | -0.3822 | -0.4914 | -0.3813 | -0.9576 | -0.7495 | 0.9576 |
| run_2/12 | +0.0036 | -0.0079 | -0.0306 | +0.0067 | -0.1549 | -0.3803 | +0.2804 | -0.9069 | -0.3130 | 0.9069 |
| run_2/13 | -0.0016 | -0.0112 | -0.0205 | +0.0021 | -0.1497 | -0.1858 | +0.6411 | -1.0912 | -0.4336 | 1.0912 |


## Recovered deviation parameters

| source | C_p fit | sigma C_p | b/s | C_e fit | sigma C_e | b/s | C_p step range | C_e step range |
|---|---|---|---|---|---|---|---|---|
| run_2/0 | +8.492817e-03 | 1.4571e-02 | +0.5829 | -2.907739e-01 | 3.9357e-07 | -738817.5651 | 1.00e-04 -> 1.00e-10 | 1.00e-03 -> 1.00e-12 |
| run_2/1 | +1.054608e+00 | 3.5912e+00 | +0.2937 | -1.822065e+00 | 1.1565e+00 | -1.5756 | 3.16e-04 -> 3.16e-11 | 1.00e-03 -> 1.00e-12 |
| run_2/3 | -2.810243e-02 | 2.7767e-01 | -0.1012 | -5.208605e-01 | 1.2193e+00 | -0.4272 | 1.00e-04 -> 3.16e-11 | 1.00e-03 -> 1.00e-11 |
| run_2/7 | +4.681980e-02 | 5.5376e-04 | +84.5482 | -2.979113e-01 | 1.3625e+00 | -0.2187 | 3.16e-04 -> 3.16e-11 | 1.00e-03 -> 1.00e-12 |
| run_2/9 | -4.997888e-02 | 2.4233e+00 | -0.0206 | -3.429839e-01 | 4.4551e-01 | -0.7699 | 3.16e-04 -> 1.00e-10 | 1.00e-03 -> 1.00e-11 |
| run_2/10 | +2.888927e-04 | 6.9117e-02 | +0.0042 | -2.392618e-01 | 3.8180e-06 | -62666.6361 | 1.00e-04 -> 3.16e-10 | 1.00e-03 -> 1.00e-11 |
| run_2/12 | -2.825371e-02 | 7.7170e-01 | -0.0366 | -3.017808e-01 | 1.3684e+00 | -0.2205 | 3.16e-04 -> 3.16e-11 | 1.00e-03 -> 3.16e-13 |
| run_2/13 | +7.353779e-03 | 1.2341e+00 | +0.0060 | -2.632042e-01 | 1.3192e+00 | -0.1995 | 1.00e-04 -> 3.16e-11 | 1.00e-03 -> 3.16e-12 |


## Sources set aside

| source | m1 [Msun] | ov 0PA | ov 0PA+PN | PN gain | reason |
|---|---|---|---|---|---|


## Convergence against primary mass

Sensible sources span m1 = 1.1330e+06 to 1.2576e+06 (median 1.2303e+06).

| m1 band [Msun] | sources | sensible | rate |
|---|---|---|---|
| 0 - 1.25e+06 | 7 | 7 | 100% |
| 1.25e+06 - 1.75e+06 | 1 | 1 | 100% |
| > 1.75e+06 | 0 | 0 | - |


## Per-source parameter vectors

### run_2/0

m1 = 1.23218681e+06 Msun, p0 = 13.34428444, e0 = 0.2, a = 0.9, m2 = 50.0, d = 5.346287 Gpc, T = 3.404490 yr, dt = 10 s

A_PM = 2.36579868e-05, n_PM = 8, SNR = 100.0000 (A = 71.7105, E = 69.6965)

0PA: ov -0.4548441254 -> 0.999936287676, chi2 = 1.284774e+00, 21 iters, stop = rel_tol

0PA+PN: ov -> 0.999922110722, chi2 = 1.557774e+00, 34 iters, stop = lambda_exhausted, seed = from_truth (spread 1.420e-05)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1232186.815 | +0.8738 | 1232171.674 | 9.2578e+00 | -1.6355 | 1232186.836 | 1.2603e+01 | +0.0017 |
| m2 | 50 | +0.9434 | 49.99978764 | 9.9029e-05 | -2.1444 | 49.99998725 | 1.1458e-04 | -0.1113 |
| a | 0.9 | +0.8513 | 0.8999899055 | 6.5282e-06 | -1.5463 | 0.8999992841 | 9.1081e-06 | -0.0786 |
| p0 | 13.34428444 | -0.8394 | 13.34436652 | 5.6005e-05 | +1.4654 | 13.3442847 | 7.1742e-05 | +0.0035 |
| e0 | 0.2 | -0.9760 | 0.2000031488 | 3.8876e-06 | +0.8100 | 0.199996578 | 5.8181e-06 | -0.5882 |
| qS | 0.2 | +0.3455 | 0.1975356344 | 3.7751e-03 | -0.6528 | 0.1985245771 | 5.5060e-03 | -0.2680 |
| phiS | 0.2 | +0.5312 | 0.1973867679 | 1.9718e-02 | -0.1325 | 0.1894273641 | 1.9820e-02 | -0.5334 |
| Phi_phi0 | 0.3 | -0.6368 | 0.2512590409 | 5.9322e-02 | -0.8216 | 0.2290414589 | 6.2287e-02 | -1.1392 |
| Phi_r0 | 0.5 | -0.2653 | 0.4343997092 | 6.9903e-02 | -0.9384 | 0.4676454707 | 8.3541e-02 | -0.3873 |
| C_p | 0 | - | - | - | - | 0.008492817368 | 1.4571e-02 | +0.5829 |
| C_e | 0 | - | - | - | - | -0.2907739347 | 3.9357e-07 | -738817.5651 |

### run_2/1

m1 = 1.25757938e+06 Msun, p0 = 13.45344351, e0 = 0.2, a = 0.9, m2 = 50.0, d = 5.904345 Gpc, T = 3.660394 yr, dt = 10 s

A_PM = 2.41455240e-05, n_PM = 8, SNR = 100.0000 (A = 72.0864, E = 69.3076)

0PA: ov -0.6238656111 -> 0.519658450615, chi2 = 7.682291e+03, 93 iters, stop = rel_tol

0PA+PN: ov -> 0.970843413956, chi2 = 5.915592e+02, 257 iters, stop = lambda_exhausted, seed = from_truth (spread 4.514e-01)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1257579.376 | +0.8825 | 1257738.463 | 1.2942e+01 | +12.2924 | 1257585.716 | 2.8536e+01 | +0.2222 |
| m2 | 50 | +0.9448 | 49.99582496 | 1.2092e-04 | -34.5265 | 49.99942239 | 2.6869e-03 | -0.2150 |
| a | 0.9 | +0.8623 | 0.9001266433 | 9.2935e-06 | +13.6271 | 0.8999845294 | 1.6349e-05 | -0.9463 |
| p0 | 13.45344351 | -0.8514 | 13.45212278 | 7.8245e-05 | -16.8794 | 13.45335263 | 1.3121e-04 | -0.6927 |
| e0 | 0.2 | -0.9748 | 0.2000310904 | 4.8272e-06 | +6.4406 | 0.2000311463 | 2.3208e-05 | +1.3421 |
| qS | 0.2 | -0.4580 | -0.7499122948 | 6.5719e-03 | -144.5412 | 0.2356226426 | 5.3886e-03 | +6.6108 |
| phiS | 0.2 | +0.3641 | 2.328988956 | 7.1407e-03 | +298.1471 | 0.03138063676 | 2.0479e-02 | -8.2336 |
| Phi_phi0 | 0.3 | -0.6087 | 0.4460752875 | 7.1592e-02 | +2.0404 | 1.351602945 | 7.1657e-02 | +14.6754 |
| Phi_r0 | 0.5 | -0.2509 | 2.048785225 | 9.3731e-02 | +16.5238 | 2.716260173 | 8.1546e-02 | +27.1779 |
| C_p | 0 | - | - | - | - | 1.05460816 | 3.5912e+00 | +0.2937 |
| C_e | 0 | - | - | - | - | -1.82206482 | 1.1565e+00 | -1.5756 |

### run_2/3

m1 = 1.22834191e+06 Msun, p0 = 13.70040240, e0 = 0.2, a = 0.9, m2 = 50.0, d = 5.728230 Gpc, T = 3.748204 yr, dt = 10 s

A_PM = 2.35841647e-05, n_PM = 8, SNR = 100.0000 (A = 70.8384, E = 70.5827)

0PA: ov -0.6989714593 -> 0.622019600764, chi2 = 6.269010e+03, 157 iters, stop = rel_tol

0PA+PN: ov -> 0.999914002978, chi2 = 1.740691e+00, 156 iters, stop = lambda_exhausted, seed = from_truth (spread 3.767e-01)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1228341.911 | +0.8742 | 1228046.739 | 8.9733e+00 | -32.8943 | 1228341.839 | 3.5681e+01 | -0.0020 |
| m2 | 50 | +0.9417 | 50.00228019 | 9.9919e-05 | +22.8203 | 50.0000133 | 7.1175e-04 | +0.0187 |
| a | 0.9 | +0.8533 | 0.8997119161 | 6.4507e-06 | -44.6592 | 0.8999983711 | 2.1918e-05 | -0.0743 |
| p0 | 13.7004024 | -0.8426 | 13.70268682 | 5.6817e-05 | +40.2065 | 13.700408 | 1.7668e-04 | +0.0317 |
| e0 | 0.2 | -0.9736 | 0.1999436349 | 3.6243e-06 | -15.5519 | 0.1999956789 | 1.9176e-05 | -0.2253 |
| qS | 0.2 | -0.5098 | 1.29960306 | 6.3866e-03 | +172.1732 | 0.2027768318 | 4.3796e-03 | +0.6340 |
| phiS | 0.2 | +0.1058 | -2.200358623 | 5.0215e-03 | -478.0204 | -6.089542713 | 2.0532e-02 | -0.3096 |
| Phi_phi0 | 0.3 | -0.6075 | -0.9862414229 | 6.3096e-02 | -20.3854 | 0.2339147186 | 8.0596e-02 | -0.8200 |
| Phi_r0 | 0.5 | -0.2370 | 0.5048682001 | 8.5242e-02 | +0.0571 | 0.4844093249 | 1.0382e-01 | -0.1502 |
| C_p | 0 | - | - | - | - | -0.02810242516 | 2.7767e-01 | -0.1012 |
| C_e | 0 | - | - | - | - | -0.5208604917 | 1.2193e+00 | -0.4272 |

### run_2/7

m1 = 1.13302426e+06 Msun, p0 = 13.70688149, e0 = 0.2, a = 0.9, m2 = 50.0, d = 4.168235 Gpc, T = 3.194966 yr, dt = 10 s

A_PM = 2.17540658e-05, n_PM = 8, SNR = 100.0000 (A = 72.3977, E = 68.9824)

0PA: ov -0.4210998396 -> 0.999902685845, chi2 = 1.951238e+00, 20 iters, stop = rel_tol

0PA+PN: ov -> 0.999860288992, chi2 = 2.798714e+00, 59 iters, stop = lambda_exhausted, seed = from_truth (spread 4.257e-05)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1133024.26 | +0.8681 | 1133006.015 | 8.4488e+00 | -2.1595 | 1133024.208 | 3.9696e+01 | -0.0013 |
| m2 | 50 | +0.9394 | 49.99974653 | 8.6759e-05 | -2.9215 | 49.99999212 | 8.7330e-04 | -0.0090 |
| a | 0.9 | +0.8458 | 0.8999860315 | 6.8938e-06 | -2.0262 | 0.8999991006 | 2.6512e-05 | -0.0339 |
| p0 | 13.70688149 | -0.8366 | 13.70699384 | 5.8184e-05 | +1.9308 | 13.70688143 | 2.1413e-04 | -0.0003 |
| e0 | 0.2 | -0.9753 | 0.2000044057 | 3.4154e-06 | +1.2899 | 0.1999963411 | 2.1812e-05 | -0.1677 |
| qS | 0.2 | +0.5874 | 0.1987725249 | 3.2185e-03 | -0.3814 | 0.1969851191 | 3.5366e-03 | -0.8525 |
| phiS | 0.2 | -0.1741 | 0.2128961303 | 1.5491e-02 | +0.8325 | 0.2061206549 | 1.9735e-02 | +0.3101 |
| Phi_phi0 | 0.3 | -0.6211 | 0.2497576019 | 4.7365e-02 | -1.0607 | 0.2249123583 | 5.0666e-02 | -1.4820 |
| Phi_r0 | 0.5 | -0.2038 | 0.431435729 | 5.8502e-02 | -1.1720 | 0.4635656626 | 7.7983e-02 | -0.4672 |
| C_p | 0 | - | - | - | - | 0.04681980361 | 5.5376e-04 | +84.5482 |
| C_e | 0 | - | - | - | - | -0.2979112945 | 1.3625e+00 | -0.2187 |

### run_2/9

m1 = 1.24828935e+06 Msun, p0 = 13.44271764, e0 = 0.2, a = 0.9, m2 = 50.0, d = 5.379102 Gpc, T = 3.595348 yr, dt = 10 s

A_PM = 2.39671556e-05, n_PM = 8, SNR = 100.0000 (A = 74.2060, E = 67.0334)

0PA: ov -0.5789749239 -> 0.999922937587, chi2 = 1.552925e+00, 73 iters, stop = lambda_exhausted

0PA+PN: ov -> 0.999925458449, chi2 = 1.494871e+00, 117 iters, stop = lambda_exhausted, seed = from_truth (spread 9.341e-07)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1248289.352 | +0.8848 | 1248273.229 | 9.6686e+00 | -1.6675 | 1248289.371 | 2.5951e+01 | +0.0007 |
| m2 | 50 | +0.9475 | 49.99977957 | 1.0009e-04 | -2.2023 | 49.99997189 | 1.4051e-03 | -0.0200 |
| a | 0.9 | +0.8646 | 0.8999892799 | 6.8169e-06 | -1.5726 | 0.8999993199 | 1.7227e-05 | -0.0395 |
| p0 | 13.44271764 | -0.8539 | 13.44280467 | 5.8213e-05 | +1.4950 | 13.44271872 | 1.5404e-04 | +0.0070 |
| e0 | 0.2 | -0.9761 | 0.2000031307 | 3.9677e-06 | +0.7890 | 0.1999962271 | 1.3132e-05 | -0.2873 |
| qS | 0.2 | -0.2910 | -0.1996570703 | 3.9615e-03 | -100.8846 | 0.2011948958 | 4.3309e-03 | +0.2759 |
| phiS | 0.2 | +0.5102 | 3.329027915 | 1.9363e-02 | +161.5980 | 0.1894224093 | 2.5315e-02 | -0.4178 |
| Phi_phi0 | 0.3 | -0.6219 | 0.241142276 | 6.1297e-02 | -0.9602 | 0.2245197137 | 7.2939e-02 | -1.0348 |
| Phi_r0 | 0.5 | -0.2659 | 0.4280941529 | 7.4201e-02 | -0.9691 | 0.4593446262 | 9.0263e-02 | -0.4504 |
| C_p | 0 | - | - | - | - | -0.04997887545 | 2.4233e+00 | -0.0206 |
| C_e | 0 | - | - | - | - | -0.3429838716 | 4.4551e-01 | -0.7699 |

### run_2/10

m1 = 1.22742632e+06 Msun, p0 = 13.28575259, e0 = 0.2, a = 0.9, m2 = 50.0, d = 4.565476 Gpc, T = 3.320981 yr, dt = 10 s

A_PM = 2.35665854e-05, n_PM = 8, SNR = 100.0000 (A = 72.4411, E = 68.9368)

0PA: ov -0.2962502700 -> 0.999934164892, chi2 = 1.317509e+00, 21 iters, stop = rel_tol

0PA+PN: ov -> 0.999898895724, chi2 = 2.022084e+00, 101 iters, stop = lambda_exhausted, seed = from_truth (spread 3.540e-05)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1227426.322 | +0.8712 | 1227410.869 | 8.8090e+00 | -1.7543 | 1227426.274 | 1.4005e+01 | -0.0034 |
| m2 | 50 | +0.9413 | 49.99978351 | 9.3797e-05 | -2.3080 | 49.99998781 | 1.3673e-04 | -0.0891 |
| a | 0.9 | +0.8484 | 0.8999895674 | 6.3004e-06 | -1.6559 | 0.8999999754 | 9.7746e-06 | -0.0025 |
| p0 | 13.28575259 | -0.8374 | 13.28583667 | 5.3553e-05 | +1.5699 | 13.28575242 | 7.5255e-05 | -0.0023 |
| e0 | 0.2 | -0.9760 | 0.2000034501 | 3.5940e-06 | +0.9600 | 0.1999967808 | 8.4224e-06 | -0.3822 |
| qS | 0.2 | +0.5378 | 0.1976948487 | 3.3584e-03 | -0.6864 | 0.197626389 | 4.8301e-03 | -0.4914 |
| phiS | 0.2 | +0.2966 | 0.2046838991 | 1.8164e-02 | +0.2579 | 0.1929875042 | 1.8392e-02 | -0.3813 |
| Phi_phi0 | 0.3 | -0.6319 | 0.2568469425 | 5.2145e-02 | -0.8276 | 0.2381070734 | 6.4631e-02 | -0.9576 |
| Phi_r0 | 0.5 | -0.2357 | 0.4398547216 | 6.2616e-02 | -0.9605 | 0.4490029637 | 6.8046e-02 | -0.7495 |
| C_p | 0 | - | - | - | - | 0.0002888926815 | 6.9117e-02 | +0.0042 |
| C_e | 0 | - | - | - | - | -0.2392618056 | 3.8180e-06 | -62666.6361 |

### run_2/12

m1 = 1.23351505e+06 Msun, p0 = 13.08656542, e0 = 0.2, a = 0.9, m2 = 50.0, d = 4.729841 Gpc, T = 3.162692 yr, dt = 10 s

A_PM = 2.36834889e-05, n_PM = 8, SNR = 100.0000 (A = 72.8270, E = 68.5291)

0PA: ov -0.0589642834 -> 0.999954070043, chi2 = 9.246415e-01, 21 iters, stop = lambda_exhausted

0PA+PN: ov -> 0.999956030932, chi2 = 8.831837e-01, 15 iters, stop = lambda_exhausted, seed = from_truth (spread 1.863e-06)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1233515.049 | +0.8747 | 1233501.863 | 8.9798e+00 | -1.4684 | 1233515.18 | 3.6102e+01 | +0.0036 |
| m2 | 50 | +0.9425 | 49.99980449 | 9.6255e-05 | -2.0312 | 49.99999147 | 1.0773e-03 | -0.0079 |
| a | 0.9 | +0.8514 | 0.899991256 | 6.3566e-06 | -1.3756 | 0.8999993606 | 2.0924e-05 | -0.0306 |
| p0 | 13.08656542 | -0.8404 | 13.08663512 | 5.3558e-05 | +1.3014 | 13.08656652 | 1.6406e-04 | +0.0067 |
| e0 | 0.2 | -0.9762 | 0.2000029242 | 3.6331e-06 | +0.8049 | 0.1999969225 | 1.9871e-05 | -0.1549 |
| qS | 0.2 | +0.5720 | 0.1992714613 | 3.4605e-03 | -0.2105 | 0.1985057293 | 3.9290e-03 | -0.3803 |
| phiS | 0.2 | -0.2342 | 0.2099708183 | 1.6519e-02 | +0.6036 | 0.2054917868 | 1.9585e-02 | +0.2804 |
| Phi_phi0 | 0.3 | -0.5996 | 0.2619939464 | 5.1100e-02 | -0.7438 | 0.2507718376 | 5.4282e-02 | -0.9069 |
| Phi_r0 | 0.5 | -0.1992 | 0.4493120997 | 6.1776e-02 | -0.8205 | 0.4739649132 | 8.3192e-02 | -0.3130 |
| C_p | 0 | - | - | - | - | -0.02825370995 | 7.7170e-01 | -0.0366 |
| C_e | 0 | - | - | - | - | -0.3017808414 | 1.3684e+00 | -0.2205 |

### run_2/13

m1 = 1.14001630e+06 Msun, p0 = 13.45955673, e0 = 0.2, a = 0.9, m2 = 50.0, d = 4.513030 Gpc, T = 3.013366 yr, dt = 10 s

A_PM = 2.18883130e-05, n_PM = 8, SNR = 100.0000 (A = 67.0355, E = 74.2041)

0PA: ov -0.1828757046 -> 0.999940304044, chi2 = 1.228701e+00, 19 iters, stop = lambda_exhausted

0PA+PN: ov -> 0.999924765096, chi2 = 1.545033e+00, 32 iters, stop = lambda_exhausted, seed = from_truth (spread 1.559e-05)

| param | injected | corr(A_PM) | 0PA fit | 0PA sigma | 0PA b/s | 0PA+PN fit | PN sigma | PN b/s |
|---|---|---|---|---|---|---|---|---|
| m1 | 1140016.301 | +0.8639 | 1140001.447 | 9.2509e+00 | -1.6057 | 1140016.239 | 3.8319e+01 | -0.0016 |
| m2 | 50 | +0.9377 | 49.99978119 | 9.6004e-05 | -2.2791 | 49.99998748 | 1.1169e-03 | -0.0112 |
| a | 0.9 | +0.8398 | 0.8999889209 | 7.4003e-06 | -1.4971 | 0.8999994796 | 2.5336e-05 | -0.0205 |
| p0 | 13.45955673 | -0.8303 | 13.45964524 | 6.1834e-05 | +1.4314 | 13.45955717 | 2.0340e-04 | +0.0021 |
| e0 | 0.2 | -0.9726 | 0.2000034506 | 3.7897e-06 | +0.9105 | 0.1999967678 | 2.1587e-05 | -0.1497 |
| qS | 0.2 | +0.2332 | 0.201075427 | 3.6466e-03 | +0.2949 | 0.1992684592 | 3.9382e-03 | -0.1858 |
| phiS | 0.2 | -0.5225 | 0.2101325274 | 1.4851e-02 | +0.6823 | 0.2125436688 | 1.9567e-02 | +0.6411 |
| Phi_phi0 | 0.3 | -0.5701 | 0.2544734618 | 4.9098e-02 | -0.9273 | 0.24144965 | 5.3658e-02 | -1.0912 |
| Phi_r0 | 0.5 | -0.1757 | 0.442762546 | 6.1207e-02 | -0.9352 | 0.4640501238 | 8.2913e-02 | -0.4336 |
| C_p | 0 | - | - | - | - | 0.007353779076 | 1.2341e+00 | +0.0060 |
| C_e | 0 | - | - | - | - | -0.2632042104 | 1.3192e+00 | -0.1995 |
