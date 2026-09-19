# harder environmental points: where does 0PA+PN beat 0PA?

Every converged PN variant against its own 0PA baseline.  "bias better on"
counts parameters where PN's absolute bias is smaller than 0PA's, with angles
unwrapped first.  `e0` is held at its injected value throughout.

PN exported for source(s): 8, 13.
Source 16 is left out of this analysis; source 14 has not been run.

| source | PN run | ov 0PA | ov PN | PN >= 0PA | bias better on | C_p |
|---|---|---|---|---|---|---|
| 8 | DE in a +-7 sigma box about the injected point | 0.927343 | 0.999999 | yes | 7/7 | 18.84 |
| 13 | DE (2000 iters) in a +-5 sigma box about the injected point, then an LM climb | 0.999852 | 0.999975 | yes | 2/7 | 74.55 |

## PN improved the fit (overlap held, bias reduced)

| source | PN run | ov 0PA | ov PN | PN >= 0PA | bias better on | C_p |
|---|---|---|---|---|---|---|
| 8 | DE in a +-7 sigma box about the injected point | 0.927343 | 0.999999 | yes | 7/7 | 18.84 |
| 13 | DE (2000 iters) in a +-5 sigma box about the injected point, then an LM climb | 0.999852 | 0.999975 | yes | 2/7 | 74.55 |

## bias reduced on some parameters, but the overlap collapsed

These are not improvements: the PN search stopped somewhere far worse, and a
smaller error on one parameter there is where it happened to land.

_none_

## ties: the PN fit is the 0PA fit

`C_p` stayed at zero, where the PN template is exactly the 0PA one, so both
fits agree to the optimiser's arithmetic.

_none_

## not represented here

The LM-seeded run on source 8 (started at the 0PA best fit with `C_p = 0`)
crashed at the save step, so it has no results file to read.  From its log it
gained overlap (+6.0e-3) while `C_p` ran to 468, and it was worse in absolute
bias on 5 of 7 parameters.
