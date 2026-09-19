# harder environmental point idx 13 -- DE (2000 iters) in a +-5 sigma box about the injected point, then an LM climb

Injected signal carries the environmental flux deviation.  `e0` is held at its
injected value; A and E channels only.  Angles are unwrapped before the bias.

| | 0PA | 0PA+PN |
|---|---|---|
| overlap | 0.9998516514 | 0.9999753184 |
| chi2 | 1.188259e+01 | 2.031127e+00 |
| wall time | 3.20 h | 6.89 h |

PN >= 0PA in overlap: **yes**.  In absolute bias PN is better on 2 of 7 parameters (`phiS`, `Phi_phi0`).

Fitted `C_p` = 74.5479.

## absolute bias

| param | injected | 0PA fit | 0PA+PN fit | \|bias\| 0PA | \|bias\| PN | ratio |
|---|---|---|---|---|---|---|
| `m1` | 2067045.419 | 2067858.162 | 2063358.947 | 8.1274e+02 | 3.6865e+03 | 4.54 |
| `m2` | 50 | 49.98906906 | 50.07279921 | 1.0931e-02 | 7.2799e-02 | 6.66 |
| `a` | 0.9 | 0.9000360486 | 0.8997117229 | 3.6049e-05 | 2.8828e-04 | 8 |
| `p0` | 10.30948048 | 10.30667829 | 10.32213794 | 2.8022e-03 | 1.2657e-02 | 4.52 |
| `qS` | 0.2 | 0.1999220689 | 0.1978778345 | 7.7931e-05 | 2.1222e-03 | 27.2 |
| `phiS` | 0.2 | 0.1769516893 | 0.1989401244 | 2.3048e-02 | 1.0599e-03 | 0.046 |
| `Phi_phi0` | 0.3 | 0.1730683742 | 0.2464432787 | 1.2693e-01 | 5.3557e-02 | 0.422 |
