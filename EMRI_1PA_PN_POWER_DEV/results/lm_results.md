# LM arms, 0PA + power-law PN

## Fits at T = 2.5 yr

| point | arm | overlap | mismatch gain vs PN | C_p | C_e | n_p | n_e | m2 | iters | stop |
|---|---|---|---|---|---|---|---|---|---|---|
| idx0 | best_full | 0.9999999839 | 1.06x | +19.556 | +2.690 | +0.0416 | +0.5331 | 10.000091 | 28 | lambda_exhausted |
| idx0 | best_ladder | 0.9999999844 | 1.1x | +20.174 | +2.502 | -0.0191 | +0.9288 | 10.000095 | 15 | lambda_exhausted |
| idx0 | inj_ladder | not reached | | | | | | | | |
| idx4 | best_full | 0.9999743943 | 1.01x | -36.552 | +14.690 | -0.0088 | -0.2027 | 9.998643 | 20 | lambda_exhausted |
| idx4 | best_ladder | 0.9999964328 | 7.26x | -1.136 | +18.398 | -2.7000 | -3.2919 | 9.999694 | 771 | lambda_exhausted |
| idx4 | inj_ladder | not reached | | | | | | | | |
| idx8 | best_full | not reached | | | | | | | | |
| idx8 | best_ladder | not reached | | | | | | | | |
| idx8 | inj_ladder | not reached | | | | | | | | |
| idx12 | best_full | not reached | | | | | | | | |
| idx12 | best_ladder | not reached | | | | | | | | |
| idx12 | inj_ladder | not reached | | | | | | | | |
| idx19 | best_full | not reached | | | | | | | | |
| idx19 | best_ladder | not reached | | | | | | | | |
| idx19 | inj_ladder | not reached | | | | | | | | |

## Spread between arms, in sigma of the best arm

| point | pair | m1 | m2 | a | p0 | e0 | C_p | C_e | n_p | n_e |
|---|---|---|---|---|---|---|---|---|---|---|
| idx0 | best_full vs best_ladder | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| idx4 | best_full vs best_ladder | 0.01 | 0.03 | 0.01 | 0.01 | 0.00 | 1.00 | 0.00 | 0.06 | 0.02 |

## Ladder rungs

| point | arm | T | start overlap | end overlap | n_p | stop |
|---|---|---|---|---|---|---|
| idx0 | best_ladder | 2.2 | 0.9999999900 | 0.9999999980 | -0.1243 | lambda_exhausted |
| idx0 | best_ladder | 2.3 | 0.9999999971 | 0.9999999979 | -0.1272 | rel_tol |
| idx0 | best_ladder | 2.4 | 0.9999999924 | 0.9999999970 | -0.1285 | lambda_exhausted |
| idx0 | best_ladder | 2.45 | 0.9999999884 | 0.9999999946 | -0.0769 | lambda_exhausted |
| idx0 | best_ladder | 2.47 | 0.9999999905 | 0.9999999921 | -0.0760 | lambda_exhausted |
| idx0 | best_ladder | 2.5 | 0.9999999730 | 0.9999999844 | -0.0191 | lambda_exhausted |
| idx0 | inj_ladder | 2.2 | -0.0065215992 | 0.0334718384 | -147.5551 | running |
| idx4 | best_ladder | 2.2 | 0.9999857086 | 0.9999996244 | +0.2930 | lambda_exhausted |
| idx4 | best_ladder | 2.3 | 0.9999989375 | 0.9999994082 | +0.2667 | lambda_exhausted |
| idx4 | best_ladder | 2.4 | 0.9999713790 | 0.9999968365 | +0.1547 | lambda_exhausted |
| idx4 | best_ladder | 2.45 | 0.9999847776 | 0.9999942496 | +0.0732 | lambda_exhausted |
| idx4 | best_ladder | 2.47 | 0.9999870429 | 0.9999907545 | +0.0241 | lambda_exhausted |
| idx4 | best_ladder | 2.5 | 0.9998553559 | 0.9999964328 | -2.7000 | lambda_exhausted |
| idx4 | inj_ladder | 2.2 | -0.0961898542 | -0.0961898542 | +161.5516 | running |
