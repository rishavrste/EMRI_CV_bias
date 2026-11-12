# 0PA + PN + C_p-driven frequency term, LM from the physical 1PA seed

pdot += q C_p PN_p,  edot += q e C_e PN_e,  Omega_phi, Omega_r += -(15/56) q C_p (1-e^2)^1.5 / p^2

## Fits

| point | arm | overlap | 1 - O | gain vs PN | gain vs PN_power | gain vs PN+env | C_p | C_e |
|---|---|---|---|---|---|---|---|---|
| idx0 | full | 0.9999999909 | 9.134e-09 | 1.88x | 1.71x | 1.87x | -6.8672 | +6.78741 |
| idx4 | full | 0.7390241749 | 2.610e-01 | 9.93e-05x | 1.37e-05x | 2.83e-05x | +4.67871 | -695.752 |

## Conditioning of the 11-parameter Fisher (Jacobi-rescaled)

| point | cond | corr C_p-C_e | corr C_p-m1 | corr C_p-m2 | corr C_p-a | corr C_p-p0 | corr C_e-e0 |
|---|---|---|---|---|---|---|---|
| idx0 | 4.83e+10 | +0.9773 | -0.9634 | +0.3818 | -0.9360 | +0.9537 | +0.9738 |
| idx4 | 1.01e+09 | -0.1450 | -0.6073 | +0.7000 | -0.0079 | +0.7147 | +0.9786 |

## nD critical SNR, intrinsic parameters (m1, m2, a, p0, e0), each at its own Fisher

| fit | 1 - O | D at SNR 20 | rho_crit 90% | rho_crit 99% | rho_crit 90%, steps doubled |
|---|---|---|---|---|---|
| idx0 0PA | 1.385e-05 | 78.6 | 0.773 | 0.988 |  |
| idx0 0PA+PN | 1.716e-08 | 5.91 | 10.3 | 13.1 |  |
| idx0 0PA+PN_power | 1.564e-08 | 3.59 | 16.9 | 21.6 |  |
| idx0 0PA+PN+env | 1.712e-08 | 3.58 | 17 | 21.7 |  |
| idx0 0PA+PN+freq | 9.134e-09 | 4.04 | 15 | 19.2 | 15 |
| idx4 0PA | 5.853e-04 | 29 | 2.1 | 2.68 |  |
| idx4 0PA+PN | 2.591e-05 | 1.53 | 39.7 | 50.7 |  |
| idx4 0PA+PN_power | 3.567e-06 | 1.09 | 55.7 | 71.2 |  |
| idx4 0PA+PN+env | 7.397e-06 | 0.387 | 157 | 201 |  |
| idx4 0PA+PN+freq | 2.610e-01 | 24.4 | 2.5 | 3.19 | 2.5 |

## Best-fit vectors

```
idx0: m1=1000025.9364320938, m2=9.999857783692788, a=-0.8999189049434788, p0=13.067070354714962, e0=0.10000747882996214, qS=1.0471943413353744, phiS=0.7854204457157667, Phi_phi0=0.09968251500023075, Phi_r0=0.30021216703217124, C_p=-6.86720415435483, C_e=6.787412418759294
idx4: m1=1000802.3568162394, m2=10.008168941291473, a=0.9003824960585387, p0=9.084267645934151, e0=0.09833217448260585, qS=0.9831936009320151, phiS=0.6946468179653119, Phi_phi0=1.0819532245957926, Phi_r0=0.976948744910652, C_p=4.678708424274366, C_e=-695.751593462994
```
