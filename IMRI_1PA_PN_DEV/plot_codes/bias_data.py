"""1D and nD bias per grid point, from the Fisher matrices of fisher/fisher_at_best.py (via
fisher/compare_bias.py).

    load_bias(snr) -> one dict per grid point (common.load_points fields) plus
        z_0pa    : bias / sigma, 0PA fit, its own sigma                (PHYS)
        z_pn     : bias / sigma, 0PA + PN fit, its own sigma           (PHYS + C_p, C_e)
        z_pn_0pa : 0PA + PN bias in units of the 0PA sigma             (PHYS)
        b_0pa, s_0pa / b_pn, s_pn : bias = best fit - injection and the Fisher sigma, per parameter
        D_0pa, D_pn_all, D_pn_phys, D_pn_C : nD bias D = sqrt(D^2) (compare_bias.nd_bias)
        n_0pa, n_pn_all, n_pn_phys, n_pn_C : the matching Gaussian-equivalent significance
        ..._np : the same without the phases (Phi_phi0, Phi_r0 marginalised); ND_SETS[variant][key]
                 is the parameter set of each. The sky angles stay in both.
        cond_0pa, cond_pn : Jacobi-rescaled condition numbers
"""
import numpy as np

from common import C, load_points                         # puts IMRI_1PA_PN_DEV, fisher/ on sys.path
import compare_bias as CB                                 # noqa: E402

PHYS = CB.PHYS
ALL = CB.ALL
NOPH = [n for n in PHYS if n not in C.PHASES]
# nD parameter sets per variant: "" with the phases, "_np" with them marginalised
ND_SETS = {"": {"0pa": PHYS, "pn_all": ALL, "pn_phys": PHYS, "pn_C": C.DEV_PARAMS},
           "_np": {"0pa": NOPH, "pn_all": NOPH + C.DEV_PARAMS, "pn_phys": NOPH,
                   "pn_C": C.DEV_PARAMS}}


def nd(rec, subset):
    D2 = CB.nd_bias(rec, subset)
    return float(np.sqrt(D2)), CB.gauss_sigma(D2, len(subset))


def point_bias(p, scale):
    r1, r2 = CB.load("0pa", p["point"], scale), CB.load("pn", p["point"], scale)
    out = dict(p, z_0pa={n: r1["b"][n] / CB.sigma(r1, n) for n in PHYS},
               z_pn={n: r2["b"][n] / CB.sigma(r2, n) for n in ALL},
               z_pn_0pa={n: r2["b"][n] / CB.sigma(r1, n) for n in PHYS},
               b_0pa={n: r1["b"][n] for n in PHYS}, s_0pa={n: CB.sigma(r1, n) for n in PHYS},
               b_pn={n: r2["b"][n] for n in ALL}, s_pn={n: CB.sigma(r2, n) for n in ALL},
               cond_0pa=r1["cond_hat"], cond_pn=r2["cond_hat"])
    for variant, sets in ND_SETS.items():
        for key, subset in sets.items():
            rec = r1 if key == "0pa" else r2
            out[f"D_{key}{variant}"], out[f"n_{key}{variant}"] = nd(rec, subset)
    return out


def load_bias(snr=CB.SNR0):
    return [point_bias(p, snr / CB.SNR0) for p in load_points()]
