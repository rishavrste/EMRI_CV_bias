"""1D and nD bias per grid point, from the Fisher matrices of fisher_at_best.py (via compare_bias).

    load_bias(snr) -> one dict per grid point (common.load_points fields, flags included) plus
        z_1pa     : bias / sigma, 1PA-only fit, its own sigma           (PHYS)
        z_dev     : bias / sigma, 1PA + deviation fit, its own sigma    (PHYS + C_p, C_e)
        z_dev_1pa : 1PA + deviation bias in units of the 1PA-only sigma (PHYS)
        b_1pa, s_1pa / b_dev, s_dev : bias = best fit - injection and the Fisher sigma, per parameter
        D_1pa, D_dev_all, D_dev_phys, D_dev_C : nD bias D = sqrt(D^2) (compare_bias.nd_bias)
        n_1pa, n_dev_all, n_dev_phys, n_dev_C : the matching Gaussian-equivalent significance
        ..._np : the same without the phases (Phi_phi0, Phi_r0 marginalised); ND_SETS[variant][key]
                 is the parameter set of each
        cond_1pa, cond_dev : Jacobi-rescaled condition numbers
"""
import numpy as np

from common import C, load_points                         # puts IMRI_TAIL_dev on sys.path
import compare_bias as CB                                 # noqa: E402

PHYS = CB.PHYS
ALL = C.PARAMS
NOPH = [n for n in PHYS if n not in C.PHASES]
# nD parameter sets per variant: "" with the phases, "_np" with them marginalised
ND_SETS = {"": {"1pa": PHYS, "dev_all": ALL, "dev_phys": PHYS, "dev_C": C.DEV_PARAMS},
           "_np": {"1pa": NOPH, "dev_all": NOPH + C.DEV_PARAMS, "dev_phys": NOPH,
                   "dev_C": C.DEV_PARAMS}}


def nd(rec, subset):
    D2 = CB.nd_bias(rec, subset)
    return float(np.sqrt(D2)), CB.gauss_sigma(D2, len(subset))


def point_bias(p, scale):
    r1, r2 = CB.load("1pa", p["idx"], scale), CB.load("1pa_dev", p["idx"], scale)
    out = dict(p, z_1pa={n: r1["b"][n] / CB.sigma(r1, n) for n in PHYS},
               z_dev={n: r2["b"][n] / CB.sigma(r2, n) for n in ALL},
               z_dev_1pa={n: r2["b"][n] / CB.sigma(r1, n) for n in PHYS},
               b_1pa={n: r1["b"][n] for n in PHYS}, s_1pa={n: CB.sigma(r1, n) for n in PHYS},
               b_dev={n: r2["b"][n] for n in ALL}, s_dev={n: CB.sigma(r2, n) for n in ALL},
               cond_1pa=r1["cond_hat"], cond_dev=r2["cond_hat"])
    for variant, sets in ND_SETS.items():
        for key, subset in sets.items():
            rec = r1 if key == "1pa" else r2
            out[f"D_{key}{variant}"], out[f"n_{key}{variant}"] = nd(rec, subset)
    return out


def load_bias(snr=20.0):
    return [point_bias(p, snr / 20.0) for p in load_points()]
