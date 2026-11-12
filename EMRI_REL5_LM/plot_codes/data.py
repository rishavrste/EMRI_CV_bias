"""Per-point fits of 0PA, 0PA + PN and 0PA + REL5_1, each at its own Fisher (SNR 20, T = 2.5 yr).

    load_points() -> one dict per grid point, sorted by idx:
        idx, a, e0
        fits[tag] for tag in TAGS: names, bias (best fit - injection, phases wrapped), sigma (marginal,
            every other fitted parameter marginalised), mismatch 1 - O, rho_crit {cl: value},
            max1d (max over INTRINSIC of |b|/sigma, its parameter)
Fishers: EMRI_1PA_PN_DEV/results/fisher_at_best/{0pa,pn} and results/fisher_at_best here.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # EMRI_REL5_LM, for config and make_summary
import config as C                                       # noqa: E402
import make_summary as MS                                # noqa: E402

TAGS = {"0pa": "0PA", "pn": "0PA + PN", "rel5": "0PA + REL5_1"}
PHYS = list(C.PHYS)
DEV = list(C.DEV_SLOTS)
A_VALUES = [-0.9, -0.5, 0.0, 0.5, 0.9]
E_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]


def fisher_json(tag, point):
    if tag == "rel5":
        return MS.new_fit(point)
    return MS.load(MS.OLD_FISHERS / tag / f"fisher_idx{point}.json")


def summarise(d):
    names, G = d["names"], np.asarray(d["fisher"])
    sd = np.sqrt(np.abs(np.diag(MS.covariance(G))))
    D = MS.nd_distance(G, names, d["bias"])
    return dict(names=names, bias=dict(zip(names, d["bias"])), sigma=dict(zip(names, sd)),
                mismatch=1 - d["final_eval"]["overlap"], D=D,
                rho_crit={cl: MS.rho_crit(D, cl) for cl in MS.CLS},
                max1d=MS.max_1d(G, names, d["bias"]))


def point_record(point):
    row = C.signal_row(point)
    return dict(idx=int(point), a=round(float(row[C.COL["a"]]), 2), e0=round(float(row[C.COL["e0"]]), 2),
                fits={tag: summarise(fisher_json(tag, point)) for tag in TAGS})


def load_points():
    return [point_record(p) for p in C.POINTS]


def grid(points, value):
    """5 x 5 array of value(point): rows e0 = 0.1 .. 0.5, columns a = -0.9 .. 0.9."""
    g = np.full((len(E_VALUES), len(A_VALUES)), np.nan)
    for p in points:
        g[E_VALUES.index(p["e0"]), A_VALUES.index(p["a"])] = value(p)
    return g
