"""Controls and paths: 1PA signal vs 0PA + PN deviation with a C_p-driven frequency term, 2nd-generation TDI, A and E.

Signal  : 1PA, (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation.
Template: 0PA, (False, False, False), plus C_p, C_e free.

Deviation (SuperKludge_r @ PN_power_deviation, few/trajectory/ode/flux.py, working tree):
    pdot      += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot      += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5
    Omega_phi += -(15/56) q C_p (1-e^2)^1.5 / p^2,   Omega_r gets the same term      q = m1 m2 / (m1 + m2)^2
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e].
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SK_FILES = Path("/scratch/e1583490/SK_files")      # holds overlap.py and data/
OUT_ROOT = HERE / "results"
OLD_BEST_FITS = HERE.parent / "EMRI_1PA_PN_DEV" / "results" / "best_fits.json"
SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
SUPERKLUDGE_BRANCH = "PN_power_deviation"          # the frequency term lives in its working tree

sys.path.insert(0, str(SK_FILES))
import overlap as ovl                                          # noqa: E402  (SK_files/overlap.py)

# --- case ------------------------------------------------------------------
GRID = "EMRI"
TEMPLATE = "pnfreq"
POINTS = ("0", "4")

SIGNAL_FLAGS = (True, False, False)      # 1PA injection
TEMPLATE_FLAGS = (False, False, False)   # 0PA

# Deviation coefficients in additional_args order (slots 5, 6). The signal has none.
DEV_SLOTS = ["C_p", "C_e"]
DEV_INJ = {n: 0.0 for n in DEV_SLOTS}

# Sky position free; distance, spin orientation, Phi_theta0 and chi2 at the injected values.
PHYS = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
FIT_PARAMS = PHYS + DEV_SLOTS

# --- arms ------------------------------------------------------------------
# Seed "physical": the injected parameters, with C_p, C_e the projection of the exact 1PA pdot, edot onto the
# C_p, C_e shapes at the injected (a, p0, e0, chi2); see physical_c.py.
# "full" climbs at T_FULL directly. "ladder" runs only when "full" ends below ACCEPT_FLOOR: each rung starts at
# the final point of the rung before.
T_FULL = 2.5
LADDER = (2.2, 2.3, 2.4, 2.45, 2.47, 2.5)
# Retry seeds (idx4): the physical parameters of an EMRI_1PA_PN_DEV best fit ("pn" or "0pa"), with the physical
# C_p, C_e as above (the PN fit's own C is not used: C_p now also shifts the frequencies).
SEED_ARMS = {"physical": ("full", "ladder"),
             "pn_fit": ("full_pn", "ladder_pn"),
             "0pa_fit": ("full_0pa", "ladder_0pa")}
ARMS = {arm: dict(seed=seed, rungs=rungs)
        for seed, (full, ladder) in SEED_ARMS.items()
        for arm, rungs in ((full, (T_FULL,)), (ladder, LADDER))}
# "full" worked if it reached the 0PA best-fit overlap of EMRI_1PA_PN_DEV: C = 0 is nested, so a fit below
# it is a failed climb.
ACCEPT_FLOOR = "0pa"


def signal_row(point):
    """The injection of a grid point as a row in the COL layout."""
    return ovl.signal_array(GRID)[int(point)]


def point_label(point):
    return f"idx{point}"


def injected_value(sig_row, name):
    """Injected value of a fitted parameter: the signal row, or 0 for a deviation coefficient."""
    return DEV_INJ[name] if name in DEV_INJ else float(sig_row[COL[name]])


# --- evaluation stack: an SK_files overlap.py setup --------------------------
SETUP = "rishav"                         # FEW defaults, pad_output + odd_len, AET -> AE, f > 0:
                                         # the signal SNR is exactly 20
GEN = ovl.SETUPS[SETUP]["template"]      # signal and template share it in the rishav setup
DROP_DC = ovl.SETUPS[SETUP]["drop_dc"]   # never use the DC bin in a fit

# --- LM controls -----------------------------------------------------------
DER_ORDER, NDELTA = 6, 12                # SEF stencil
RECOMPUTE_DELTAS_EVERY = 3              # re-scan SEF's stable steps this often
OVERLAP_TARGET = 0.9999999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-6, 1000, 30, 1e-9
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_UP = 1e-12, 1e8, 10.0
REGULARISE = False                      # lm.py option, unused here
# SEF's step search is relative to the parameter value; |value| < NEAR_ZERO[name] searches ZERO_STEPS instead.
NEAR_ZERO = {"C_p": 1e-10, "C_e": 1e-10, "a": 1e-2}
ZERO_STEPS = np.geomspace(1e-4, 1e-9, NDELTA)
ALWAYS_ABSOLUTE = {}

# Physical box a proposed step is pulled back into; phiS, phases and C are unbounded.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "qS": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "qS": 3.141592653589793}

COL = ovl.COL
ARGS14 = ovl.ARGS14
