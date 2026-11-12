"""Controls and paths: 1PA signal vs 0PA + REL5_1 deviation, EMRI grid, 2nd-generation TDI, A and E.

Signal  : 1PA, (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation.
Template: 0PA, (False, False, False), plus the five REL5_1 coefficients free.

Deviation (SuperKludge_r @ PN_v_3, few/trajectory/ode/flux.py), X = pdot, edot, Omega_phi, Omega_r:
    X += eta * X_0PA(p, e) * (c_X * v + K * v^4),      v = p^-1/2,  eta = m1 m2 / (m1 + m2)^2
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, c_p, c_e, c_Omphi, c_Omr, K].
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SK_FILES = Path("/scratch/e1583490/SK_files")      # holds overlap.py and data/
OUT_ROOT = HERE / "results"
SEEDS_FILE = OUT_ROOT / "seeds.json"               # make_seeds.py
OLD_BEST_FITS = HERE.parent / "EMRI_1PA_PN_DEV" / "results" / "best_fits.json"   # 0PA fits, same TDI setup
EMRI_V3 = HERE.parent / "EMRI_v_3"                 # forms_v3.py, phase_fit_v3.py (CPU seeds)
SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
SUPERKLUDGE_BRANCH = "PN_v_3"

sys.path.insert(0, str(SK_FILES))
import overlap as ovl                                          # noqa: E402  (SK_files/overlap.py)

# --- case ------------------------------------------------------------------
GRID = "EMRI"
TEMPLATE = "rel5"
POINTS = tuple(str(i) for i in range(25))

SIGNAL_FLAGS = (True, False, False)      # 1PA injection
TEMPLATE_FLAGS = (False, False, False)   # 0PA

# Deviation coefficients in additional_args order (slots 5-9). The signal has none.
DEV_SLOTS = ["c_p", "c_e", "c_Omphi", "c_Omr", "K"]
DEV_INJ = {n: 0.0 for n in DEV_SLOTS}

# Sky position free; distance, spin orientation, Phi_theta0 and chi2 at the injected values.
PHYS = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
FIT_PARAMS = PHYS + DEV_SLOTS

# --- arms ------------------------------------------------------------------
# Seed: the injected parameters, with the REL5_1 coefficients fitted to the 1PA phases at the injection
# (make_seeds.py). "full" climbs at T_FULL; "ladder" runs only when "full" ends below the 0PA best-fit overlap
# (EMRI_1PA_PN_DEV): the deviation is nested at 0, so a fit below it is a failed climb.
T_FULL = 2.5
LADDER = (2.2, 2.3, 2.4, 2.45, 2.47, 2.5)
ARMS = {"full": (T_FULL,), "ladder": LADDER, "reseed": (T_FULL,)}    # reseed: run_lm.py --coeff-seed-from
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

# --- LM controls (as EMRI_1PA_freq) -----------------------------------------
DER_ORDER, NDELTA = 6, 12                # SEF stencil
RECOMPUTE_DELTAS_EVERY = 3              # re-scan SEF's stable steps this often
OVERLAP_TARGET = 0.9999999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-6, 1000, 30, 1e-9
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_UP = 1e-12, 1e8, 10.0
REGULARISE = False                      # lm.py option, unused here
# SEF's step search is relative to the parameter value; |value| < NEAR_ZERO[name] searches ZERO_STEPS instead
# (a = 0 cells would otherwise get ~1e-9 spin steps).
NEAR_ZERO = {**{n: 1e-3 for n in DEV_SLOTS}, "a": 1e-2}
ZERO_STEPS = np.geomspace(1e-4, 1e-9, NDELTA)
ALWAYS_ABSOLUTE = {}

# Physical box a proposed step is pulled back into; phiS, phases and coefficients are unbounded.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "qS": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "qS": 3.141592653589793}

COL = ovl.COL
ARGS14 = ovl.ARGS14
