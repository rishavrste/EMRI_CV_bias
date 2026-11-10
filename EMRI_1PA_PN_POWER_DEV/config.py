"""Controls and paths: 1PA signal vs 0PA + power-law PN deviation, three LM arms, 2nd-generation TDI, A and E.

Signal  : 1PA, (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation.
Template: 0PA, (False, False, False), plus C_p, C_e, n_p, n_e free.

Deviation (SuperKludge_r @ PN_power_deviation, few/trajectory/ode/flux.py):
    pdot += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5 (p/10)^n_p
    edot += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5 (p/10)^n_e      q = m1 m2 / (m1 + m2)^2
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, n_p, n_e].

Setup and LM controls as in EMRI_1PA_PN_DEV; results go to results/ here.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SK_FILES = Path("/scratch/e1583490/SK_files")      # holds overlap.py and data/
OUT_ROOT = HERE / "results"
OLD_BEST_FITS = HERE.parent / "EMRI_1PA_PN_DEV" / "results" / "best_fits.json"
SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
SUPERKLUDGE_BRANCH = "PN_power_deviation"          # the n_p, n_e slots above live on this branch

sys.path.insert(0, str(SK_FILES))
import overlap as ovl                                          # noqa: E402  (SK_files/overlap.py)

# --- case ------------------------------------------------------------------
GRID = "EMRI"
TEMPLATE = "pnpow"
POINTS = ("0", "4", "8", "12", "19")

SIGNAL_FLAGS = (True, False, False)      # 1PA injection
TEMPLATE_FLAGS = (False, False, False)   # 0PA

# Deviation coefficients in additional_args order (slots 5-8). The signal has none.
DEV_SLOTS = ["C_p", "C_e", "n_p", "n_e"]
DEV_INJ = {n: 0.0 for n in DEV_SLOTS}

# Sky position free; distance, spin orientation, Phi_theta0 and chi2 at the injected values.
PHYS = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]
FIT_PARAMS = PHYS + DEV_SLOTS

# --- arms ------------------------------------------------------------------
# seed "best": the PN best fit of EMRI_1PA_PN_DEV with n_p = n_e = 0.
# seed "inj" : the injection with C_p = C_e = INJ_C, n_p = n_e = 0 (at C = 0, d/dn vanishes).
# rungs: observation times [yr]; each rung starts at the final point of the rung before.
T_FULL = 2.5
LADDER = (2.2, 2.3, 2.4, 2.45, 2.47, 2.5)
INJ_C = 1.0
ARMS = {"best_full": dict(seed="best", rungs=(T_FULL,)),
        "best_ladder": dict(seed="best", rungs=LADDER),
        "inj_ladder": dict(seed="inj", rungs=LADDER)}


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
# LAMBDA0 is 1e-2 in EMRI_1PA_PN_DEV. With n free the scaled Fisher reaches eigenvalues ~1e-11, so a useful
# step needs lam <= 1e-4 (diagnose_step.py, idx4), and LM only raises lam after a rejected step.
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-6, 1000, 30, 1e-9
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_UP = 1e-12, 1e8, 10.0
REGULARISE = False                      # lm.py option, unused here
# SEF's step search is relative to the parameter value. |value| < NEAR_ZERO[name] searches the absolute
# steps ZERO_STEPS instead. n_p, n_e always search N_STEPS: they sit at or near 0, and a relative step
# there is below what the waveform resolves (the n term is ~1e-5 of pdot).
NEAR_ZERO = {"C_p": 1e-10, "C_e": 1e-10, "a": 1e-2}
ZERO_STEPS = np.geomspace(1e-4, 1e-9, NDELTA)
N_STEPS = np.geomspace(1e-1, 1e-5, NDELTA)
ALWAYS_ABSOLUTE = {"n_p": N_STEPS, "n_e": N_STEPS}

# Physical box a proposed step is pulled back into; phiS, phases, C and n are unbounded.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "qS": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "qS": 3.141592653589793}

COL = ovl.COL
ARGS14 = ovl.ARGS14
