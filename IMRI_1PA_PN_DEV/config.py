"""Controls and paths: IMRI grid, 1PA signal vs 0PA and 0PA + PN deviation templates, T-ladder LM,
2nd-generation TDI with A and E only (the 1PA-vs-2PA setup; the old fits in src/IMRI used 1st
generation, A, E, T, at dt 5).

Signal  : 1PA,  (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation.
Template: 0PA,  (False, False, False); "pn" adds the PN deviation, C_p and C_e free.

Deviation (SuperKludge_r @ hybrid, few/trajectory/ode/flux.py, outside the evolve_1PA block):
    pdot += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5        q = m1 m2 / (m1 + m2)^2
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included,
                   C_p, C_e, del_0_p, del_0_e]; the simple deviation (del_0_p, del_0_e) stays 0.

Code adapted from EMRI_1PA_PN_DEV; results go to results/ here.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SK_FILES = Path("/scratch/e1583490/SK_files")      # holds overlap.py and data/
FISHER_RESULTS = HERE.parent / "Fisher_results"     # parse_results.py: the old 1st-gen best fits
OUT_ROOT = HERE / "results"
SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
SUPERKLUDGE_BRANCH = "hybrid"                     # the C_p / C_e slots above live on this branch

sys.path.insert(0, str(SK_FILES))
import overlap as ovl                                          # noqa: E402  (SK_files/overlap.py)

# --- case ------------------------------------------------------------------
GRID = "IMRI"
TEMPLATES = ("0pa", "pn")

SIGNAL_FLAGS = (True, False, False)      # 1PA injection
TEMPLATE_FLAGS = (False, False, False)   # 0PA

# Deviation coefficients in additional_args order (slots 5-8). Only the PN pair is ever fitted;
# the signal has none, so the injected value of each is 0.
DEV_SLOTS = ["C_p", "C_e", "del_0_p", "del_0_e"]
DEV_PARAMS = ["C_p", "C_e"]
DEV_INJ = {n: 0.0 for n in DEV_SLOTS}

# Free parameters, as in the old src/IMRI fits: sky position free, distance, spin orientation and
# Phi_theta0 at the injected values. chi2 (secondary spin) is a 1PA effect: held at the injection.
PHYS = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]


def fit_params(template):
    return PHYS + (DEV_PARAMS if template == "pn" else [])


# Points: the 25 SK_files IMRI grid cells "0".."24" (m1 1e6, m2 1e3, T 1 yr, dt 10, chi2 0.95,
# signal SNR 20). dt 10 as in SK_files (user, 2026-10-05): the grid aliases there (AGENTS.md 3), but
# dt 5 puts the 2nd-generation TDI PSD null at ~0.06 Hz in band.
POINTS = [str(i) for i in range(25)]


def signal_row(point):
    """The injection of a point as a row in the COL layout."""
    return ovl.signal_array(GRID)[int(point)]


def point_label(point):
    """idx4; used in file names."""
    return f"idx{point}"


def injected_value(sig_row, name):
    """Injected value of a fitted parameter: the signal row, or 0 for a deviation coefficient."""
    return DEV_INJ[name] if name in DEV_INJ else float(sig_row[COL[name]])


def case_name(template, t_fracs, seed="old"):
    """{GRID}_{template}_{setup}_{seed}_Tfrac{f1-f2-...}; the rungs are fractions of each point's T."""
    return f"{GRID}_{template}_{SETUP}_{seed}_Tfrac" + "-".join(f"{f:g}" for f in t_fracs)


# --- seeds -----------------------------------------------------------------
# Both written by make_seeds.py as {template: {point: {theta, overlap_old, chi2_old}}}.
# "old": the best fit of the same template in src/IMRI/results_combined.txt (1st-generation TDI,
#        A, E, T, dt 5). Its PN dev_1, dev_2 are C_p, C_e.
# "sk" : the SK_files 0PA LM best fit (LM_iterations/best_fit_IMRI_0pa.json; 2nd-generation AE,
#        but fitted to a 2PA signal) for both templates. It has no sky angles: qS, phiS start at the
#        injection; pn starts at C_p = C_e = 0 (user, 2026-10-05). overlap_old is SK's own score.
SEEDS_OLD = HERE / "seeds_old_1stgen.json"
SEEDS_SK = HERE / "seeds_sk.json"
SEEDS = {"old": SEEDS_OLD, "sk": SEEDS_SK}
OLD_MODEL = {"0pa": "0PA", "pn": "PN"}
OLD_NAMES = {"dev_1": "C_p", "dev_2": "C_e"}
SK_BEST_0PA = SK_FILES / "LM_iterations" / "best_fit_IMRI_0pa.json"

# Rungs of the ladder, fractions of T = 1 yr (user, 2026-10-05).
T_FRACS = (0.9, 0.95, 1.0)

# --- evaluation stack: an SK_files overlap.py setup --------------------------
SETUP = "rishav"                         # FEW defaults, pad_output + odd_len, AET -> AE, f > 0:
                                         # the signal SNR is exactly 20
GEN = ovl.SETUPS[SETUP]["template"]      # signal and template share it in the rishav setup
DROP_DC = ovl.SETUPS[SETUP]["drop_dc"]   # never use the DC bin in a fit

# --- LM controls -----------------------------------------------------------
DER_ORDER, NDELTA = 6, 12                # SEF stencil
RECOMPUTE_DELTAS_EVERY = 3              # re-scan SEF's stable steps this often
OVERLAP_TARGET = 0.9999999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-2, 1000, 30, 1e-9
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_UP = 1e-12, 1e8, 10.0
REGULARISE = False                      # lm.py option, unused here
# SEF's step search is relative to the parameter value, so a parameter near zero gets steps below what
# the waveform resolves (the spin at the a = 0 points). |value| < NEAR_ZERO[name] searches the absolute
# steps ZERO_STEPS instead (SEF's own at-zero grid); model.derivs applies it. C_p = C_e = 0 at the
# start of every sk pn ladder.
NEAR_ZERO = {"C_p": 1e-10, "C_e": 1e-10, "a": 1e-2}
ZERO_STEPS = np.geomspace(1e-4, 1e-9, NDELTA)

# Physical box a proposed step is pulled back into; phiS, phases, C_p and C_e are unbounded.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "qS": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "qS": 3.141592653589793}

COL = ovl.COL
ARGS14 = ovl.ARGS14

PHASES = ("Phi_phi0", "Phi_r0")                    # biases in these are wrapped into (-pi, pi]

# --- best fits, Fisher, bias ------------------------------------------------
BEST_FITS = OUT_ROOT / "best_fits.json"            # best_fits.py: the full-T fit of each template
FISHER_DIR = OUT_ROOT / "fisher_at_best"           # fisher_at_best.py: {template}/fisher_{label}.json
