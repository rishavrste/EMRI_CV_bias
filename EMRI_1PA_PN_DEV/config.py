"""Controls and paths: EMRI grid, 1PA signal vs 0PA and 0PA + PN deviation templates, T-ladder LM,
2nd-generation TDI with A and E only (the 1PA-vs-2PA setup; the old fits in src/EMRI used 1st
generation, A, E, T).

Signal  : 1PA,  (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, False), no deviation.
Template: 0PA,  (False, False, False); "pn" adds the PN deviation, C_p and C_e free.

Deviation (SuperKludge_r @ hybrid, few/trajectory/ode/flux.py, outside the evolve_1PA block):
    pdot += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5        q = m1 m2 / (m1 + m2)^2
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included,
                   C_p, C_e, del_0_p, del_0_e]; the simple deviation (del_0_p, del_0_e) stays 0.

Code adapted from 1PA_vs_2PA/IMRI_TAIL_dev; results go to results/ here.
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
GRID = "EMRI"
TEMPLATES = ("0pa", "pn")

SIGNAL_FLAGS = (True, False, False)      # 1PA injection
TEMPLATE_FLAGS = (False, False, False)   # 0PA

# Deviation coefficients in additional_args order (slots 5-8). Only the PN pair is ever fitted;
# the signal has none, so the injected value of each is 0.
DEV_SLOTS = ["C_p", "C_e", "del_0_p", "del_0_e"]
DEV_PARAMS = ["C_p", "C_e"]
DEV_INJ = {n: 0.0 for n in DEV_SLOTS}

# Free parameters, as in the old src/EMRI fits: sky position free, distance, spin orientation and
# Phi_theta0 at the injected values. chi2 (secondary spin) is a 1PA effect: held at the injection.
PHYS = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]


def fit_params(template):
    return PHYS + (DEV_PARAMS if template == "pn" else [])


# Points: the 25 SK_files grid cells ("0".."24") and the old src/EMRI ad-hoc injection, which is
# outside the grid (its own T, dt, chi2 and distance: SNR ~44.7), as a row in the COL layout.
ADHOC = {"adhoc_A": dict(m1=1e6, m2=10.0, a=0.9, p0=7.5, e0=0.5, x0=1.0, dist=5.0,
                         qS=0.7853981633974483, phiS=1.0, qK=1.0, phiK=1.0471975511965976,
                         Phi_phi0=0.9, Phi_theta0=0.5, Phi_r0=0.4, dt=5.0, T=1.0, chi2=0.0)}
POINTS = [str(i) for i in range(25)] + list(ADHOC)


def signal_row(point):
    """The injection of a point as a row in the COL layout."""
    if point in ADHOC:
        row = np.zeros(max(COL.values()) + 1)
        for n, v in ADHOC[point].items():
            row[COL[n]] = v
        return row
    return ovl.signal_array(GRID)[int(point)]


def point_label(point):
    """idx4 for a grid cell, the name otherwise; used in file names."""
    return f"idx{point}" if point.isdigit() else point


def injected_value(sig_row, name):
    """Injected value of a fitted parameter: the signal row, or 0 for a deviation coefficient."""
    return DEV_INJ[name] if name in DEV_INJ else float(sig_row[COL[name]])


def case_name(template, t_fracs, seed="old"):
    """{GRID}_{template}_{setup}_{seed}_Tfrac{f1-f2-...}; the rungs are fractions of each point's T."""
    return f"{GRID}_{template}_{SETUP}_{seed}_Tfrac" + "-".join(f"{f:g}" for f in t_fracs)


# --- seeds -----------------------------------------------------------------
# "old": the best fit of the same template in src/EMRI/results_compiled.txt (1st-generation TDI,
# A, E, T), copied to SEEDS_OLD by make_seeds.py. Its PN dev_1, dev_2 are C_p, C_e.
SEEDS_OLD = HERE / "seeds_old_1stgen.json"
OLD_MODEL = {"0pa": "0PA", "pn": "PN"}
OLD_NAMES = {"dev_1": "C_p", "dev_2": "C_e"}
# "row": PN re-climb off the C = 0 ridge, written by make_seeds_row.py: the point's own 0PA ladder fit
# plus the median C_p, C_e of its row-mates (same a) where PN bought more than ROW_MIN_GAIN.
SEEDS_ROW = HERE / "seeds_row.json"
ROW_MIN_GAIN = 2.0
SEEDS = {"old": SEEDS_OLD, "row": SEEDS_ROW}
# "ramp" (run_cp_ramp.py): from the point's own 0PA fit, step C_p by RAMP_STEP out to +-RAMP_MAX with
# C_e = 0, re-fitting PHYS at each step; a direction stops once 1 - O exceeds RAMP_LOSS x its C_p = 0
# value. The best step seeds the free PN climb.
RAMP_STEP, RAMP_MAX, RAMP_LOSS = 2.5, 40.0, 100.0
RAMP_MAX_ITERS, RAMP_REL_TOL = 30, 1e-6
# "rampfine": the ladder continued from a finished rung of another case, for fits that lost the overlap
# on too long a jump in T (ramp idx12, 17: 2.25 -> 2.5 yr). SEEDS_FIT[seed] = (that case's seed, its
# t_fracs, the index of its rung to start from).
SEEDS_FIT = {"rampfine": ("ramp", (0.9, 1.0), 0)}
# "best": the final full-T fit of the template in BEST_FITS (fisher/best_fits.py), for a re-climb at the
# a = 0 points with the at-zero spin steps (NEAR_ZERO below).
SEED_BEST = "best"

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
# the waveform resolves (the spin at the a = 0 points: best fit ~1e-4, steps ~3e-9, noisy d/da).
# |value| < NEAR_ZERO[name] searches the absolute steps ZERO_STEPS instead (SEF's own at-zero grid);
# model.derivs applies it in the LM and the Fisher alike.
NEAR_ZERO = {"C_p": 1e-10, "C_e": 1e-10, "a": 1e-2}
ZERO_STEPS = np.geomspace(1e-4, 1e-9, NDELTA)

# Physical box a proposed step is pulled back into; phiS, phases, C_p and C_e are unbounded.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "qS": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "qS": 3.141592653589793}

COL = ovl.COL
ARGS14 = ovl.ARGS14

# --- best fits, Fisher, bias ------------------------------------------------
BEST_FITS = OUT_ROOT / "best_fits.json"            # best_fits.py: the full-T fit of each template
FISHER_DIR = OUT_ROOT / "fisher_at_best"           # fisher_at_best.py: {template}/fisher_{label}.json
PHASES = ("Phi_phi0", "Phi_r0")                    # biases in these are wrapped into (-pi, pi]
