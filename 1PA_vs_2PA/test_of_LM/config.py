"""Controls and paths for the injection-seeded LM test (IMRI_TAIL, 2PA signal vs 1PA / 0PA template).

Every other script in this folder takes its settings from here.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SK_FILES = Path("/scratch/e1583490/SK_files")      # holds overlap.py and data/
OUT_ROOT = HERE / "results"

sys.path.insert(0, str(SK_FILES))
import overlap as ovl                                          # noqa: E402  (SK_files/overlap.py)

# --- case ------------------------------------------------------------------
GRID = "IMRI_TAIL"
IDX_DEFAULT = (6, 9)

# Waveform flags (evolve_1PA, evolve_primary, evolve_2PA); the secondary spin chi2 comes first.
SIGNAL_FLAGS = (True, False, True)       # 2PA injection
TEMPLATE_PA = "1pa"                      # --template: analysis template, "1pa" or "0pa"


def template_flags():
    """(evolve_1PA, evolve_primary, evolve_2PA) of the analysis template."""
    return (TEMPLATE_PA == "1pa", False, False)

# Free parameters (8). Distance, sky angles and Phi_theta0 stay at the injected values.
PARAMS = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0", "chi2"]


PHASES = ("Phi_phi0", "Phi_r0")


def fit_params(fix_chi2=False, fix_phases=False):
    """The fitted parameters; held ones (chi2, the two phases) stay at their injected values.
    A 0PA template does not depend on chi2, so chi2 is never fitted with it."""
    held = ({"chi2"} if fix_chi2 or TEMPLATE_PA == "0pa" else set()) | (set(PHASES) if fix_phases else set())
    return [p for p in PARAMS if p not in held]


def case_name(fix_chi2=False, dist_div=1.0, fix_phases=False):
    """Results folder: IMRI_TAIL_{1pa,0pa}[_fixchi2][_fixphases][_distdiv{D}][_reg]."""
    name = f"{GRID}_{TEMPLATE_PA}" + ("_fixchi2" if fix_chi2 else "") + ("_fixphases" if fix_phases else "")
    name += f"_distdiv{dist_div:g}" if dist_div != 1.0 else ""
    return name + ("_reg" if REGULARISE else "")


# --- evaluation stack: the SK_files `doc` setup ------------------------------
GEN = ovl.FIT_GEN                        # err 1e-11, mode threshold 1e-5, pad_output, AE
DROP_DC = False                          # every rfft bin

# --- LM controls -----------------------------------------------------------
DER_ORDER, NDELTA = 6, 12                # SEF stencil
RECOMPUTE_DELTAS_EVERY = 3              # re-scan SEF's stable steps this often
OVERLAP_TARGET = 0.9999999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-2, 1000, 30, 1e-9
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_UP = 1e-12, 1e8, 10.0
REGULARISE = False                      # --regularise: step through Sigma_reg (see lm.py)

# Physical box a proposed step is pulled back into.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "chi2": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "chi2": 0.999}

COL = ovl.COL
ARGS14 = ovl.ARGS14
