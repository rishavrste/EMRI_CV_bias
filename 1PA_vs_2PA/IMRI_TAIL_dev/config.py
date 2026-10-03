"""Controls and paths: IMRI_TAIL, 2PA signal vs 1PA + deviation template, T-ladder LM.

Signal  : 2PA,  (evolve_1PA, evolve_primary, evolve_2PA) = (True, False, True), no deviation.
Template: 1PA + deviation, (True, False, False), deviation_included = True, C_p and C_e free.

Deviation (SuperKludge_r @ 2PA_PN_deviation, few/trajectory/ode/flux.py, inside the evolve_1PA block):
    pdot += q^2 C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot += q^2 e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5        q = m1 m2 / (m1 + m2)^2
additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e].

Code adapted from SK_files/LM_iterations/code/t_ladder; results go to results/ here.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SK_FILES = Path("/scratch/e1583490/SK_files")      # holds overlap.py and data/
OUT_ROOT = HERE / "results"
SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
SUPERKLUDGE_BRANCH = "2PA_PN_deviation"           # the deviation slots above live on this branch

sys.path.insert(0, str(SK_FILES))
import overlap as ovl                                          # noqa: E402  (SK_files/overlap.py)

# --- case ------------------------------------------------------------------
GRID = "IMRI_TAIL"
TEMPLATE = "1pa_dev"

SIGNAL_FLAGS = (True, False, True)       # 2PA injection
TEMPLATE_FLAGS = (True, False, False)    # 1PA; the deviation only acts when evolve_1PA is on

# Deviation coefficients, in additional_args order (slots 5, 6). The signal has none, so the
# injected value of each is 0.
DEV_PARAMS = ["C_p", "C_e"]
DEV_INJ = {"C_p": 0.0, "C_e": 0.0}

# Free parameters. Distance, sky angles and Phi_theta0 stay at the injected values.
PARAMS = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0", "chi2"] + DEV_PARAMS


def fit_params(fix_chi2=False):
    """The fitted parameters; with fix_chi2 the secondary spin stays at its injected value."""
    return [p for p in PARAMS if not (fix_chi2 and p == "chi2")]


def injected_value(sig_row, name):
    """Injected value of a fitted parameter: the signal row, or 0 for a deviation coefficient."""
    return DEV_INJ[name] if name in DEV_INJ else float(sig_row[COL[name]])


def case_name(fix_chi2, dist_div, t_steps, seed="inj", chi2_at="inj"):
    """{GRID}_{TEMPLATE}[_fixchi2[seed]][_distdiv{D}][_dropdc]_{setup}_{seed}_{T1-T2-...}

    "_fixchi2seed": chi2 held at the seed's value instead of the injected one."""
    name = f"{GRID}_{TEMPLATE}" + ("_fixchi2" if fix_chi2 else "")
    name += "seed" if fix_chi2 and chi2_at == "seed" else ""
    name += f"_distdiv{dist_div:g}" if dist_div != 1.0 else ""
    name += ("_dropdc" if DROP_DC else "") + f"_{SETUP}_{seed}_"
    return name + "-".join(f"{t:g}" for t in t_steps)


# --- seeds -----------------------------------------------------------------
# Where the first rung starts. Any fitted parameter the source lacks (chi2, C_p, C_e) starts at
# its injected value.
#   inj      : the injection, C_p = C_e = 0
#   1pabest  : the SK_files 1PA-only best fit (chi2 free there), C_p = C_e = 0
#   dev...   : the final point of an earlier run of this grid, results/{case}/lm_idx{i}.json; a run
#              that held chi2 fixed supplies its fixed value
BEST_1PA = SK_FILES / "LM_iterations" / "best_fit_IMRI_TAIL_1pa.json"
SEED_CASES = {
    "devinj": "IMRI_TAIL_1pa_dev_fixchi2_distdiv10_dropdc_rishav_inj_0.2-0.25",
    "dev1pabest": "IMRI_TAIL_1pa_dev_fixchi2_distdiv10_dropdc_rishav_1pabest_0.25",
    "devchi2seed": "IMRI_TAIL_1pa_dev_fixchi2seed_distdiv10_dropdc_rishav_1pabest_0.242-0.25",
}
SEEDS = ["inj", "1pabest", *SEED_CASES]


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

# Physical box a proposed step is pulled back into; C_p and C_e are unbounded.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0, "chi2": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99, "chi2": 0.999}

PHASES = ("Phi_phi0", "Phi_r0")         # biases in these are wrapped into (-pi, pi]

# Best fits: the SK_files 1PA-only file, and this grid's chosen 1PA + deviation fits
# (best_fit_dev.py).
BEST_DEV = OUT_ROOT / "best_fit_IMRI_TAIL_1pa_dev.json"

COL = ovl.COL
ARGS14 = ovl.ARGS14
