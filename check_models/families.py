"""Deviation families of SuperKludge_r @ PN_freq_evol, the grid points, and the trajectory of a model.

additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, model, c1, ..., c8]
    model 0  PN_freq    C_p, C_e                 Omega_phi, Omega_r += -(15/56) q C_p (1-e^2)^1.5 / p^2
    model 1  A          C_p, C_e, C_Om           one Omega amplitude for Omega_phi and Omega_r
    model 2  B          C_p, C_e, C_Omphi, C_Omr
    model 3  power law  B and n_p, n_e, n_Omphi, n_Omr: each term times (10/p)^n
Each family contains the one before it, so each fit is seeded from the optimum of the previous family.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

SUPERKLUDGE = Path("/home/svu/e1583490/packages_to_install/SuperKludge_r")
BRANCH = "PN_freq_evol"
SK_FILES = Path("/scratch/e1583490/SK_files")
sys.path.insert(0, str(SK_FILES))
import overlap as ovl                     # noqa: E402  (SK_files/overlap.py: grid rows and COL layout)

# EMRI grid points by index; "I<idx>" is the IMRI grid (m2 = 1e3, T = 1 yr)
POINTS = ("0", "1", "2", "3", "4", "7", "10", "19",   # e0 = 0.1 spin row, idx7 (a=0, e0=0.2), idx10, idx19
          "I4")                                         # IMRI idx4 (a=+0.9, e0=0.1), the most biased IMRI cell
SIGNAL_FLAGS = (1, 0, 0)                  # 1PA injection
TEMPLATE_FLAGS = (0, 0, 0)                # 0PA template
PHYS_FREE = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0"]   # phases decide only these

# name -> (model id, coefficient names); "0PA" has no deviation
FAMILIES = {
    "0PA": (None, []),
    "PN_freq": (0, ["C_p", "C_e"]),
    "A": (1, ["C_p", "C_e", "C_Om"]),
    "B": (2, ["C_p", "C_e", "C_Omphi", "C_Omr"]),
    "power_law": (3, ["C_p", "C_e", "C_Omphi", "C_Omr", "n_p", "n_e", "n_Omphi", "n_Omr"]),
}


def branch_check():
    branch = subprocess.run(["git", "-C", str(SUPERKLUDGE), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print(f"SuperKludge_r branch: {branch}", flush=True)
    if branch != BRANCH:
        raise SystemExit(f"SuperKludge_r is on {branch!r}, need {BRANCH!r}")


# --- grid point ------------------------------------------------------------------------
def grid_index(point):
    """("EMRI", 4) for "4", ("IMRI", 4) for "I4"."""
    return ("IMRI", int(point[1:])) if point.startswith("I") else ("EMRI", int(point))


def signal_row(point):
    grid, idx = grid_index(point)
    return ovl.signal_array(grid)[idx]


def injection(point):
    row = signal_row(point)
    return {n: float(row[ovl.COL[n]]) for n in PHYS_FREE + ["chi2", "T"]}


# --- nesting: the seed of a family from the optimum of the one before -----------------
def lift(family, prev_coeffs):
    """Coefficients of `family` that reproduce the previous family's optimum."""
    c = dict(prev_coeffs)
    if family == "A":
        c["C_Om"] = -(15 / 56) * c["C_p"]
    elif family == "B":
        c["C_Omphi"] = c["C_Omr"] = c.pop("C_Om")
    elif family == "power_law":
        c.update(n_p=0.0, n_e=0.0, n_Omphi=0.0, n_Omr=0.0)
    return c


def add_args(chi2, flags, family, coeffs):
    model, names = FAMILIES[family]
    if model is None:
        return [chi2, *flags, 0, 0] + [0.0] * 8
    c = [float(coeffs[n]) for n in names]
    return [chi2, *flags, 1, model] + c + [0.0] * (8 - len(c))


# --- trajectory ------------------------------------------------------------------------
_TRAJ = {}


def phases(theta, args, t_eval, T):
    """Phi_phi, Phi_r at times t_eval [s]; None if the trajectory ends before t_eval[-1]."""
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode.flux import SuperKludgeFlux
    traj = _TRAJ.setdefault("sk", EMRIInspiral(func=SuperKludgeFlux))
    m = (theta["m1"], theta["m2"], theta["a"], theta["p0"], theta["e0"], 1.0)
    kw = dict(Phi_phi0=theta["Phi_phi0"], Phi_r0=theta["Phi_r0"], T=T, dt=10.0)
    t_end = traj(*m, *args, **kw)[0][-1]
    if t_end < t_eval[-1] * (1 - 1e-9):
        return None
    out = traj(*m, *args, **kw, upsample=True, new_t=np.minimum(t_eval, t_end))
    return out[4], out[6]
