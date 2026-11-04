"""Spin check: the EMRI idx22 injection (a = 0, e0 = 0.5) repeated at small spins, 0PA and 0PA + PN fits.

Each spin gets its own injection, otherwise idx22's:
    p0   : the 1PA trajectory (the signal's flags) plunges at T = 2.5 yr (get_p_at_t)
    dist : rescaled so the 1PA signal has SNR 20 in the rishav setup (A, E, DC bin dropped)
Seed of each fit: the new injection + (idx22 best fit - idx22 injection) of the same template, then
the EMRI_1PA_PN_DEV T ladder 0.9 -> 1 of T. Fisher at the full-T best fit -> critical SNR.

Signal, template, inner product, LM and SEF are EMRI_1PA_PN_DEV's (config, model, lm), unchanged
except for the spin step grid below.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "EMRI_1PA_PN_DEV"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "fisher"))
import config as C                                  # noqa: E402  (EMRI_1PA_PN_DEV/config.py)

SPINS = [-1e-3, -5e-2, -1e-1, 0.0, 1e-1, 5e-2, 1e-3]
BASE_POINT = "22"                                   # idx22: a = 0, e0 = 0.5
T_FRACS = (0.9, 1.0)                                # rungs, fractions of the stored T (2.5 yr)
SNR = 20.0
RESULTS = HERE / "results"

# SEF's default spin step search is relative to a, so small spins get unresolved steps (the
# fisher_spin_scan at a = +0.01 chose 1.9e-9, sigma_a 70x off). Every spin here takes the absolute
# at-zero grid C.ZERO_STEPS (1e-4 .. 1e-9) for a, in the LM and the Fisher alike.
C.NEAR_ZERO["a"] = 0.2


def label(spin):
    """a-0.001, a+0, a+0.1: used in directory names."""
    return f"a{spin:+g}"
