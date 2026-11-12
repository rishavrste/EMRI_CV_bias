"""Does the PN + frequency deviation in SuperKludgeFlux.modify_rhs equal the leading (lowest power of v = p^-1/2)
term of the SuperKludge 1PA rates in few/trajectory/ode/SKequatorialfluxes.py?

Leading terms read off SKequatorialfluxes.py, with s = 5 a - 3 chi2:
    pdot1PA    -> (28/15) s (1-e^2)^1.5 (8 + 7 e^2) / p^3.5           (v^7 term)
    edot1PA    -> (4/45)  s e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5     (v^9 term)
    OmegaPhi1PA, Omegar1PA -> -(1/2) s (1-e^2)^1.5 / p^2               (v^4 term, identical for both)
Deviation (flux.py):
    pdot  += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot  += q C_e e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5
    Omega_phi, Omega_r += -(15/56) q C_p (1-e^2)^1.5 / p^2
So the deviation is the leading 1PA term exactly when C_p = (28/15) s, C_e = (4/45) s, and then
-(15/56) C_p = -(1/2) s.

Checks
    1. the leading coefficients above, numerically: the SK function / its deviation shape at p -> infinity
    2. the coded frequency term at C_p = C_p^LO equals the leading OmegaPhi1PA, Omegar1PA term
    3. how much of the full 1PA rate the leading term carries along idx0 / idx4 p values
Run:  python consistency_check/check_lo_terms.py > consistency_check/logs/check_lo_terms.log
"""
import sys
from pathlib import Path

import numpy as np
from few.utils.geodesic import get_separatrix
from few.trajectory.ode.SKequatorialfluxes import OmegaPhi1PA, Omegar1PA, edot1PA, pdot1PA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                   # noqa: E402
from physical_c import physical_c, pn_shapes          # noqa: E402

CASES = [(a, chi2, e) for a in (-0.9, 0.0, 0.9) for chi2 in (0.0, 0.95) for e in (0.1, 0.4)]
P_FAR = 1e8                                           # next term is down by v = 1e-4 relative


# --- leading-order C ----------------------------------------------------------
def spin_combo(a, chi2):
    return 5 * a - 3 * chi2


def c_lo(a, chi2):
    """C_p, C_e for which the deviation is the leading 1PA term."""
    s = spin_combo(a, chi2)
    return 28 / 15 * s, 4 / 45 * s


def freq_shape(p, e):
    return (1 - e * e) ** 1.5 / p ** 2


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    return ok


# --- checks ------------------------------------------------------------------
def check_limits(a, chi2, e):
    """SK 1PA / deviation shape at large p against the read-off leading coefficients."""
    cp, ce = c_lo(a, chi2)
    sp, se = pn_shapes(P_FAR, e)
    fs = freq_shape(P_FAR, e)
    got = dict(C_p=pdot1PA(a, P_FAR, e, chi2) / sp, C_e=edot1PA(a, P_FAR, e, chi2) / se,
               Omega_phi=OmegaPhi1PA(a, P_FAR, e, chi2) / fs, Omega_r=Omegar1PA(a, P_FAR, e, chi2) / fs)
    want = dict(C_p=cp, C_e=ce, Omega_phi=-(15 / 56) * cp, Omega_r=-(15 / 56) * cp)
    tag = f"a={a:+.1f} chi2={chi2:.2f} e={e:.1f}"
    if spin_combo(a, chi2) == 0:          # leading term vanishes; next order dominates, nothing to compare
        print(f"[SKIP] {tag}: 5a - 3chi2 = 0")
        return []
    return [report(f"{tag}: {k:9s} limit {got[k]:+.6f}  expected {want[k]:+.6f}",
                   np.isclose(got[k], want[k], rtol=1e-3)) for k in want]


def leading_share(point):
    """Ratio leading term / full 1PA term along p in [p_plunge, p0] at the injected a, e0, chi2 of a point."""
    row = C.signal_row(point)
    a, p0, e, chi2 = (float(row[C.COL[n]]) for n in ("a", "p0", "e0", "chi2"))
    cp, ce = c_lo(a, chi2)
    pc = physical_c(point)
    print(f"\n{C.point_label(point)}: a={a} p0={p0:.4f} e0={e} chi2={chi2}  5a-3chi2={spin_combo(a, chi2):+.4f}")
    print(f"  C^LO       : C_p={cp:+.5f}  C_e={ce:+.5f}  (C_e/C_p = 1/21)")
    print(f"  C projected: C_p={pc['C_p']:+.5f}  C_e={pc['C_e']:+.5f}  (full 1PA at p0, physical_c.py)")
    print("    p     | LO/full pdot | LO/full edot | LO/full Omega_phi | LO/full Omega_r")
    for p in np.linspace(p0, get_separatrix(a, e, 1.0) + 0.1, 6):   # injection down to the plunge
        sp, se = pn_shapes(p, e)
        lo_f = -(15 / 56) * cp * freq_shape(p, e)
        print(f"  {p:7.3f} | {cp * sp / pdot1PA(a, p, e, chi2):12.4f} | {ce * se / edot1PA(a, p, e, chi2):12.4f} | "
              f"{lo_f / OmegaPhi1PA(a, p, e, chi2):17.4f} | {lo_f / Omegar1PA(a, p, e, chi2):15.4f}")


def main():
    results = [ok for case in CASES for ok in check_limits(*case)]
    print(f"\n{sum(results)} of {len(results)} leading-order checks pass")
    for pt in C.POINTS:
        leading_share(pt)


if __name__ == "__main__":
    main()
