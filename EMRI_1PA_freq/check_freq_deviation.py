"""CPU checks of the PN + frequency deviation in SuperKludgeFlux.modify_rhs (SuperKludge_r working tree).

    1. the Omega_phi, Omega_r shift equals -(15/56) q C_p (1-e^2)^1.5 / p^2, and is the same for both
    2. C_e changes edot and leaves both frequencies alone
    3. C_p = C_e = 0 with the deviation on is bit-identical to the deviation off (GR)
    4. at the physical C_p (physical_c.py) the shift is compared with the exact 1PA OmegaPhi1PA, Omegar1PA
Run:  python check_freq_deviation.py > logs/check_freq_deviation.log
"""
import numpy as np
from few.trajectory.ode.flux import SuperKludgeFlux
from few.trajectory.ode.SKequatorialfluxes import OmegaPhi1PA, Omegar1PA

import config as C
from physical_c import physical_c

M1, M2 = 1e6, 10.0
Q = M1 * M2 / (M1 + M2) ** 2


def rhs(flux, a, p, e, chi2, dev):
    """modify_rhs output at (p, e) for a fixed dummy vacuum rhs; dev = None switches the deviation off."""
    on = dev is not None
    dev = dev or {}
    flux.add_fixed_parameters(M1, M2, a, additional_args=[chi2, 0, 0, 0, on, dev.get("C_p", 0.0), dev.get("C_e", 0.0)])
    ydot = np.array([-1e-6, -1e-7, 0.0, 0.03, 0.03, 0.02, 0.0, 0.0])
    flux.modify_rhs(ydot, np.array([p, e, 1.0, 0, 0, 0, 0.0, 0.0]))
    return ydot


def freq_unit(p, e):
    return Q * (1 - e * e) ** 1.5 / p ** 2


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    return ok


def main():
    flux, results = SuperKludgeFlux(), []
    for a, p, e in ((-0.9, 13.0, 0.1), (0.9, 9.0, 0.1), (0.5, 12.0, 0.3)):
        base = rhs(flux, a, p, e, 0.95, None)
        dp = rhs(flux, a, p, e, 0.95, {"C_p": 7.0})
        expect = -(15 / 56) * 7.0 * freq_unit(p, e)
        results.append(report(f"a={a} p={p} e={e}: dOmega_phi = formula",
                              np.isclose(dp[3] - base[3], expect, rtol=1e-9, atol=0)))   # shift ~1e-5 of the base: ~1e-12 is round-off
        results.append(report(f"a={a} p={p} e={e}: dOmega_r = dOmega_phi", dp[5] - base[5] == dp[3] - base[3]))
        de = rhs(flux, a, p, e, 0.95, {"C_e": 50.0})
        results.append(report(f"a={a} p={p} e={e}: C_e leaves Omega alone, changes edot",
                              de[3] == base[3] and de[5] == base[5] and de[1] != base[1]))
        zero = rhs(flux, a, p, e, 0.95, {"C_p": 0.0, "C_e": 0.0})
        results.append(report(f"a={a} p={p} e={e}: C = 0 is GR", np.array_equal(zero, base)))

    print("\nphysical seed vs exact 1PA frequency shift, units q (1-e^2)^1.5 / p^2")
    for pt in C.POINTS:
        row = C.signal_row(pt)
        a, p, e, chi2 = (float(row[C.COL[n]]) for n in ("a", "p0", "e0", "chi2"))
        c = physical_c(pt)
        f = (1 - e * e) ** 1.5 / p ** 2
        print(f"{C.point_label(pt)}: C_p={c['C_p']:.6g} C_e={c['C_e']:.6g}  coded dOmega={-(15 / 56) * c['C_p']:.4f}  "
              f"exact dOmega_phi={float(OmegaPhi1PA(a, p, e, chi2)) / f:.4f}  "
              f"dOmega_r={float(Omegar1PA(a, p, e, chi2)) / f:.4f}")
    print(f"\n{sum(results)} of {len(results)} checks pass")


if __name__ == "__main__":
    main()
