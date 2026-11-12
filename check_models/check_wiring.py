"""CPU checks of the model-switch deviation in SuperKludgeFlux (SuperKludge_r @ PN_freq_evol, working tree).

additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, model, c1, ..., c8]
    1. model 0 adds the PN_freq terms: q C_p, q C_e on the PN shapes, -(15/56) q C_p on Omega_phi and Omega_r
    2. model 1 with C_Om = -(15/56) C_p equals model 0
    3. model 2 with C_Omphi = C_Omr = C_Om equals model 1
    4. model 3 with n = 0 equals model 2; with n != 0 each term is multiplied by (10/p)^n
    5. all c = 0, or deviation off, is GR (bit-identical)
Run:  python check_wiring.py > logs/check_wiring.log
"""
import numpy as np
from few.trajectory.ode.flux import SuperKludgeFlux

from families import branch_check

M1, M2, CHI2 = 1e6, 10.0, 0.95
Q = M1 * M2 / (M1 + M2) ** 2
POINTS = ((-0.9, 13.0, 0.1), (0.9, 9.0, 0.1), (0.5, 12.0, 0.3))


def rhs(a, p, e, model=None, c=()):
    """modify_rhs output at (p, e) on a fixed dummy vacuum rhs; model = None switches the deviation off."""
    flux = SuperKludgeFlux()
    on = model is not None
    c = list(c) + [0.0] * (8 - len(c))
    flux.add_fixed_parameters(M1, M2, a, additional_args=[CHI2, 0, 0, 0, on, model or 0, *c])
    ydot = np.array([-1e-6, -1e-7, 0.0, 0.03, 0.03, 0.02, 0.0, 0.0])
    flux.modify_rhs(ydot, np.array([p, e, 1.0, 0, 0, 0, 0.0, 0.0]))
    return ydot


def report(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    return ok


def checks_at(a, p, e):
    tag = f"a={a:+.1f} p={p} e={e}"
    f = (1 - e * e) ** 1.5
    sp, se, so = Q * f * (8 + 7 * e * e) / p ** 3.5, Q * e * f * (304 + 121 * e * e) / p ** 4.5, Q * f / p ** 2
    base = rhs(a, p, e)
    cp, ce, cphi, cr, n = -13.7, 6.8, 2.1, -1.3, (0.7, -1.1, 1.9, -0.4)
    m0 = rhs(a, p, e, 0, (cp, ce))
    want0 = np.array([cp * sp, ce * se, -(15 / 56) * cp * so, -(15 / 56) * cp * so])
    got0 = (m0 - base)[[0, 1, 3, 5]]
    m1 = rhs(a, p, e, 1, (cp, ce, -(15 / 56) * cp))
    m2 = rhs(a, p, e, 2, (cp, ce, cphi, cr))
    m3_0 = rhs(a, p, e, 3, (cp, ce, cphi, cr, 0, 0, 0, 0))
    m3 = rhs(a, p, e, 3, (cp, ce, cphi, cr, *n))
    want3 = np.array([cp * sp, ce * se, cphi * so, cr * so]) * (10 / p) ** np.array(n)
    return [report(f"{tag}: model 0 = PN_freq formula", np.allclose(got0, want0, rtol=1e-9, atol=0)),
            report(f"{tag}: model 1 at C_Om = -(15/56) C_p = model 0", np.allclose(m1, m0, rtol=1e-14, atol=0)),
            report(f"{tag}: model 2 at C_Omphi = C_Omr = model 1",
                   np.allclose(rhs(a, p, e, 2, (cp, ce, -(15 / 56) * cp, -(15 / 56) * cp)), m1, rtol=1e-14, atol=0)),
            report(f"{tag}: model 3 at n = 0 = model 2", np.array_equal(m3_0, m2)),
            report(f"{tag}: model 3 = (10/p)^n times each term",
                   np.allclose((m3 - base)[[0, 1, 3, 5]], want3, rtol=1e-9, atol=0)),
            report(f"{tag}: c = 0 is GR for every model",
                   all(np.array_equal(rhs(a, p, e, m, ()), base) for m in range(4)))]


def main():
    branch_check()
    results = [ok for pt in POINTS for ok in checks_at(*pt)]
    print(f"\n{sum(results)} of {len(results)} checks pass")


if __name__ == "__main__":
    main()
