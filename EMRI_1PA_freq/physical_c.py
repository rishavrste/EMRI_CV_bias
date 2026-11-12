"""C_p, C_e implied by the exact 1PA rates: the projection of pdot1PA, edot1PA onto the deviation shapes,

    C_p = pdot1PA / [(1-e^2)^1.5 (8 + 7 e^2) / p^3.5],    C_e = edot1PA / [e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5],

at the injected (a, p0, e0, chi2) of a grid point. These are the values the deviation would take if it were
the 1PA term at the start of the inspiral.
Run:  python physical_c.py
"""
from few.trajectory.ode.SKequatorialfluxes import edot1PA, pdot1PA

import config as C


def pn_shapes(p, e):
    f = (1 - e * e) ** 1.5
    return f * (8 + 7 * e * e) / p ** 3.5, e * f * (304 + 121 * e * e) / p ** 4.5


def physical_c(point):
    """{C_p, C_e} from the exact 1PA rates at the injection of a point."""
    row = C.signal_row(point)
    a, p, e, chi2 = (float(row[C.COL[n]]) for n in ("a", "p0", "e0", "chi2"))
    shape_p, shape_e = pn_shapes(p, e)
    return dict(C_p=float(pdot1PA(a, p, e, chi2)) / shape_p, C_e=float(edot1PA(a, p, e, chi2)) / shape_e)


if __name__ == "__main__":
    for pt in C.POINTS:
        row = C.signal_row(pt)
        print(C.point_label(pt), {n: float(row[C.COL[n]]) for n in ("a", "p0", "e0", "chi2")}, physical_c(pt))
