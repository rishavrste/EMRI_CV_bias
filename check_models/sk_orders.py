"""The PN terms of each SuperKludge 1PA function, and fluxes that add chosen terms to 0PA.

The terms are taken exactly as written in few/trajectory/ode/SKequatorialfluxes.py (sk_pn_terms.py splits each
function's return expression into its four addends):
    pdot1PA = v^7 + v^8 + v^9 + v^10 terms,   edot1PA = v^9 ... v^12,   OmegaPhi1PA, Omegar1PA = v^4 ... v^7
Order i = 0 is the leading 1PA term (the PN_freq / A / B shapes); i = 1, 2, 3 are +0.5PN, +1PN, +1.5PN beyond it.
(An earlier version recovered the same terms from a 4x4 Vandermonde solve; they agree to 8e-13.)
"""
import numpy as np
from few.trajectory.ode.flux import SuperKludgeFlux
from few.trajectory.ode.SKequatorialfluxes import OmegaPhi1PA, Omegar1PA, edot1PA, pdot1PA

import sk_pn_terms

CHANNELS = {"pdot": (pdot1PA, 7), "edot": (edot1PA, 9), "Omega_phi": (OmegaPhi1PA, 4), "Omega_r": (Omegar1PA, 4)}
ORDERS = (0, 1, 2, 3)
ORDER_NAMES = sk_pn_terms.ORDER_NAMES


def order_terms(channel, a, p, e, chi2):
    """(4, ...) array: the four PN terms of a 1PA function as written, by increasing power of v."""
    return sk_pn_terms.pn_terms(channel, a, p, e, chi2)


def self_check(a, p, e, chi2):
    """The four written terms add up to the full 1PA function in every channel."""
    sk_pn_terms.self_check(a, p, e, chi2)


def make_order_flux(orders):
    """SuperKludgeFlux class that adds to its rhs only the given 1PA orders (all four channels).
    Use with 0PA flags: the class adds q * sum_{i in orders} term_i itself."""
    orders = tuple(orders)

    class OrderFlux(SuperKludgeFlux):
        def modify_rhs(self, ydot, y, **kwargs):
            super().modify_rhs(ydot, y, **kwargs)
            p, e = float(y[0]), float(y[1])
            a = self.a + y[-1]
            for slot, ch in zip((0, 1, 3, 5), CHANNELS):
                ydot[slot] += self.massratio * order_terms(ch, a, p, e, self.chi2)[list(orders)].sum()

    OrderFlux.__name__ = "SuperKludgeFlux"     # EMRIInspiral keys its SuperKludge set-up on this name
    return OrderFlux
