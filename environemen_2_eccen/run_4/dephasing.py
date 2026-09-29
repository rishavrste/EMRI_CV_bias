"""Azimuthal and radial dephasing, Delta Phi_phi and Delta Phi_r.

Every quantity is (template) - (environmental signal), both trajectories run over the
same observation time T and compared on one common time grid.  Two variants:

    orbital   the phase accumulated by the inspiral alone, both starting from zero.
              This is what the disk (or a mis-set template) does to the orbit.
    total     orbital + the initial phases Phi_phi0, Phi_r0.  This is the phase the
              waveform actually carries, so it is what the fit is trying to zero out:
              a fitted Phi_phi0 offset can cancel an orbital dephasing.

Only trajectories are integrated, so this runs on the CPU in seconds.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqAccFlux, SuperKludgeFlux
from few.utils.constants import YRSID_SI

# The trajectory flux behind each template family in env_cv.SPECS.
FLUX_OF = {"vacuum": KerrEccEqAccFlux, "pn": SuperKludgeFlux}


def trajectory(flux, params, extra_args, T):
    """(t, Phi_phi, Phi_r) of one inspiral, orbital phases starting from zero."""
    out = EMRIInspiral(func=flux)(*[params[n] for n in ("m1", "m2", "a", "p0", "e0", "xI0")],
                                  *extra_args, T=T)
    return out[0], out[4], out[6]


def signal_trajectory(cfg, T):
    return trajectory(KerrEccEqAccFlux, cfg.injection,
                      [cfg.disk["Sigma0"], cfg.disk["h0"], cfg.disk["Sigma_p"]], T)


def template_trajectory(cfg, template, names, theta, tail, T):
    params = dict(cfg.injection)
    params.update({n: v for n, v in zip(names, theta) if n in params})
    return trajectory(FLUX_OF[template], params, list(tail(theta)), T), params


def dephasing(sig, tmpl, phase0_sig, phase0_tmpl, n_grid=4000):
    """Delta Phi(t) = template - signal on a common grid; final and max |.| values."""
    (ts, phs, rs), (tt, pht, rt) = sig, tmpl
    t = np.linspace(0.0, min(ts[-1], tt[-1]), n_grid)
    out = dict(t_common_yr=float(t[-1] / YRSID_SI))
    for key, a_sig, a_tmpl, i0 in (("Phi_phi", phs, pht, 0), ("Phi_r", rs, rt, 1)):
        orbital = CubicSpline(tt, a_tmpl)(t) - CubicSpline(ts, a_sig)(t)
        total = orbital + (phase0_tmpl[i0] - phase0_sig[i0])
        out[key] = dict(orbital_final=float(orbital[-1]),
                        orbital_max_abs=float(np.abs(orbital).max()),
                        total_final=float(total[-1]),
                        total_max_abs=float(np.abs(total).max()))
    return out


def dephasing_at(cfg, template, names, theta, tail, T):
    """Dephasing of the template at theta against the environmental signal."""
    sig = signal_trajectory(cfg, T)
    tmpl, params = template_trajectory(cfg, template, names, theta, tail, T)
    inj = cfg.injection
    return dephasing(sig, tmpl, (inj["Phi_phi0"], inj["Phi_r0"]),
                     (params["Phi_phi0"], params["Phi_r0"]))


def format_dephasing(label, d):
    return (f"[DEPH] {label:<14} dPhi_phi = {d['Phi_phi']['total_final']:+.6e} rad "
            f"(orbital {d['Phi_phi']['orbital_final']:+.6e}),  "
            f"dPhi_r = {d['Phi_r']['total_final']:+.6e} rad "
            f"(orbital {d['Phi_r']['orbital_final']:+.6e})  over {d['t_common_yr']:.6f} yr")
