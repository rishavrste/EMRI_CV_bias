"""Signal, template, inner product and SEF derivatives for one grid point.

    build_grid(xp, use_gpu, T, dt)   -> response, inner product and SEF for observation time T
    build_point(Gd, sig_row, names)  -> make_h, chi2r, derivs (1PA signal, 0PA [+ PN] template)
    evaluate(P, vec)                 -> overlap, chi2 = <s-h|s-h>, lnL, rho_s, rho_h
"""
import numpy as np

import config as C
from config import ARGS14, COL, DER_ORDER, DEV_SLOTS, GEN, NDELTA, ovl


# --- grid ------------------------------------------------------------------
def point_T_dt(sig_row):
    """Stored observation time [yr] and cadence [s] of a point."""
    return float(sig_row[COL["T"]]), float(sig_row[COL["dt"]])


def build_sef(T, dt, use_gpu):
    from few.waveform import GenerateEMRIWaveform
    from few.waveform.waveform import SuperKludgeWaveform
    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import get_sensitivity, A2TDISens, E2TDISens
    from stableemrifisher.fisher import StableEMRIFisher

    gen_kwargs = {k: GEN[k] for k in ("inspiral_kwargs", "sum_kwargs", "mode_selector_kwargs")
                  if GEN[k] is not None}
    resp_kwargs = dict(Tobs=T, dt=dt, index_lambda=8, index_beta=7, t0=10000.0,
                       flip_hx=True, is_ecliptic_latitude=False, remove_garbage="zero",
                       orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                       force_backend="cuda12x" if use_gpu else "cpu", order=20,
                       tdi="2nd generation", tdi_chan="AE")   # A, E only, as in waveform_AE
    channels = [A2TDISens, E2TDISens]
    return StableEMRIFisher(
        waveform_class=SuperKludgeWaveform, waveform_class_kwargs=gen_kwargs,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=resp_kwargs,
        noise_model=get_sensitivity, noise_kwargs=[{"sens_fn": c} for c in channels],
        channels=channels, T=T, dt=dt, stats_for_nerds=False, deriv_type="stable",
        use_gpu=use_gpu, stability_plot=False, der_order=DER_ORDER, Ndelta=NDELTA,
        return_derivatives=False)


def build_grid(xp, use_gpu, T, dt):
    """Response, inner product and SEF for observation time T [yr] and cadence dt [s]; a T below a
    point's stored one observes the same source for less time."""
    Gd = dict(T=T, dt=dt, xp=xp, resp=ovl.build_response(T, dt, use_gpu, GEN),
              inner=ovl.make_inner(xp, dt, C.DROP_DC), sef=build_sef(T, dt, use_gpu))
    print(f"[{C.GRID}] T={T} yr dt={dt} s drop_dc={C.DROP_DC} use_gpu={use_gpu} gen={GEN}", flush=True)
    return Gd


# --- one point -------------------------------------------------------------
def add_args(chi2, flags, dev):
    """The trailing waveform arguments: [chi2, evolve_1PA, evolve_primary, evolve_2PA,
    deviation_included, C_p, C_e, del_0_p, del_0_e]. dev = None switches the deviation off."""
    on = dev is not None
    return [chi2, *flags, on] + [float(dev[n]) if on else 0.0 for n in DEV_SLOTS]


def waveform_AE(Gd, p14, args):
    resp, xp = Gd["resp"], Gd["xp"]
    out = resp(*[p14[n] for n in ARGS14], *args, T=Gd["T"], dt=Gd["dt"])
    return [xp.asarray(c) for c in out[:2]]


def zero_step_range(names, vec):
    """C.ZERO_STEPS for every parameter with |value| < C.NEAR_ZERO[name]."""
    return {n: C.ZERO_STEPS for n, v in zip(names, vec)
            if n in C.NEAR_ZERO and abs(float(v)) < C.NEAR_ZERO[n]}


def build_point(Gd, sig_row, names, fixed=None):
    """Model functions at one grid point: 1PA signal from sig_row, 0PA template fitted in names.

    A deviation coefficient not in names stays at its value in fixed, else 0, so names without
    C_p, C_e and no fixed give the plain 0PA template. chi2 (secondary spin) is the injected value
    in signal and template alike.
    """
    base = {n: float(sig_row[COL[n]]) for n in ARGS14}
    held = dict(fixed or {})
    chi2 = float(sig_row[COL["chi2"]])
    s = waveform_AE(Gd, base, add_args(chi2, C.SIGNAL_FLAGS, None))
    inner, sef, xp = Gd["inner"], Gd["sef"], Gd["xp"]

    def split(vec):
        """(14 waveform parameters, deviation coefficients) at vec."""
        v = dict(zip(names, map(float, vec)))
        p = dict(base)
        p.update({n: x for n, x in v.items() if n in ARGS14})
        return p, {n: v.get(n, held.get(n, C.DEV_INJ[n])) for n in DEV_SLOTS}

    def make_h(vec):
        p, dev = split(vec)
        return waveform_AE(Gd, p, add_args(chi2, C.TEMPLATE_FLAGS, dev))

    def residual(h):
        return [a - b for a, b in zip(s, h)]

    def chi2r(vec):
        try:
            r = residual(make_h(vec))
            return inner(r, r)
        except Exception:                                   # noqa: BLE001  failed waveform
            return 1e30

    def derivs(vec, deltas, delta_range=None):
        """d_i h at vec from SEF, as a list of [A, E] arrays, and the steps SEF used. The
        deviation coefficients are add_param_args entries, which SEF differentiates by name.
        delta_range: optional {name: trial steps} for SEF's step search, in place of its default
        grid (relative to the parameter value); a parameter near zero (zero_step_range) gets the
        absolute at-zero grid unless delta_range names it."""
        if deltas is None:
            delta_range = {**zero_step_range(names, vec), **(delta_range or {})} or None
        p, dev = split(vec)
        wp = {("xI0" if n == "x0" else n): x for n, x in p.items()}
        f1, fp, f2 = C.TEMPLATE_FLAGS
        apa = {"chi2": chi2, "1PA": f1, "evolve_primary": fp, "2PA": f2,
               "deviation_included": True, **dev}               # dict order = additional_args order
        dH, _ = sef(wave_params=wp, param_names=names, add_param_args=apa, deltas=deltas,
                    live_dangerously=False, stability_plot=False, der_order=DER_ORDER,
                    Ndelta=(NDELTA if deltas is None else None), delta_range=delta_range,
                    return_derivatives=True,
                    plunge_check=False, filename=None)
        return [[xp.asarray(dH[j][0]), xp.asarray(dH[j][1])] for j in range(len(names))], sef.deltas

    return dict(names=list(names), s=s, make_h=make_h, residual=residual, chi2r=chi2r,
                derivs=derivs, inner=inner)


def evaluate(P, vec):
    """Overlap, chi2 = <s-h|s-h> and lnL at one point."""
    inner, s = P["inner"], P["s"]
    h = P["make_h"](vec)
    ss, hh, sh = inner(s, s), inner(h, h), inner(s, h)
    r = P["residual"](h)
    return dict(overlap=float(sh / np.sqrt(ss * hh)), chi2=float(inner(r, r)),
                lnL=float(sh - 0.5 * hh), rho_s=float(np.sqrt(ss)), rho_h=float(np.sqrt(hh)))
