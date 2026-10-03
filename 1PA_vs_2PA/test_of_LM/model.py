"""Signal, template, inner product and SEF derivatives for one grid point.

    build_grid(xp, use_gpu)        -> response, inner product and SEF shared by the grid
    build_point(Gd, sig_row)       -> make_h, chi2r, derivs for one point (2PA signal, C.TEMPLATE_PA template)
    evaluate(P, vec)               -> overlap, chi2 = <s-h|s-h>, lnL, rho_s, rho_h
"""
import numpy as np

import config as C
from config import (ARGS14, COL, DER_ORDER, DROP_DC, GEN, NDELTA, PARAMS, SIGNAL_FLAGS,
                    ovl)


# --- grid ------------------------------------------------------------------
def grid_T_dt():
    sig = ovl.signal_array(C.GRID)
    T, dt = float(sig[0, COL["T"]]), float(sig[0, COL["dt"]])
    assert np.all(sig[:, COL["T"]] == T) and np.all(sig[:, COL["dt"]] == dt)
    return T, dt


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
                       tdi="2nd generation", tdi_chan=GEN["tdi_chan"])
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


def build_grid(xp, use_gpu, T=None):
    """Response, inner product and SEF; T and dt are common to the 25 rows. T (years) overrides
    the stored observation time; the signal is then the same source observed for T."""
    T_grid, dt = grid_T_dt()
    T = T_grid if T is None else T
    Gd = dict(T=T, dt=dt, xp=xp, resp=ovl.build_response(T, dt, use_gpu, GEN),
              inner=ovl.make_inner(xp, dt, DROP_DC), sef=build_sef(T, dt, use_gpu))
    print(f"[{C.GRID}] T={T} yr dt={dt} s use_gpu={use_gpu} gen={GEN}", flush=True)
    return Gd


# --- one point -------------------------------------------------------------
def waveform_AE(Gd, p14, chi2, flags):
    resp, xp = Gd["resp"], Gd["xp"]
    out = resp(*[p14[n] for n in ARGS14], chi2, *flags, T=Gd["T"], dt=Gd["dt"])
    return [xp.asarray(c) for c in out[:2]]


def build_point(Gd, sig_row, names=PARAMS, dist_div=1.0):
    """Model functions at one grid point: 2PA signal from sig_row, template (C.TEMPLATE_PA)
    fitted in names.

    If "chi2" is not in names the secondary spin is held at its injected value. dist_div divides
    the distance of signal and template alike: SNR x dist_div, chi2 x dist_div^2, overlap unchanged.
    """
    base = {n: float(sig_row[COL[n]]) for n in ARGS14}
    base["dist"] /= dist_div
    chi2_inj = float(sig_row[COL["chi2"]])
    s = waveform_AE(Gd, base, chi2_inj, SIGNAL_FLAGS)
    flags = C.template_flags()
    inner, sef, xp = Gd["inner"], Gd["sef"], Gd["xp"]

    def split(vec):
        p = dict(base)
        p.update({n: float(v) for n, v in zip(names, vec) if n != "chi2"})
        return p, (float(vec[names.index("chi2")]) if "chi2" in names else chi2_inj)

    def make_h(vec):
        p, chi2 = split(vec)
        return waveform_AE(Gd, p, chi2, flags)

    def residual(h):
        return [a - b for a, b in zip(s, h)]

    def chi2r(vec):
        try:
            r = residual(make_h(vec))
            return inner(r, r)
        except Exception:                                   # noqa: BLE001  failed waveform
            return 1e30

    def derivs(vec, deltas):
        """d_i h at vec from SEF, as a list of [A, E] arrays, and the steps SEF used."""
        p, chi2 = split(vec)
        wp = {("xI0" if n == "x0" else n): v for n, v in p.items()}
        apa = {"chi2": chi2, "1PA": flags[0], "evolve_primary": flags[1], "2PA": flags[2]}
        dH, _ = sef(wave_params=wp, param_names=names, add_param_args=apa, deltas=deltas,
                    live_dangerously=False, stability_plot=False, der_order=DER_ORDER,
                    Ndelta=(NDELTA if deltas is None else None), return_derivatives=True,
                    plunge_check=False, filename=None)
        return [[xp.asarray(dH[j][0]), xp.asarray(dH[j][1])] for j in range(len(names))], sef.deltas

    return dict(names=list(names), s=s, make_h=make_h, residual=residual, chi2r=chi2r,
                derivs=derivs, inner=inner)


def evaluate(P, vec):
    """Overlap, chi2 = <s-h|s-h> and lnL at one point, all at the injected distance."""
    inner, s = P["inner"], P["s"]
    h = P["make_h"](vec)
    ss, hh, sh = inner(s, s), inner(h, h), inner(s, h)
    r = P["residual"](h)
    return dict(overlap=float(sh / np.sqrt(ss * hh)), chi2=float(inner(r, r)),
                lnL=float(sh - 0.5 * hh), rho_s=float(np.sqrt(ss)), rho_h=float(np.sqrt(hh)))
