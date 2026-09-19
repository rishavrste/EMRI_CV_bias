"""Step 1 of the environmental EMRI study: build and freeze the 25-source population.

This script does NOT run any Fisher or any CV climb.  It only produces the population that
step 2 will consume, so that the expensive run always starts from a fixed, inspectable set
of sources.

What is sampled, and what is held fixed
---------------------------------------
Sampled per source:
    m1        ~ U(1e6, 1e7)
    T_plunge  ~ U(2, 4) yr, and p0 is then solved so the inspiral plunges at that time
    dist        rescaled after the fact so the environmental injection has SNR = 200
Everything else is fixed
    m2 = 50, a = 0.9, e0 = 0.0, xI0 = 1, qS = 0.2, phiS = 0.2, qK = 0.8, phiK = 0.8,
    Phi_phi0 = 0.3, Phi_theta0 = 0.5, Phi_r0 = 0.5

The population is CIRCULAR (e0 = 0), which is what the environmental deviation as
implemented is valid for: it is a torque on Ldot, and on a circular orbit L = L_circ(p)
fixes both Ldot = (dL/dp) pdot and Edot = Omega_phi Ldot, so the deviation reduces to
pdot -> env * pdot.  See the SuperKludgeFlux class docstring.  At finite e the deviation
has to be applied to Edot and Ldot before the Jacobian instead, which this population does
not do -- so e0 = 0 is a requirement here, not a simplification.

disclaimer: Environmental model selected here:
    A_PM = 1.92e-5 * (m1 / 1e6)     n_PM = 8      A_GC = 1e-12      n_GC = 4
A_PM is m1-dependent, so it is recomputed per source: over m1 in [1e6, 1e7] it spans
1.92e-5 to 1.92e-4, a factor of 10 in the size of the deviation across the population.

disclaimer: A_GC is carried in the injection but is NOT freed in the step-2 Fisher: at
p ~ 12 a coefficient of 1e-12 produces a fractional flux change of ~1e-8, a dead derivative
that makes the matrix singular.

p0 is solved with the *environmental* trajectory, so "plunges at T_plunge" refers to the
signal that is actually injected, not to its GR counterpart. As we consider that as the true injected model of this study

Cadence
-------
dt = 10.  The SNRs are printed side by side.  If T carries a non-negligible share 
of SNR^2 at dt = 10 but not at dt = 5, dt = 10 is aliasing and DT must be lowered before step 2 is run.
If both contain significant power, we will just ignore the T channel for our analysis, but it is a useful check on the waveform sampling.
For now, just using 2 channels to make life easier
Output
------
environmental_tests/population.json -- 25 entries, each carrying the full 14 source
parameters (with the rescaled dist), the environmental tail, T, T_plunge and the SNR
achieved. 
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SuperKludgeFlux
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix
from few.utils.constants import YRSID_SI

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    xp, use_gpu = cp, True
except Exception as exc:
    xp, use_gpu = np, False
    print(f"[INFO] No usable GPU ({type(exc).__name__}), falling back to NumPy on CPU.")

print(f"[INFO] use_gpu = {use_gpu}")


# --- controls --------------------------------------------------------------
N_SOURCES = 25   #-> no. of sources to sample and freeze for the step-2 Fisher run.

# SEED = 42  # for reproducibility, but the population is not meant to be representative of any astrophysical distribution
SEED = 43
#just used shubham's paper for that
M1_MIN, M1_MAX = 1e6, 1e7
T_PLUNGE_MIN, T_PLUNGE_MAX = 2.0, 4.0      # years
TARGET_SNR = 200.0   #using fixed SNR for all sources

DT = 10.0
T_SAFETY = 0.99                            # same margin as test.ipynb
NCHANNELS = 2                              # A, E for the production SNR
F_MIN, F_MAX = 1e-5, 1e-1      #lisa senstivity band, used to compute the SNR and the PSD

MAX_DRAWS = 200                            # give up rather than loop forever
DT_CHECK = 5.0                             # cadence to cross-check the dt = 10 choice against

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "population.json")

param_names_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]

# the test.ipynb source, everything except m1, p0 and dist
FIXED = dict(m2=50.0, a=0.9, e0=0.0, xI0=1.0,
             qS=0.2, phiS=0.2, qK=0.8, phiK=0.8,
             Phi_phi0=0.3, Phi_theta0=0.5, Phi_r0=0.5)
DIST_REF = 1.5                             # distance the waveform is generated at before rescaling

# SuperKludge tail, everything off except the environmental block
CHI2 = 0.0
EVOLVE_1PA = EVOLVE_PRIMARY = EVOLVE_2PA = False
DEVIATION_INCLUDED = False
DEV_PN_P = DEV_PN_E = 0.0
N_PM = 8.0
A_GC = 1e-12
N_GC = 4.0
# A_GC = 0
# N_GC = 0

channels = [A1TDISens, E1TDISens, T1TDISens]


def a_pm_of(m1):
    """A_PM = 1.92e-5 * M_6, shubham's paper"""
    return 1.92e-5 * (m1 / 1e6)


def env_tail(m1, environmental_included):
    """The 12 extra SuperKludgeFlux arguments for this case"""
    return [CHI2, EVOLVE_1PA, EVOLVE_PRIMARY, EVOLVE_2PA,
            DEVIATION_INCLUDED, DEV_PN_P, DEV_PN_E,
            environmental_included, a_pm_of(m1), N_PM, A_GC, N_GC]


# --- p0 from a target plunge time -----------------------------------------
def p0_for_plunge(m1, T_target):
    """p0 such that the environmental inspiral reaches the separatrix at T_target years."""
    traj = EMRIInspiral(func=SuperKludgeFlux)
    add_args = env_tail(m1, True)
    return float(get_p_at_t(traj, T_target,
                            [m1, FIXED["m2"], FIXED["a"], FIXED["e0"], FIXED["xI0"]] + add_args,
                            bounds=None))


def plunge_time(m1, p0):
    """Evolve and report where the integrator actually stopped -- the check on p0_for_plunge."""
    traj = EMRIInspiral(func=SuperKludgeFlux)
    out = traj(m1, FIXED["m2"], FIXED["a"], p0, FIXED["e0"], FIXED["xI0"],
               *env_tail(m1, True), T=T_PLUNGE_MAX + 2.0)
    t, p, e = out[0], out[1], out[2]
    sep = float(get_separatrix(FIXED["a"], e[-1], FIXED["xI0"]))
    return dict(t_plunge_yr=float(t[-1]) / YRSID_SI,
                p_final=float(p[-1]), e_final=float(e[-1]), separatrix=sep,
                dp_to_separatrix=float(p[-1]) - sep)


# --- waveform + response ---------------------------------------------------
def build_response(T, dt, tdi_chan):
    waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform,
                                          sum_kwargs=dict(pad_output=True, odd_len=True),
                                          return_list=False, use_gpu=use_gpu)
    return ResponseWrapper(
        waveform_gen=waveform_model, Tobs=T, t0=10000.0, dt=dt,
        index_lambda=8, index_beta=7, flip_hx=True,
        is_ecliptic_latitude=False, remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend="cuda12x" if use_gpu else "cpu",
        order=20, tdi="1st generation", tdi_chan=tdi_chan)


def make_freq_mask(n, dt):
    """Boolean mask over rfftfreq with the DC bin dropped and other bins outside [F_MIN, F_MAX] zeroed out."""
    f = xp.fft.rfftfreq(n, dt)
    return ((f > F_MIN) & (f < F_MAX))[1:]


def snr_of(m1, p0, dist, T, dt, nchannels):
    """SNR of the environmental injection, and the per-channel breakdown."""
    tdi_chan = {2: "AE", 3: "AET"}[nchannels]
    response = build_response(T=T, dt=dt, tdi_chan=tdi_chan)

    src = dict(FIXED)
    src.update(m1=m1, p0=p0, dist=dist)
    args = [src[n] for n in param_names_14] + env_tail(m1, True)
    h = xp.array(response(*args))[0:nchannels, :]

    noise_kwargs = [{"sens_fn": ch} for ch in channels[:nchannels]]
    PSD = xp.array(generate_PSD(waveform=h, dt=dt, noise_PSD=get_sensitivity,
                                channels=channels[:nchannels],
                                noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    mask = make_freq_mask(len(h[0]), dt)

    def _ip(a, b, P):
        """ using the inner_product from stableemrifisher.utils.utility """
        v = inner_product(a, b, P, dt, freq_mask=mask, use_gpu=use_gpu)
        return float(v.get() if hasattr(v, "get") else v)

    total = np.sqrt(_ip(h, h, PSD))
    per_chan = [np.sqrt(_ip(h[i:i + 1], h[i:i + 1], PSD[i:i + 1, :]))
                for i in range(nchannels)]
    n_samples = int(len(h[0]))
    del h, PSD, mask
    if use_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    return total, per_chan, n_samples


# --- sample ----------------------------------------------------------------
def sample_population():
    rng = np.random.default_rng(SEED)
    sources, draws, rejected = [], 0, []

    while len(sources) < N_SOURCES and draws < MAX_DRAWS:
        draws += 1
        m1 = float(rng.uniform(M1_MIN, M1_MAX))
        T_target = float(rng.uniform(T_PLUNGE_MIN, T_PLUNGE_MAX))

        try:
            p0 = p0_for_plunge(m1, T_target)
            if not np.isfinite(p0):
                raise ValueError(f"get_p_at_t returned {p0}")
            chk = plunge_time(m1, p0)
        except Exception as exc:
            rejected.append(dict(draw=draws, m1=m1, T_target=T_target,
                                 reason=f"{type(exc).__name__}: {exc}"))
            print(f"[REJECT] draw {draws}: m1 = {m1:.4e}, T_target = {T_target:.3f} -> {exc}")
            continue

        # the solved p0 must actually reproduce the requested plunge time
        if abs(chk["t_plunge_yr"] - T_target) > 0.05:
            rejected.append(dict(draw=draws, m1=m1, T_target=T_target,
                                 reason=f"plunge {chk['t_plunge_yr']:.4f} != target {T_target:.4f}"))
            print(f"[REJECT] draw {draws}: m1 = {m1:.4e}, wanted {T_target:.3f} yr, "
                  f"got {chk['t_plunge_yr']:.3f} yr")
            continue

        idx = len(sources)
        T = T_SAFETY * chk["t_plunge_yr"]

        t0 = time.time()
        snr_ref, per_chan, n_samples = snr_of(m1, p0, DIST_REF, T, DT, NCHANNELS)
        # SNR is inversely proportional to distance, so one waveform fixes the scaling exactly.
        dist = DIST_REF * snr_ref / TARGET_SNR

        src = dict(FIXED)
        src.update(m1=m1, p0=p0, dist=dist)

        sources.append(dict(
            idx=idx, m1=m1, p0=p0, dist=dist, T=T,
            T_target_yr=T_target, t_plunge_yr=chk["t_plunge_yr"],
            p_final=chk["p_final"], e_final=chk["e_final"],
            separatrix=chk["separatrix"], dp_to_separatrix=chk["dp_to_separatrix"],
            A_PM=a_pm_of(m1), n_PM=N_PM, A_GC=A_GC, n_GC=N_GC,
            source_params={n: src[n] for n in param_names_14},
            snr_at_dist_ref=snr_ref, dist_ref=DIST_REF, target_snr=TARGET_SNR,
            snr_per_channel_at_dist_ref=per_chan,
            dt=DT, nchannels=NCHANNELS, n_samples=n_samples))

        print(f"[SRC {idx:2d}] m1 = {m1:.6e}  p0 = {p0:.6f}  t_plunge = {chk['t_plunge_yr']:.4f} yr  "
              f"T = {T:.4f} yr  N = {n_samples:,d}  SNR(d={DIST_REF}) = {snr_ref:.4f}  "
              f"-> dist = {dist:.6f} Gpc   [{time.time() - t0:.1f} s]")

        write_output(sources, rejected, draws, dt_check=None)

    if len(sources) < N_SOURCES:
        raise RuntimeError(f"only {len(sources)} sources after {draws} draws; widen the bounds")
    return sources, rejected, draws


# --- cadence cross-check ---------------------------------------------------
def dt_check(sources):
    """Re-score the lightest and heaviest source at DT and DT_CHECK, with A, E and T.

    T is a null channel for a signal that is properly sampled, so a large T share at DT but
    not at DT_CHECK is aliasing and means DT is too coarse.
    """
    picks = [min(sources, key=lambda s: s["m1"]), max(sources, key=lambda s: s["m1"])]
    rows = []
    for s in picks:
        for dt in (DT, DT_CHECK):
            total, per_chan, n = snr_of(s["m1"], s["p0"], s["dist"], s["T"], dt, 3)
            share_T = 100.0 * (per_chan[2] / total) ** 2
            rows.append(dict(idx=s["idx"], m1=s["m1"], dt=dt, n_samples=n,
                             snr_total=total, snr_A=per_chan[0], snr_E=per_chan[1],
                             snr_T=per_chan[2], T_percent_of_snr2=share_T))
            print(f"[DTCHK] idx {s['idx']:2d}  m1 = {s['m1']:.4e}  dt = {dt:>4.1f}  "
                  f"N = {n:>12,d}  SNR = {total:>9.4f}  A = {per_chan[0]:>8.4f}  "
                  f"E = {per_chan[1]:>8.4f}  T = {per_chan[2]:>8.4f}  "
                  f"(T = {share_T:.2f}% of SNR^2)")
    return rows


def write_output(sources, rejected, draws, dt_check):
    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        seed=SEED, n_sources=len(sources), n_draws=draws,
        m1_range=[M1_MIN, M1_MAX], t_plunge_range=[T_PLUNGE_MIN, T_PLUNGE_MAX],
        target_snr=TARGET_SNR, dt=DT, T_safety=T_SAFETY, nchannels=NCHANNELS,
        freq_band=[F_MIN, F_MAX], fixed_params=FIXED,
        environmental=dict(A_PM_formula="1.92e-5 * (m1 / 1e6)", n_PM=N_PM,
                           A_GC=A_GC, n_GC=N_GC),
        sources=sources, rejected=rejected, dt_check=dt_check)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    t_start = time.time()
    print(f"[CONF] {N_SOURCES} sources, seed {SEED}, m1 ~ U({M1_MIN:.0e}, {M1_MAX:.0e}), "
          f"t_plunge ~ U({T_PLUNGE_MIN}, {T_PLUNGE_MAX}) yr, SNR -> {TARGET_SNR:g}")
    print(f"[CONF] dt = {DT:g} s, T = {T_SAFETY:g} x t_plunge, nchannels = {NCHANNELS}, "
          f"band = [{F_MIN:.1e}, {F_MAX:.1e}] Hz")

    sources, rejected, draws = sample_population()

    print(f"\n[POP] {len(sources)} sources from {draws} draws ({len(rejected)} rejected)")
    print(f"\n{'idx':>3} | {'m1':>12} | {'p0':>10} | {'t_plunge':>9} | {'T':>9} | "
          f"{'A_PM':>11} | {'dist':>10} | {'N samples':>12}")
    print("-" * 96)
    for s in sources:
        print(f"{s['idx']:>3} | {s['m1']:>12.5e} | {s['p0']:>10.5f} | {s['t_plunge_yr']:>9.4f} | "
              f"{s['T']:>9.4f} | {s['A_PM']:>11.5e} | {s['dist']:>10.5f} | {s['n_samples']:>12,d}")

    print(f"\n[DTCHK] cross-checking dt = {DT:g} against dt = {DT_CHECK:g}, with A, E and T")
    rows = dt_check(sources)
    write_output(sources, rejected, draws, rows)

    print(f"\n[DONE] written {OUT}  ({time.time() - t_start:.1f} s total)")
