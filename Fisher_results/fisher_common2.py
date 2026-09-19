"""Shared GPU machinery for computing Fisher matrices + injection SNR at fixed
(already-optimized) best-fit points parsed out of results_combined.txt /
results_compiled.txt by parse_results.py.

Adapted from src/IMRI/gauss_cv_imri_dt5_rerun.py's build_context()/fisher_derivs()
and src/EMRI's fisher_common.py (bias_inference_emri/new_results_EMRI), but
simplified to a single non-iterative Fisher evaluation (and one SNR evaluation
per point) at a point that is already the CV/Nelder-Mead optimum, instead of a
full Levenberg-Marquardt climb.

Deviation wiring (see src/IMRI/README.md and src/EMRI/README.md):
  - hybrid branch:
      PN:      add_param_args += {"dev_1": d1, "dev_2": d2, "del_0_p": 0.0, "del_0_e": 0.0}
      simple:  add_param_args += {"C_p": 0.0, "C_e": 0.0, "dev_1": d1, "dev_2": d2}
  - dev_a_pe branch:
      simple_pe: add_param_args += {"dev_1": d1, "dev_2": d2}
  - 0PA (either branch): deviation_included=False, no dev kwargs at all.

The two branches are mutually exclusive SuperKludge_r git checkouts, so this
module must be run once per branch (matching whichever branch is currently
`pip install -e`'d) -- see compute_fishers.py --branch and Fisher_results/README.md.
"""
import numpy as np

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import generate_PSD, inner_product

try:
    import cupy
    USE_GPU = True
except ImportError:
    cupy = None
    USE_GPU = False
    print("[fisher_common2] CuPy not found, using NumPy (CPU).")

PARAM_NAMES_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
CHANNELS = [A1TDISens, E1TDISens, T1TDISens]
F_MIN = 1e-5


def _wresp_kwargs(T, dt, use_gpu):
    return dict(Tobs=T, t0=10000.0, dt=dt, index_lambda=8, index_beta=7, flip_hx=True,
                is_ecliptic_latitude=False, remove_garbage="zero",
                orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                force_backend="cuda12x" if use_gpu else "cpu",
                order=20, tdi="1st generation", tdi_chan="AET")


def build_sef(T, dt, use_gpu=USE_GPU):
    noise_kwargs = [{"sens_fn": ch} for ch in CHANNELS]
    return StableEMRIFisher(
        waveform_class=SuperKludgeWaveform,
        waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=_wresp_kwargs(T, dt, use_gpu),
        stats_for_nerds=False, use_gpu=use_gpu,
        deriv_type="stable",
        noise_model=get_sensitivity,
        noise_kwargs=noise_kwargs,
        channels=CHANNELS,
        T=T, dt=dt,
        stability_plot=False,
        der_order=8, Ndelta=35,
        plunge_check=True, return_derivatives=False,
    )


def dev_add_kwargs(model, chi2_sec, dev_1, dev_2):
    apa = {"chi2": chi2_sec, "evolve_1PA": False, "evolve_primary": False,
           "evolve_2PA": False, "deviation_included": model != "0PA"}
    if model == "PN":
        apa.update({"dev_1": dev_1, "dev_2": dev_2, "del_0_p": 0.0, "del_0_e": 0.0})
    elif model == "simple":
        apa.update({"C_p": 0.0, "C_e": 0.0, "dev_1": dev_1, "dev_2": dev_2})
    elif model == "simple_pe":
        apa.update({"dev_1": dev_1, "dev_2": dev_2})
    elif model != "0PA":
        raise ValueError(f"Unknown model {model!r}")
    return apa


def _wave_params(case):
    sp = case["signal_param"]
    wp = {n: sp[n] for n in PARAM_NAMES_14}
    wp.update(dict(zip(case["param_names"], case["x_bf"])))
    return wp


def compute_fisher(case, use_gpu=USE_GPU):
    """Fisher matrix (len(param_names) x len(param_names)) at case['x_bf']."""
    sef = build_sef(T=case["T"], dt=case["dt"], use_gpu=use_gpu)
    wp = _wave_params(case)
    apa = dev_add_kwargs(case["model"], case["chi2_sec"], wp.get("dev_1", 0.0), wp.get("dev_2", 0.0))
    fisher = sef(
        wave_params={n: wp[n] for n in PARAM_NAMES_14},
        param_names=case["param_names"],
        add_param_args=apa,
        live_dangerously=False,
        stability_plot=False,
        der_order=8, Ndelta=35,
        return_derivatives=False,
    )
    return np.asarray(fisher, dtype=float)


def compute_injection_snr(case, use_gpu=USE_GPU):
    """SNR of the true 1PA-like injected signal (signal_param, dev=0), used as the
    reference SNR0 for the critical-SNR analysis -- same definition as
    ctx['snr'] in gauss_cv_imri_dt5_rerun.py: sqrt(<s|s>) of the highpass-clipped
    injection, evaluated with evolve_1PA=True, deviation off."""
    xp = cupy if use_gpu else np
    T, dt = case["T"], case["dt"]
    sp = case["signal_param"]
    channels = CHANNELS
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    wfm = GenerateEMRIWaveform(SuperKludgeWaveform,
                               sum_kwargs=dict(pad_output=True, odd_len=True),
                               return_list=False, use_gpu=use_gpu)
    wresp = ResponseWrapper(waveform_gen=wfm, **_wresp_kwargs(T, dt, use_gpu))

    p = {n: sp[n] for n in PARAM_NAMES_14}
    tail = [case["chi2_sec"], True, False, False, False]  # evolve_1PA=True, deviation off
    args = [p[n] for n in PARAM_NAMES_14] + tail
    n_ch = len(channels)
    h = xp.array(wresp(*args))[:n_ch, :]

    n = h.shape[-1]
    f = xp.fft.rfftfreq(n, dt)
    s = xp.fft.irfft(xp.fft.rfft(h, axis=-1) * (f >= F_MIN), n=n, axis=-1)
    PSD = xp.array(generate_PSD(waveform=s, dt=dt, noise_PSD=get_sensitivity,
                                channels=channels, noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    fmask = (xp.fft.rfftfreq(n, dt) > F_MIN)[1:]
    ss = inner_product(s, s, PSD=PSD, dt=dt, freq_mask=fmask, use_gpu=use_gpu)
    ss = float(ss.get()) if hasattr(ss, "get") else float(ss)
    return float(np.sqrt(max(ss, 0.0)))
