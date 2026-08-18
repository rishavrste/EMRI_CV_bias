"""Cutler-Vallisneri systematic-bias estimate for an EMRI/IMRI source.

Given a *true* waveform model and an *approximate* recovery template, this script
computes the parameter bias incurred by recovering the signal with the
approximate template, using the linear-signal approximation of
Cutler & Vallisneri (2007), Eq. (29) [arXiv:gr-qc/0703086].
"""

import numpy as np

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product, fishinv
from stableemrifisher.fisher import StableEMRIFisher

from misc import _is_pos_def
from params import cfg

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np
    print("[INFO] CuPy not found, using NumPy instead.")


# Low-frequency cutoff (Hz) applied to every noise-weighted inner product.
F_MIN = 1e-5


def _to_float(x):
    """Return a plain Python float from a NumPy or CuPy scalar."""
    return float(x.get()) if hasattr(x, "get") else float(x)


def to_cpu(x):
    """Return a host NumPy array from a NumPy or CuPy array.

    Large time-series arrays stay on the GPU (CuPy); only the small Fisher-matrix
    quantities are brought back here, because their linear algebra
    (``np.linalg.inv``, ``fishinv``, ``np.linalg.cholesky``) is NumPy-only.
    """
    return x.get() if hasattr(x, "get") else np.asarray(x)


def make_freq_mask(n, dt, fmin):
    """Boolean frequency mask (f > fmin, with the f=0 bin dropped) on the active backend.

    Passed to ``inner_product`` as ``freq_mask=`` so the mask lives on the same
    device (NumPy/CuPy) as the waveforms; otherwise ``inner_product`` would build
    a NumPy mask and index CuPy arrays with it, raising a TypeError.
    """
    return (xp.fft.rfftfreq(n, dt) > fmin)[1:]


def highpass_clip(waveform, dt, fmin):
    """Zero all frequency content of a time-domain waveform below ``fmin`` (Hz).

    Applied per channel via an rFFT -> mask -> irFFT round trip so that every
    downstream quantity (PSD, SNR, Fisher, bias) sees the same band-limited data.
    """
    n = waveform.shape[-1]
    freq = xp.fft.rfftfreq(n, dt)
    spectrum = xp.fft.rfft(waveform, axis=-1) * (freq >= fmin)
    return xp.fft.irfft(spectrum, n=n, axis=-1)


def cutler_vallisneri_bias(
    waveform_true,
    waveform_approx,
    partials_approx,
    fisher_approx,
    base_params,
    psd,
    dt,
    param_names=None,
    fmin=F_MIN,
    use_gpu=False,
):
    r"""Estimate the systematic parameter bias from using an approximate template.

    Linear-signal (Cutler-Vallisneri) approximation,
    Cutler & Vallisneri 2007, Eq. (29):

        delta(theta^i) = (Gamma^-1)^{ij} < d_j h_AP | h_TR - h_AP >
        theta_BF       = theta_fid + delta(theta)

    ``Gamma`` is the Fisher matrix of the approximate template and ``d_j h_AP``
    its partial derivatives. Every approximate-model quantity here
    (``waveform_approx``, ``partials_approx``, ``fisher_approx``) must be
    evaluated at the SAME fiducial point ``theta_fid = base_params``, and the
    bias is added back to that same point.

    Parameters
    ----------
    waveform_true, waveform_approx : ndarray, shape (n_channels, n_samples)
        Time-domain waveforms of the true and approximate templates. They must
        share the same length -- pad the shorter series before calling.
    partials_approx : ndarray, shape (n_params, n_channels, n_samples)
        Time-domain partial derivatives of the approximate template.
    fisher_approx : ndarray, shape (n_params, n_params)
        Fisher matrix of the approximate template (Gamma).
    base_params : array_like, shape (n_params,)
        Fiducial point theta_fid the CV expansion is taken about -- the SAME
        point where ``fisher_approx`` and ``partials_approx`` were evaluated
        (here the approximate-model MLP), in the ordering/units of
        ``fisher_approx``. The result is ``base_params + delta``.
    psd : ndarray, shape (n_channels, n_freqs)
        One-sided noise PSD per channel.
    dt : float
        Sampling interval of the time series.
    param_names : sequence of str, optional
        Names of the inferred parameters, used only for logging.
    fmin : float, optional
        Low-frequency cutoff of the inner product.
    use_gpu : bool, optional
        Whether the waveform arrays live on the GPU (CuPy).

    Returns
    -------
    best_fit : ndarray, shape (n_params,)
        Biased best-fit parameters theta_BF, in the units of ``fisher_approx``.
    delta : ndarray, shape (n_params,)
        The parameter bias delta(theta).
    """
    if waveform_true.shape != waveform_approx.shape:
        raise ValueError(
            "true and approximate waveforms must have the same shape; got "
            f"{waveform_true.shape} vs {waveform_approx.shape}. "
            "Pad the shorter series before calling."
        )

    # Fisher-matrix layer is NumPy; the waveform arrays stay on the GPU.
    fisher_approx = to_cpu(fisher_approx)
    base_params = to_cpu(base_params).astype(float)

    n_params = len(fisher_approx)
    residual = waveform_true - waveform_approx  # h_TR - h_AP
    freq_mask = make_freq_mask(residual.shape[-1], dt, fmin)

    # < d_j h_AP | h_TR - h_AP >, one scalar per inferred parameter.
    overlaps = np.array([
        _to_float(inner_product(
            partials_approx[j], residual, PSD=psd, dt=dt,
            freq_mask=freq_mask, use_gpu=use_gpu,
        ))
        for j in range(n_params)
    ])

    delta = np.linalg.inv(fisher_approx) @ overlaps

    if param_names is not None:
        for name, d in zip(param_names, delta):
            print(f"[CV] bias in {name}: {d:.6e}")

    return base_params + delta, delta


# ---------------------------------------------------------------------------
# Configuration and true (injected) parameters
# ---------------------------------------------------------------------------
(m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
 Phi_phi0, Phi_theta0, Phi_r0, dt, T, chi2, dev_1, dev_2) = cfg.signal_row

mlp_points = cfg.mlp_points
use_gpu = cfg.use_gpu
nchannels = cfg.n_channels
params_to_infer = cfg.params_to_infer

# Response-wrapper geometry constants.
t0 = 10000.0
index_lambda = 8   # position of phiS in the parameter vector
index_beta = 7     # position of qS in the parameter vector
order = 20
tdi_gen = "1st generation"

# ---------------------------------------------------------------------------
# TDI channels and noise model
# ---------------------------------------------------------------------------
channels = [A1TDISens, E1TDISens, T1TDISens]
if nchannels == 3:
    tdi_chan = "AET"
elif nchannels == 2:
    tdi_chan = "AE"
else:
    raise ValueError(
        f"Unsupported number of channels: {nchannels}. Only 2 (A, E) or "
        "3 (A, E, T) are supported."
    )
channels = channels[:nchannels]
noise_kwargs = [{"sens_fn": ch} for ch in channels]


def build_response_kwargs():
    """Return a fresh set of ResponseWrapper keyword arguments.

    A new ``orbits`` object is created each call so the standalone wrapper and
    the one built inside ``StableEMRIFisher`` never share mutable state.
    """
    return dict(
        Tobs=T,
        t0=t0,
        dt=dt,
        index_lambda=index_lambda,
        index_beta=index_beta,
        flip_hx=True,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend="cuda12x" if use_gpu else "cpu",
        order=order,
        tdi=tdi_gen,
        tdi_chan=tdi_chan,
    )


# ---------------------------------------------------------------------------
# True waveform (injection): the 1PA model evaluated at the injected parameters
# ---------------------------------------------------------------------------
waveform_model = GenerateEMRIWaveform(
    SuperKludgeWaveform,
    sum_kwargs=dict(pad_output=True, odd_len=True),
    return_list=False,
    use_gpu=use_gpu,
)
waveform_response = ResponseWrapper(
    waveform_gen=waveform_model, **build_response_kwargs()
)
print("[INFO] Finished loading modules and building ResponseWrapper")

# Source parameters passed to the response wrapper:
#   14 EMRI parameters, then chi2 and the model-control flags
#   (evolve_1PA, evolve_primary, evolve_2PA, deviation_included, dev_1, dev_2).
# The truth uses the 1PA model with no beyond-GR deviations.
true_wave_params = [
    m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0,
    chi2, True, False, False, False, dev_1, dev_2,
]
waveform_true = xp.array(waveform_response(*true_wave_params))[:nchannels, :]
# Band-limit the true waveform below F_MIN before it feeds the PSD, SNR, and bias.
waveform_true = highpass_clip(waveform_true, dt, F_MIN)
print("[INFO] Finished generating true waveform")

# ---------------------------------------------------------------------------
# Noise PSD and SNR check
# ---------------------------------------------------------------------------
PSD_funcs = xp.array(generate_PSD(
    waveform=waveform_true,
    dt=dt,
    noise_PSD=get_sensitivity,
    channels=channels,
    noise_kwargs=noise_kwargs,
    use_gpu=use_gpu,
))
print("shape check - waveform_true:", xp.shape(waveform_true),
      "PSD_funcs:", xp.shape(PSD_funcs))

freq_mask = make_freq_mask(waveform_true.shape[-1], dt, F_MIN)
snr = np.sqrt(_to_float(inner_product(
    waveform_true, waveform_true, PSD_funcs, dt, freq_mask=freq_mask, use_gpu=use_gpu)))
print(f"[TRUE] SNR: {snr:.6f}")
for i in range(PSD_funcs.shape[0]):
    snr_i = np.sqrt(_to_float(inner_product(
        waveform_true[i:i + 1], waveform_true[i:i + 1],
        PSD_funcs[i:i + 1], dt, freq_mask=freq_mask, use_gpu=use_gpu)))
    print(f"[TRUE] SNR - channel {i}: {snr_i:.6f}")

# ---------------------------------------------------------------------------
# Approximate template: Fisher matrix and partial derivatives (StableEMRIFisher)
# ---------------------------------------------------------------------------
sef = StableEMRIFisher(
    waveform_class=SuperKludgeWaveform,
    waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
    waveform_generator=GenerateEMRIWaveform,
    waveform_generator_kwargs=dict(return_list=False),
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=build_response_kwargs(),
    stats_for_nerds=True,
    use_gpu=use_gpu,
    deriv_type="stable",
    noise_model=get_sensitivity,
    noise_kwargs=noise_kwargs,
    channels=channels,
    T=T,
    dt=dt,
    stability_plot=False,
    der_order=6,
    Ndelta=12,
    plunge_check=True,
    return_derivatives=True,
)

emri_kwargs = {"T": T, "dt": dt}

# The approximate template is evaluated at the maximum-likelihood point.
# (cfg validates cfg.analysis against its supported set on construction.)
fisher_params = mlp_points[0:14]
evolve_1PA = cfg.analysis == "1PA"
deviation_included = "dev" in cfg.analysis  # 0PA_dev_simple, 0PA_dev_PN

additional_kwargs = {
    "chi2": chi2,
    "evolve_1PA": evolve_1PA,
    "evolve_primary": False,
    "evolve_2PA": False,
    "deviation_included": deviation_included,
    "dev_1": dev_1,
    "dev_2": dev_2,
}
pars_list_com = list(fisher_params) + [
    chi2, evolve_1PA, False, False, deviation_included, dev_1, dev_2,
]

SNR = sef.SNRcalc_SEF(*pars_list_com, **emri_kwargs, use_gpu=use_gpu)
print("SNR: ", SNR)

param_names_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
param_dict = dict(zip(param_names_14, fisher_params))

Fisher = sef(
    wave_params=param_dict,
    param_names=params_to_infer,
    add_param_args=additional_kwargs,
    live_dangerously=False,
    stability_plot=True,
    der_order=8,
    Ndelta=20,
)

# SEF returns (derivatives, Fisher_matrix) when return_derivatives=True.
# The Fisher matrix is already a host NumPy float64 array; keep it on the CPU.
fisher_matrix = np.asarray(Fisher[-1], dtype=float)
print(f"[FISHER] {fisher_matrix!r}")
print(f"[FISHER_IS_PD] {_is_pos_def(fisher_matrix)}")
fisher_inv = fishinv(param_dict["m1"], fisher_matrix, index_of_M=0)
print(f"[FISHER_INV] {fisher_inv!r}")
print(f"[FISHER_INV_IS_PD] {_is_pos_def(fisher_inv)}")
fisher_std = np.sqrt(np.diag(fisher_inv))
print(f"[FISHER_STD] {fisher_std!r}")

# ---------------------------------------------------------------------------
# Cutler-Vallisneri bias
# ---------------------------------------------------------------------------
waveform_approx = xp.array(sef.waveform)   # 0PA template (GPU), (n_channels, N)
partials_approx = xp.array(Fisher[0])      # derivatives (GPU), (n_params, n_channels, N)
print("shape of waveform_approx:", waveform_approx.shape)

# CV expands about the point where Gamma/derivatives were evaluated: the MLP.
mlp_by_name = dict(param_dict)                          # 14 EMRI params at the MLP
mlp_by_name.update(dev_1=mlp_points[17], dev_2=mlp_points[18])
base_inferred = np.array([mlp_by_name[name] for name in params_to_infer])

# True (injected) values of the inferred parameters, for comparison only.
truth_by_name = dict(zip(param_names_14, cfg.signal_row[:14]))
truth_by_name.update(dev_1=dev_1, dev_2=dev_2)
true_inferred = np.array([truth_by_name[name] for name in params_to_infer])

best_fit, bias = cutler_vallisneri_bias(
    waveform_true=waveform_true,
    waveform_approx=waveform_approx,
    partials_approx=partials_approx,
    fisher_approx=fisher_matrix,
    base_params=base_inferred,          # theta_BF = MLP + delta
    psd=PSD_funcs,
    dt=dt,
    param_names=params_to_infer,
    use_gpu=use_gpu,
)

# best_fit = MLP + delta is the predicted 0PA best-fit; compare it to the truth.
residual = best_fit - true_inferred
print(f"[CV] bias vector (delta): {bias!r}")
print("[CV] param        MLP            delta        best_fit        true"
      "        best_fit-true   (in sigma)")
for name, mlp_v, d, bf, tr, res, sig in zip(
        params_to_infer, base_inferred, bias, best_fit, true_inferred,
        residual, fisher_std):
    print(f"[CV] {name:10s} {mlp_v: .6e} {d: .3e} {bf: .6e} {tr: .6e} "
          f"{res: .3e} {res / sig: .2f}")
