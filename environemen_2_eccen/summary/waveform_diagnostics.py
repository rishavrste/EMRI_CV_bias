"""Diagnostics for a suspect injection: where in time and frequency its SNR comes from.

Built for set 4, whose in-band SNR changes with dt (351 at dt = 5, 316 at dt = 2.5, 82 at
dt = 10) while set 5's does not.  Everything reuses the pipeline's own signal, PSD and inner
product conventions (env_cv.build_scoring, stableemrifisher generate_PSD / inner_product), so
the SNR density here integrates to exactly the SNR the fits saw.

Used by snr_channels_dt.ipynb.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from env_cv import (Config, SPECS, build_scoring, make_model, observation_time,  # noqa: E402
                    log)
from fisher_bestfit import CONFIGS                                              # noqa: E402
from few.trajectory.inspiral import EMRIInspiral                                # noqa: E402
from few.trajectory.ode.flux import KerrEccEqAccFlux                            # noqa: E402
from few.utils.constants import YRSID_SI                                        # noqa: E402
from lisatools.sensitivity import get_sensitivity                               # noqa: E402
from stableemrifisher.utils import generate_PSD                                 # noqa: E402

CHANNELS = "AET"


def to_np(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def load_signal(tag, dt=5.0, fmax=0.1):
    """The set's environmental injection exactly as the fits saw it, plus its PSD (CPU)."""
    cfg = Config(os.path.join(ROOT, CONFIGS[tag][1]))
    cfg.dt, cfg.fmax = dt, fmax
    T = observation_time(cfg)
    scoring = build_scoring(cfg, T)
    sig = to_np(scoring["signal"])
    psd = generate_PSD(waveform=sig, dt=dt, noise_PSD=get_sensitivity,
                       channels=scoring["channels"], noise_kwargs=scoring["noise_kwargs"],
                       use_gpu=False)
    return dict(tag=tag, cfg=cfg, T=T, dt=dt, fmin=cfg.fmin, fmax=fmax, snr=scoring["snr"],
                signal=sig, psd=np.array([to_np(p) for p in psd]),
                t=np.arange(sig.shape[-1]) * dt, scoring=scoring)


def load_vacuum(d):
    """The vacuum-GR waveform at the injected parameters (disk off), same response and T."""
    cfg = d["cfg"]
    model = make_model(cfg, d["scoring"], d["T"], SPECS["vacuum"](cfg))
    return to_np(model["template"](cfg.theta_inj))


def snr_density(d, signal=None):
    """Per-channel dSNR^2 per frequency bin and its cumulative sum, pipeline conventions:
    <a|a> = 4 df sum |dt rfft(a)|^2 / PSD over fmin < f < fmax, DC dropped."""
    sig = d["signal"] if signal is None else signal
    n = sig.shape[-1]
    f = np.fft.rfftfreq(n, d["dt"])[1:]
    df = 1.0 / (n * d["dt"])
    mask = (f > d["fmin"]) & (f < d["fmax"])
    hf = d["dt"] * np.fft.rfft(sig, axis=-1)[:, 1:]
    dens = 4 * df * np.abs(hf) ** 2 / d["psd"]
    dens[:, ~mask] = 0.0
    return dict(f=f, density=dens, cumulative=np.cumsum(dens, axis=-1),
                snr=float(np.sqrt(dens.sum())),
                channel_snr={c: float(np.sqrt(dens[k].sum())) for k, c in enumerate(CHANNELS)})


def glitch_report(d, signal=None):
    """Per channel: non-finite samples, where |x| and |dx| peak, and how extreme the largest
    sample-to-sample jump is against the typical one.  A smooth inspiral has a jump ratio of
    a few; a discontinuity or spike stands out by orders of magnitude."""
    sig = d["signal"] if signal is None else signal
    out = {}
    for k, c in enumerate(CHANNELS[:sig.shape[0]]):
        x = sig[k]
        dx = np.abs(np.diff(x))
        j = int(np.argmax(dx))
        out[c] = dict(nonfinite=int((~np.isfinite(x)).sum()),
                      t_max_abs_days=float(d["t"][int(np.argmax(np.abs(x)))] / 86400),
                      max_abs=float(np.max(np.abs(x))),
                      t_max_jump_days=float(d["t"][j] / 86400),
                      jump_ratio=float(dx[j] / np.median(dx)))
    return out


def trajectory(tag, disk=True, T=None):
    """p(t), e(t) and the phases of the injection's trajectory, with or without the disk."""
    cfg = Config(os.path.join(ROOT, CONFIGS[tag][1]))
    inj, dk = cfg.injection, cfg.disk
    disk_args = (dk["Sigma0"], dk["h0"], dk["Sigma_p"]) if disk else (0.0, dk["h0"], dk["Sigma_p"])
    traj = EMRIInspiral(func=KerrEccEqAccFlux)
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(
        *[inj[n] for n in ("m1", "m2", "a", "p0", "e0", "xI0")], *disk_args,
        T=T if T is not None else 10.0)
    log(f"[TRAJ] {tag} disk={disk}: {len(t)} steps, t_end = {t[-1] / YRSID_SI:.6f} yr, "
        f"e: {e[0]:.6g} -> {e[-1]:.6g}, p: {p[0]:.6g} -> {p[-1]:.6g}")
    return dict(t_yr=np.asarray(t) / YRSID_SI, p=np.asarray(p), e=np.asarray(e),
                Phi_phi=np.asarray(Phi_phi), Phi_r=np.asarray(Phi_r))


TDI_NULL_HZ = 299792458.0 / (2 * 2.5e9)       # c / (2L): first-generation A/E PSD null


def null_contribution(d, halfwidth=2e-3, fmax_clean=0.05):
    """How much of the SNR^2 comes from around the TDI null at c/(2L) ~ 0.05996 Hz.

    Returns, per channel and in total, the SNR^2 inside [null - halfwidth, null + halfwidth],
    its share of the in-band SNR^2, and the SNR with the band capped at `fmax_clean` (below
    the null).  A resolved signal has essentially nothing there; set 4 had ~95%."""
    s = snr_density(d)
    near = np.abs(s["f"] - TDI_NULL_HZ) < halfwidth
    clean = s["f"] < fmax_clean
    tot = s["density"].sum()
    per = {c: dict(near_null=float(s["density"][k, near].sum()),
                   share=float(s["density"][k, near].sum() / tot))
           for k, c in enumerate(CHANNELS)}
    return dict(snr=s["snr"], snr_below_clean=float(np.sqrt(s["density"][:, clean].sum())),
                near_null=float(s["density"][:, near].sum()),
                share=float(s["density"][:, near].sum() / tot), channels=per,
                f=s["f"], density=s["density"])
