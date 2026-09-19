"""Per-channel (A, E, T) SNR of the 1PA injection for every IMRI case.

For each case the 1PA signal is generated once and the optimal SNR is computed
SEPARATELY in each TDI channel:
    SNR_X = sqrt(<s_X|s_X>)   with that channel's own PSD,
and the quadrature total  SNR_tot = sqrt(SNR_A^2 + SNR_E^2 + SNR_T^2)
(which is what the 3-channel inner product returns).

Covers: the full 25-point a x e0 grid (m2=1e3, dt=10, T=1.0), the two IMRI-tails
points (m2=1e4, dt=10, T=0.25), and the ad-hoc case (m2=5e3, dt=5, T=1.0).

Output: snr_channels_imri.json  + a printed table.
Run:  python snr_channels_imri.py
"""
import json, os
from datetime import datetime, timezone
import numpy as np

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product

try:
    import cupy as cp; xp = cp
except ImportError:
    xp = np; print("[INFO] CuPy not found, using NumPy.")

F_MIN, NCH, USE_GPU = 1e-5, 3, True
CHAN = [A1TDISens, E1TDISens, T1TDISens]
NAMES = ["A", "E", "T"]
P14 = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
       "Phi_phi0","Phi_theta0","Phi_r0"]

SIGNAL_ARRAY = "/scratch/e1583490/SK_files/data/signal/signal_parameter_array_IMRI.npy"


def _f(x): return float(x.get()) if hasattr(x, "get") else float(x)
def fmask(n, dt): return (xp.fft.rfftfreq(n, dt) > F_MIN)[1:]
def hp(w, dt):
    n = w.shape[-1]; f = xp.fft.rfftfreq(n, dt)
    return xp.fft.irfft(xp.fft.rfft(w, axis=-1) * (f >= F_MIN), n=n, axis=-1)


def snr_per_channel(sp, dt, T, chi2):
    """Return (per-channel SNR list, total SNR) for the 1PA injection."""
    rkw = dict(Tobs=T, t0=10000.0, dt=dt, index_lambda=8, index_beta=7, flip_hx=True,
               is_ecliptic_latitude=False, remove_garbage="zero",
               orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
               force_backend="cuda12x" if USE_GPU else "cpu",
               order=20, tdi="1st generation", tdi_chan="AET")
    wfm = GenerateEMRIWaveform(SuperKludgeWaveform,
                               sum_kwargs=dict(pad_output=True, odd_len=True),
                               return_list=False, use_gpu=USE_GPU)
    wresp = ResponseWrapper(waveform_gen=wfm, **rkw)
    args = [sp[n] for n in P14] + [chi2, True, False, False, False]   # 1PA, no deviation
    s = hp(xp.array(wresp(*args))[:NCH, :], dt)
    noise_kwargs = [{"sens_fn": c} for c in CHAN[:NCH]]
    PSD = xp.array(generate_PSD(waveform=s, dt=dt, noise_PSD=get_sensitivity,
                                channels=CHAN[:NCH], noise_kwargs=noise_kwargs,
                                use_gpu=USE_GPU))
    fm = fmask(s.shape[-1], dt)
    per = []
    for i in range(NCH):                      # each channel with its OWN PSD
        v = _f(inner_product(s[i:i+1], s[i:i+1], PSD=PSD[i:i+1], dt=dt,
                             freq_mask=fm, use_gpu=USE_GPU))
        per.append(float(np.sqrt(max(v, 0.0))))
    tot = _f(inner_product(s, s, PSD=PSD, dt=dt, freq_mask=fm, use_gpu=USE_GPU))
    return per, float(np.sqrt(tot))


def build_cases():
    cases = []
    sig = np.load(SIGNAL_ARRAY)
    for i in range(25):                       # full a x e0 grid
        r = sig[i]
        cases.append(dict(name=f"grid_idx{i}", group="grid", dt=10.0, T=1.0, chi2=0.95,
            sp={"m1": 1e6, "m2": 1e3, "a": float(r[2]), "p0": float(r[3]), "e0": float(r[4]),
                "xI0": 1.0, "dist": float(r[6]), "qS": float(r[7]), "phiS": float(r[8]),
                "qK": float(r[9]), "phiK": float(r[10]), "Phi_phi0": float(r[11]),
                "Phi_theta0": float(r[12]), "Phi_r0": float(r[13])}))
    base = dict(qS=1.04719755, phiS=0.78539816, qK=0.62831853, phiK=0.52359878,
                Phi_phi0=0.1, Phi_theta0=0.2, Phi_r0=0.3, xI0=1.0)
    cases.append(dict(name="pt4", group="tails", dt=10.0, T=0.25, chi2=0.95,
        sp={**base, "m1": 1e6, "m2": 1e4, "a": 0.9, "p0": 29.2602456, "e0": 0.10, "dist": 272.11852}))
    cases.append(dict(name="pt20", group="tails", dt=10.0, T=0.25, chi2=0.95,
        sp={**base, "m1": 1e6, "m2": 1e4, "a": -0.9, "p0": 30.522071, "e0": 0.50, "dist": 94.592671}))
    cases.append(dict(name="adhoc_A", group="adhoc", dt=5.0, T=1.0, chi2=0.95,
        sp={"m1": 1e6, "m2": 5.0e3, "a": 0.70, "p0": 25.0, "e0": 0.25, "xI0": 1.0, "dist": 12.0,
            "qS": 0.7853981633974483, "phiS": 1.0, "qK": 1.0, "phiK": 1.0471975511965976,
            "Phi_phi0": 0.9, "Phi_theta0": 0.5, "Phi_r0": 0.4}))
    return cases


JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snr_channels_imri.json")


def main():
    out = {"description": "per-TDI-channel optimal SNR of the 1PA injection",
           "channels": NAMES, "f_min": F_MIN,
           "generated_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    print(f"{'case':12} {'grp':6} {'a':>5} {'e0':>4} | {'SNR_A':>9} {'SNR_E':>9} {'SNR_T':>9} | "
          f"{'SNR_tot':>9} | {'A%':>6} {'E%':>6} {'T%':>6}")
    print("-" * 104)
    for c in build_cases():
        try:
            per, tot = snr_per_channel(c["sp"], c["dt"], c["T"], c["chi2"])
        except Exception as e:
            print(f"{c['name']:12} {c['group']:6} FAILED: {type(e).__name__}: {e}")
            out["cases"].append(dict(name=c["name"], group=c["group"], error=str(e)))
            continue
        fr = [100.0 * (p / tot) ** 2 if tot > 0 else float("nan") for p in per]  # power fraction
        print(f"{c['name']:12} {c['group']:6} {c['sp']['a']:+5.1f} {c['sp']['e0']:4.2f} | "
              f"{per[0]:9.4f} {per[1]:9.4f} {per[2]:9.4f} | {tot:9.4f} | "
              f"{fr[0]:6.2f} {fr[1]:6.2f} {fr[2]:6.2f}")
        out["cases"].append(dict(name=c["name"], group=c["group"], a=c["sp"]["a"], e0=c["sp"]["e0"],
            dt=c["dt"], T=c["T"], snr_A=per[0], snr_E=per[1], snr_T=per[2], snr_total=tot,
            power_frac_percent=dict(zip(NAMES, fr))))
        with open(JSON_PATH, "w") as f:
            json.dump(out, f, indent=2)
    print(f"\n[saved] {JSON_PATH}")


if __name__ == "__main__":
    main()
