"""0PA + SIMPLE deviation: CV -> Nelder-Mead(500) -> CV, for all three EMRI cases.

Start at each case's CV-refined 0PA best fit with dev_1 = dev_2 = 0, turn on the two
simple-deviation coefficients (del_0_p, del_0_e -- multiplicative flux rescaling), and try
to improve the overlap with the 1PA signal by:
    1. adaptive-LM CV climb (Fisher gradient),
    2. a 500-step Nelder-Mead refinement (sigma-scaled), accepted only if it helps,
    3. one more CV climb.

Simple deviation -> additional_args[7]=del_0_p=dev_1, [8]=del_0_e=dev_2  (SuperKludgeFlux).

Run:  python gauss_cv_simple_dev.py
"""

import numpy as np
from scipy.optimize import minimize

from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product, fishinv
from stableemrifisher.fisher import StableEMRIFisher

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np
    print("[INFO] CuPy not found, using NumPy instead.")

# --- deviation wiring: SIMPLE = del_0_p (idx 7), del_0_e (idx 8) ------------
DEV_NAME = "simple"


def dev_tail(dev_1, dev_2):
    """[C_p, C_e, del_0_p, del_0_e] slice of additional_args for the simple deviation."""
    return [0.0, 0.0, dev_1, dev_2]


def dev_apa(dev_1, dev_2):
    """add_param_args entries after deviation_included, ordered to hit idx 5,6,7,8."""
    return {"C_p": 0.0, "C_e": 0.0, "dev_1": dev_1, "dev_2": dev_2}


# --- controls --------------------------------------------------------------
F_MIN = 1e-5
NDELTA = 12
RECOMPUTE_DELTAS_EVERY = 5
OVERLAP_TARGET = 0.999999
LAMBDA0, LM_MAX_ITERS, MAX_INNER, REL_TOL = 1e-2, 120, 30, 1e-9
NM_MAXITER, NM_STEP = 500, 2.0

use_gpu = True
nchannels = 3
param_names_14 = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
                  "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
params_to_infer = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0",
                   "dev_1", "dev_2"]

# start = CV-refined 0PA best fit (9 params) + [dev_1=0, dev_2=0]
CASES = [
    dict(name="idx0", dt=5.0, T=1.0, chi2=0.0, dev_nm_overlap=0.9982902463962956,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 7.5, "e0": 0.5, "xI0": 1.0,
                       "dist": 5.0, "qS": 0.7853981633974483, "phiS": 1.0, "qK": 1.0,
                       "phiK": 1.0471975511965976, "Phi_phi0": 0.9, "Phi_theta0": 0.5,
                       "Phi_r0": 0.4, "dev_1": 0.0, "dev_2": 0.0},
         start=[1.00014867e+06, 1.00010513e+01, 9.00096370e-01, 7.49942614e+00,
                4.99974688e-01, 7.83388290e-01, 9.99626083e-01, 9.09269038e-01,
                3.96038198e-01, 0.0, 0.0]),
    dict(name="idx9", dt=10.0, T=2.5, chi2=0.95, dev_nm_overlap=0.9980454212618244,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.9, "p0": 9.07414088, "e0": 0.2, "xI0": 1.0,
                       "dist": 5.0, "qS": 1.04719755, "phiS": 0.785398163, "qK": 0.628318531,
                       "phiK": 0.523598776, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start=[9.99994929e+05, 1.00000755e+01, 9.00001697e-01, 9.07417360e+00,
                1.99997410e-01, 1.04653506e+00, 7.83187637e-01, 1.59230827e-01,
                9.90108700e-02, 0.0, 0.0]),
    dict(name="idx13", dt=10.0, T=2.5, chi2=0.95, dev_nm_overlap=0.9998873034358231,
         signal_param={"m1": 1e6, "m2": 10.0, "a": 0.5, "p0": 9.97066819, "e0": 0.3, "xI0": 1.0,
                       "dist": 5.0, "qS": 1.04719755, "phiS": 0.785398163, "qK": 0.628318531,
                       "phiK": 0.523598776, "Phi_phi0": 0.1, "Phi_theta0": 0.2,
                       "Phi_r0": 0.3, "dev_1": 0.0, "dev_2": 0.0},
         start=[1.00003623e+06, 1.00001339e+01, 5.00052129e-01, 9.97041651e+00,
                2.99991839e-01, 1.04533235e+00, 7.83864901e-01, 1.13205146e-01,
                2.54180379e-01, 0.0, 0.0]),
]


# --- helpers ---------------------------------------------------------------
def _to_float(x):
    return float(x.get()) if hasattr(x, "get") else float(x)


def make_freq_mask(n, dt, fmin):
    return (xp.fft.rfftfreq(n, dt) > fmin)[1:]


def highpass_clip(w, dt, fmin):
    n = w.shape[-1]
    f = xp.fft.rfftfreq(n, dt)
    return xp.fft.irfft(xp.fft.rfft(w, axis=-1) * (f >= fmin), n=n, axis=-1)


def build_context(case):
    sp, dt, T, chi2 = case["signal_param"], case["dt"], case["T"], case["chi2"]
    channels = [A1TDISens, E1TDISens, T1TDISens][:nchannels]
    tdi_chan = {2: "AE", 3: "AET"}[nchannels]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    def rkw():
        return dict(Tobs=T, t0=10000.0, dt=dt, index_lambda=8, index_beta=7, flip_hx=True,
                    is_ecliptic_latitude=False, remove_garbage="zero",
                    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                    force_backend="cuda12x" if use_gpu else "cpu",
                    order=20, tdi="1st generation", tdi_chan=tdi_chan)

    wfm = GenerateEMRIWaveform(SuperKludgeWaveform,
                               sum_kwargs=dict(pad_output=True, odd_len=True),
                               return_list=False, use_gpu=use_gpu)
    wresp = ResponseWrapper(waveform_gen=wfm, **rkw())

    def resp_args(vec, evolve_1pa, deviation_on):
        p = {n: sp[n] for n in param_names_14}
        p.update(dict(zip(params_to_infer, vec)))
        return [p[n] for n in param_names_14] + [
            chi2, evolve_1pa, False, False, deviation_on, *dev_tail(vec[9], vec[10])]

    def make0(vec):                                    # 0PA + deviation template
        return xp.array(wresp(*resp_args(vec, False, True)))[:nchannels, :]

    true_inf = np.array([sp[n] for n in params_to_infer])
    s = highpass_clip(xp.array(wresp(*resp_args(true_inf, True, False)))[:nchannels, :], dt, F_MIN)
    PSD = xp.array(generate_PSD(waveform=s, dt=dt, noise_PSD=get_sensitivity,
                                channels=channels, noise_kwargs=noise_kwargs, use_gpu=use_gpu))
    fmask = make_freq_mask(s.shape[-1], dt, F_MIN)

    def ip(a, b):
        return _to_float(inner_product(a, b, PSD=PSD, dt=dt, freq_mask=fmask, use_gpu=use_gpu))

    def ov(vec):
        h = make0(vec)
        return ip(s, h) / np.sqrt(ip(s, s) * ip(h, h))

    def chi2r(vec):
        try:
            r = s - make0(vec)
            return ip(r, r)
        except Exception:
            return 1e30

    sef = StableEMRIFisher(
        waveform_class=SuperKludgeWaveform,
        waveform_class_kwargs=dict(sum_kwargs=dict(pad_output=True, odd_len=True)),
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=dict(return_list=False),
        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=rkw(),
        stats_for_nerds=False, use_gpu=use_gpu, deriv_type="stable",
        noise_model=get_sensitivity, noise_kwargs=noise_kwargs, channels=channels,
        T=T, dt=dt, stability_plot=False, der_order=6, Ndelta=NDELTA,
        plunge_check=True, return_derivatives=True)

    def fisher_derivs(vec, dl):
        wp = {n: sp[n] for n in param_names_14}
        wp.update(dict(zip(params_to_infer, vec)))
        apa = {"chi2": chi2, "evolve_1PA": False, "evolve_primary": False,
               "evolve_2PA": False, "deviation_included": True}
        apa.update(dev_apa(vec[9], vec[10]))
        F = sef(wave_params={n: wp[n] for n in param_names_14}, param_names=params_to_infer,
                add_param_args=apa, deltas=dl, live_dangerously=False, stability_plot=False,
                der_order=8, Ndelta=(NDELTA if dl is None else None))
        return np.asarray(F[-1], dtype=float), xp.array(F[0]), sef.deltas

    return dict(make0=make0, ov=ov, chi2r=chi2r, ip=ip, s=s,
                fisher_derivs=fisher_derivs, snr=float(np.sqrt(ip(s, s))))


# --- optimisers ------------------------------------------------------------
def lm_climb(ctx, theta0, tag=""):
    """Adaptive Levenberg-Marquardt CV climb. Returns (theta, overlap, sigma)."""
    ov, chi2r, fisher_derivs, s, ip, make0 = (ctx["ov"], ctx["chi2r"], ctx["fisher_derivs"],
                                              ctx["s"], ctx["ip"], ctx["make0"])
    npar = len(theta0)
    cur = np.array(theta0, dtype=float)
    lam, nu, dl, sigma = LAMBDA0, 2.0, None, np.ones(npar)
    for it in range(LM_MAX_ITERS):
        if it % RECOMPUTE_DELTAS_EVERY == 0:
            dl = None
        G, dH, deltas = fisher_derivs(cur, dl)
        if dl is None:
            dl = deltas
        h = make0(cur)
        r = s - h
        g = np.array([ip(dH[j], r) for j in range(npar)])
        sigma = np.sqrt(np.abs(np.diag(fishinv(cur[0], G, index_of_M=0))))
        c0 = ip(r, r)
        ovc = ip(s, h) / np.sqrt(ip(s, s) * ip(h, h))
        if ovc > OVERLAP_TARGET:
            break
        dvec = np.abs(np.diag(G)) + 1e-30
        delta, ok, rel = np.zeros(npar), False, 0.0
        for _ in range(MAX_INNER):
            try:
                delta = np.linalg.solve(G + lam * np.diag(dvec), g)
            except np.linalg.LinAlgError:
                lam *= nu; nu *= 2.0; continue
            pred = float(delta @ (g + lam * dvec * delta))
            rho = (c0 - chi2r(cur + delta)) / pred if pred > 0 else -1.0
            if rho > 0.0:
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3); nu = 2.0
                rel = (c0 - chi2r(cur + delta)) / c0; ok = True; break
            lam *= nu; nu *= 2.0
        print(f"    CV{tag} it {it:>3} lam={lam:.1e} ov={ovc:.7f} chi2={c0:.3e} rel={rel:.1e}")
        if not ok:
            break
        cur = cur + delta
        if rel < REL_TOL:
            break
    return cur, ov(cur), sigma


def nm_refine(ctx, theta0, sigma, maxiter):
    """Nelder-Mead in sigma-scaled coords, minimising chi2; accept only if overlap improves."""
    chi2r, ov = ctx["chi2r"], ctx["ov"]
    n = len(theta0)
    simplex = np.vstack([np.zeros(n)] + [NM_STEP * np.eye(n)[i] for i in range(n)])
    res = minimize(lambda x: chi2r(theta0 + x * sigma), np.zeros(n), method="Nelder-Mead",
                   options=dict(initial_simplex=simplex, maxiter=maxiter,
                                xatol=1e-4, fatol=1e-4, adaptive=True))
    cand = theta0 + res.x * sigma
    ok = ov(cand) > ov(theta0)
    print(f"    NM  nfev={res.nfev} ov {ov(theta0):.7f} -> {ov(cand):.7f} (accepted={ok})")
    return cand if ok else theta0


# --- per-case driver: CV -> (only if it stalls) NM(500) -> CV ---------------
def run_case(case):
    ctx = build_context(case)
    theta = np.array(case["start"], dtype=float)
    print(f"[{case['name']}] SNR={ctx['snr']:.2f}  overlap(start, dev=0) = {ctx['ov'](theta):.7f}")
    theta, ov1, sigma = lm_climb(ctx, theta, tag="#1")
    if ov1 < OVERLAP_TARGET:                    # CV stalled below target -> one NM, then CV
        theta = nm_refine(ctx, theta, sigma, NM_MAXITER)
        theta, ov1, sigma = lm_climb(ctx, theta, tag="#2")
    return dict(name=case["name"], snr=ctx["snr"], ov_start=float(ctx["ov"](case["start"])),
                ov_final=float(ctx["ov"](theta)), dev_nm_overlap=case["dev_nm_overlap"],
                params=theta.copy())


def main():
    results = []
    for case in CASES:
        print("=" * 70)
        results.append(run_case(case))

    print("\n" + "=" * 70)
    print(f"model = 0PA + {DEV_NAME} deviation")
    print(f"{'case':6} {'SNR':>7} {'ov@start':>11} {'ov@final':>11} {'ov@NM(dev)':>12} {'improved?':>10}")
    for r in results:
        imp = "YES" if r["ov_final"] > r["ov_start"] + 1e-8 else "no"
        print(f"{r['name']:6} {r['snr']:>7.2f} {r['ov_start']:>11.7f} {r['ov_final']:>11.7f} "
              f"{r['dev_nm_overlap']:>12.7f} {imp:>10}")

    print("\noptimized 11-param points  [m1, m2, a, p0, e0, qS, phiS, Phi_phi0, Phi_r0, dev_1, dev_2]:")
    for r in results:
        print(f"[{r['name']}] = [{', '.join(f'{v:.8e}' for v in r['params'])}]")


if __name__ == "__main__":
    main()
