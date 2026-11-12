"""idx0 trajectories and their rates for the signal and the fitted templates, to see why the PN + frequency
deviation does not lower the absolute bias.

Models (each on its own trajectory, T = 2.5 yr):
    1PA                     injection, evolve_1PA                                  (the signal)
    0PA injection           injection, 0PA
    0PA fit                 EMRI_1PA_PN_DEV best fit "0pa"
    0PA+PN fit              EMRI_1PA_PN_DEV best fit "pn"   (pdot, edot deviation only, no frequency term)
    0PA+PN_freq fit         results/full/lm_idx0.json       (pdot, edot and frequency deviation)
    0PA+PN_freq inj, C_LO   injection, C = leading 1PA term: C_p = (28/15)(5a-3chi2), C_e = (4/45)(5a-3chi2)
    0PA+PN_freq inj, C_proj injection, C = full 1PA rate projected at p0 (physical_c.py, the LM seed)

Along each trajectory: p, e, Phi_phi, Phi_r, and the ODE right-hand side pdot, edot, Omega_phi, Omega_r (the
deviation included), in seconds, plus dOmega/dt from the dense trajectory.
    -> consistency_check/results/trajectories_idx0.npz
Run (CPU):  python consistency_check/trajectories_idx0.py > consistency_check/logs/trajectories_idx0.log
"""
import json
import sys
from pathlib import Path

import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SuperKludgeFlux

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C                               # noqa: E402
from physical_c import physical_c                 # noqa: E402

POINT = "0"
T_YR = 2.5
N_T = 6000                                        # dense time samples per trajectory
OUT = HERE / "results" / f"trajectories_idx{POINT}.npz"
SIGNAL_FLAGS, TEMPLATE_FLAGS = C.SIGNAL_FLAGS, C.TEMPLATE_FLAGS
INTRINSIC = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0"]


# --- the PN template without the frequency term ---------------------------------
class PNOnlyFlux(SuperKludgeFlux):
    """The EMRI_1PA_PN_DEV PN deviation: the working-tree rhs with its frequency term taken back out."""

    def modify_rhs(self, ydot, y, **kwargs):
        super().modify_rhs(ydot, y, **kwargs)
        if self.deviation_included:
            p, e = y[:2]
            shift = self.massratio * (-(15 / 56) * self.C_p) * (1 - e * e) ** 1.5 / p ** 2
            ydot[3] -= shift
            ydot[5] -= shift


PNOnlyFlux.__name__ = "SuperKludgeFlux"          # EMRIInspiral keys its SuperKludge set-up on this name


# --- model definitions ------------------------------------------------------------
def injection():
    row = C.signal_row(POINT)
    return {n: float(row[C.COL[n]]) for n in INTRINSIC}


def old_fit(template):
    fit = json.loads(C.OLD_BEST_FITS.read_text())[template][POINT]["theta_final"]
    return {n: float(fit[n]) for n in fit}


def freq_fit():
    return json.loads((C.OUT_ROOT / "full" / f"lm_idx{POINT}.json").read_text())["theta_final"]


def c_leading(theta, chi2):
    s = 5 * theta["a"] - 3 * chi2
    return dict(C_p=28 / 15 * s, C_e=4 / 45 * s)


def models():
    """label -> (parameters, evolve flags, deviation dict or None, flux class)."""
    chi2 = float(C.signal_row(POINT)[C.COL["chi2"]])
    inj = injection()
    pn, fq = old_fit("pn"), freq_fit()
    return {
        "1PA": (inj, SIGNAL_FLAGS, None, SuperKludgeFlux),
        "0PA injection": (inj, TEMPLATE_FLAGS, None, SuperKludgeFlux),
        "0PA fit": (old_fit("0pa"), TEMPLATE_FLAGS, None, SuperKludgeFlux),
        "0PA+PN fit": (pn, TEMPLATE_FLAGS, {k: pn[k] for k in C.DEV_SLOTS}, PNOnlyFlux),
        "0PA+PN_freq fit": (fq, TEMPLATE_FLAGS, {k: fq[k] for k in C.DEV_SLOTS}, SuperKludgeFlux),
        "0PA+PN_freq inj, C_LO": (inj, TEMPLATE_FLAGS, c_leading(inj, chi2), SuperKludgeFlux),
        "0PA+PN_freq inj, C_proj": (inj, TEMPLATE_FLAGS, physical_c(POINT), SuperKludgeFlux),
    }


def add_args(chi2, flags, dev):
    on = dev is not None
    return [chi2, *flags, on] + [float(dev[n]) if on else 0.0 for n in C.DEV_SLOTS]


# --- one trajectory ------------------------------------------------------------------
def trajectory(theta, args, flux_cls):
    """Dense (t, p, e, Phi_phi, Phi_r) over T_YR or until the plunge."""
    traj = EMRIInspiral(func=flux_cls)
    kw = dict(Phi_phi0=theta["Phi_phi0"], Phi_r0=theta["Phi_r0"], T=T_YR, dt=10.0)
    m = (theta["m1"], theta["m2"], theta["a"], theta["p0"], theta["e0"], 1.0)
    t_end = traj(*m, *args, **kw)[0][-1]
    t, p, e, _, phi, _, phr = traj(*m, *args, **kw, upsample=True, new_t=np.linspace(0.0, t_end, N_T))[:7]
    return dict(t=t, p=p, e=e, Phi_phi=phi, Phi_r=phr)


def rhs_along(tr, theta, args, flux_cls):
    """ODE rhs (geometric units) at every sample: pdot, edot, Omega_phi, Omega_r."""
    flux = flux_cls()
    flux.add_fixed_parameters(theta["m1"], theta["m2"], theta["a"], additional_args=list(args))
    out = np.zeros((tr["t"].size, 4))
    for k, (p, e) in enumerate(zip(tr["p"], tr["e"])):
        yd = flux(np.array([p, e, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        out[k] = yd[[0, 1, 3, 5]]
    return out


def to_seconds(tr, rhs):
    """Geometric rhs -> per second: fitted from dp/dt (slow time, carries q) and dPhi/dt (fast time)."""
    t = tr["t"]
    inner = slice(5, -5)
    k_p = np.median(np.gradient(tr["p"], t)[inner] / rhs[inner, 0])
    k_phi = np.median(np.gradient(tr["Phi_phi"], t)[inner] / rhs[inner, 2])
    k_r = np.median(np.gradient(tr["Phi_r"], t)[inner] / rhs[inner, 3])
    return k_p, k_phi, k_r


def one_model(label, theta, flags, dev, flux_cls, chi2):
    args = add_args(chi2, flags, dev)
    tr = trajectory(theta, args, flux_cls)
    rhs = rhs_along(tr, theta, args, flux_cls)
    k_p, k_phi, k_r = to_seconds(tr, rhs)
    tr.update(pdot=rhs[:, 0] * k_p, edot=rhs[:, 1] * k_p, Omega_phi=rhs[:, 2] * k_phi, Omega_r=rhs[:, 3] * k_r)
    tr.update(Omegadot_phi=np.gradient(tr["Omega_phi"], tr["t"]), Omegadot_r=np.gradient(tr["Omega_r"], tr["t"]))
    dev_s = "none" if dev is None else ", ".join(f"{k}={v:+.5f}" for k, v in dev.items())
    print(f"{label:24s} p {tr['p'][0]:.5f} -> {tr['p'][-1]:.5f}  e {tr['e'][0]:.5f} -> {tr['e'][-1]:.5f}  "
          f"t_end {tr['t'][-1] / 3.15581e7:.4f} yr  dev: {dev_s}", flush=True)
    q = theta["m1"] * theta["m2"] / (theta["m1"] + theta["m2"]) ** 2
    print(f"{'':24s} 1/(M MTSUN) from dPhi_phi/dt {k_phi:.8e}, dPhi_r/dt {k_r:.8e}; "
          f"dp/dt factor / (q k_phi) - 1 = {k_p / (q * k_phi) - 1:.1e}", flush=True)
    return tr


def main():
    chi2 = float(C.signal_row(POINT)[C.COL["chi2"]])
    data = {}
    for label, (theta, flags, dev, flux_cls) in models().items():
        tr = one_model(label, theta, flags, dev, flux_cls, chi2)
        data.update({f"{label}|{k}": v for k, v in tr.items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, labels=np.array(list(models())), **data)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
