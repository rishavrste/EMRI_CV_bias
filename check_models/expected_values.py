"""Expected deviation coefficients of each family, and the dephasing they leave at the injection.

1. Along the 1PA trajectory at the injection (N_T samples uniform in t), fit each family's terms to the exact 1PA
   rates of SKequatorialfluxes (pdot1PA, edot1PA, OmegaPhi1PA, Omegar1PA). Shapes: s_p, s_e, s_Om (flux.py).
     PN_freq   : C_p from pdot, C_e from edot; Omega tied to -(15/56) C_p           (linear least squares)
     A         : C_p, C_e as above; C_Om from Omega_phi and Omega_r together          (linear)
     B         : C_p, C_e; C_Omphi, C_Omr separately                                  (linear)
     power_law : per channel C (10/p)^n, nonlinear least squares from the B value and n = 0
   Uniform-in-t samples weight each rate as it enters the accumulated phase.
2. One trajectory per family at the injection with these coefficients: dephasing against 1PA.
    -> results/expected_idx{p}.json
Run:  python expected_values.py --point 0 > logs/expected_idx0.log
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from families import POINTS, SIGNAL_FLAGS, TEMPLATE_FLAGS, add_args, branch_check, injection, phases

HERE = Path(__file__).resolve().parent
YR = 3.15581e7
N_T = 2000
FAMILY_ORDER = ("0PA", "PN_freq", "A", "B", "power_law")


# --- targets and shapes along the 1PA trajectory ----------------------------------------
def track(inj):
    """t, p, e of the 1PA trajectory at the injection."""
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode.flux import SuperKludgeFlux
    traj = EMRIInspiral(func=SuperKludgeFlux)
    m = (inj["m1"], inj["m2"], inj["a"], inj["p0"], inj["e0"], 1.0)
    args = add_args(inj["chi2"], SIGNAL_FLAGS, "0PA", {})
    kw = dict(Phi_phi0=inj["Phi_phi0"], Phi_r0=inj["Phi_r0"], T=inj["T"], dt=10.0)
    t_end = traj(*m, *args, **kw)[0][-1]
    t, p, e = traj(*m, *args, **kw, upsample=True, new_t=np.linspace(0, t_end, N_T))[:3]
    return t, p, e


def targets_and_shapes(inj, p, e):
    from few.trajectory.ode.SKequatorialfluxes import OmegaPhi1PA, Omegar1PA, edot1PA, pdot1PA
    a, chi2 = inj["a"], inj["chi2"]
    f = (1 - e * e) ** 1.5
    shapes = dict(p=f * (8 + 7 * e * e) / p ** 3.5, e=e * f * (304 + 121 * e * e) / p ** 4.5, Om=f / p ** 2)
    target = dict(p=pdot1PA(a, p, e, chi2), e=edot1PA(a, p, e, chi2),
                  phi=OmegaPhi1PA(a, p, e, chi2), r=Omegar1PA(a, p, e, chi2))
    return target, shapes


# --- fits of the rates --------------------------------------------------------------------
def lin(y, s):
    """Best C in y = C s, and the rms of the remainder relative to rms(y)."""
    c = float(np.dot(y, s) / np.dot(s, s))
    return c, float(np.sqrt(np.mean((y - c * s) ** 2) / np.mean(y ** 2)))


def power(y, s, p, c0):
    """Best (C, n) in y = C s (10/p)^n."""
    sol = least_squares(lambda x: (y - x[0] * s * (10 / p) ** x[1]) / np.abs(y).max(), [c0, 0.0], method="lm")
    c, n = sol.x
    return float(c), float(n), float(np.sqrt(np.mean((y - c * s * (10 / p) ** n) ** 2) / np.mean(y ** 2)))


def expected_coeffs(target, shapes, p):
    """family -> (coefficients, relative rms remainder per channel p, e, phi, r)."""
    cp, rp = lin(target["p"], shapes["p"])
    ce, re = lin(target["e"], shapes["e"])
    om = shapes["Om"]
    tied = -(15 / 56) * cp
    r_tied = [float(np.sqrt(np.mean((target[k] - tied * om) ** 2) / np.mean(target[k] ** 2))) for k in ("phi", "r")]
    c_om, _ = lin(np.concatenate([target["phi"], target["r"]]), np.concatenate([om, om]))
    r_a = [float(np.sqrt(np.mean((target[k] - c_om * om) ** 2) / np.mean(target[k] ** 2))) for k in ("phi", "r")]
    c_phi, r_phi = lin(target["phi"], om)
    c_r, r_r = lin(target["r"], om)
    pl = {k: power(target[k], shapes[sk], p, c0) for k, sk, c0 in
          (("p", "p", cp), ("e", "e", ce), ("phi", "Om", c_phi), ("r", "Om", c_r))}
    return {
        "0PA": ({}, [1.0, 1.0, 1.0, 1.0]),
        "PN_freq": (dict(C_p=cp, C_e=ce), [rp, re, *r_tied]),
        "A": (dict(C_p=cp, C_e=ce, C_Om=c_om), [rp, re, *r_a]),
        "B": (dict(C_p=cp, C_e=ce, C_Omphi=c_phi, C_Omr=c_r), [rp, re, r_phi, r_r]),
        "power_law": (dict(C_p=pl["p"][0], C_e=pl["e"][0], C_Omphi=pl["phi"][0], C_Omr=pl["r"][0],
                           n_p=pl["p"][1], n_e=pl["e"][1], n_Omphi=pl["phi"][1], n_Omr=pl["r"][1]),
                      [pl[k][2] for k in ("p", "e", "phi", "r")]),
    }


# --- dephasing at the injection ------------------------------------------------------------
def dephasing(inj, fams):
    t = np.linspace(0.0, inj["T"] * YR * 0.999, 400)
    sig = phases(inj, add_args(inj["chi2"], SIGNAL_FLAGS, "0PA", {}), t, inj["T"])
    out = {}
    for fam in FAMILY_ORDER:
        mod = phases(inj, add_args(inj["chi2"], TEMPLATE_FLAGS, fam, fams[fam][0]), t, inj["T"])
        if mod is None:
            out[fam] = None
            continue
        dphi, dr = sig[0] - mod[0], sig[1] - mod[1]
        out[fam] = dict(rms_phi=float(np.sqrt(np.mean(dphi ** 2))), rms_r=float(np.sqrt(np.mean(dr ** 2))),
                        end_phi=float(dphi[-1]), end_r=float(dr[-1]),
                        max_phi=float(np.abs(dphi).max()), max_r=float(np.abs(dr).max()))
    return out


def report(point, fams, deph):
    print(f"\nidx{point}: expected coefficients (1PA rate fit along the injected 1PA trajectory)")
    for fam in FAMILY_ORDER[1:]:
        print(f"  {fam:10s} " + ", ".join(f"{k}={v:+.5g}" for k, v in fams[fam][0].items()))
    print(f"\nidx{point}: rate remainder rms(1PA - family)/rms(1PA), and dephasing at the injection [rad]")
    print(f"  {'family':10s} | pdot  | edot  | Om_phi | Om_r  | rms dPhi_phi | rms dPhi_r | end dPhi_phi | end dPhi_r")
    for fam in FAMILY_ORDER:
        r, d = fams[fam][1], deph[fam]
        dd = (f"{d['rms_phi']:12.4f} | {d['rms_r']:10.4f} | {d['end_phi']:+12.4f} | {d['end_r']:+10.4f}"
              if d else "plunged before T")
        print(f"  {fam:10s} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:6.3f} | {r[3]:.3f} | " + dd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", required=True, choices=POINTS)
    point = ap.parse_args().point
    branch_check()
    inj = injection(point)
    t, p, e = track(inj)
    target, shapes = targets_and_shapes(inj, p, e)
    fams = expected_coeffs(target, shapes, p)
    deph = dephasing(inj, fams)
    report(point, fams, deph)
    path = HERE / "results" / f"expected_idx{point}.json"
    path.write_text(json.dumps(dict(point=point, injection=inj, p_range=[float(p[0]), float(p[-1])],
                                    families={f: dict(coeffs=fams[f][0], rate_remainder=fams[f][1],
                                                      dephasing=deph[f]) for f in FAMILY_ORDER}), indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
