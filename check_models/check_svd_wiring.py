"""Check the SVD deviation in SuperKludge_r flux.py (branch PN_freq_evol) against the tested fit_forms.py version.

additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_1, C_2, C_3, C_4]
    1. basis   : flux.py SVD_BASIS equals fit_forms.build_basis() (first four components)
    2. rates   : deviation_terms(p, e) equals eta shape_X sum_j C_j b_jX(v) from fit_forms, over a (p, e) grid
    3. GR      : deviation_included = 1 with C = 0 gives the same trajectory as deviation_included = 0
    4. traject.: the flux.py trajectory with the EMRI idx0 SVD_4 coefficients equals fit_forms' DevFlux trajectory
Run:  python check_svd_wiring.py > logs/check_svd_wiring.log
"""
import numpy as np

import fit_forms as F
from families import branch_check

C_TEST = np.array([-26.764032978015077, -4.039489321421882, 0.8984620392021068, 0.4335202146255203])  # EMRI idx0
INJ = dict(m1=1e6, m2=10.0, a=-0.9, p0=13.0673756813688, e0=0.1, Phi_phi0=0.1, Phi_r0=0.3, chi2=0.95, T=2.5)
RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def flux_object(c):
    from few.trajectory.ode.flux import SuperKludgeFlux
    fl = SuperKludgeFlux()
    fl.add_fixed_parameters(INJ["m1"], INJ["m2"], INJ["a"], [INJ["chi2"], 0, 0, 0, 1, *c])
    return fl


def check_basis(beta):
    from few.trajectory.ode.flux import SuperKludgeFlux
    err = np.max(np.abs(SuperKludgeFlux.SVD_BASIS - beta[:4]))
    report("basis", err < 1e-12, f"max |flux.py - fit_forms| = {err:.2e}")


def check_rates(beta):
    fl = flux_object(C_TEST)
    worst = 0.0
    for p in np.linspace(5, 31, 14):
        for e in np.linspace(0.05, 0.5, 6):
            ours = np.asarray(fl.deviation_terms(p, e))
            sh = F.shapes(p, e)
            ref = fl.massratio * np.array([sh[ch] for ch in F.CH]) * F.form_R("SVD_4", C_TEST, np.array([p]), beta)[:, 0]
            worst = max(worst, np.max(np.abs(ours - ref) / np.abs(ref)))
    report("rates", worst < 1e-12, f"max relative difference over 84 (p, e) = {worst:.2e}")


def trajectory(flux_cls, args):
    from few.trajectory.inspiral import EMRIInspiral
    traj = EMRIInspiral(func=flux_cls)
    out = traj(INJ["m1"], INJ["m2"], INJ["a"], INJ["p0"], INJ["e0"], 1.0, *args,
               Phi_phi0=INJ["Phi_phi0"], Phi_r0=INJ["Phi_r0"], T=INJ["T"], dt=10.0)
    return np.array(out[0]), np.array(out[4]), np.array(out[6])


def end_phase_diff(a, b):
    return max(abs(a[1][-1] - b[1][-1]), abs(a[2][-1] - b[2][-1]))


def check_gr():
    from few.trajectory.ode.flux import SuperKludgeFlux
    off = trajectory(SuperKludgeFlux, [INJ["chi2"], 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0])
    on0 = trajectory(SuperKludgeFlux, [INJ["chi2"], 0, 0, 0, 1, 0.0, 0.0, 0.0, 0.0])
    d = end_phase_diff(off, on0)
    report("GR limit", d == 0.0, f"end |dPhi| with C = 0 = {d:.2e} rad")


def check_trajectory(beta):
    from few.trajectory.ode.flux import SuperKludgeFlux
    ours = trajectory(SuperKludgeFlux, [INJ["chi2"], 0, 0, 0, 1, *C_TEST])
    ref = trajectory(F.make_dev_flux("SVD_4", C_TEST, beta), [INJ["chi2"], 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0])
    zero = trajectory(SuperKludgeFlux, [INJ["chi2"], 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0])
    d = end_phase_diff(ours, ref)
    report("trajectory", d < 1e-6, f"end |dPhi| flux.py vs fit_forms = {d:.2e} rad "
                                   f"(deviation itself moves Phi_phi by {ours[1][-1] - zero[1][-1]:+.3f} rad)")


def main():
    branch_check()
    beta, _, _ = F.build_basis()
    check_basis(beta)
    check_rates(beta)
    check_gr()
    check_trajectory(beta)
    print(f"{sum(RESULTS)}/{len(RESULTS)} pass", flush=True)


if __name__ == "__main__":
    main()
