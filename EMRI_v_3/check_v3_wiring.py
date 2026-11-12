"""Checks of the deviation in flux.py on SuperKludge_r branch PN_v_3, now REL5_1 (it was V3 for job 713215).

    1. flux.py rates : deviation_terms(p, rates_0pa) equals eta X_0PA (c_X v + K v^4) over a (p, e) grid
    2. GR limit      : deviation_included = 1 with all coefficients 0 gives the 0PA trajectory exactly
    3. flux vs class : the REL5_1 trajectory through flux.py equals forms_v3's own DevFlux with the same x
    4. 0PA shape     : the forms_v3 REL DevFlux adds eta * 0PA rate * R, checked at one point against rates_0pa
    5. f = 0 basis   : forms_v3.build_basis(0) equals check_models/fit_forms.build_basis()
Run:  python check_v3_wiring.py > logs/check_v3_wiring.log
"""
import numpy as np

import forms_v3 as V

X_TEST = np.array([0.330241, 11.8206, 1.93908, 1.01968, 62.8424])   # c_p, c_e, c_Omphi, c_Omr, K (idx0 free fit)
INJ = dict(m1=1e6, m2=10.0, a=-0.9, p0=13.0673756813688, e0=0.1, Phi_phi0=0.1, Phi_r0=0.3, chi2=0.95, T=2.5)
RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def flux(dev, c=()):
    from few.trajectory.ode.flux import SuperKludgeFlux
    fl = SuperKludgeFlux()
    fl.add_fixed_parameters(INJ["m1"], INJ["m2"], INJ["a"], [INJ["chi2"], 0, 0, 0, dev, *c])
    return fl


def check_rates(forms):
    fl, worst = flux(1, X_TEST), 0.0
    for p in np.linspace(5, 31, 14):
        for e in np.linspace(0.05, 0.5, 6):
            base = V.rates_0pa(INJ["m1"], INJ["m2"], INJ["a"], INJ["chi2"], np.array([p]), np.array([e]))[:, 0]
            ours = fl.deviation_terms(p, base)
            ref = fl.massratio * base * np.einsum("k,kc->c", X_TEST, V.design(forms["REL5_1"], np.array([p ** -0.5]))[..., 0])
            worst = max(worst, np.max(np.abs(ours - ref) / np.abs(ref)))
    report("flux.py rates", worst < 1e-12, f"max relative difference over 84 (p, e) = {worst:.2e}")


def check_trajectories(forms):
    from few.trajectory.ode.flux import SuperKludgeFlux
    t = np.linspace(0, 2.5 * 3.15581e7, 200)
    args = lambda dev, c=(): [INJ["chi2"], 0, 0, 0, dev, *c, *[0.0] * (5 - len(c))]
    off = V.phases(INJ, SuperKludgeFlux, args(0), t)
    on0 = V.phases(INJ, SuperKludgeFlux, args(1), t)
    d = max(abs(off[0][-1] - on0[0][-1]), abs(off[1][-1] - on0[1][-1]))
    report("GR limit", d == 0.0, f"end |dPhi| with all coefficients 0 = {d:.2e} rad")
    via_flux = V.phases(INJ, SuperKludgeFlux, args(1, X_TEST), t)
    via_class = V.phases(INJ, V.make_dev_flux(forms["REL5_1"], X_TEST), args(0), t)
    d = max(abs(via_flux[0][-1] - via_class[0][-1]), abs(via_flux[1][-1] - via_class[1][-1]))
    report("flux vs class", d < 1e-6, f"end |dPhi| = {d:.2e} rad (deviation moves Phi_phi by "
                                      f"{via_flux[0][-1] - off[0][-1]:+.3f} rad)")


def check_rel(forms):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cls = V.make_dev_flux(forms["REL5_1"], x)
    fl = cls()
    fl.add_fixed_parameters(INJ["m1"], INJ["m2"], INJ["a"], [INJ["chi2"], 0, 0, 0, 0])
    p, e = 10.0, 0.2
    y = np.array([p, e, 1.0, 0.0, 0.0])
    got = fl(y).copy()[list(V.SLOTS)]
    base = V.rates_0pa(INJ["m1"], INJ["m2"], INJ["a"], INJ["chi2"], np.array([p]), np.array([e]))[:, 0]
    want = base * (1 + fl.massratio * np.einsum("k,kc->c", x, V.design(forms["REL5_1"], np.array([p ** -0.5]))[..., 0]))
    err = np.max(np.abs(got - want) / np.abs(want))
    report("0PA-relative", err < 1e-12, f"max relative difference = {err:.2e}")


def check_basis():
    import fit_forms
    ref, _, _ = fit_forms.build_basis()
    ours, _ = V.build_basis(0.0)
    err = np.max(np.abs(ours[:4] - ref[:4]))
    report("f = 0 basis", err < 1e-10, f"max |forms_v3 - fit_forms| = {err:.2e}")


def main():
    V.branch_check()
    forms = V.make_forms({0.0: V.build_basis(0.0)[0]})
    check_rates(forms)
    check_trajectories(forms)
    check_rel(forms)
    check_basis()
    print(f"{sum(RESULTS)}/{len(RESULTS)} pass", flush=True)


if __name__ == "__main__":
    main()
