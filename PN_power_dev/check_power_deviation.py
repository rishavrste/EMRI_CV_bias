"""Consistency checks of the power-law PN deviation in SuperKludge_r (CPU, no waveform).

    pdot += q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5 * (p/10)^n_p
    edot += q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5 * (p/10)^n_e     q = m1 m2 / (m1 + m2)^2

additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, n_p, n_e]

The reference for the old model is flux_head_reference.py, a copy of the committed hybrid flux.py.
Run:  python check_power_deviation.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SuperKludgeFlux
from few.utils.constants import MTSUN_SI, YRSID_SI
from few.utils.geodesic import get_separatrix

HERE = Path(__file__).resolve().parent
POINTS = HERE.parent / "EMRI_1PA_PN_DEV" / "results" / "optimized_points.json"
N_RANDOM = 300
CHECK_POINTS = (4, 24, 0)
T_CHECK = 2.0
RESULTS = []


def report(name, value, limit):
    ok = bool(value <= limit)
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (limit {limit:.0e})", flush=True)


def load_head_class():
    spec = importlib.util.spec_from_file_location("few.trajectory.ode.flux_head",
                                                  HERE / "flux_head_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SuperKludgeFlux


def args_for(C_p=0.0, C_e=0.0, s_p=0.0, s_e=0.0, on=True, chi2=0.0, pa1=False):
    """additional_args; s_p, s_e fill slots 7, 8 (n_p, n_e, or del_0_p, del_0_e in the old code)."""
    return [chi2, pa1, False, False, on, C_p, C_e, s_p, s_e]


def rhs(ode, m1, m2, a, p, e, args):
    ode.add_fixed_parameters(m1, m2, a, additional_args=args)
    return np.array(ode(np.array([p, e, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])))


def term_p(q, C, n, p, e):
    """Written out separately from flux.py."""
    return q * C * (1 - e * e) ** 1.5 * (8 + 7 * e * e) * p ** (n - 3.5) * 10.0 ** (-n)


def term_e(q, C, n, p, e):
    return q * C * e * (1 - e * e) ** 1.5 * (304 + 121 * e * e) * p ** (n - 4.5) * 10.0 ** (-n)


def draw_points(rng):
    for _ in range(N_RANDOM):
        a, e = rng.uniform(-0.9, 0.9), rng.uniform(0.02, 0.55)
        p = get_separatrix(abs(a), e, np.sign(a) or 1.0) + rng.uniform(0.5, 12.0)
        yield (10 ** rng.uniform(5, 6.5), 10 ** rng.uniform(0.5, 3.2), a, p, e,
               rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(-2, 2), rng.uniform(-2, 2))


def check_rhs(new, old):
    rng = np.random.default_rng(1)
    worst = dict.fromkeys(("c0", "n0", "formula", "off", "short", "cross", "other", "pa1"), 0.0)
    for m1, m2, a, p, e, Cp, Ce, np_, ne in draw_points(rng):
        q = m1 * m2 / (m1 + m2) ** 2
        off = rhs(new, m1, m2, a, p, e, args_for(on=False))
        vac_head = rhs(old, m1, m2, a, p, e, args_for(on=False))
        worst["c0"] = max(worst["c0"], np.abs(rhs(new, m1, m2, a, p, e, args_for(0, 0, np_, ne)) - off).max())
        worst["n0"] = max(worst["n0"], np.abs(rhs(new, m1, m2, a, p, e, args_for(Cp, Ce))
                                              - rhs(old, m1, m2, a, p, e, args_for(Cp, Ce))).max(),
                          np.abs(off - vac_head).max())
        full = rhs(new, m1, m2, a, p, e, args_for(Cp, Ce, np_, ne))
        tp, te = term_p(q, Cp, np_, p, e), term_e(q, Ce, ne, p, e)
        worst["formula"] = max(worst["formula"], abs((full[0] - off[0]) - tp) / abs(off[0]),
                               abs((full[1] - off[1]) - te) / abs(off[1]))
        worst["off"] = max(worst["off"], np.abs(rhs(new, m1, m2, a, p, e, args_for(Cp, Ce, np_, ne, on=False)) - off).max())
        worst["short"] = max(worst["short"], np.abs(rhs(new, m1, m2, a, p, e, args_for(Cp, Ce)[:7])
                                                    - rhs(new, m1, m2, a, p, e, args_for(Cp, Ce))).max())
        only_p = rhs(new, m1, m2, a, p, e, args_for(Cp, 0.0, np_, ne))
        only_e = rhs(new, m1, m2, a, p, e, args_for(0.0, Ce, np_, ne))
        worst["cross"] = max(worst["cross"], abs(only_p[1] - off[1]), abs(only_e[0] - off[0]))
        worst["other"] = max(worst["other"], np.abs(np.delete(full - off, [0, 1])).max())
        pa_off = rhs(new, m1, m2, a, p, e, args_for(on=False, pa1=True))
        pa_on = rhs(new, m1, m2, a, p, e, args_for(Cp, Ce, np_, ne, pa1=True))
        worst["pa1"] = max(worst["pa1"], abs((pa_on[0] - pa_off[0]) - tp) / abs(off[0]))
    report("C = 0, any n, identical to deviation off", worst["c0"], 0.0)
    report("n = 0 identical to committed hybrid C_p, C_e", worst["n0"], 0.0)
    report("added term vs separately written formula (rel. to vacuum rate)", worst["formula"], 1e-13)
    report("deviation_included False ignores C, n", worst["off"], 0.0)
    report("slots 7, 8 absent gives n = 0", worst["short"], 0.0)
    report("C_p, n_p never touch edot; C_e, n_e never touch pdot", worst["cross"], 0.0)
    report("x, phases, frequencies, primary unchanged", worst["other"], 0.0)
    report("deviation adds on top of 1PA unchanged", worst["pa1"], 1e-13)


def check_pivot_and_dn(new):
    m1, m2, a, e, C = 1e6, 10.0, 0.5, 0.2, 30.0
    base = rhs(new, m1, m2, a, 10.0, e, args_for(C, C, 0.0, 0.0))
    moved = rhs(new, m1, m2, a, 10.0, e, args_for(C, C, 1.7, -1.3))
    report("at p = 10 the term does not depend on n", np.abs(base - moved).max() / abs(base[0]), 1e-15)
    off = rhs(new, m1, m2, a, 12.0, e, args_for(on=False))[0]
    h = 0.2                       # the term is 1e-5 of pdot: a small step only measures roundoff
    up, dn = (rhs(new, m1, m2, a, 12.0, e, args_for(C, 0.0, s * h, 0.0))[0] for s in (1, -1))
    q = m1 * m2 / (m1 + m2) ** 2
    exact = term_p(q, C, 0.0, 12.0, e) * np.sinh(h * np.log(1.2)) / h    # central difference of (p/10)^n
    report("central difference in n matches term * sinh(h ln(p/10)) / h", abs((up - dn) / (2 * h) / exact - 1), 1e-9)
    print(f"      deviation/vacuum pdot at p=12: {term_p(q, C, 0.0, 12.0, e) / off:.3e}", flush=True)


def own_phases(ode, m1, m2, a, p0, e0, T, dev, t_eval=None):
    """scipy integration of the vacuum RHS plus the term written above, same scaled time as FEW."""
    C_p, C_e, n_p, n_e = dev
    q = m1 * m2 / (m1 + m2) ** 2
    ode.add_fixed_parameters(m1, m2, a, additional_args=args_for(on=False))
    tau = T * YRSID_SI / ((m1 + m2) * MTSUN_SI) * q

    def f(_, y):
        p, e = y[:2]
        v = ode(np.array([p, e, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        return [v[0] + term_p(q, C_p, n_p, p, e), v[1] + term_e(q, C_e, n_e, p, e), v[3], v[5]]

    sol = solve_ivp(f, (0.0, tau), [p0, e0, 0.0, 0.0], method="DOP853", rtol=1e-11, atol=1e-12,
                    t_eval=t_eval)
    return sol.y[:, -1] if t_eval is None else sol.y, q


def few_end(traj, m1, m2, a, p0, e0, T, args):
    t, p, e, x, phi, th, r = traj(m1, m2, a, p0, e0, 1.0, *args, T=T, dt=10.0)[:7]
    return dict(t=t[-1], p=p[-1], e=e[-1], phi=phi[-1], r=r[-1])


def load_cases():
    """(label, 0PA + PN best fit) of the check points."""
    points = json.loads(POINTS.read_text())["points"]
    for idx in CHECK_POINTS:
        yield idx, points[idx]["best_fit"]["0PA+PN"]["params"]


def check_trajectories(new_traj, old_traj):
    for label, b in load_cases():
        m1, m2, a, p0, e0, C_p, C_e = (b[k] for k in ("m1", "m2", "a", "p0", "e0", "C_p", "C_e"))
        print(f"-- idx{label}: a {a}, e0 {e0}, p0 {p0:.4f}, C_p {C_p:.3f}, C_e {C_e:.3f}", flush=True)
        old = few_end(old_traj, m1, m2, a, p0, e0, T_CHECK, args_for(C_p, C_e))
        new = few_end(new_traj, m1, m2, a, p0, e0, T_CHECK, args_for(C_p, C_e))
        report("n = 0 trajectory identical to committed hybrid", abs(new["phi"] - old["phi"]), 0.0)
        vac = few_end(new_traj, m1, m2, a, p0, e0, T_CHECK, args_for(on=False))
        c0 = few_end(new_traj, m1, m2, a, p0, e0, T_CHECK, args_for(0.0, 0.0, 0.8, -0.8))
        report("C = 0, n != 0 trajectory identical to vacuum", abs(c0["phi"] - vac["phi"]), 0.0)
        for n in (-1.0, -0.5, 0.5, 1.0):
            dev = (C_p, C_e, n, n)
            got = few_end(new_traj, m1, m2, a, p0, e0, T_CHECK, args_for(*dev))
            own, q = own_phases(new_traj.func, m1, m2, a, p0, e0, T_CHECK, dev)
            report(f"n = {n:+.1f}: Phi_phi FEW vs scipy [rad] (dephasing from vacuum "
                   f"{got['phi'] - vac['phi']:+.2f})", abs(got["phi"] - own[2] / q), 5e-2)
            back = new_traj(m1, m2, a, got["p"], got["e"], 1.0, *args_for(*dev), T=T_CHECK, dt=10.0,
                            integrate_backwards=True)
            report(f"n = {n:+.1f}: forward then backward returns to p0", abs(back[1][-1] - p0), 1e-7)


def main():
    print("few:", sys.modules["few"].__file__, flush=True)
    head_class = load_head_class()
    new_traj, old_traj = EMRIInspiral(func=SuperKludgeFlux), EMRIInspiral(func=head_class)
    check_rhs(new_traj.func, old_traj.func)
    check_pivot_and_dn(new_traj.func)
    check_trajectories(new_traj, old_traj)
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} passed", flush=True)


if __name__ == "__main__":
    main()
