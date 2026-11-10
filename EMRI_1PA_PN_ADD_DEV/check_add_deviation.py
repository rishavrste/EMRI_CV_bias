"""Consistency checks of the PN + environmental deviation in SuperKludge_r (CPU, no waveform).

    pdot = pdot_GR (1 + A_p (p/10)^n_p) + q C_p (1-e^2)^1.5 (8 + 7 e^2) / p^3.5
    edot = edot_GR (1 + A_e (p/10)^n_e) + q e C_e (1-e^2)^1.5 (304 + 121 e^2) / p^4.5

additional_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA, deviation_included, C_p, C_e, A_p, A_e, n_p, n_e]
Run:  python check_add_deviation.py
"""
import json
import sys

import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SuperKludgeFlux
from few.utils.constants import YRSID_SI
from few.utils.geodesic import get_separatrix

import config as C

N_RANDOM = 300
T_CHECK = 2.5
RESULTS = []


def report(name, value, limit):
    ok = bool(value <= limit)
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (limit {limit:.0e})", flush=True)


def args_for(C_p=0.0, C_e=0.0, A_p=0.0, A_e=0.0, n_p=0.0, n_e=0.0, on=True, pa1=False):
    return [0.0, pa1, False, False, on, C_p, C_e, A_p, A_e, n_p, n_e]


def rhs(ode, m1, m2, a, p, e, args):
    ode.add_fixed_parameters(m1, m2, a, additional_args=args)
    return np.array(ode(np.array([p, e, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])))


# --- the model, written out separately from flux.py --------------------------
def pn_p(q, C_p, p, e):
    return q * C_p * (1 - e * e) ** 1.5 * (8 + 7 * e * e) / p ** 3.5


def pn_e(q, C_e, p, e):
    return q * C_e * e * (1 - e * e) ** 1.5 * (304 + 121 * e * e) / p ** 4.5


def env(A, n, p):
    return A * (p / 10.0) ** n


# --- right-hand side ----------------------------------------------------------
def draw_points(rng):
    for _ in range(N_RANDOM):
        a, e = rng.uniform(-0.9, 0.9), rng.uniform(0.02, 0.55)
        p = get_separatrix(abs(a), e, np.sign(a) or 1.0) + rng.uniform(0.5, 12.0)
        dev = dict(C_p=rng.uniform(-60, 60), C_e=rng.uniform(-60, 60), A_p=10 ** rng.uniform(-7, -2),
                   A_e=-10 ** rng.uniform(-7, -2), n_p=rng.uniform(-6, 6), n_e=rng.uniform(-6, 6))
        yield 10 ** rng.uniform(5, 6.5), 10 ** rng.uniform(0.5, 3.2), a, p, e, dev


def check_rhs(ode):
    rng = np.random.default_rng(1)
    worst = dict.fromkeys(("zero", "off", "formula", "a0", "c0", "cross", "other", "short", "pa1"), 0.0)

    def bump(key, value):
        worst[key] = max(worst[key], float(value))

    for m1, m2, a, p, e, d in draw_points(rng):
        q = m1 * m2 / (m1 + m2) ** 2
        R = lambda **kw: rhs(ode, m1, m2, a, p, e, args_for(**kw))      # noqa: E731
        off = R(on=False)
        full = R(**d)
        want_p = off[0] * (1 + env(d["A_p"], d["n_p"], p)) + pn_p(q, d["C_p"], p, e)
        want_e = off[1] * (1 + env(d["A_e"], d["n_e"], p)) + pn_e(q, d["C_e"], p, e)
        bump("zero", np.abs(R() - off).max())
        bump("off", np.abs(R(on=False, **d) - off).max())
        bump("formula", max(abs(full[0] - want_p) / abs(off[0]), abs(full[1] - want_e) / abs(off[1])))
        no_a = R(C_p=d["C_p"], C_e=d["C_e"], n_p=d["n_p"], n_e=d["n_e"])
        bump("a0", np.abs(no_a - R(C_p=d["C_p"], C_e=d["C_e"])).max())
        only_env = R(A_p=d["A_p"], A_e=d["A_e"], n_p=d["n_p"], n_e=d["n_e"])
        bump("c0", max(abs(only_env[0] / off[0] - 1 - env(d["A_p"], d["n_p"], p)),
                       abs(only_env[1] / off[1] - 1 - env(d["A_e"], d["n_e"], p))))
        only_p = R(C_p=d["C_p"], A_p=d["A_p"], n_p=d["n_p"])
        only_e = R(C_e=d["C_e"], A_e=d["A_e"], n_e=d["n_e"])
        bump("cross", max(abs(only_p[1] - off[1]), abs(only_e[0] - off[0])))
        bump("other", np.abs(np.delete(full - off, [0, 1])).max())
        short = rhs(ode, m1, m2, a, p, e, args_for(C_p=d["C_p"], C_e=d["C_e"])[:7])
        bump("short", np.abs(short - R(C_p=d["C_p"], C_e=d["C_e"])).max())
        pa_off, pa_on = R(on=False, pa1=True), R(pa1=True, **d)
        bump("pa1", abs((pa_on[0] - pa_off[0]) - (want_p - off[0])) / abs(off[0]))
    report("all six zero, deviation on, identical to deviation off", worst["zero"], 0.0)
    report("deviation_included False ignores all six", worst["off"], 0.0)
    report("rates vs separately written formula (rel. to vacuum rate)", worst["formula"], 1e-13)
    report("A = 0: n has no effect", worst["a0"], 0.0)
    report("C = 0: rate / vacuum - 1 equals A (p/10)^n", worst["c0"], 1e-12)
    report("C_p, A_p, n_p never touch edot; C_e, A_e, n_e never touch pdot", worst["cross"], 0.0)
    report("x, phases, frequencies, primary unchanged", worst["other"], 0.0)
    report("slots 7-10 absent gives A = 0", worst["short"], 0.0)
    report("environment scales the 0PA rate only; 1PA added on top unchanged", worst["pa1"], 1e-13)


def check_pivot_and_edot(ode):
    m1, m2, a = 1e6, 10.0, 0.5
    base = rhs(ode, m1, m2, a, 10.0, 0.2, args_for(5.0, 5.0, 1e-4, 1e-4, 0.0, 0.0))
    moved = rhs(ode, m1, m2, a, 10.0, 0.2, args_for(5.0, 5.0, 1e-4, 1e-4, 3.7, -2.1))
    report("at p = 10 the rates do not depend on n", np.abs(base - moved).max() / abs(base[0]), 1e-15)
    circ = rhs(ode, m1, m2, a, 10.0, 0.0, args_for(5.0, 5.0, 1e-3, 1e-3, 2.0, 2.0))
    report("edot vanishes at e = 0 with every deviation on", abs(circ[1]), 0.0)


# --- trajectories ------------------------------------------------------------------
def few_end(traj, b, args, T=T_CHECK, **kw):
    t, p, e, x, phi, th, r = traj(b["m1"], b["m2"], b["a"], b["p0"], b["e0"], 1.0, *args, T=T, dt=10.0, **kw)[:7]
    return dict(t=t[-1], p=p[-1], e=e[-1], phi=phi[-1], n=len(t))


def check_trajectories(traj):
    fits = json.loads(C.OLD_BEST_FITS.read_text())["pn"]
    for point in C.POINTS:
        b = fits[point]["theta_final"]
        pn = (b["C_p"], b["C_e"])
        print(f"-- idx{point}: a {b['a']:+.3f} e0 {b['e0']:.3f} p0 {b['p0']:.4f} C_p {pn[0]:.3f} C_e {pn[1]:.3f}",
              flush=True)
        ref = few_end(traj, b, args_for(*pn))
        same = few_end(traj, b, args_for(*pn, 0.0, 0.0, 2.0, -2.0))
        report("A = 0 trajectory identical to the PN model", abs(same["phi"] - ref["phi"]), 0.0)
        print(f"      PN model: p {b['p0']:.4f} -> {ref['p']:.4f}, e -> {ref['e']:.4f}", flush=True)
        for A in (1e-6, 1e-5, -1e-5):
            for n in (0.0, 4.0, -4.0):
                args = args_for(*pn, A, A, n, n)
                got = few_end(traj, b, args)
                back = few_end(traj, dict(b, p0=got["p"], e0=got["e"]), args, T=got["t"] / YRSID_SI,
                               integrate_backwards=True)
                print(f"      A {A:+.0e} n {n:+.0f}: dPhi_phi vs PN {got['phi'] - ref['phi']:+.3e} rad, "
                      f"back to p0 {abs(back['p'] - b['p0']):.1e}", flush=True)
                report(f"A {A:+.0e} n {n:+.0f}: forward then backward returns to p0", abs(back["p"] - b["p0"]), 1e-7)


def main():
    print("few:", sys.modules["few"].__file__, flush=True)
    traj = EMRIInspiral(func=SuperKludgeFlux)
    check_rhs(traj.func)
    check_pivot_and_edot(traj.func)
    check_trajectories(traj)
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} passed", flush=True)


if __name__ == "__main__":
    main()
