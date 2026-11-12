"""CPU proxy for the absolute bias of the forms_v3 deviations (SuperKludge_r branch PN_v_3).

Residual r = [Phi_phi^1PA - Phi_phi, Phi_r^1PA - Phi_r] at N_T times over the signal's span, against the 1PA
trajectory at the injection (same as check_models/phase_fit.py):
    (i)  at_injection : physical parameters at the injection, only the deviation coefficients fitted, seeded from the
                        expected coefficients (forms_v3.fit, the 1PA rate fit along the track).
    (ii) free         : coefficients and m1, m2, a, p0, e0, Phi_phi0, Phi_r0 fitted, seeded from (i).
                        The parameter shifts are the bias proxy (phases only, no noise weighting: a ranking).
Forms: 0PA (no coefficients), B (4), REL5_1 (5), REL5_1S and REL5_1S2 (5, shifted by v0 = p0^-1/2 of the trial
point, so they move with the fitted p0). The coefficients travel in the PN_v_3 deviation slots
additional_args[5:10]; slot_flux() replaces flux.py's own deviation by the form's.
    -> results/phase_fit_{grid}_idx{p}_{form}.json (rewritten after each fit)
Run:  python phase_fit_v3.py --grid EMRI --point 4 --form REL5_1 > logs/phase_fit_EMRI_idx4_REL5_1.log
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares

import forms_v3 as V

FORMS = ("0PA", "B", "REL5_1", "REL5_1S", "REL5_1S2")
PHYS_FREE = ["m1", "m2", "a", "p0", "e0", "Phi_phi0", "Phi_r0"]
N_T = 400
T_FRACTION = 0.99                  # compare up to this fraction of the signal's end (keeps clear of the plunge)
MAX_NFEV = 60                      # per least_squares call, in units of (n_params + 1) evaluations
FAIL = 1e3                         # residual when a trial trajectory ends before the last comparison time
STEP = dict(m1=1e-8, m2=1e-7, a=1e-6, p0=1e-7, e0=1e-6, Phi_phi0=1e-5, Phi_r0=1e-5)   # relative, scipy convention
COEFF_STEP = 1e-4
N_SLOTS = 5


# --- template ---------------------------------------------------------------------------------
def slot_flux(form):
    """SuperKludgeFlux whose deviation is `form`, coefficients read from additional_args[5:5+n] (dev_c)."""
    from few.trajectory.ode.flux import SuperKludgeFlux
    kind, n = form[0], len(form[1])

    class SlotFlux(SuperKludgeFlux):
        V0 = None                                             # v0 = p0^-1/2 of the current trial (shifted forms)

        def deviation_terms(self, *args):                    # switch off flux.py's own deviation
            return np.zeros(4)

        def modify_rhs(self, ydot, y, **kwargs):
            base = ydot[list(V.SLOTS)].copy()                 # 0PA rates (evolve_1PA = 0 for the template)
            super().modify_rhs(ydot, y, **kwargs)
            if not self.deviation_included:
                return
            p, e = float(y[0]), float(y[1])
            S = V.lead_shapes(p, e) if kind == "lead" else base
            g = V.design(form, np.array([p ** -0.5]), SlotFlux.V0)[..., 0]    # (n, 4)
            ydot[list(V.SLOTS)] += self.massratio * S * (self.dev_c[:n] @ g)

    SlotFlux.__name__ = "SuperKludgeFlux"     # EMRIInspiral keys its SuperKludge set-up on this name
    return SlotFlux


class Problem:
    """Phase residual of one form against the 1PA signal of one grid point."""

    def __init__(self, grid, point, form_name, forms):
        from few.trajectory.inspiral import EMRIInspiral
        from few.trajectory.ode.flux import SuperKludgeFlux
        self.inj = V.injection(grid, point)
        self.form = forms.get(form_name)
        self.n_c = 0 if self.form is None else len(self.form[1])
        self.flux = SuperKludgeFlux if self.form is None else slot_flux(self.form)
        self.traj = EMRIInspiral(func=self.flux)
        sig_traj = EMRIInspiral(func=SuperKludgeFlux)
        args = [self.inj["chi2"], 1, 0, 0, 0] + [0.0] * N_SLOTS
        t_end = sig_traj(*self.masses(self.inj), *args, **self.kw(self.inj))[0][-1]
        self.t = np.linspace(0.0, T_FRACTION * t_end, N_T)
        out = sig_traj(*self.masses(self.inj), *args, **self.kw(self.inj), upsample=True, new_t=self.t)
        self.sig = np.concatenate([out[4], out[6]])
        self.n_eval = 0

    @staticmethod
    def masses(theta):
        return theta["m1"], theta["m2"], theta["a"], theta["p0"], theta["e0"], 1.0

    def kw(self, theta):
        return dict(Phi_phi0=theta["Phi_phi0"], Phi_r0=theta["Phi_r0"], T=self.inj["T"], dt=10.0)

    def residual(self, theta, coeffs):
        self.n_eval += 1
        c = list(coeffs) + [0.0] * (N_SLOTS - len(coeffs))
        args = [self.inj["chi2"], 0, 0, 0, int(self.n_c > 0), *c]
        self.flux.V0 = theta["p0"] ** -0.5
        try:
            t_end = self.traj(*self.masses(theta), *args, **self.kw(theta))[0][-1]
            if t_end < self.t[-1] * (1 - 1e-9):
                return np.full(self.sig.size, FAIL)
            out = self.traj(*self.masses(theta), *args, **self.kw(theta), upsample=True, new_t=self.t)
        except Exception:                                     # noqa: BLE001  invalid trial point
            return np.full(self.sig.size, FAIL)
        return self.sig - np.concatenate([out[4], out[6]])


# --- fits -------------------------------------------------------------------------------------
def summary(r):
    phi, rr = np.split(r, 2)
    return dict(rms_phi=float(np.sqrt(np.mean(phi ** 2))), rms_r=float(np.sqrt(np.mean(rr ** 2))),
                end_phi=float(phi[-1]), end_r=float(rr[-1]))


def fit(P, theta0, c0, free_phys):
    pnames = PHYS_FREE if free_phys else []
    x0 = np.array([theta0[n] for n in pnames] + list(c0))
    steps = np.array([STEP[n] for n in pnames] + [COEFF_STEP] * len(c0))

    def split(x):
        return dict(theta0, **dict(zip(pnames, x[:len(pnames)]))), x[len(pnames):]

    def res(x):
        return P.residual(*split(x))

    start = summary(res(x0))
    if x0.size == 0:
        return dict(start=start, final=start, nfev=1, coeffs=[], shift={}), theta0, np.array([])
    t0 = time.time()
    sol = least_squares(res, x0, method="trf", x_scale="jac", diff_step=steps,
                        max_nfev=MAX_NFEV * (x0.size + 1), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    theta, c = split(sol.x)
    return dict(start=start, final=summary(sol.fun), nfev=int(sol.nfev), status=sol.message,
                seconds=time.time() - t0, coeffs=list(map(float, c)), theta=theta,
                shift={n: float(theta[n] - P.inj[n]) for n in pnames}), theta, c


def log_fit(tag, out):
    s, f = out["start"], out["final"]
    print(f"[{tag}] rms dPhi_phi {s['rms_phi']:.4g} -> {f['rms_phi']:.4g}  rms dPhi_r {s['rms_r']:.4g} -> "
          f"{f['rms_r']:.4g}  end ({f['end_phi']:+.4g}, {f['end_r']:+.4g}) rad  nfev={out['nfev']}  "
          f"{out.get('seconds', 0):.0f} s", flush=True)
    if out["coeffs"]:
        print("    coeffs: " + ", ".join(f"{v:+.6g}" for v in out["coeffs"]), flush=True)
    if out["shift"]:
        print("    shift : " + ", ".join(f"{k}={v:+.4g}" for k, v in out["shift"].items()), flush=True)


def expected_coeffs(grid, point, form):
    """The form's coefficients from the 1PA rate fit along the injected track (forms_v3.fit)."""
    if form is None:
        return np.array([])
    tr = V.load_track(grid, point)
    v, S, target = V.track_inputs(tr, need_0pa=form[0] == "0pa")
    return V.fit(form, S[form[0]], v, target)[0]


def run(grid, point, form_name):
    forms = V.make_forms({})
    P = Problem(grid, point, form_name, forms)
    path = V.RES / f"phase_fit_{grid}_idx{point}_{form_name}.json"
    seed = expected_coeffs(grid, point, P.form)
    data = dict(grid=grid, point=point, form=form_name, injection=P.inj, n_t=N_T, seed=list(map(float, seed)))
    print(f"{grid} idx{point} {form_name}: injection {P.inj}\n  seed {data['seed']}", flush=True)
    for tag, free in (("at_injection", False), ("free", True)):
        out, _, c = fit(P, P.inj, seed, free_phys=free)
        data[tag] = out
        log_fit(tag, out)
        V.RES.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
        seed = c
    print(f"wrote {path}  ({P.n_eval} trajectories)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, choices=("EMRI", "IMRI"))
    ap.add_argument("--point", required=True)
    ap.add_argument("--form", required=True, choices=FORMS)
    args = ap.parse_args()
    V.branch_check()
    run(args.grid, args.point, args.form)


if __name__ == "__main__":
    main()
