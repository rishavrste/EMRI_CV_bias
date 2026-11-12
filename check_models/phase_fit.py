"""CPU ranking of the deviation families by how well they can absorb 1PA, and what that costs in absolute bias.

For one grid point, against the 1PA trajectory at the injection, residual r = [Phi_phi^1PA - Phi_phi, Phi_r^1PA - Phi_r]
at N_T times over [0, T]:
    (i)  at_injection : physical parameters at the injection, only the deviation coefficients fitted.
                        Residual dephasing = can the family reach the signal at the true parameters?
    (ii) free         : coefficients and m1, m2, a, p0, e0, Phi_phi0, Phi_r0 fitted.
                        Parameter shifts = a CPU proxy for the absolute bias (phases only, no noise weighting:
                        a ranking, not final numbers). 0PA and PN_freq are included to compare with their LM biases.
Seeds: (i) the family's expected coefficients (results/expected_idx{p}.json, expected_values.py);
       (ii) the injection with the family's (i) optimum.
    -> results/phase_fit_idx{p}_{family}.json (rewritten after each of the two fits)
Run:  python phase_fit.py --point 0 --family B > logs/phase_fit_idx0_B.log
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from families import FAMILIES, PHYS_FREE, POINTS, SIGNAL_FLAGS, TEMPLATE_FLAGS, add_args, branch_check, injection, phases

HERE = Path(__file__).resolve().parent
YR = 3.15581e7
N_T = 400
MAX_NFEV = 60                               # per least_squares call, in units of (n_params + 1) evaluations
FAIL = 1e3                                   # residual when a trial trajectory plunges before T
# finite-difference step, relative to max(1, |x|) (scipy convention)
STEP = dict(m1=1e-8, m2=1e-7, a=1e-6, p0=1e-7, e0=1e-6, Phi_phi0=1e-5, Phi_r0=1e-5)
COEFF_STEP = 1e-4


# --- residual --------------------------------------------------------------------------
class Problem:
    """Phase residual of one family against the 1PA signal of one point."""

    def __init__(self, point):
        self.inj = injection(point)
        self.T = self.inj["T"]
        self.t = np.linspace(0.0, self.T * YR * 0.999, N_T)
        sig = phases(self.inj, add_args(self.inj["chi2"], SIGNAL_FLAGS, "0PA", {}), self.t, self.T)
        self.sig = np.concatenate(sig)
        self.n_eval = 0

    def residual(self, family, theta, coeffs):
        self.n_eval += 1
        args = add_args(self.inj["chi2"], TEMPLATE_FLAGS, family, coeffs)
        try:
            out = phases(theta, args, self.t, self.T)
        except Exception:                                    # noqa: BLE001  invalid trial point
            out = None
        return np.full(self.sig.size, FAIL) if out is None else self.sig - np.concatenate(out)


def summary(r):
    n = r.size // 2
    phi, rr = r[:n], r[n:]
    return dict(rms_phi=float(np.sqrt(np.mean(phi ** 2))), rms_r=float(np.sqrt(np.mean(rr ** 2))),
                max_phi=float(np.abs(phi).max()), max_r=float(np.abs(rr).max()),
                end_phi=float(phi[-1]), end_r=float(rr[-1]))


# --- fits ------------------------------------------------------------------------------
def fit(P, family, theta0, coeffs0, free_phys):
    """least_squares over the coefficients (and the physical parameters if free_phys)."""
    cnames = FAMILIES[family][1]
    pnames = PHYS_FREE if free_phys else []
    names = pnames + cnames
    x0 = np.array([theta0[n] for n in pnames] + [coeffs0[n] for n in cnames])
    steps = np.array([STEP[n] for n in pnames] + [COEFF_STEP] * len(cnames))

    def split(x):
        theta = dict(theta0, **dict(zip(pnames, x[:len(pnames)])))
        return theta, dict(zip(cnames, x[len(pnames):]))

    def res(x):
        return P.residual(family, *split(x))

    start = summary(res(x0))
    if not names:
        return dict(names=[], x=[], start=start, final=start, nfev=1, status="no parameters"), theta0, {}
    t0 = time.time()
    sol = least_squares(res, x0, method="trf", x_scale="jac", diff_step=steps,
                        max_nfev=MAX_NFEV * (len(names) + 1), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    theta, coeffs = split(sol.x)
    out = dict(names=names, x=sol.x.tolist(), start=start, final=summary(sol.fun), nfev=int(sol.nfev),
               status=sol.message, seconds=time.time() - t0,
               coeffs=coeffs, shift={n: theta[n] - P.inj[n] for n in pnames})
    return out, theta, coeffs


def log_fit(tag, family, out):
    s, f = out["start"], out["final"]
    print(f"[{tag} {family}] rms dPhi_phi {s['rms_phi']:.4g} -> {f['rms_phi']:.4g}  rms dPhi_r {s['rms_r']:.4g} -> "
          f"{f['rms_r']:.4g}  end ({f['end_phi']:+.4g}, {f['end_r']:+.4g}) rad  nfev={out['nfev']}  "
          f"{out.get('seconds', 0):.0f} s", flush=True)
    if out.get("coeffs"):
        print("    coeffs: " + ", ".join(f"{k}={v:+.6g}" for k, v in out["coeffs"].items()), flush=True)
    if out.get("shift"):
        print("    shift : " + ", ".join(f"{k}={v:+.4g}" for k, v in out["shift"].items()), flush=True)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def expected_coeffs(point, family):
    """The family's expected coefficients from expected_values.py (1PA rate fit along the injected trajectory)."""
    path = HERE / "results" / f"expected_idx{point}.json"
    return json.loads(path.read_text())["families"][family]["coeffs"]


def run_point(point, family):
    """(i) coefficients only, from the expected values; (ii) coefficients and physical parameters, from (i)."""
    P = Problem(point)
    path = HERE / "results" / f"phase_fit_idx{point}_{family}.json"
    data = dict(point=point, family=family, injection=P.inj, n_t=N_T, seed=expected_coeffs(point, family))
    print(f"idx{point} {family}: injection {P.inj}\n  seed {data['seed']}", flush=True)

    out, _, coeffs = fit(P, family, P.inj, data["seed"], free_phys=False)
    data["at_injection"] = out
    log_fit("at_injection", family, out)
    save(path, data)

    out, _, _ = fit(P, family, P.inj, coeffs, free_phys=True)
    data["free"] = out
    log_fit("free", family, out)
    save(path, data)
    print(f"wrote {path}  ({P.n_eval} trajectories)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", required=True, choices=POINTS)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    args = ap.parse_args()
    branch_check()
    run_point(args.point, args.family)


if __name__ == "__main__":
    main()
