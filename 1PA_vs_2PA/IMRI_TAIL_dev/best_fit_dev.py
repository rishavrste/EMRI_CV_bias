"""Chosen 1PA + deviation best fit per grid point -> config.BEST_DEV.

The highest T = 0.25 overlap over every finished results/*/lm_idx{i}.json, except where CHOSEN
names the run (user's choice, 2026-10-03). A run that held chi2 fixed gets that value filled in, so
every theta_final carries all of config.PARAMS.

Run:  python best_fit_dev.py
"""
import json

import config as C

# idx -> (case, reason) overriding the max-overlap pick
CHOSEN = {
    4: ("IMRI_TAIL_1pa_dev_distdiv10_dropdc_rishav_1pabest_0.25",
        "chi2-free run of a tied pair (same overlap as the fixed-chi2 run)"),
    20: ("IMRI_TAIL_1pa_dev_fixchi2seed_distdiv10_dropdc_rishav_1pabest_0.242-0.25",
         "chi2 held at the 1PA value; a 1.2e-8 higher fit needs chi2 = 0.998"),
}


def finished_runs():
    """{idx: [(case, record), ...]} over every finished T = 0.25 result."""
    runs = {}
    for f in sorted(C.OUT_ROOT.glob("*/lm_idx*.json")):
        out = json.loads(f.read_text())
        if "final_eval" in out:
            runs.setdefault(out["idx"], []).append((f.parent.name, out))
    return runs


def pick(idx, cands):
    if idx in CHOSEN:
        case, reason = CHOSEN[idx]
        return next(c for c in cands if c[0] == case), reason
    return max(cands, key=lambda c: c[1]["final_eval"]["overlap"]), "max overlap"


def full_theta(out):
    """theta_final over all of config.PARAMS, chi2 filled in where it was held fixed."""
    th = dict(out["theta_final"])
    th.setdefault("chi2", out["chi2_fixed"] if out.get("chi2_fixed") is not None
                  else out["chi2_spin_inj"] if "chi2_spin_inj" in out else 0.95)
    return {n: th[n] for n in C.PARAMS}


def record(idx, case, out, reason, n_runs):
    sig_row = C.ovl.signal_array(C.GRID)[idx]
    return dict(idx=idx, a_inj=out["a_inj"], e0_inj=out["e0_inj"], best_from=case, why=reason,
                n_runs=n_runs, chi2_fitted="chi2" in out["theta_final"], params=C.PARAMS,
                theta_final=full_theta(out), final_eval=out["final_eval"],
                theta_inj={n: C.injected_value(sig_row, n) for n in C.PARAMS})


def main():
    runs = finished_runs()
    best = {}
    for idx in sorted(runs):
        (case, out), reason = pick(idx, runs[idx])
        best[str(idx)] = record(idx, case, out, reason, len(runs[idx]))
        print(f"idx{idx:<2d} O={out['final_eval']['overlap']:.15f}  {reason:12.12s}  {case}")
    C.BEST_DEV.write_text(json.dumps(best, indent=1))
    print(f"-> {C.BEST_DEV}")


if __name__ == "__main__":
    main()
