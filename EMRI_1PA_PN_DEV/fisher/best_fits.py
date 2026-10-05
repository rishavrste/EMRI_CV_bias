"""The full-T best fit of each template at every grid point -> config.BEST_FITS, for fisher_at_best.py
and the bias plots.

    0pa : the seed-old ladder fit
    pn  : the highest-overlap finished fit over make_results_table.PN_SEEDS (old, row, ramp, rampfine, best)
Each record is the LM output without its history, so theta_final, theta_inj, final_eval, a_inj, e0_inj,
T, dt and case are all there.

Run (from fisher/):  python best_fits.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # EMRI_1PA_PN_DEV: config, model, lm

import config as C                                  # noqa: E402
from make_results_table import GRID_POINTS, best_pn, load  # noqa: E402

T_FRACS = [0.9, 1.0]


def strip(d):
    return {k: v for k, v in d.items() if k not in ("history", "theta_current", "sigma_current")}


def main():
    best = {"0pa": {}, "pn": {}}
    for point in GRID_POINTS:
        z, p = load("0pa", T_FRACS, point), best_pn(T_FRACS, point)
        assert z is not None and p is not None, f"idx{point}: a fit is missing or running"
        best["0pa"][point], best["pn"][point] = strip(z), strip(p)
        print(f"idx{point:<2s} 0PA O={z['final_eval']['overlap']:.12f}  "
              f"PN O={p['final_eval']['overlap']:.12f}  {p['case']}")
    C.BEST_FITS.write_text(json.dumps(best, indent=1))
    print(f"-> {C.BEST_FITS}")


if __name__ == "__main__":
    main()
