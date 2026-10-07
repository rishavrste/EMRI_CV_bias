"""The full-T best fit of each template at every grid point -> config.BEST_FITS, for fisher_at_best.py
and the bias plots.

For each template and point: the highest-overlap finished full-T fit over every run of that template
in results/ (all T ladders and seeds: results/IMRI_{template}_*/lm_idx{i}.json; user, 2026-10-07).
The fits that lost the basin at full T (overlap ~0.75-0.83) drop out this way. Each record is the LM
output without its history, so theta_final, theta_inj, final_eval, a_inj, e0_inj, T, dt and case are
all there.

Run (from fisher/):  python best_fits.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # IMRI_1PA_PN_DEV: config

import config as C                                  # noqa: E402


def strip(d):
    return {k: v for k, v in d.items() if k not in ("history", "theta_current", "sigma_current")}


def finished_fits(template, point):
    """Every finished full-T fit of template at point, over all cases."""
    paths = sorted(C.OUT_ROOT.glob(f"{C.GRID}_{template}_*/lm_{C.point_label(point)}.json"))
    fits = [json.loads(p.read_text()) for p in paths]
    return [d for d in fits if not d.get("running")]


def best_fit(template, point):
    fits = finished_fits(template, point)
    assert fits, f"{template} {C.point_label(point)}: no finished fit"
    return max(fits, key=lambda d: d["final_eval"]["overlap"]), len(fits)


def main():
    best = {t: {} for t in C.TEMPLATES}
    for point in C.POINTS:
        line = f"{C.point_label(point):<5s}"
        for t in C.TEMPLATES:
            d, n = best_fit(t, point)
            best[t][point] = strip(d)
            line += f"  {t} O={d['final_eval']['overlap']:.12f} ({d['case']}, best of {n})"
        print(line)
    C.BEST_FITS.write_text(json.dumps(best, indent=1))
    print(f"-> {C.BEST_FITS}")


if __name__ == "__main__":
    main()
