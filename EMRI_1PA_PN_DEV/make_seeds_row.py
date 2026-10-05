"""Row seeds for PN fits stalled on the C = 0 ridge -> config.SEEDS_ROW, for run_lm_T_ladder.py --seed row.

Each point's seed is its own 0PA ladder fit (full T, seed "old") plus C_p, C_e = the median over its
row-mates (same a) whose PN fit bought (1 - O_0PA)/(1 - O_PN) > config.ROW_MIN_GAIN. The median keeps
a single off-cluster row-mate (idx10, C_p +50 against ~+20) from dragging the seed.

Run:  python make_seeds_row.py --points 7 12 17 20
"""
import argparse
import json

import numpy as np

import config as C
from make_results_table import GRID_POINTS, load

T_FRACS = [0.9, 1.0]


def gain(z, p):
    return (1 - z["final_eval"]["overlap"]) / (1 - p["final_eval"]["overlap"])


def row_dev(point, fits):
    """Median C_p, C_e of the row-mates of point that left the ridge, and who they are."""
    a = fits[point][0]["a_inj"]
    mates = [q for q, (z, p) in fits.items()
             if q != point and z["a_inj"] == a and gain(z, p) > C.ROW_MIN_GAIN]
    assert mates, f"idx{point}: no row-mate with PN gain > {C.ROW_MIN_GAIN}"
    dev = {n: float(np.median([fits[q][1]["theta_final"][n] for q in mates])) for n in C.DEV_PARAMS}
    return dev, mates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=GRID_POINTS, required=True)
    args = ap.parse_args()
    fits = {q: [load(t, T_FRACS, q) for t in C.TEMPLATES] for q in GRID_POINTS}
    seeds = {"pn": {}}
    for point in args.points:
        dev, mates = row_dev(point, fits)
        theta = dict(fits[point][0]["theta_final"], **dev)
        seeds["pn"][point] = dict(theta=theta, row_mates=mates, from_0pa=C.case_name("0pa", T_FRACS))
        print(f"idx{point} a={fits[point][0]['a_inj']:+.1f}: C_p={dev['C_p']:+.4f} "
              f"C_e={dev['C_e']:+.4f} from idx{', idx'.join(mates)}")
    C.SEEDS_ROW.write_text(json.dumps(seeds, indent=1))
    print(f"-> {C.SEEDS_ROW}")


if __name__ == "__main__":
    main()
