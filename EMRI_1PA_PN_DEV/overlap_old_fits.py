"""Overlap of the old 1st-generation best fits (config.SEEDS_OLD) in this setup: 2nd-generation TDI,
A and E, rishav generator, DC bin dropped, each point's full T (grid 2.5 yr, adhoc_A 1 yr). No fitting.
    -> results/overlap_old_fits.json      per point and template: theta, old and new overlap, chi2
       results/overlap_old_fits.md        copy-ready table
Points already in the JSON are kept, so a later run can add points.

Run (GPU):  python overlap_old_fits.py [--points 0 1 adhoc_A ...]
"""
import argparse
import json

import config as C
from model import build_grid, build_point, evaluate, point_T_dt
from run_lm_T_ladder import check_branch, seed_theta

OUT_JSON = C.OUT_ROOT / "overlap_old_fits.json"
OUT_MD = C.OUT_ROOT / "overlap_old_fits.md"


def score_point(Gd, point, old):
    sig_row = C.signal_row(point)
    row = dict(point=point, a=float(sig_row[C.COL["a"]]), e0=float(sig_row[C.COL["e0"]]))
    for t in C.TEMPLATES:
        names = C.fit_params(t)
        theta = seed_theta(point, names, t)
        ev = evaluate(build_point(Gd, sig_row, names), theta)
        row[t] = dict(theta=dict(zip(names, map(float, theta))), eval=ev,
                      overlap_old=old[t][point]["overlap_old"])
        print(f"{C.point_label(point):7s} {t:3s} O_old(1st gen AET)={row[t]['overlap_old']:.10f}  "
              f"O_new(2nd gen AE)={ev['overlap']:.10f}  chi2={ev['chi2']:.6e}  "
              f"rho_s={ev['rho_s']:.6f}", flush=True)
    return row


def markdown(rows):
    out = ["| point | a | e0 | 0PA O (1st gen, AET) | 0PA O (2nd gen, AE) | 0PA + PN O (1st gen, AET) "
           "| 0PA + PN O (2nd gen, AE) | C_p | C_e |", "|---" * 9 + "|"]
    for r in rows:
        p, n = r["0pa"], r["pn"]
        out.append(f"| {r['point']} | {r['a']:+.1f} | {r['e0']:.1f} | {p['overlap_old']:.10f} | "
                   f"{p['eval']['overlap']:.10f} | {n['overlap_old']:.10f} | {n['eval']['overlap']:.10f} | "
                   f"{n['theta']['C_p']:+.4f} | {n['theta']['C_e']:+.4f} |")
    return "\n".join(out) + "\n"


def load_rows():
    """Rows already scored, by point; the first run (job 644900) keyed them by integer idx."""
    if not OUT_JSON.exists():
        return {}
    rows = json.loads(OUT_JSON.read_text())
    for r in rows:
        r.setdefault("point", str(r.pop("idx", None)))
    return {r["point"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", choices=C.POINTS, default=C.POINTS)
    args = ap.parse_args()
    check_branch()
    old = json.loads(C.SEEDS_OLD.read_text())
    xp, use_gpu = C.ovl.load_backend()
    rows, grids = load_rows(), {}
    for point in args.points:
        T, dt = point_T_dt(C.signal_row(point))
        if (T, dt) not in grids:
            grids[(T, dt)] = build_grid(xp, use_gpu, T, dt)
        rows[point] = score_point(grids[(T, dt)], point, old)
        C.OUT_ROOT.mkdir(parents=True, exist_ok=True)
        ordered = [rows[p] for p in C.POINTS if p in rows]
        OUT_JSON.write_text(json.dumps(ordered, indent=1))  # rewritten after every point
    OUT_MD.write_text(markdown(ordered))
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
