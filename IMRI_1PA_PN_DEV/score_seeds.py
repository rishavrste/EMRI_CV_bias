"""Overlap of the ladder seeds (config.SEEDS[--seed]) in this setup: 2nd-generation TDI, A and E,
rishav generator, DC bin dropped, full T = 1 yr. No fitting.
    -> results/seed_overlap/{seed}_idx{i}.json   per template: theta, stored and new overlap, chi2
       results/seed_overlap/{seed}.md            copy-ready table of every point scored so far
"stored" is the seed's own score: 1st-generation AET at dt 5 (old), or SK_files' fit to its 2PA
signal (sk). One JSON per point, so jobs on different points can run at once.

Run (GPU):  python score_seeds.py --seed old [--points 0 1 ...]
"""
import argparse
import json

import config as C
from model import build_grid, build_point, evaluate, point_T_dt
from run_lm_T_ladder import check_branch, seed_theta


OUT_DIR = C.OUT_ROOT / "seed_overlap"


def point_file(seed, point):
    return OUT_DIR / f"{seed}_{C.point_label(point)}.json"


def score_point(Gd, point, seed, stored):
    sig_row = C.signal_row(point)
    row = dict(point=point, a=float(sig_row[C.COL["a"]]), e0=float(sig_row[C.COL["e0"]]))
    for t in C.TEMPLATES:
        names = C.fit_params(t)
        theta = seed_theta(point, names, t, seed)
        ev = evaluate(build_point(Gd, sig_row, names), theta)
        row[t] = dict(theta=dict(zip(names, map(float, theta))), eval=ev,
                      overlap_stored=stored[t][point]["overlap_old"])
        print(f"{C.point_label(point):5s} {t:3s} seed {seed}: O_stored={row[t]['overlap_stored']:.10f}  "
              f"O_new(2nd gen AE)={ev['overlap']:.10f}  chi2={ev['chi2']:.6e}  "
              f"rho_s={ev['rho_s']:.6f}", flush=True)
    return row


def markdown(rows, seed):
    out = [f"# Seed {seed}, scored in 2nd-generation TDI AE, rishav setup, T 1 yr, dt 10", "",
           "| point | a | e0 | 0PA O stored | 0PA O new | 0PA chi2 new | 0PA + PN O stored "
           "| 0PA + PN O new | 0PA + PN chi2 new | C_p | C_e |", "|---" * 11 + "|"]
    for r in rows:
        p, n = r["0pa"], r["pn"]
        out.append(f"| {C.point_label(r['point'])} | {r['a']:+.1f} | {r['e0']:.1f} | "
                   f"{p['overlap_stored']:.10f} | {p['eval']['overlap']:.10f} | {p['eval']['chi2']:.4e} | "
                   f"{n['overlap_stored']:.10f} | {n['eval']['overlap']:.10f} | {n['eval']['chi2']:.4e} | "
                   f"{n['theta']['C_p']:+.4f} | {n['theta']['C_e']:+.4f} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", choices=list(C.SEEDS), required=True)
    ap.add_argument("--points", nargs="+", choices=C.POINTS, default=C.POINTS)
    args = ap.parse_args()
    check_branch()
    stored = json.loads(C.SEEDS[args.seed].read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xp, use_gpu = C.ovl.load_backend()
    T, dt = point_T_dt(C.signal_row(C.POINTS[0]))
    Gd = build_grid(xp, use_gpu, T, dt)                     # every grid cell has T 1 yr, dt 10
    for point in args.points:
        point_file(args.seed, point).write_text(
            json.dumps(score_point(Gd, point, args.seed, stored), indent=1))
    rows = [json.loads(f.read_text()) for p in C.POINTS if (f := point_file(args.seed, p)).exists()]
    out_md = OUT_DIR / f"{args.seed}.md"
    out_md.write_text(markdown(rows, args.seed))
    print(f"-> {OUT_DIR}/{args.seed}_idx*.json\n-> {out_md}")


if __name__ == "__main__":
    main()
