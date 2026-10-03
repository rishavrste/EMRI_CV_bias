"""Do the spin trends survive without the marginal / suspect points?

Per spin row: median gain, mean and spread of C_p, median worst and best r-factor and the median r
of each parameter, over all points and again with the flagged ones (common.add_flags) removed.
    -> results/plot/fit/trend_check.txt

Run:  python check_trends.py
"""
import numpy as np

from common import A_VALUES, LABEL, R_PARAMS, load_points, out_file


def row_stats(pts):
    if not pts:
        return None
    cp = [p["C_p"] for p in pts]
    return dict(n=len(pts), gain=np.median([p["gain"] for p in pts]), cp_mean=np.mean(cp),
                cp_sd=np.std(cp), worst=np.median([max(p["r"].values()) for p in pts]),
                best=np.median([min(p["r"].values()) for p in pts]),
                r={n: np.median([p["r"][n] for p in pts]) for n in R_PARAMS})


def fmt(a, label, s):
    if s is None:
        return f"{a:+.1f} {label:10s}  (no points left)"
    return (f"{a:+.1f} {label:10s} n={s['n']}  gain {s['gain']:6.2f}  C_p {s['cp_mean']:+7.2f} "
            f"± {s['cp_sd']:5.2f}  worst r {s['worst']:6.2f}  best r {s['best']:5.2f} | "
            + " ".join(f"{LABEL[n]} {s['r'][n]:5.2f}" for n in R_PARAMS))


def main():
    pts = load_points()
    lines = ["flagged: " + ", ".join(
        f"idx{p['idx']}({'M' if p['marginal'] else ''}{'S' if p['suspect'] else ''})"
        for p in pts if p["marginal"] or p["suspect"]), "",
        "medians per spin row; r = |dev offset| / |1PA offset| per parameter", ""]
    for a in A_VALUES:
        row = [p for p in pts if p["a"] == a]
        clean = [p for p in row if not (p["marginal"] or p["suspect"])]
        lines += [fmt(a, "all", row_stats(row)), fmt(a, "unflagged", row_stats(clean)), ""]
    text = "\n".join(lines)
    out_file("fit", "trend_check.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
