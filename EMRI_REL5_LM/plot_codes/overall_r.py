"""Overall r-factor: how far a template's best fit sits from the injection, relative to the 0PA best fit,
over a whole parameter set S (PHYS = all nine physical parameters, or INTRINSIC = m1, m2, a, p0, e0).

    r_geo    = geometric mean over S of r_i = |b_T,i| / |b_0PA,i|              (unit-free, no Fisher)
    r_fisher = sqrt(b_T^T M b_T) / sqrt(b_0PA^T M b_0PA),  M = (Sigma_0PA[S, S])^-1
               the 0PA Fisher metric on S with the rest marginalised (for S = PHYS, M = Gamma_0PA),
               so both templates are measured with the same yardstick
r < 1: the template's best fit is closer to the injection than the 0PA one.  b = best fit - injection,
phases wrapped.  T = 0PA + PN or 0PA + REL5_1.
    -> plots/fit/overall_r.png, plots/tables/overall_r.md

Run:  python overall_r.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common import DIV, heat, out_file, save
from data import MS, PHYS, TAGS, fisher_json, grid, load_points

SETS = {"all 9 physical": PHYS, "intrinsic 5": PHYS[:5]}
TEMPLATES = ("pn", "rel5")
LOG_LIM = 2.0


def bias_vec(p, t, names):
    return np.array([p["fits"][t]["bias"][n] for n in names])


def r_geo(p, t, names):
    return float(np.exp(np.mean(np.log(np.abs(bias_vec(p, t, names)) / np.abs(bias_vec(p, "0pa", names))))))


def metric(point, names):
    d = fisher_json("0pa", point)
    k = [d["names"].index(n) for n in names]
    S = MS.covariance(np.asarray(d["fisher"]))[np.ix_(k, k)]
    return np.linalg.inv(S)


def r_fisher(p, t, names):
    M = metric(str(p["idx"]), names)
    b, b0 = bias_vec(p, t, names), bias_vec(p, "0pa", names)
    return float(np.sqrt(b @ M @ b) / np.sqrt(b0 @ M @ b0))


KINDS = {"geometric mean": r_geo, "0PA Fisher metric": r_fisher}


def figure(pts):
    rows = [(k, s) for k in KINDS for s in SETS]
    fig, axes = plt.subplots(len(rows), len(TEMPLATES), figsize=(12.5, 4.6 * len(rows)), layout="constrained")
    norm = TwoSlopeNorm(0.0, -LOG_LIM, LOG_LIM)
    for row, (kind, sname) in zip(axes, rows):
        for ax, t in zip(row, TEMPLATES):
            g = grid(pts, lambda p: KINDS[kind](p, t, SETS[sname]))
            heat(ax, fig, np.clip(np.log10(g), -LOG_LIM, LOG_LIM), np.vectorize(lambda v: f"{v:.2g}")(g), DIV,
                 norm, f"{TAGS[t]}: r ({kind}, {sname})", "log10 r  (r < 1: closer than 0PA)")
    fig.suptitle("Overall r-factor vs 0PA: shift of the best fit toward the injection", x=0.01, ha="left")
    return fig


def table(pts):
    cols = [(kind, s, t) for kind in KINDS for s in SETS for t in TEMPLATES]
    head = "| idx | a | e0 | " + " | ".join(f"{TAGS[t]}, {k}, {s}" for k, s, t in cols) + " |"
    lines = [head, "|---" * (3 + len(cols)) + "|"]
    vals = {c: [] for c in cols}
    for p in pts:
        cells = []
        for c in cols:
            v = KINDS[c[0]](p, c[2], SETS[c[1]])
            vals[c].append(v)
            cells.append(f"{v:.3g}")
        lines.append(f"| {p['idx']} | {p['a']:+.1f} | {p['e0']:.1f} | " + " | ".join(cells) + " |")
    lines.append("| median | | | " + " | ".join(f"{np.median(vals[c]):.3g}" for c in cols) + " |")
    lines.append("| r < 1 (of 25) | | | " + " | ".join(str(sum(v < 1 for v in vals[c])) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    pts = load_points()
    save(figure(pts), "fit", "overall_r.png")
    text = table(pts)
    out = out_file("tables", "overall_r.md")
    out.write_text(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
