"""Copy-ready comparison table of 0PA, 0PA + PN and 0PA + REL5_1 at every grid point -> plots/tables/comparison.md

Per point and template: 1 - O, nD rho_crit at 90% and 99% CL (intrinsic m1, m2, a, p0, e0; everything else
marginalised), worst 1D |b|/sigma at SNR 20 over the same five, and the bias b = best fit - injection of the five.

Run:  python make_table.py
"""
from common import out_file
from data import PHYS, TAGS, load_points

INTRINSIC = PHYS[:5]


def row(p, t):
    f = p["fits"][t]
    v, n = f["max1d"]
    rc = " | ".join(f"{f['rho_crit'][cl]:.3g}" for cl in sorted(f["rho_crit"]))
    b = " | ".join(f"{f['bias'][k]:+.2e}" for k in INTRINSIC)
    return f"| idx{p['idx']} | {p['a']:+.1f} | {p['e0']:.1f} | {TAGS[t]} | {f['mismatch']:.2e} | {rc} | {v:.3g} ({n}) | {b} |"


def main():
    head = ["| point | a | e0 | template | 1 − O | ρ_crit 90% | ρ_crit 99% | worst 1D b/σ | "
            + " | ".join(f"b {k}" for k in INTRINSIC) + " |", "|---" * (10 + len(INTRINSIC) - 2) + "|"]
    lines = head + [row(p, t) for p in load_points() for t in TAGS]
    text = "\n".join(lines) + "\n"
    out = out_file("tables", "comparison.md")
    out.write_text(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
