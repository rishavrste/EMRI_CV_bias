"""Markdown summary table of the parsed EMRI/IMRI optimization results: best
overlap and (dev_1, dev_2) values per point/model. Reads only from
results_combined.txt / results_compiled.txt via parse_results.py (falls back
to a computed SNR_<point>.npy for the couple of EMRI grid points that don't
quote an SNR in the text file). Writes results_table.md.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_results import parse_all  # noqa: E402


def snr_of(system, point, fallback):
    if fallback is not None:
        return fallback
    p = os.path.join(HERE, system, f"SNR_{point}.npy")
    return float(np.load(p)) if os.path.exists(p) else None


def fmt_dev(c):
    if c is None or "dev_1" not in c["param_names"]:
        return "--"
    pn = c["param_names"]
    d1, d2 = c["x_bf"][pn.index("dev_1")], c["x_bf"][pn.index("dev_2")]
    return f"{d1:.4g}, {d2:.4g}"


def fmt_ov(c):
    return f"{c['overlap']:.7f}" if c else "--"


def fmt(x, spec):
    return format(x, spec) if x is not None else "--"


def main():
    cases = parse_all()
    by_key = {}
    for c in cases:
        by_key.setdefault((c["system"], c["point"]), {})[c["model"]] = c

    lines = [
        "| system | point | a | e0 | SNR | 0PA overlap | PN overlap | PN dev_1,dev_2 | "
        "simple overlap | simple dev_1,dev_2 | simple_pe overlap | simple_pe dev_1,dev_2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system, point in sorted(by_key, key=lambda k: (k[0], k[1] != "adhoc_A", k[1])):
        models = by_key[(system, point)]
        any_c = next(iter(models.values()))
        sp = any_c["signal_param"]
        snr = snr_of(system, point, any_c["snr"])
        m0, mpn, msi, mpe = (models.get(k) for k in ("0PA", "PN", "simple", "simple_pe"))
        lines.append(
            f"| {system} | {point} | {sp['a']:+.2f} | {sp['e0']:.2f} | {fmt(snr, '.2f')} | "
            f"{fmt_ov(m0)} | {fmt_ov(mpn)} | {fmt_dev(mpn)} | "
            f"{fmt_ov(msi)} | {fmt_dev(msi)} | {fmt_ov(mpe)} | {fmt_dev(mpe)} |"
        )

    out = os.path.join(HERE, "results_table.md")
    with open(out, "w") as f:
        f.write("# EMRI / IMRI optimization results (best overlap + deviation values)\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
