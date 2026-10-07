"""Copy-ready tables of the T-ladder fits (run_lm_T_ladder.py), full-T rung. For each template the
best finished fit over the seeds (old, sk) is used, and the table names the one used; the per-seed
overlaps are listed alongside.
    -> results/lm_results.md      overlap, chi2, mismatch ratio, C_p, C_e and every theta_final

Run:  python make_results_table.py [--t-fracs 0.9 0.95 1]
"""
import argparse
import json

import config as C

OUT_MD = C.OUT_ROOT / "lm_results.md"


def load(template, t_fracs, point, seed):
    path = C.OUT_ROOT / C.case_name(template, t_fracs, seed) / f"lm_{C.point_label(point)}.json"
    d = json.loads(path.read_text()) if path.exists() else None
    return None if d is None or d.get("running") else d


def seed_of(d):
    return d["case"].split("_")[3]


def fits_at(t_fracs, point):
    """{template: {seed: finished fit}} at one point."""
    return {t: {s: d for s in C.SEEDS if (d := load(t, t_fracs, point, s))} for t in C.TEMPLATES}


def best(per_seed):
    return max(per_seed.values(), key=lambda d: d["final_eval"]["overlap"]) if per_seed else None


def summary_table(rows):
    seeds = list(C.SEEDS)
    head = (["point", "a", "e0", "0PA O", "0PA chi2", "0PA + PN O", "0PA + PN chi2",
             "(1-O_0PA)/(1-O_PN)", "C_p", "C_e", "seed 0PA / PN"]
            + [f"O {t} {s}" for t in C.TEMPLATES for s in seeds])
    out = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for point, fits in rows.items():
        z, p = best(fits["0pa"]), best(fits["pn"])
        oz, op = z["final_eval"]["overlap"], p["final_eval"]["overlap"]
        per = [f"{fits[t][s]['final_eval']['overlap']:.10f}" if s in fits[t] else "-"
               for t in C.TEMPLATES for s in seeds]
        out.append("| " + " | ".join(
            [C.point_label(point), f"{z['a_inj']:+.1f}", f"{z['e0_inj']:.1f}", f"{oz:.10f}",
             f"{z['final_eval']['chi2']:.4e}", f"{op:.10f}", f"{p['final_eval']['chi2']:.4e}",
             f"{(1 - oz) / (1 - op):.1f}", f"{p['theta_final']['C_p']:+.4f}",
             f"{p['theta_final']['C_e']:+.4f}", f"{seed_of(z)} / {seed_of(p)}"] + per) + " |")
    return out


def theta_lines(rows):
    out = []
    for point, fits in rows.items():
        for t in C.TEMPLATES:
            d = best(fits[t])
            th = ", ".join(f"{n}={v!r}" for n, v in d["theta_final"].items())
            out.append(f"{C.point_label(point)} {t} ({seed_of(d)} seed): O={d['final_eval']['overlap']!r} "
                       f"chi2={d['final_eval']['chi2']!r}  {th}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-fracs", nargs="+", type=float, default=list(C.T_FRACS))
    args = ap.parse_args()
    rows, missing = {}, []
    for point in C.POINTS:
        fits = fits_at(args.t_fracs, point)
        if all(fits[t] for t in C.TEMPLATES):
            rows[point] = fits
        absent = [f"{t} {s}" for t in C.TEMPLATES for s in C.SEEDS if s not in fits[t]]
        if absent:
            missing.append(f"{C.point_label(point)} ({', '.join(absent)})")
    text = (["# T-ladder LM fits, IMRI, full T 1 yr, dt 10, 2nd-generation TDI AE, rishav setup, "
             "signal SNR 20", ""]
            + summary_table(rows) + ["", "## theta_final (best seed)", "", "```"] + theta_lines(rows)
            + ["```"])
    if missing:
        text += ["", "Missing or still running: " + "; ".join(missing)]
    OUT_MD.write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"-> {OUT_MD}")


if __name__ == "__main__":
    main()
