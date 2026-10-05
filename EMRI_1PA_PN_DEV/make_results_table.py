"""Copy-ready tables of the T-ladder fits (run_lm_T_ladder.py, run_cp_ramp.py), full-T rung, grid points
only. 0PA is the seed-old fit; 0PA + PN is the best finished fit over the seeds in PN_SEEDS (old, row,
ramp, rampfine, best), and the table names the one used.
adhoc_A is left out: at dt 5 its band reaches the 2nd-generation TDI PSD null at ~0.06 Hz, where one
bin carries rho_s ~ 4.9e5, so its chi2 and SNR are meaningless in this setup.
    -> results/lm_results.md      overlap, chi2, mismatch ratio, C_p, C_e and every theta_final

Run:  python make_results_table.py [--t-fracs 0.9 1]
"""
import argparse
import json

import config as C

OUT_MD = C.OUT_ROOT / "lm_results.md"
GRID_POINTS = [p for p in C.POINTS if p not in C.ADHOC]
# PN seed -> its t_fracs, if not the table's (rampfine continues the ramp fit from 2.25 yr: 0.95, 1; best
# re-climbs the best fit at full T with the at-zero spin steps: 1)
PN_SEEDS = {"old": None, "row": None, "ramp": None, "rampfine": [0.95, 1.0], "best": [1.0]}


def load(template, t_fracs, point, seed="old"):
    path = C.OUT_ROOT / C.case_name(template, t_fracs, seed) / f"lm_{C.point_label(point)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def best_pn(t_fracs, point):
    """The highest-overlap finished PN fit over PN_SEEDS; None if the old-seed fit is missing."""
    fits = [d for d in (load("pn", f or t_fracs, point, s) for s, f in PN_SEEDS.items())
            if d is not None and not d.get("running")]
    if not any(d["case"] == C.case_name("pn", t_fracs) for d in fits):
        return None
    return max(fits, key=lambda d: d["final_eval"]["overlap"])


def seed_of(d):
    return d["case"].split("_")[3]


def summary_table(fits):
    out = ["| point | a | e0 | 0PA O | 0PA chi2 | 0PA + PN O | 0PA + PN chi2 | (1-O_0PA)/(1-O_PN) "
           "| C_p | C_e | PN seed | stop 0PA / PN |", "|---" * 12 + "|"]
    for point, (z, p) in fits.items():
        oz, op = z["final_eval"]["overlap"], p["final_eval"]["overlap"]
        out.append(f"| {C.point_label(point)} | {z['a_inj']:+.1f} | {z['e0_inj']:.1f} | {oz:.10f} | "
                   f"{z['final_eval']['chi2']:.4e} | {op:.10f} | {p['final_eval']['chi2']:.4e} | "
                   f"{(1 - oz) / (1 - op):.1f} | {p['theta_final']['C_p']:+.4f} | "
                   f"{p['theta_final']['C_e']:+.4f} | {seed_of(p)} | {z['info']['stop_reason']} / "
                   f"{p['info']['stop_reason']} |")
    return out


def theta_lines(fits):
    out = []
    for point, pair in fits.items():
        for d in pair:
            th = ", ".join(f"{n}={v!r}" for n, v in d["theta_final"].items())
            seed = f" ({seed_of(d)} seed)" if d["template"] == "pn" else ""
            out.append(f"{C.point_label(point)} {d['template']}{seed}: O={d['final_eval']['overlap']!r} "
                       f"chi2={d['final_eval']['chi2']!r}  {th}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-fracs", nargs="+", type=float, default=[0.9, 1.0])
    args = ap.parse_args()
    fits, missing = {}, []
    for point in GRID_POINTS:
        pair = [load("0pa", args.t_fracs, point), best_pn(args.t_fracs, point)]
        if None in pair or any(d.get("running") for d in pair):
            missing.append(C.point_label(point))
        else:
            fits[point] = pair
    text = (["# T-ladder LM fits, full T, 2nd-generation TDI AE, rishav setup, signal SNR 20", ""]
            + summary_table(fits) + ["", "## theta_final", "", "```"] + theta_lines(fits) + ["```"])
    if missing:
        text += ["", f"Missing or still running: {', '.join(missing)}"]
    OUT_MD.write_text("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"-> {OUT_MD}")


if __name__ == "__main__":
    main()
