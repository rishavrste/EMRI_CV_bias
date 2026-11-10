"""Compare the LM arms -> results/lm_results.md: final fits at T = 2.5 yr, spread between arms, ladder rungs.

mismatch gain = (1 - O of the PN best fit of EMRI_1PA_PN_DEV) / (1 - O of the arm).
Spread: |difference from the highest-overlap arm| in units of that arm's sigma_final.
Run:  python make_results_table.py
"""
import json

import config as C

SHOW = ("m1", "m2", "a", "p0", "e0", "C_p", "C_e", "n_p", "n_e")


def load(arm, stem, point):
    path = C.OUT_ROOT / arm / f"{stem}_{C.point_label(point)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def theta_of(d):
    return d.get("theta_final") or d["theta_current"]


def overlap_of(d):
    if not d.get("running"):
        return d["final_eval"]["overlap"]
    return d["history"][-1]["overlap"] if d.get("history") else d["start_eval"]["overlap"]


def old_overlap(point):
    return json.loads(C.OLD_BEST_FITS.read_text())["pn"][point]["final_eval"]["overlap"]


def fit_rows(point):
    rows, base = [], old_overlap(point)
    for arm in C.ARMS:
        d = load(arm, "lm", point)
        if d is None:
            rows.append(f"| idx{point} | {arm} | not reached | | | | | | | | |")
            continue
        o, t = overlap_of(d), theta_of(d)
        state = "running" if d.get("running") else d["info"]["stop_reason"]
        iters = len(d.get("history", []))
        rows.append(f"| idx{point} | {arm} | {o:.10f} | {(1 - base) / (1 - o):.3g}x | {t['C_p']:+.3f} | "
                    f"{t['C_e']:+.3f} | {t['n_p']:+.4f} | {t['n_e']:+.4f} | {t['m2']:.6f} | {iters} | {state} |")
    return rows


def spread_rows(point):
    fits = {a: load(a, "lm", point) for a in C.ARMS}
    fits = {a: d for a, d in fits.items() if d and not d.get("running")}
    if len(fits) < 2:
        return []
    top = max(fits, key=lambda a: overlap_of(fits[a]))
    sig, ref = fits[top]["sigma_final"], fits[top]["theta_final"]
    rows = []
    for arm, d in fits.items():
        if arm != top:
            cells = " | ".join(f"{abs(d['theta_final'][n] - ref[n]) / sig[n]:.2f}" for n in SHOW)
            rows.append(f"| idx{point} | {arm} vs {top} | {cells} |")
    return rows


def rung_rows(point):
    rows = []
    for arm, spec in C.ARMS.items():
        if len(spec["rungs"]) < 2:
            continue
        for k, t in enumerate(spec["rungs"]):
            d = load(arm, "lm" if k == len(spec["rungs"]) - 1 else f"lm_T{t:g}", point)
            if d is not None:
                state = "running" if d.get("running") else d["info"]["stop_reason"]
                rows.append(f"| idx{point} | {arm} | {t:g} | {d['start_eval']['overlap']:.10f} | "
                            f"{overlap_of(d):.10f} | {theta_of(d)['n_p']:+.4f} | {state} |")
    return rows


def main():
    out = ["# LM arms, 0PA + power-law PN", "", "## Fits at T = 2.5 yr", "",
           "| point | arm | overlap | mismatch gain vs PN | C_p | C_e | n_p | n_e | m2 | iters | stop |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in C.POINTS:
        out += fit_rows(p)
    out += ["", "## Spread between arms, in sigma of the best arm", "",
            "| point | pair | " + " | ".join(SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
    for p in C.POINTS:
        out += spread_rows(p)
    out += ["", "## Ladder rungs", "", "| point | arm | T | start overlap | end overlap | n_p | stop |",
            "|---|---|---|---|---|---|---|"]
    for p in C.POINTS:
        out += rung_rows(p)
    C.OUT_ROOT.mkdir(exist_ok=True)
    (C.OUT_ROOT / "lm_results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
