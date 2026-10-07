"""Starting points of the T ladder, 25 IMRI grid cells -> config.SEEDS_OLD, config.SEEDS_SK.

    old: the old 0PA-vs-1PA best fit of each template (src/IMRI/results_combined.txt, 1st-generation
         TDI, A, E, T, dt 5), read by Fisher_results/parse_results.py
    sk : the SK_files 0PA LM best fit (config.SK_BEST_0PA), qS, phiS at the injection, and for pn
         C_p = C_e = 0

Run once, outside the container: the jobs cannot see Fisher_results/ (singularity binds only the
working directory).

Run:  python make_seeds.py
"""
import json
import sys

import config as C

sys.path.insert(0, str(C.FISHER_RESULTS))
from parse_results import parse_all                            # noqa: E402


def point_of(case):
    """config.POINTS key of an old IMRI grid case, None if it is not one."""
    p = case["point"].removeprefix("grid_idx")
    return p if case["system"] == "IMRI" and case["point"].startswith("grid_idx") else None


def check_injection(case, point):
    """The old run must have injected the SK_files grid row of the same point."""
    row = C.signal_row(point)
    got = {("x0" if n == "xI0" else n): v for n, v in case["signal_param"].items()
           if not n.startswith("dev_")}
    got.update(T=case["T"], chi2=case["chi2_sec"])
    bad = {n: (v, float(row[C.COL[n]])) for n, v in got.items() if v != float(row[C.COL[n]])}
    assert not bad, f"idx{point}: old injection differs from the SK_files grid: {bad}"


def old_seeds():
    seeds = {t: {} for t in C.TEMPLATES}
    for case in parse_all():
        point = point_of(case)
        if point is None:
            continue
        check_injection(case, point)
        for t in C.TEMPLATES:
            if case["model"] == C.OLD_MODEL[t]:
                th = {C.OLD_NAMES.get(n, n): v for n, v in zip(case["param_names"], case["x_bf"])}
                seeds[t][point] = dict(theta=th, overlap_old=case["overlap"], chi2_old=case["chi2_val"])
    return seeds


def sk_seeds():
    sk = json.loads(C.SK_BEST_0PA.read_text())
    seeds = {t: {} for t in C.TEMPLATES}
    for point in C.POINTS:
        row = C.signal_row(point)
        fit = sk[point]
        base = {n: (fit["theta_final"][n] if n in fit["theta_final"] else float(row[C.COL[n]]))
                for n in C.PHYS}
        for t in C.TEMPLATES:
            th = dict(base, **{n: C.DEV_INJ[n] for n in C.fit_params(t) if n in C.DEV_PARAMS})
            seeds[t][point] = dict(theta=th, overlap_old=fit["final_eval"]["overlap"],
                                   chi2_old=fit["final_eval"]["chi2"], best_from=fit["best_from"])
    return seeds


def write(path, seeds):
    for t in C.TEMPLATES:
        missing = [n for i, s in seeds[t].items() for n in C.fit_params(t) if n not in s["theta"]]
        assert sorted(seeds[t]) == sorted(C.POINTS) and not missing, (path, t, len(seeds[t]), missing)
    path.write_text(json.dumps(seeds, indent=1))
    print(f"-> {path} ({', '.join(f'{t}: {len(seeds[t])}' for t in C.TEMPLATES)})")


def main():
    write(C.SEEDS_OLD, old_seeds())
    write(C.SEEDS_SK, sk_seeds())


if __name__ == "__main__":
    main()
