"""Old 0PA-vs-1PA best fits, 25 grid cells and adhoc_A (src/EMRI/results_compiled.txt, 1st-generation
TDI, A, E, T) -> config.SEEDS_OLD, the starting points of the T ladder. Run once, outside the container: the jobs
cannot see Fisher_results/ (singularity binds only the working directory).

Run:  python make_seeds.py
"""
import json
import sys

import config as C

sys.path.insert(0, str(C.FISHER_RESULTS))
from parse_results import parse_all                            # noqa: E402


def point_of(case):
    """config.POINTS key of an old EMRI case, None if it is not one of ours."""
    p = case["point"].removeprefix("grid_idx")
    return p if case["system"] == "EMRI" and p in C.POINTS else None


def check_adhoc(case):
    """The old ad-hoc injection must be config.ADHOC's."""
    want = C.ADHOC[case["point"]]
    got = {("x0" if n == "xI0" else n): v for n, v in case["signal_param"].items()}
    got.update(dt=case["dt"], T=case["T"], chi2=case["chi2_sec"])
    bad = {n: (got[n], v) for n, v in want.items() if got[n] != v}
    assert not bad, f"{case['point']}: config.ADHOC differs from the old run: {bad}"


def main():
    seeds = {t: {} for t in C.TEMPLATES}
    for case in parse_all():
        point = point_of(case)
        if point is None:
            continue
        if point in C.ADHOC:
            check_adhoc(case)
        for t in C.TEMPLATES:
            if case["model"] == C.OLD_MODEL[t]:
                th = {C.OLD_NAMES.get(n, n): v for n, v in zip(case["param_names"], case["x_bf"])}
                seeds[t][point] = dict(
                    theta=th, overlap_old=case["overlap"], chi2_old=case["chi2_val"])
    for t in C.TEMPLATES:
        missing = [n for i, s in seeds[t].items() for n in C.fit_params(t) if n not in s["theta"]]
        assert sorted(seeds[t]) == sorted(C.POINTS) and not missing, (t, len(seeds[t]), missing)
    C.SEEDS_OLD.write_text(json.dumps(seeds, indent=1))
    print(f"-> {C.SEEDS_OLD} ({', '.join(f'{t}: {len(seeds[t])}' for t in C.TEMPLATES)})")


if __name__ == "__main__":
    main()
