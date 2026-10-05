"""Final fits per template and grid point -> config.BEST_FREE[template].

Each is the chi2-free LM climb at T 0.25 of run_lm_T_ladder.py, results/{config.free_case(t)}/
lm_idx{i}.json, seeded from config.BEST_1PA (1pa) or config.BEST_DEV (1pa_dev), except the points
in config.FREE_OVERRIDE, which take that case's lm_idx{i}.json (a basin test). The climb only
accepts steps that raise the overlap, so the final overlap is never below its seed's; the table
printed here shows by how much it moved.

Run:  python best_fit_free.py
"""
import json

import config as C
from best_fit_dev import record

SEED_FILE = {"1pa": C.BEST_1PA, "1pa_dev": C.BEST_DEV}


def collect(template):
    case = C.free_case(template)
    seeds = json.loads(SEED_FILE[template].read_text())
    params = C.fit_params(False, template)
    best = {}
    override = C.FREE_OVERRIDE.get(template, {})
    for idx in range(25):
        src = override.get(idx, case)
        f = C.OUT_ROOT / src / f"lm_idx{idx}.json"
        out = json.loads(f.read_text()) if f.exists() else {}
        if "final_eval" not in out or out.get("running"):
            print(f"[{template}] idx{idx:<2d} MISSING or unfinished: {f}")
            continue
        o, o_seed = out["final_eval"]["overlap"], seeds[str(idx)]["final_eval"]["overlap"]
        print(f"[{template}] idx{idx:<2d} O {o_seed:.12f} -> {o:.12f} ({o - o_seed:+.2e})  "
              f"iters={out['info']['n_iter']} stop={out['info']['stop_reason']}")
        why = "basin test (config.FREE_OVERRIDE)" if idx in override else "final chi2-free climb"
        best[str(idx)] = record(idx, src, out, why, 1, params)
    C.BEST_FREE[template].write_text(json.dumps(best, indent=1))
    print(f"-> {C.BEST_FREE[template]}  ({len(best)} points)")


def main():
    for template in C.TEMPLATES:
        collect(template)


if __name__ == "__main__":
    main()
