"""summary/export_fishers.py for fisher_bestfit_2g.json: one self-contained file per (set, model).

    python export_fishers_2g.py   -> fishers/<set>_<model>.json, fishers/all_fishers.npz

Same fields as the summary export; `settings` additionally records the 2nd-gen changes
(TDI generation, A and E only, fmax 0.05 Hz), which the 1st-gen configs it names do not carry.
CPU, seconds.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "summary"))
sys.path.insert(0, ROOT)
from export_fishers import FILE_TAG, export_model                               # noqa: E402

SOURCE = os.path.join(HERE, "fisher_bestfit_2g.json")
OUT_DIR = os.path.join(HERE, "fishers")
SECOND_GEN = dict(tdi="2nd generation", channels="AE", nchannels=2, fmax=0.05,
                  note="1st-gen config below, overridden by tdi / channels / fmax here")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    arrays = {}
    for s in json.load(open(SOURCE)):
        for t, m in s["models"].items():
            rec = export_model(s, t, m)
            rec["settings"] = dict(rec["settings"], **SECOND_GEN)
            key = f"{s['tag'].replace(' ', '')}_{FILE_TAG[t]}"
            json.dump(rec, open(os.path.join(OUT_DIR, f"{key}.json"), "w"), indent=1)
            for field in ("fisher", "covariance", "theta", "theta_inj", "sigma"):
                arrays[f"{key}_{field}"] = np.array(rec[field], float)
            arrays[f"{key}_names"] = np.array(rec["names"])
    np.savez(os.path.join(OUT_DIR, "all_fishers.npz"), **arrays)
    print(f"wrote {OUT_DIR}/: {len(arrays) // 6} Fisher files + all_fishers.npz")


if __name__ == "__main__":
    main()
