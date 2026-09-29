"""Export every Fisher matrix of a fisher_bestfit*.json re-score, one self-contained file per
(set, model), with everything needed to reproduce it or reuse it without the pipeline.

    python export_fishers.py fisher_bestfit.json         -> fishers/native/
    python export_fishers.py fisher_bestfit_snr200.json  -> fishers/snr200/

Each fishers/<tag>/<set>_<model>.json holds
  names, theta (best fit, phases wrapped to within pi of the injection), theta_inj,
  fisher (Gamma, in `names` order), covariance (Gamma^-1, as sigma was computed),
  sigma, bias, bias_over_sigma, overlap, chi2, SNR, the full injection (incl. distance),
  disk, T, and the waveform / Fisher settings and config files that produced it;
and fishers/<tag>/all_fishers.npz has the same matrices as arrays, keyed "<set>_<model>_*".
CPU, seconds.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from fisher_bestfit import CONFIGS                                          # noqa: E402
from stableemrifisher.utils import fishinv                                  # noqa: E402

FILE_TAG = {"vacuum": "0PA", "pn": "0PA_PN"}
TEMPLATE = {
    "vacuum": "FastKerrEccentricEquatorialAccretionFlux, disk arguments dropped (vacuum GR)",
    "pn": "SuperKludge 0PA + 2.5PN deviation; additional_args = [chi2, evolve_1PA, "
          "evolve_primary, evolve_2PA, deviation_included, C_p(dev_1), C_e(dev_2), del_0_p, "
          "del_0_e] = [0, False, False, False, True, dev_1, dev_2, 0, 0]",
}


def settings(cfg_path):
    cfg = json.load(open(os.path.join(ROOT, cfg_path)))
    return dict(config=cfg_path, waveform=cfg["waveform"], fisher=cfg["fisher"],
                deriv_type="stable", pn=cfg.get("pn"))


def export_model(s, t, m):
    G = np.array(m["fisher"], float)
    cov = np.asarray(fishinv(m["theta"][0], G, index_of_M=0), float)
    return dict(
        set=s["tag"], model=FILE_TAG[t], template=TEMPLATE[t], folder=s["folder"],
        names=m["names"], theta=m["theta"], theta_inj=m["theta_inj"],
        fisher=G.tolist(), covariance=cov.tolist(), sigma=m["sigma"], bias=m["bias"],
        bias_over_sigma=m["bias_over_sigma"], overlap=m["overlap"], chi2=m["chi2"],
        snr=s["snr"], injection=s["injection"], disk=s["disk"], T=s["T"],
        fisher_health=m["fisher_health"],
        settings=settings(CONFIGS[s["tag"]][0 if t == "vacuum" else 1]))


def main(path):
    tag = os.path.basename(path).replace("fisher_bestfit", "").replace(".json", "").strip("_")
    out_dir = os.path.join(HERE, "fishers", tag or "native")
    os.makedirs(out_dir, exist_ok=True)
    arrays = {}
    for s in json.load(open(path)):
        for t, m in s["models"].items():
            rec = export_model(s, t, m)
            key = f"{s['tag'].replace(' ', '')}_{FILE_TAG[t]}"
            json.dump(rec, open(os.path.join(out_dir, f"{key}.json"), "w"), indent=1)
            for field in ("fisher", "covariance", "theta", "theta_inj", "sigma"):
                arrays[f"{key}_{field}"] = np.array(rec[field], float)
            arrays[f"{key}_names"] = np.array(rec["names"])
    np.savez(os.path.join(out_dir, "all_fishers.npz"), **arrays)
    print(f"wrote {out_dir}/: {len(arrays) // 6} Fisher files + all_fishers.npz")


if __name__ == "__main__":
    main(sys.argv[1])
