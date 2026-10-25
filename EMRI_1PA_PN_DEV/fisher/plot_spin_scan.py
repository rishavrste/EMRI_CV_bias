"""1D widths and 2D marginals of the spin-scan Fishers (fisher_spin_scan.py) against the a = 0 one.

Three panels, every Fisher compared with the reference a = 0 Fisher at the same point:
  1. sigma_i(a) / sigma_i(0) per parameter                       (1D marginal widths)
  2. rho_ij(a) against rho_ij(0) for every pair                     (2D marginal orientation)
  3. sqrt(det Sigma_ij(a) / det Sigma_ij(0)) of every 2x2 block     (2D marginal area)
The stored best-fit Fisher (fisher_at_best/{template}/fisher_{label}.json) is drawn as a check.

    python plot_spin_scan.py --template pn --point 22
    -> results/fisher_at_best/{template}_spin_scan/spin_scan_{label}.png
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# same as config.FISHER_DIR, without importing the waveform stack
FISHER_DIR = Path(__file__).resolve().parents[1] / "results" / "fisher_at_best"
REF_SPIN = 0.0
# hue by |a|, filled for a > 0, hollow for a < 0
CASES = [(0.001, "#2a78d6", True), (-0.001, "#2a78d6", False),
         (0.01, "#eb6834", True), (-0.01, "#eb6834", False)]
BEST_COLOR, INK, GRID = "#1baf7a", "#52514e", "#e4e3df"
LABELS = {"m1": "m_1", "m2": "m_2", "a": "a", "p0": "p_0", "e0": "e_0", "qS": r"\theta_S",
          "phiS": r"\phi_S", "Phi_phi0": r"\Phi_{\phi,0}", "Phi_r0": r"\Phi_{r,0}",
          "C_p": "C_p", "C_e": "C_e"}


def load(path):
    d = json.loads(Path(path).read_text())
    return d["names"], np.linalg.inv(np.array(d["fisher"])), d


def spin_file(template, label, spin):
    return FISHER_DIR / f"{template}_spin_scan" / f"fisher_{label}_a{spin:+g}.json"


def compare(cov, ref):
    """1D width ratios, (rho, rho_ref) per pair, 2D area ratio per pair."""
    s, s0 = np.sqrt(np.diag(cov)), np.sqrt(np.diag(ref))
    pairs = list(combinations(range(len(s)), 2))
    rho = np.array([cov[i, j] / (s[i] * s[j]) for i, j in pairs])
    rho0 = np.array([ref[i, j] / (s0[i] * s0[j]) for i, j in pairs])
    area = np.array([np.sqrt(np.linalg.det(cov[np.ix_(p, p)]) / np.linalg.det(ref[np.ix_(p, p)]))
                     for p in map(list, pairs)])
    return s / s0, rho, rho0, area


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def draw_case(axes, x, res, color, filled, label, k):
    width, rho, rho0, area = res
    kw = dict(color=color, mfc=color if filled else "white", mec=color, mew=1.5, ms=8, ls="none")
    marker = "o" if filled else "s"
    axes[0].plot(x + (k - 2) * 0.12, width, marker=marker, label=label, **kw)
    axes[1].plot(rho0, rho, marker=marker, **kw)
    jitter = np.random.default_rng(k).uniform(-0.25, 0.25, area.size)
    axes[2].plot(k + jitter, area, marker=marker, **kw)


def plot(template, point):
    label = f"idx{point}"
    names, ref, _ = load(spin_file(template, label, REF_SPIN))
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})

    cases = [(f"a = {spin:+g}" + (" (relative step)" if abs(spin) >= 0.01 else ""),
              load(spin_file(template, label, spin))[1], color, filled) for spin, color, filled in CASES]
    _, best_cov, best = load(FISHER_DIR / template / f"fisher_{label}.json")
    a_best = best["theta"][names.index("a")]
    cases.append((f"stored best fit, a = {a_best:.2e}", best_cov, BEST_COLOR, True))
    for k, (lab, cov, color, filled) in enumerate(cases):
        draw_case(axes, x, compare(cov, ref), color, filled, lab, k)

    ax = axes[0]
    ax.set_yscale("log")
    ax.axhline(1.0, color=INK, lw=1)
    ax.set_xticks(x, [rf"${LABELS.get(n, n)}$" for n in names], fontsize=12)
    ax.set_ylabel(r"$\sigma_i(a)\,/\,\sigma_i(a=0)$", fontsize=13)
    ax.set_title("1D marginal widths", fontsize=14, loc="left")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=12, loc="lower center",
               ncol=len(cases), bbox_to_anchor=(0.5, -0.06))

    ax = axes[1]
    ax.plot([-1, 1], [-1, 1], color=INK, lw=1)
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal")
    ax.set_xlabel(r"$\rho_{ij}(a=0)$", fontsize=13)
    ax.set_ylabel(r"$\rho_{ij}(a)$", fontsize=13)
    ax.set_title(f"2D marginal correlations ({len(names) * (len(names) - 1) // 2} pairs)",
                 fontsize=14, loc="left")

    ax = axes[2]
    ax.set_yscale("log")
    ax.axhline(1.0, color=INK, lw=1)
    ax.set_xticks(range(len(cases)), [c[0].split(" (")[0].replace("stored best fit, ", "best fit\n")
                                      for c in cases], fontsize=11)
    ax.set_ylabel(r"$\sqrt{\det\Sigma_{ij}(a)\,/\,\det\Sigma_{ij}(a=0)}$", fontsize=13)
    ax.set_title("2D marginal areas", fontsize=14, loc="left")

    for ax in axes:
        style(ax)
    fig.suptitle(f"EMRI {label}, {template.upper()} Fisher at the best fit with a swapped, "
                 f"against a = 0 (signal SNR 20)", fontsize=15, x=0.01, ha="left")
    fig.tight_layout()
    out = FISHER_DIR / f"{template}_spin_scan" / f"spin_scan_{label}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", default="pn", choices=["0pa", "pn"])
    ap.add_argument("--point", default="22")
    args = ap.parse_args()
    plot(args.template, args.point)


if __name__ == "__main__":
    main()
