"""Bias-inference plots (Mahalanobis distance, critical SNR, bias summary,
corner plots) for the EMRI_CV_bias IMRI + EMRI results, built from the Fisher
matrices computed by compute_fishers.py.

Ported from bias_inference_emri/new_results_EMRI/combined_plot.py +
fisher_common.py -- see that file's module docstring for the full methodology
(Jacobi-preconditioned inversion, dev_1/dev_2 marginalization for the
Mahalanobis distance, critical-SNR rescaling, deviation-benefit ratios). The
only change here is that the case list is read generically from
parse_results.parse_all() instead of 3 hardcoded EMRI benchmark points, so it
covers every point in both results_combined.txt (IMRI) and
results_compiled.txt (EMRI), split into separate plot sets per system (the two
systems have different masses/params and aren't meant to be compared directly).

Run (only needs numpy/scipy/matplotlib -- no GPU/FEW required, just the
Fisher_*.npy / SNR_*.npy files already computed by compute_fishers.py):
    python plot_bias.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_results import parse_all  # noqa: E402

PLOTDIR = os.path.join(HERE, "plots")
CL_LEVELS = ((0.90, ""), (0.99, "_99CL"))

LABELS_BY_NAME = {
    "m1": r"$M_1\ (M_\odot)$", "m2": r"$M_2\ (M_\odot)$", "a": r"$a$",
    "p0": r"$p_0\ (M)$", "e0": r"$e_0$", "qS": r"$q_S$", "phiS": r"$\phi_S$",
    "Phi_phi0": r"$\Phi_{\phi_0}$", "Phi_r0": r"$\Phi_{r_0}$",
    "dev_1": r"$dev_1$", "dev_2": r"$dev_2$",
}
CASE_COLOR = {"0PA": "#0F6E56", "PN": "#534AB7", "simple": "#D85A30", "simple_pe": "#C7A62A"}
INTRINSIC_EXTRINSIC_PARAMS = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"]


def make_point_hatches(point_order):
    """Assign every point a distinct hatch pattern. A fixed 7-entry list would
    silently repeat (and thus visually alias two different points) once a
    system has more than 7 points -- IMRI has 16, EMRI has 15 -- so build
    singles/doubles/triples of the base symbols instead, which comfortably
    covers up to ~30 distinct points."""
    base = "/\\|-+xoO.*"
    combos = [None] + [c for c in base] + [c * 2 for c in base] + [c * 3 for c in base]
    if len(point_order) > len(combos):
        raise ValueError(f"{len(point_order)} points exceeds {len(combos)} available hatch patterns")
    return {p: combos[i] for i, p in enumerate(point_order)}


def z_threshold(cl):
    return norm.ppf(0.5 + cl / 2)


def chi2_2d_threshold(cl):
    return chi2.ppf(cl, df=2)


def jacobi_inverse(matrix):
    d = np.sqrt(np.diag(matrix))
    Dinv = np.diag(1.0 / d)
    matrix_scaled = Dinv @ matrix @ Dinv
    cond_before = np.linalg.cond(matrix)
    cond_after = np.linalg.cond(matrix_scaled)
    inv_scaled = np.linalg.inv(matrix_scaled)
    inv = Dinv @ inv_scaled @ Dinv
    return inv, cond_before, cond_after


def load_dataset(system):
    """-> dict keyed by (point, model) -> data dict with bf/true/cov/sigma/fisher/
    param_names/snr, only for cases whose Fisher_*.npy exists on disk."""
    cases = [c for c in parse_all() if c["system"] == system]
    out = {}
    by_point = {}
    missing = []
    for c in cases:
        d = os.path.join(HERE, system)
        fpath = os.path.join(d, f"Fisher_{c['point']}_{c['model']}.npy")
        spath = os.path.join(d, f"SNR_{c['point']}.npy")
        if not (os.path.exists(fpath) and os.path.exists(spath)):
            missing.append((c["point"], c["model"]))
            continue
        fisher = np.load(fpath)
        cov, cb, ca = jacobi_inverse(fisher)
        param_names = c["param_names"]
        true = np.array([c["signal_param"][n] for n in param_names], dtype=float)
        bf = np.array(c["x_bf"], dtype=float)
        sigma = np.sqrt(np.diag(cov))
        snr0 = float(np.load(spath))
        data = dict(bf=bf, true=true, cov=cov, sigma=sigma, fisher=fisher,
                    param_names=param_names, cond_before=cb, cond_after=ca,
                    snr0=snr0, point=c["point"], model=c["model"])
        out[(c["point"], c["model"])] = data
        by_point.setdefault(c["point"], []).append(c["model"])
    if missing:
        print(f"[{system}] {len(missing)} cases missing Fisher/SNR .npy "
              f"(run compute_fishers.py first): {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
    point_order = sorted(by_point, key=lambda p: (p != "adhoc_A", p))
    return out, point_order, by_point


def mahalanobis_distance(data, cl):
    param_names = data["param_names"]
    n_phys = len(INTRINSIC_EXTRINSIC_PARAMS)
    delta = data["bf"] - data["true"]
    if len(param_names) > n_phys:
        assert param_names[:n_phys] == INTRINSIC_EXTRINSIC_PARAMS
        cov_phys = data["cov"][:n_phys, :n_phys]
        fisher_phys, _, _ = jacobi_inverse(cov_phys)
        delta = delta[:n_phys]
    else:
        fisher_phys = data["fisher"]
    d2 = float(delta @ fisher_phys @ delta)
    d2_threshold = chi2.ppf(cl, df=n_phys)
    return dict(D=np.sqrt(max(d2, 0.0)), D_threshold=np.sqrt(d2_threshold), ndim=n_phys)


def worst_marginal_z(data):
    z = (data["bf"] - data["true"]) / data["sigma"]
    idx = int(np.argmax(np.abs(z)))
    return float(z[idx]), data["param_names"][idx]


def critical_snr(data, cl):
    snr0 = data["snr0"]
    maha = mahalanobis_distance(data, cl)
    rho_nD = snr0 * np.sqrt(maha["D_threshold"] ** 2 / maha["D"] ** 2) if maha["D"] > 0 else np.inf
    z0, worst_name = worst_marginal_z(data)
    z_thr = z_threshold(cl)
    rho_1D = snr0 * z_thr / abs(z0) if z0 != 0 else np.inf
    return dict(rho_nD=rho_nD, rho_1D=rho_1D, snr0=snr0, D=maha["D"], z0=z0, worst_param=worst_name)


# ===========================================================================
# Corner plots
# ===========================================================================
def draw_ellipse(ax, mean, cov, i, j, color, chi2_val):
    cov2 = cov[np.ix_([i, j], [i, j])]
    mean2 = mean[[i, j]]
    eigvals, eigvecs = np.linalg.eigh(cov2)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0)
    eigvecs = eigvecs[:, order]
    theta = np.linspace(0, 2 * np.pi, 400)
    r1, r2 = np.sqrt(eigvals[0] * chi2_val), np.sqrt(eigvals[1] * chi2_val)
    ellipse = eigvecs @ np.vstack([r1 * np.cos(theta), r2 * np.sin(theta)]) + mean2[:, None]
    ax.plot(ellipse[0], ellipse[1], color=color, lw=1.3)


def corner_plot(dataset, point, models, title, fname, cl, outdir, param_subset=None):
    chi2_val = chi2_2d_threshold(cl)
    datasets = {m: dataset[(point, m)] for m in models if (point, m) in dataset}
    if not datasets:
        return
    if param_subset is not None:
        param_names = list(param_subset)
    else:
        param_names = max((d["param_names"] for d in datasets.values()), key=len)
    true = np.array([datasets[list(datasets)[0]]["true"][datasets[list(datasets)[0]]["param_names"].index(n)]
                      if n in datasets[list(datasets)[0]]["param_names"] else np.nan
                      for n in param_names])
    # more robust: pull truth per-param from whichever dataset actually has it
    true = np.array([next(d["true"][d["param_names"].index(n)] for d in datasets.values()
                           if n in d["param_names"]) for n in param_names])
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, n_params, figsize=(2.1 * n_params, 2.1 * n_params))
    if n_params == 1:
        axes = np.array([[axes]])
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j >= i:
                ax.axis("off")
                continue
            for m, data in datasets.items():
                names = data["param_names"]
                if param_names[i] not in names or param_names[j] not in names:
                    continue
                ii, jj = names.index(param_names[i]), names.index(param_names[j])
                draw_ellipse(ax, data["bf"], data["cov"], jj, ii, CASE_COLOR[m], chi2_val)
                ax.scatter(data["bf"][jj], data["bf"][ii], color=CASE_COLOR[m], s=10, zorder=3)
            ax.axvline(true[j], color="black", lw=0.8, ls="--", alpha=0.6)
            ax.axhline(true[i], color="black", lw=0.8, ls="--", alpha=0.6)
            ax.scatter(true[j], true[i], color="black", marker="*", s=60, zorder=5,
                       edgecolor="white", linewidth=0.6)
            if i == n_params - 1:
                ax.set_xlabel(LABELS_BY_NAME.get(param_names[j], param_names[j]), fontsize=10)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(LABELS_BY_NAME.get(param_names[i], param_names[i]), fontsize=10)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    legend_handles = [plt.Line2D([0], [0], color=CASE_COLOR[m], lw=2, label=m) for m in datasets]
    legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.3, label=f"{cl * 100:.0f}% CL"))
    legend_handles.append(plt.Line2D([0], [0], color="black", marker="*", ls="--", lw=0.8,
                                     markersize=10, markeredgecolor="white", label="truth"))
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98),
               fontsize=11, frameon=True)
    fig.suptitle(title, fontsize=13, y=1.005)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, fname), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[saved] {os.path.join(outdir, fname)}")


# ===========================================================================
# Cross-point plots
# ===========================================================================
def bias_summary_plot(dataset, point_order, models, cl, outdir, fname):
    z_thr = z_threshold(cl)
    all_param_names = INTRINSIC_EXTRINSIC_PARAMS + ["dev_1", "dev_2"]
    x_positions = {name: idx for idx, name in enumerate(all_param_names)}
    point_markers_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p"]
    point_markers = {p: point_markers_cycle[i % len(point_markers_cycle)] for i, p in enumerate(point_order)}
    n_models = len(models)
    case_offsets = {m: (i - (n_models - 1) / 2) * 0.22 for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(13, 6))
    for m in models:
        for point in point_order:
            data = dataset.get((point, m))
            if data is None:
                continue
            z = (data["bf"] - data["true"]) / data["sigma"]
            xs = [x_positions[n] + case_offsets[m] for n in data["param_names"]]
            ax.scatter(xs, z, color=CASE_COLOR[m], marker=point_markers[point],
                       s=60, edgecolor="black", linewidth=0.4, zorder=3)

    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(z_thr, color="red", lw=1.0, ls="--")
    ax.axhline(-z_thr, color="red", lw=1.0, ls="--")
    ax.set_xticks(range(len(all_param_names)))
    ax.set_xticklabels([LABELS_BY_NAME.get(n, n) for n in all_param_names], fontsize=11)
    ax.set_ylabel(r"normalized bias  $z = (\hat\theta - \theta_{true})/\sigma_{\rm Fisher}$")
    ax.set_title(f"Normalized bias across all points and models ({cl * 100:.0f}% CL)")

    case_handles = [plt.Line2D([0], [0], marker="o", color=CASE_COLOR[m], lw=0,
                                markersize=8, label=m) for m in models]
    threshold_handle = plt.Line2D([0], [0], color="red", lw=1.0, ls="--",
                                   label=f"{cl * 100:.0f}% threshold (|z|={z_thr:.2f})")
    ax.legend(handles=case_handles + [threshold_handle], loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    print(f"[saved] {os.path.join(outdir, fname)}")


def mahalanobis_plot(dataset, point_order, models, cl, outdir, fname):
    n_points = len(point_order)
    bar_width = 0.8 / max(n_points, 1)
    group_centers = np.arange(len(models))
    point_hatch = make_point_hatches(point_order)

    fig, ax = plt.subplots(figsize=(max(9, len(models) * 2.5), 6))
    for p_idx, point in enumerate(point_order):
        xs = group_centers + (p_idx - (n_points - 1) / 2) * bar_width
        heights = []
        for m in models:
            data = dataset.get((point, m))
            heights.append(mahalanobis_distance(data, cl)["D"] / mahalanobis_distance(data, cl)["D_threshold"]
                            if data else 0.0)
        ax.bar(xs, heights, width=bar_width * 0.92, color=[CASE_COLOR[m] for m in models],
               edgecolor="black", linewidth=0.6, hatch=point_hatch[point], zorder=3)

    ax.axhline(1.0, color="red", lw=1.2, ls="--", zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(rf"$D / D_{{{cl * 100:.0f}}}$,  $D=\sqrt{{\Delta\theta^T F\, \Delta\theta}}$")
    ax.set_title(f"Mahalanobis distance of best fit from truth, normalized by the "
                 f"{cl * 100:.0f}% chi2 threshold")
    point_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                    hatch=point_hatch[p], label=p) for p in point_order]
    threshold_handle = plt.Line2D([0], [0], color="red", lw=1.2, ls="--", label=f"{cl * 100:.0f}% threshold")
    ax.legend(handles=point_handles + [threshold_handle], loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    print(f"[saved] {os.path.join(outdir, fname)}")


def critical_snr_plot(dataset, point_order, models, cl, outdir, fname):
    n_points = len(point_order)
    bar_width = 0.8 / max(n_points, 1)
    group_centers = np.arange(len(models))
    point_hatch = make_point_hatches(point_order)

    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(models) * 2.5) + 2.5, 10), sharex=True)
    for ax, key, panel_title in (
        (axes[0], "rho_1D", "1D (worst single marginal parameter)"),
        (axes[1], "rho_nD", "nD (joint, marginalized over dev_1/dev_2, ndim=9)"),
    ):
        for p_idx, point in enumerate(point_order):
            xs = group_centers + (p_idx - (n_points - 1) / 2) * bar_width
            heights = []
            for m in models:
                data = dataset.get((point, m))
                if data is None:
                    heights.append(0.0)
                    continue
                crit = critical_snr(data, cl)
                heights.append(crit[key] / crit["snr0"])
            ax.bar(xs, heights, width=bar_width * 0.92, color=[CASE_COLOR[m] for m in models],
                   edgecolor="black", linewidth=0.6, hatch=point_hatch[point], zorder=3)
        ax.axhline(1.0, color="red", lw=1.2, ls="--", zorder=4)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\rho_{\rm crit} / {\rm SNR}_0$")
        ax.set_title(panel_title)

    axes[1].set_xticks(group_centers)
    axes[1].set_xticklabels(models, fontsize=11)
    point_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                    hatch=point_hatch[p], label=p) for p in point_order]
    threshold_handle = plt.Line2D([0], [0], color="red", lw=1.2, ls="--",
                                   label=f"{cl * 100:.0f}% threshold (already detectable below this line)")
    # Point legend can have up to ~30 entries (one per point) -- placing it
    # top-center (as in the 3-point reference plot) collided with the
    # suptitle once generalized to systems with many more points, so it goes
    # in its own column to the right of both panels instead.
    fig.legend(handles=point_handles + [threshold_handle], loc="center left",
               bbox_to_anchor=(1.0, 0.5), ncol=1 + len(point_handles) // 20, fontsize=8)
    fig.suptitle(f"Critical SNR for detecting template mismatch ({cl * 100:.0f}% CL)")
    plt.tight_layout(rect=(0, 0, 0.82, 1))
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.join(outdir, fname)}")


def deviation_benefit_plot(dataset, point_order, dev_models, cl, outdir, fname):
    if not dev_models:
        return
    n_points = len(point_order)
    bar_width = 0.8 / max(n_points, 1)
    group_centers = np.arange(len(dev_models))
    point_hatch = make_point_hatches(point_order)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    panels = [
        ("Mahalanobis improvement\n" + r"$D_{\rm 0PA} / D_{\rm dev}$",
         lambda point, m: mahalanobis_distance(dataset[(point, "0PA")], cl)["D"]
         / mahalanobis_distance(dataset[(point, m)], cl)["D"]),
        ("Critical SNR improvement (1D)\n" + r"$\rho_{\rm crit,dev} / \rho_{\rm crit,0PA}$",
         lambda point, m: critical_snr(dataset[(point, m)], cl)["rho_1D"]
         / critical_snr(dataset[(point, "0PA")], cl)["rho_1D"]),
        ("Critical SNR improvement (nD)\n" + r"$\rho_{\rm crit,dev} / \rho_{\rm crit,0PA}$",
         lambda point, m: critical_snr(dataset[(point, m)], cl)["rho_nD"]
         / critical_snr(dataset[(point, "0PA")], cl)["rho_nD"]),
    ]
    for ax, (title, metric) in zip(axes, panels):
        for p_idx, point in enumerate(point_order):
            if (point, "0PA") not in dataset:
                continue
            xs = group_centers + (p_idx - (n_points - 1) / 2) * bar_width
            heights = [metric(point, m) if (point, m) in dataset else 0.0 for m in dev_models]
            ax.bar(xs, heights, width=bar_width * 0.92,
                   color=[CASE_COLOR[m] for m in dev_models],
                   edgecolor="black", linewidth=0.6, hatch=point_hatch[point], zorder=3)
        ax.axhline(1.0, color="red", lw=1.2, ls="--", zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(group_centers)
        ax.set_xticklabels(dev_models, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("improvement factor (>1 = better than 0PA)")

    point_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                                    hatch=point_hatch[p], label=p) for p in point_order]
    threshold_handle = plt.Line2D([0], [0], color="red", lw=1.2, ls="--", label="no improvement (=1)")
    # See critical_snr_plot: a top-anchored legend (fine for the 3-point
    # reference plot) collides with the suptitle once there are enough points
    # to need several legend rows, so it goes below the panels instead.
    n_legend = len(point_handles) + 1
    fig.legend(handles=point_handles + [threshold_handle], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), fontsize=8, ncol=min(n_legend, 8))
    fig.suptitle(f"Benefit of including a deviation term, relative to plain 0PA ({cl * 100:.0f}% CL)")
    plt.tight_layout(rect=(0, 0.16, 1, 1))
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.join(outdir, fname)}")


def print_containment_report(dataset, point_order, models, cl):
    z_thr = z_threshold(cl)
    print(f"\n=== {cl * 100:.0f}% marginal containment check (|z| <= {z_thr:.3f}) ===")
    for point in point_order:
        for m in models:
            data = dataset.get((point, m))
            if data is None:
                continue
            z = (data["bf"] - data["true"]) / data["sigma"]
            print(f"\n[{point} / {m}]  cond(raw)={data['cond_before']:.2e}"
                  f"  cond(preconditioned)={data['cond_after']:.2e}")
            for name, zi, sigma_i in zip(data["param_names"], z, data["sigma"]):
                ok = "OK" if abs(zi) <= z_thr else f"OUTSIDE {cl * 100:.0f}%"
                print(f"    {name:>10s}: z={zi:+8.3f}  sigma={sigma_i:.4g}  [{ok}]")
            maha = mahalanobis_distance(data, cl)
            ok = "OK" if maha["D"] <= maha["D_threshold"] else f"OUTSIDE {cl * 100:.0f}%"
            print(f"    {'Mahalanobis':>10s}: D={maha['D']:8.3f}  "
                  f"D_{cl * 100:.0f}(ndim={maha['ndim']})={maha['D_threshold']:.3f}  [{ok}]")


def run_system(system):
    dataset, point_order, by_point = load_dataset(system)
    if not dataset:
        print(f"[{system}] no Fisher matrices on disk yet -- run compute_fishers.py first.")
        return

    sysdir = os.path.join(PLOTDIR, system)
    corner_dir = os.path.join(sysdir, "corner_full")
    corner_5p_dir = os.path.join(sysdir, "corner_5param")
    bias_dir = os.path.join(sysdir, "bias_summary")
    maha_dir = os.path.join(sysdir, "mahalanobis")
    snr_dir = os.path.join(sysdir, "snr_crit")
    benefit_dir = os.path.join(sysdir, "deviation_benefit")

    all_models_present = sorted({m for pt in by_point.values() for m in pt},
                                 key=lambda m: {"0PA": 0, "PN": 1, "simple": 2, "simple_pe": 3}.get(m, 9))
    dev_models = [m for m in all_models_present if m != "0PA"]

    for cl, suffix in CL_LEVELS:
        print(f"\n{'=' * 70}\n[{system}] Running analysis at {cl * 100:.0f}% CL\n{'=' * 70}")
        print_containment_report(dataset, point_order, all_models_present, cl)

        for point in point_order:
            models_here = by_point[point]
            for m in models_here:
                corner_plot(dataset, point, [m], f"{system} {point}: {m} ({cl * 100:.0f}% CL)",
                            f"corner_{point}_{m}{suffix}.png", cl, corner_dir)
            if len(models_here) > 1:
                corner_plot(dataset, point, models_here,
                            f"{system} {point}: {'+'.join(models_here)} overlaid ({cl * 100:.0f}% CL)",
                            f"corner_{point}_combined{suffix}.png", cl, corner_dir)

        if "0PA" in all_models_present:
            bias_summary_plot(dataset, point_order, all_models_present, cl, bias_dir, f"bias_summary{suffix}.png")
            mahalanobis_plot(dataset, point_order, all_models_present, cl, maha_dir, f"mahalanobis_bias{suffix}.png")
            critical_snr_plot(dataset, point_order, all_models_present, cl, snr_dir, f"snr_crit{suffix}.png")
            deviation_benefit_plot(dataset, point_order, dev_models, cl, benefit_dir,
                                    f"deviation_benefit{suffix}.png")

    five = INTRINSIC_EXTRINSIC_PARAMS[:5]
    for point in point_order:
        models_here = by_point[point]
        for m in models_here:
            corner_plot(dataset, point, [m], f"{system} {point}: {m}, intrinsic params (90% CL)",
                        f"corner_{point}_{m}_5param.png", 0.90, corner_5p_dir, param_subset=five)
        if len(models_here) > 1:
            corner_plot(dataset, point, models_here,
                        f"{system} {point}: {'+'.join(models_here)} overlaid, intrinsic params (90% CL)",
                        f"corner_{point}_combined_5param.png", 0.90, corner_5p_dir, param_subset=five)


def main():
    for system in ("IMRI", "EMRI"):
        run_system(system)


if __name__ == "__main__":
    main()
