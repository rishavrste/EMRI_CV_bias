"""Plots of results/trajectories_idx0.npz (trajectories_idx0.py).

    plots/diff_vs_t.png        (1PA - model) / 1PA of pdot, edot, Omega_phi, Omega_r, dOmega_phi/dt, dOmega_r/dt,
                               at the same t
    plots/diff_vs_p.png        the same at the same p (interpolated onto the 1PA p grid)
    plots/phase_diff_vs_t.png  1PA - model of Phi_phi, Phi_r, p, e at the same t (what the waveform sees)
One row per quantity, one column per model, each panel on its own y-scale. dOmega_r/dt changes sign along the
inspiral, so it is divided by max|1PA| instead of the pointwise 1PA value.
and a table of the end-of-inspiral phase differences and the mean fractional rate differences on stdout.
Run:  python plot_trajectories_idx0.py > logs/plot_trajectories_idx0.log
      python plot_trajectories_idx0.py --only-injection     (1PA, 0PA injection, PN_freq at injection with C_proj; *_inj.png)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "results" / "trajectories_idx0.npz"
PLOTS = HERE / "plots"
YR = 3.15581e7
SIGNAL = "1PA"
INJECTION_SET = ("0PA injection", "0PA+PN_freq inj, C_proj")   # --only-injection
EDGE = 3

RATES = [("pdot", r"$\dot p$ [1/s]"), ("edot", r"$\dot e$ [1/s]"),
         ("Omega_phi", r"$\Omega_\phi$ [rad/s]"), ("Omega_r", r"$\Omega_r$ [rad/s]"),
         ("Omegadot_phi", r"$\dot\Omega_\phi$ [rad/s$^2$]"), ("Omegadot_r", r"$\dot\Omega_r$ [rad/s$^2$]")]
ORBIT = [("Phi_phi", r"$\Phi_\phi$ [rad]"), ("Phi_r", r"$\Phi_r$ [rad]"), ("p", "p"), ("e", "e")]

# reference categorical palette (dataviz palette.md), fixed order; the signal is drawn in neutral ink
COLORS = {"1PA": "#222222", "0PA injection": "#2a78d6", "0PA fit": "#eb6834", "0PA+PN fit": "#1baf7a",
          "0PA+PN_freq fit": "#e34948", "0PA+PN_freq inj, C_LO": "#4a3aa7", "0PA+PN_freq inj, C_proj": "#e87ba4"}
STYLES = {"0PA+PN_freq inj, C_LO": "--", "0PA+PN_freq inj, C_proj": ":"}


# --- loading ---------------------------------------------------------------------------
def load():
    d = np.load(DATA)
    labels = list(d["labels"])
    D = {lab: {k.split("|", 1)[1]: d[k].copy() for k in d.files if k.startswith(lab + "|")} for lab in labels}
    for tr in D.values():                       # one-sided np.gradient at the ends: drop EDGE samples
        for k in ("Omegadot_phi", "Omegadot_r"):
            tr[k][:EDGE], tr[k][-EDGE:] = np.nan, np.nan
    return labels, D


# --- differences -----------------------------------------------------------------------
def diff_at_t(sig, mod, key):
    """1PA - model at the 1PA times (both trajectories share the dense grid if they end together)."""
    return sig["t"], sig[key] - np.interp(sig["t"], mod["t"], mod[key])


def diff_at_p(sig, mod, key):
    """1PA - model at the same p; p falls monotonically, so interpolate in reversed order."""
    lo, hi = max(sig["p"].min(), mod["p"].min()), min(sig["p"].max(), mod["p"].max())
    m = (sig["p"] >= lo) & (sig["p"] <= hi)
    return sig["p"][m], sig[key][m] - np.interp(sig["p"][m], mod["p"][::-1], mod[key][::-1])


# --- figures ---------------------------------------------------------------------------
def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_figure(quantities, title):
    fig, axes = plt.subplots(len(quantities) // 2, 2, figsize=(13, 3.2 * len(quantities) // 2),
                             constrained_layout=True)
    fig.suptitle(title)
    return fig, axes.ravel()


def plot_diffs(labels, D, keys, diff, xlabel, xscale, name, title, relative):
    """Rows: quantities; columns: models. relative divides by the 1PA value at the same t or p."""
    models = [lab for lab in labels if lab != SIGNAL]
    fig, axes = plt.subplots(len(keys), len(models), figsize=(4.2 * len(models) + 1, 2.4 * len(keys)),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(title)
    for col, lab in enumerate(models):
        axes[0, col].set_title(lab, color=COLORS[lab], fontsize=10)
        for row, (key, ylab) in enumerate(keys):
            ax = axes[row, col]
            x, y = diff(D[SIGNAL], D[lab], key)
            if relative:
                y, norm = normalise(y, x, D[SIGNAL], key, xlabel)
            ax.axhline(0, color="#888888", lw=0.8)
            ax.plot(x / xscale, y, color=COLORS[lab], lw=1.6)
            style_axes(ax, xlabel if row == len(keys) - 1 else "",
                       (f"(1PA - model)/{norm}: {ylab.split(' [')[0]}" if relative else f"1PA - model: {ylab}")
                       if col == 0 else "")
            if xlabel == "p":
                ax.invert_xaxis()                              # inspiral runs left to right
    save(fig, name)


def normalise(y, x, sig, key, xlabel):
    """Divide by the 1PA value at the same x; if the 1PA quantity changes sign (dOmega_r/dt does), by max|1PA|."""
    xs, vs = signal_curve(sig, key, xlabel)
    vs = vs[np.isfinite(vs)]
    if vs.min() < 0 < vs.max():
        return y / np.abs(vs).max(), "max|1PA|"
    return y / np.interp(x, xs, signal_curve(sig, key, xlabel)[1]), "1PA"


def signal_curve(sig, key, xlabel):
    """(x, value) of the 1PA signal, x increasing, for np.interp."""
    x = sig["p"] if xlabel == "p" else sig["t"] / 1.0
    order = np.argsort(x)
    return x[order], sig[key][order]


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=130)
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


# --- table -----------------------------------------------------------------------------
def summary_table(labels, D):
    """End-of-T phase differences, and the rms fractional rate difference at the same p."""
    print("\n1PA - model at t = T (same t) and rms[(1PA - model)/(1PA - 0PA injection)] at the same p")
    print(f"{'model':24s} | dPhi_phi(T) | dPhi_r(T) | dp(T)     | de(T)     | pdot frac | edot frac | Om_phi frac | Om_r frac")
    sig, ref = D[SIGNAL], D["0PA injection"]
    for lab in labels:
        if lab == SIGNAL:
            continue
        end = [diff_at_t(sig, D[lab], k)[1][-1] for k in ("Phi_phi", "Phi_r", "p", "e")]
        frac = []
        for k in ("pdot", "edot", "Omega_phi", "Omega_r"):
            x, y = diff_at_p(sig, D[lab], k)
            _, y0 = diff_at_p(sig, ref, k)
            n = min(y.size, y0.size)
            frac.append(np.sqrt(np.mean((y[:n] / y0[:n]) ** 2)))
        print(f"{lab:24s} | {end[0]:+11.4f} | {end[1]:+9.4f} | {end[2]:+.2e} | {end[3]:+.2e} | "
              + " | ".join(f"{f:9.4f}" for f in frac))


def make_set(labels, D, suffix, title_tag):
    """The three figures for the models in labels (the 1PA signal always included)."""
    plot_diffs(labels, D, RATES, diff_at_t, "t [yr]", YR, f"diff_vs_t{suffix}.png",
               f"idx0: (1PA - model)/1PA at the same t (each on its own trajectory){title_tag}", relative=True)
    plot_diffs(labels, D, RATES, diff_at_p, "p", 1.0, f"diff_vs_p{suffix}.png",
               f"idx0: (1PA - model)/1PA at the same p{title_tag}", relative=True)
    plot_diffs(labels, D, ORBIT, diff_at_t, "t [yr]", YR, f"phase_diff_vs_t{suffix}.png",
               f"idx0: 1PA - model of phases and orbit at the same t{title_tag}", relative=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-injection", action="store_true",
                    help=f"plot only {', '.join(INJECTION_SET)}; files get the suffix _inj")
    args = ap.parse_args()
    labels, D = load()
    if args.only_injection:
        make_set([SIGNAL, *INJECTION_SET], D, "_inj", " - injection parameters")
    else:
        make_set(labels, D, "", "")
        summary_table(labels, D)


if __name__ == "__main__":
    main()
