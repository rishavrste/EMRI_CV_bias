"""Posterior sampling (PARIS) at the best fit of a template, to test the Fisher Gaussian against the
true (noise-free) posterior. Adapted from bias_inference_emri @ josh/pe-sampling, src/pe_sampling.py.

    0pa : 1PA signal vs 0PA template,        9 parameters (PHYS: sky angles included)
    pn  : 1PA signal vs 0PA + PN template,  11 parameters (PHYS + C_p, C_e)
lnL = -<s-h|s-h>/2 in the setup of the Fishers being tested (model.py: 2nd-generation TDI A and E, the
rishav inner product with the DC bin dropped, the point's stored T and dt, signal SNR 20). Distance,
qK, phiK, Phi_theta0 and chi2 held at the injection, as in the fits.

Prior: a box centred on the best fit (config.BEST_FITS), half-width PSR x the Fisher sigma of the stored
Fisher (results/fisher_at_best/{template}/fisher_{label}.json); the phases Phi_phi0, Phi_r0 capped at
PHASE_PSR x sigma (as Josh's: a wide phase window opens secondary phase modes); clipped to the physical
box config.PHYS_LO / PHYS_HI.
Seeds and the starting proposal covariance are draws from / the covariance of the Fisher Gaussian;
PARIS re-weights every sample by the true likelihood, so this sets where it starts, not the answer.

    -> results/pe/{template}_{label}[_{tag}]/sampler_state.pkl  (every CHECKPOINT_EVERY iterations; --resume)
                                     pe_results.npz     (weighted samples, Fisher, prior box)
                                     pe_summary.json    (PE vs Fisher: sigma, correlation, mode, bias)

Run (GPU, from pe/):  python pe_sampling.py --template 0pa --point 24 [--resume]
--min-ess N keeps sampling past the dlogZ stop, in ESS_CHUNK-iteration chunks, until ESS >= N (a dlogZ
stop says the evidence has settled, not that the weights are spread over enough samples).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # EMRI_1PA_PN_DEV: config, model, lm
sys.path.insert(0, "/home/svu/e1583490/packages_to_install/parismc")
import config as C                                  # noqa: E402
from fisher.fisher_at_best import bias_vector, load_best_fit  # noqa: E402
from make_results_table import GRID_POINTS  # noqa: E402
from model import build_grid, build_point, evaluate, point_T_dt  # noqa: E402
from run_lm_T_ladder import check_branch  # noqa: E402

from parismc import Sampler, SamplerConfig  # noqa: E402

PE_DIR = C.OUT_ROOT / "pe"
PSR, PHASE_PSR = 50.0, 5.0            # prior half-width in Fisher sigma; phases capped (Josh's values)
N_SEED, NUM_ITERATIONS, STOP_DLOGZ = 20, 100_000, 0.01
ESS_CHUNK = 2000                      # iterations per extra chunk while ESS < --min-ess
CHECKPOINT_EVERY = 500                # iterations between sampler_state.pkl saves (walltime safety)
SEED = 42
PARIS_CONFIG = dict(merge_confidence=0.9, alpha=5000, trail_size=int(1e3), boundary_limiting=True,
                    use_beta=True, integral_num=int(1e5), gamma=500, use_pool=False)

# Module-level state: PARIS pickles the sampler with its log-density and prior transform, so both are
# top-level functions reading these globals (set by set_context), never closures over GPU arrays.
_P = None                             # build_point model functions
_LO = None                            # prior box, lower edge
_SPAN = None                          # prior box, width


# --- likelihood and prior ----------------------------------------------------
def log_density(theta_batch):
    """lnL = -chi2/2 at physical parameters, shape (N, ndim) -> (N,); a failed waveform is -inf."""
    theta_batch = np.atleast_2d(theta_batch)
    out = np.full(len(theta_batch), -np.inf)
    for k, theta in enumerate(theta_batch):
        chi2 = _P["chi2r"](theta)
        if np.isfinite(chi2) and chi2 < 1e29:              # chi2r returns 1e30 on a failed waveform
            out[k] = -0.5 * float(chi2)
    return out


def prior_transform(u):
    """Unit hypercube -> the prior box."""
    return _LO + np.asarray(u) * _SPAN


def to_unit(theta):
    return (np.asarray(theta) - _LO) / _SPAN


def set_context(P, lo, hi):
    global _P, _LO, _SPAN
    _P, _LO, _SPAN = P, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float) - np.asarray(lo)


# --- Fisher and prior box ----------------------------------------------------
def load_fisher(template, point):
    path = C.FISHER_DIR / template / f"fisher_{C.point_label(point)}.json"
    return json.loads(path.read_text())


def fisher_cov(F):
    """Covariance, inverting F in Jacobi-rescaled coordinates (as lm.sigma_from_fisher)."""
    F = np.asarray(F, dtype=float)
    d = np.sqrt(np.diag(F))
    return np.linalg.inv(F / np.outer(d, d)) / np.outer(d, d)


def prior_box(names, theta, sigma, psr, phase_psr):
    width = np.array([(min(psr, phase_psr) if n in C.PHASES else psr) * s
                      for n, s in zip(names, sigma)])
    lo, hi = np.asarray(theta) - width, np.asarray(theta) + width
    lo = np.array([max(v, C.PHYS_LO.get(n, -np.inf)) for n, v in zip(names, lo)])
    hi = np.array([min(v, C.PHYS_HI.get(n, np.inf)) for n, v in zip(names, hi)])
    return lo, hi


# --- sampler -----------------------------------------------------------------
def seeds_from_fisher(theta, cov, n_seed, rng):
    """n_seed - 1 draws of the Fisher Gaussian plus the best fit itself, in the unit cube."""
    draws = rng.multivariate_normal(theta, cov, size=n_seed - 1)
    return np.clip(np.vstack([to_unit(draws), to_unit(theta)]), 0.0, 1.0)


def new_sampler(names, cov, n_seed):
    cov_unit = cov / np.outer(_SPAN, _SPAN)
    return Sampler(ndim=len(names), n_seed=n_seed, log_density_func=log_density,
                   init_cov_list=[cov_unit.copy() for _ in range(n_seed)],
                   prior_transform=prior_transform, config=SamplerConfig(**PARIS_CONFIG))


def checkpoint(sampler, i):
    """run_sampling callback: save the state every CHECKPOINT_EVERY iterations."""
    if i > 0 and i % CHECKPOINT_EVERY == 0:
        sampler.save_state()
        print(f"[checkpoint] iter {i}, state saved", flush=True)


def continue_sampling(sampler, out_dir, num_iterations, **kw):
    """run_sampling on a sampler that has run before. PARIS grows its arrays to current_iter +
    num_iterations, one slot short of the element count, so each continuation silently drops its last
    sample and the next growth leaves np.empty garbage there (ESS -> ~1); one slot of headroom first."""
    sampler._extend_arrays_if_needed(num_iterations + 1)
    sampler.run_sampling(num_iterations=num_iterations, savepath=str(out_dir), print_iter=100,
                         callback=checkpoint, **kw)


def ess_of(sampler):
    _, w = sampler.get_samples_with_weights(flatten=True)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def sample_to_min_ess(sampler, out_dir, args, done):
    """Fixed ESS_CHUNK-iteration chunks (no dlogZ stop: it is already met) until ESS >= args.min_ess
    or the iteration budget is spent; `done` = iterations already run in this job."""
    ess = ess_of(sampler)
    while ess < args.min_ess and done < args.num_iterations:
        print(f"[min-ess] ESS {ess:.1f} < {args.min_ess}; {ESS_CHUNK} more iterations", flush=True)
        continue_sampling(sampler, out_dir, ESS_CHUNK)
        done += ESS_CHUNK
        ess = ess_of(sampler)
    print(f"[min-ess] ESS {ess:.1f} (target {args.min_ess}) after {done} iterations in this job", flush=True)
    sampler.save_state()


def run_paris(out_dir, names, theta, cov, args):
    pkl = out_dir / "sampler_state.pkl"
    if args.resume:
        print(f"Resuming from {pkl}", flush=True)
        sampler = Sampler.load_state(str(pkl))
        # the wrapper sampler.log_density_func applies the prior transform; re-wire what it wraps
        sampler.log_density_func_original, sampler.prior_transform = log_density, prior_transform
        start = sampler.current_iter
        continue_sampling(sampler, out_dir, args.num_iterations, stop_dlogZ=args.stop_dlogz)
        sample_to_min_ess(sampler, out_dir, args, sampler.current_iter - start)
        return sampler
    seeds = seeds_from_fisher(theta, cov, args.n_seed, np.random.default_rng(SEED))
    seeds_lnL = log_density(prior_transform(seeds))
    print(f"seed lnL: best fit {seeds_lnL[-1]:.4f}, range [{seeds_lnL.min():.3f}, {seeds_lnL.max():.3f}]",
          flush=True)
    sampler = new_sampler(names, cov, args.n_seed)
    sampler.run_sampling(num_iterations=args.num_iterations, savepath=str(out_dir), print_iter=100,
                         stop_dlogZ=args.stop_dlogz, external_lhs_points=seeds,
                         external_lhs_log_densities=seeds_lnL, callback=checkpoint)
    sample_to_min_ess(sampler, out_dir, args, sampler.current_iter)
    return sampler


# --- PE vs Fisher ------------------------------------------------------------
def weighted_moments(samples, weights):
    w = weights / weights.sum()
    return np.average(samples, weights=w, axis=0), np.cov(samples.T, aweights=w)


def corr(cov):
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


def nd_bias(b, cov):
    """sqrt(b^T cov^-1 b), inverting in Jacobi-rescaled coordinates."""
    s = np.sqrt(np.diag(cov))
    return float(np.sqrt(b / s @ np.linalg.solve(corr(cov), b / s)))


def sample_lnL(sampler):
    """lnL of every sample, stored by PARIS, in the order of get_samples_with_weights(flatten=True)."""
    return np.concatenate([sampler.searched_log_densities_list[j][:sampler.element_num_list[j]]
                           for j in range(sampler.n_proc)])


def summarise(names, theta, theta_inj, cov_f, samples, weights, lnL, lnL_best):
    mean, cov_pe = weighted_moments(samples, weights)
    mode = samples[int(np.argmax(lnL))]                      # mode: the best sample PARIS found
    sig_f, sig_pe = np.sqrt(np.diag(cov_f)), np.sqrt(np.diag(cov_pe))
    bias = bias_vector(names, theta, theta_inj)
    return dict(names=names, theta_best=theta.tolist(), theta_inj=theta_inj.tolist(),
                mean_pe=mean.tolist(), mode_pe=mode.tolist(), lnL_best=lnL_best,
                lnL_mode=float(lnL.max()), sigma_fisher=sig_f.tolist(), sigma_pe=sig_pe.tolist(),
                sigma_ratio=(sig_pe / sig_f).tolist(),
                mean_shift_over_sigma_f=((mean - theta) / sig_f).tolist(),
                mode_shift_over_sigma_f=((mode - theta) / sig_f).tolist(),
                corr_fisher=corr(cov_f).tolist(), corr_pe=corr(cov_pe).tolist(),
                max_abs_corr_diff=float(np.abs(corr(cov_pe) - corr(cov_f)).max()),
                bias=bias.tolist(), bias_over_sigma_f=(bias / sig_f).tolist(),
                bias_over_sigma_pe=(bias / sig_pe).tolist(),
                nd_bias_fisher=nd_bias(bias, cov_f), nd_bias_pe=nd_bias(bias, cov_pe),
                ess=float(weights.sum() ** 2 / np.sum(weights ** 2)), n_samples=int(len(samples)))


def print_summary(tag, S):
    print(f"\n[{tag}] ESS {S['ess']:.1f} of {S['n_samples']} samples; lnL best fit {float(S['lnL_best'])!r}, "
          f"PE mode {S['lnL_mode']!r}", flush=True)
    print(f"  {'param':9s} {'sigma_F':>11s} {'sigma_PE':>11s} {'PE/F':>6s} {'(mean-bf)/sF':>12s} "
          f"{'b/sF':>7s} {'b/sPE':>7s}")
    for j, n in enumerate(S["names"]):
        print(f"  {n:9s} {S['sigma_fisher'][j]:11.4e} {S['sigma_pe'][j]:11.4e} {S['sigma_ratio'][j]:6.3f} "
              f"{S['mean_shift_over_sigma_f'][j]:+12.3f} {S['bias_over_sigma_f'][j]:+7.2f} "
              f"{S['bias_over_sigma_pe'][j]:+7.2f}")
    print(f"  nD bias: Fisher {S['nd_bias_fisher']:.3f}, PE {S['nd_bias_pe']:.3f}; "
          f"max |corr_PE - corr_F| {S['max_abs_corr_diff']:.3f}")
    print(f"  PE mode = {S['mode_pe']!r}", flush=True)


# --- one run -----------------------------------------------------------------
def run(Gd, template, point, args):
    best, fish = load_best_fit(template, point), load_fisher(template, point)
    names = list(best["theta_final"])
    assert names == fish["names"], (names, fish["names"])
    theta = np.array([best["theta_final"][n] for n in names])
    theta_inj = np.array([best["theta_inj"][n] for n in names])
    cov = fisher_cov(fish["fisher"])
    lo, hi = prior_box(names, theta, np.sqrt(np.diag(cov)), args.psr, args.phase_psr)
    P = build_point(Gd, C.signal_row(point), names)
    set_context(P, lo, hi)
    ev = evaluate(P, theta)
    tag = f"{template} {C.point_label(point)}"
    print(f"[{tag}] {len(names)} parameters {names}\n  best fit overlap={ev['overlap']!r} "
          f"(stored {best['final_eval']['overlap']!r}) chi2={ev['chi2']!r} rho_s={ev['rho_s']:.6f}",
          flush=True)
    for n, a, b in zip(names, lo, hi):
        print(f"  prior {n:9s} [{float(a)!r}, {float(b)!r}]", flush=True)

    out_dir = PE_DIR / f"{template}_{C.point_label(point)}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = run_paris(out_dir, names, theta, cov, args)
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    lnL = sample_lnL(sampler)
    assert len(lnL) == len(samples), (len(lnL), len(samples))
    S = summarise(names, theta, theta_inj, cov, samples, weights, lnL, -0.5 * ev["chi2"])
    print_summary(tag, S)
    np.savez(out_dir / "pe_results.npz", samples=samples, weights=weights, names=np.array(names),
             lnL=lnL, theta_best=theta, theta_inj=theta_inj, cov_fisher=cov, bounds_lo=lo, bounds_hi=hi)
    (out_dir / "pe_summary.json").write_text(json.dumps(dict(
        grid=C.GRID, template=template, point=point, setup=C.SETUP, drop_dc=C.DROP_DC,
        tdi="2nd generation, AE", T=Gd["T"], dt=Gd["dt"], best_eval=ev, psr=args.psr,
        phase_psr=args.phase_psr, n_seed=args.n_seed, stop_dlogz=args.stop_dlogz, min_ess=args.min_ess,
        bounds_lo=lo.tolist(), bounds_hi=hi.tolist(), **S), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", choices=C.TEMPLATES, required=True)
    ap.add_argument("--point", choices=GRID_POINTS, default="24")
    ap.add_argument("--psr", type=float, default=PSR)
    ap.add_argument("--phase-psr", type=float, default=PHASE_PSR)
    ap.add_argument("--n-seed", type=int, default=N_SEED)
    ap.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    ap.add_argument("--stop-dlogz", type=float, default=STOP_DLOGZ)
    ap.add_argument("--min-ess", type=float, default=0.0, help="keep sampling until ESS >= this")
    ap.add_argument("--tag", default="", help="output folder suffix, e.g. psr10")
    ap.add_argument("--resume", action="store_true", help="continue from sampler_state.pkl")
    args = ap.parse_args()
    check_branch()
    xp, use_gpu = C.ovl.load_backend()
    Gd = build_grid(xp, use_gpu, *point_T_dt(C.signal_row(args.point)))
    run(Gd, args.template, args.point, args)


if __name__ == "__main__":
    main()
