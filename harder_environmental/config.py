"""Every setting of the CV -> DE -> CV run, in one place.

Same arrangement as bias_inference_emri/src/config_paris.py: one Config object holds the
point, the model and the optimiser settings, and the runner reads nothing else.  Change a
number here, not in cv_de_cv.py.

The CV controls (LAMBDA0, LM_MAX_ITERS, ...) are the ones cv_population.py uses.  They are
repeated here rather than imported because the whole point of this directory is to vary
them: `apply_cv_controls` pushes them back into that module before any climb runs, so the
two stages of CV in this pipeline can be tuned independently of the population run.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_TESTS = os.path.join(os.path.dirname(HERE), "environmental_tests")

# The circular population (e0 = 0, xI0 = 1) drops the two parameters an eccentric source
# carries: Phi_r0, which has no radial motion to phase, and C_e, which deforms edot.  These
# mirror cv_population.PARAMS_0PA / PARAMS_PN exactly, so a point taken from that run and a
# point run here are the same recovery model.
PARAMS_0PA = ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0"]
PARAMS_PN = PARAMS_0PA + ["C_p"]

# Periodic parameters.  qS is polar, bounded to [0, pi], so it is not in this set.
ANGLE_PARAMS = ("phiS", "Phi_phi0")

# Physical limits the DE box is clipped to, as in bias_inference_emri/src/inference.py.
# e0 floors at 0.0, not 1e-3: the injection sits exactly at zero, so the old floor would put
# the truth outside its own search box.  The box is one-sided there by construction.
PHYS_LO = {"m1": 1e3, "m2": 1.0, "a": -0.999, "p0": 1.0, "e0": 0.0,
           "qS": 0.0, "phiS": 0.0, "Phi_phi0": 0.0}
PHYS_HI = {"a": 0.999, "e0": 0.99,
           "qS": np.pi, "phiS": 2 * np.pi, "Phi_phi0": 2 * np.pi}


class Config:
    """Settings for one CV -> DE -> CV run on one frozen source."""

    def __init__(self, **kwargs):

        # ---- the point ----
        # point_idx selects the source; the frozen json is only a cached copy of it, named
        # after the source so two points can never be confused for one another.  A missing
        # frozen file is not an error -- the runner rebuilds the point from the results file.
        # "" -> environmental_tests/results_env_cv.json, the live circular run; a name like
        # "useless/run_2" -> that archived run's results file instead.
        self.point_run = ""
        self.point_idx = 8

        # ---- recovery model ----
        # '0PA'    -> the 9 vacuum parameters
        # '0PA+PN' -> those 9 plus C_p, C_e
        # Templates never carry the environment; only the injection does.
        self.model = "0PA"

        # Parameters held at their injected value instead of being fitted.  e0 is fixed by
        # default on the hard points: the circular population injects e0 = 0 exactly, so its
        # Fisher row is a boundary derivative -- SEF warns "minimum relative error is greater
        # than 1% for e0" on every source, sigma(e0) came out anywhere from 2e-10 to 0.53
        # across the population, and on source 13 a step capped at 2 of those sigma proposed
        # e0 = 1.57, which is not a physical eccentricity.  A fixed parameter keeps its source
        # value in the waveform (build_ctx only overrides the names it is given) and drops out
        # of the Fisher, the DE box and the bias table entirely.
        self.fixed_params = ("e0",)

        # ---- stage 1: the CV step off the injected point ----
        # "One step of CV bias": a single accepted LM iteration, which is the Gauss-Newton
        # step dtheta = Gamma^-1 <dh|s-h> that the CV formalism predicts.  Its Fisher is
        # also what sets the DE box, so this stage is where sigma comes from.
        # cv1_exact=True takes the complete undamped step dtheta = Gamma^-1 <dh|s-h>, which is
        # what the CV formalism predicts.  False falls back to `cv1_iters` LM iterations, i.e.
        # the same step shrunk by lambda*diag(Gamma) and subject to a gain-ratio accept.
        self.cv1_exact = True
        # The exact step is taken only at a length that raises the overlap: 1, 1/2, 1/4, ...
        # The stage-1 point centres the DE box, so a step that lands worse than it started
        # does not just waste an iteration, it puts the truth outside the box for every later
        # stage.  False restores the pure undamped step.
        self.cv1_backtrack = True
        self.cv1_max_halvings = 12
        # Stage 1 as a trust-region climb rather than one leap.  On source 8 the single
        # undamped step walked 30 sigma (0PA) / 58 sigma (PN) from the injected point while
        # the DE box is only +-15 sigma, so the box could not contain its own starting point.
        # Capping each step and the cumulative walk keeps the box self-consistent; iterating
        # recovers the overlap the shorter steps give up.
        self.cv1_steps = 8               # 1 reproduces the old single-step stage 1
        self.cv1_step_cap_sigma = 2.0    # per-step cap, in sigma at the injected point
        self.cv1_total_cap_sigma = 5.0   # cap on the whole walk, same units
        self.cv1_iters = 1
        self.cv1_nm = False          # no Nelder-Mead refinement before DE

        # Skip stage 1 entirely and centre the DE box on the injected point.  The capped walk
        # moves the centre to a better overlap, but it also moves the prior off the truth; with
        # this the prior is +-prior_sigma_range sigma about the injected parameters themselves.
        self.box_at_injection = False

        # ---- seeding a deviation run from the 0PA answer ----
        # A PN template at C_p = 0 *is* the 0PA template, so the 0PA best fit is a feasible
        # PN point and the PN maximum can never lie below it.  DE did not respect that: on
        # both hard points it converged (success = True) to chi2 ~ 4e4 where the 0PA arm had
        # already reached 5.6e3 (source 8) and 11.9 (source 13).  Naming a 0PA results file
        # here drops DE entirely and climbs from that point with the deviation at zero, which
        # makes PN >= 0PA true by construction rather than by hope.
        #
        # Any finished run may be named, not only a 0PA one: the seed is matched by parameter
        # name, so a PN fit reseeds a PN run, and a parameter this run frees that the seed
        # file held fixed (e0) enters at its injected value.  That is the same point the seed
        # run occupied, so the climb again starts at its overlap and can only rise.
        self.seed_fit = ""               # path to any results_cv_de_cv_*.json

        # ---- stage 2: differential evolution inside the Fisher box ----
        self.de_maxiter = 1100
        self.prior_sigma_range = 15.0    # box half-width in units of the stage-1 sigma
        self.per_param_sigma = {}        # per-parameter override, e.g. {'Phi_r0': 30.0}
        self.dev_prior_sigma_range = 15.0    # C_p when model == '0PA+PN'
        self.dev_hard_bounds = None          # (lo, hi) overriding the Fisher box for C_p
        # sigma_C_p is not a usable box unit: it came out as 7e-3, 2.7 and 1.5e6 on three
        # similar sources.  Instead bound C_p by where it stops meaning anything -- the
        # calibrated step gives mismatch DEV_MISMATCH_HI, mismatch grows quadratically, so
        # the template fully decoheres near d_hi / sqrt(DEV_MISMATCH_HI).  Searching beyond
        # that samples noise, which is how the source-8 fit reached C_p = 2088.
        self.dev_bounds_from_decoherence = True
        self.dev_decoh_margin = 10.0         # multiples of the decoherence scale to allow
        self.full_phase_bounds = False   # True -> qS, phiS, Phi_phi0 span [phys_lo, phys_hi]
        self.de_objective = "chi2"       # 'chi2' (minimise the residual) | 'overlap'
        self.de_tol = 1e-8
        self.de_atol = 1e-5
        self.de_popsize = 15
        self.de_init = "sobol"
        self.de_mutation = (0.5, 1.0)
        self.de_recombination = 0.7
        self.de_polish = False           # stage 3 is the polish, and it is a better one
        self.de_seed = 42
        self.de_report_every = 200       # progress line every N objective evaluations
        self.de_x0_from_cv = True        # seed the population with the stage-1 point

        # ---- stage 3: CV from the DE winner ----
        self.cv2_iters = 150
        self.cv2_nm = True               # LM -> Nelder-Mead -> LM, as the population run does

        # ---- CV controls, pushed into cv_population before any climb ----
        self.overlap_target = 0.9999999999
        self.lambda0 = 1e-2
        self.lambda_min, self.lambda_max, self.lambda_up = 1e-12, 1e8, 10.0
        self.max_inner = 30
        self.rel_tol = 1e-9
        self.nm_maxiter, self.nm_step = 1000, 2.0
        self.recompute_deltas_every = 10

        # ---- I/O ----
        self.tag = ""                    # suffix on the output file; "" -> results_cv_de_cv.json
        self.out_dir = HERE

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"unknown config field: {k}")
            setattr(self, k, v)

    # -------------------------------------------------------------- derived

    @property
    def param_names(self):
        """The parameters being optimised, in order, minus anything held fixed."""
        base = PARAMS_0PA if self.model == "0PA" else PARAMS_PN
        return [n for n in base if n not in self.fixed_params]

    @property
    def point_json(self):
        """The frozen copy of this source, written by freeze_point.py."""
        return os.path.join(HERE, f"hard_point_{self.point_idx}.json")

    @property
    def point_results(self):
        """The population results file the point is read out of."""
        return os.path.join(ENV_TESTS, self.point_run, "results_env_cv.json")

    @property
    def dev_params(self):
        """The deviation parameters of the chosen model, empty for 0PA or if held fixed."""
        return [n for n in ([] if self.model == "0PA" else ["C_p"])
                if n not in self.fixed_params]

    @property
    def out_path(self):
        suffix = f"_{self.tag}" if self.tag else ""
        return os.path.join(self.out_dir, f"results_cv_de_cv{suffix}.json")

    def as_dict(self):
        """Every setting, for the record written beside the results."""
        out = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        out["model_params"] = self.param_names
        out["point_json"] = self.point_json
        return out


def apply_cv_controls(cfg, cvp):
    """Push the CV controls into the cv_population module the climbs live in.

    cv_population keeps them as module-level constants, which is right for a run where they
    never change; here they are the thing being varied, so they are set from the config once,
    loudly, before any climb starts.
    """
    cvp.OVERLAP_TARGET = cfg.overlap_target
    cvp.LAMBDA0 = cfg.lambda0
    cvp.LAMBDA_MIN, cvp.LAMBDA_MAX, cvp.LAMBDA_UP = cfg.lambda_min, cfg.lambda_max, cfg.lambda_up
    cvp.MAX_INNER = cfg.max_inner
    cvp.REL_TOL = cfg.rel_tol
    cvp.NM_MAXITER, cvp.NM_STEP = cfg.nm_maxiter, cfg.nm_step
    cvp.RECOMPUTE_DELTAS_EVERY = cfg.recompute_deltas_every
    print(f"[cfg] CV controls: overlap_target={cfg.overlap_target}, lambda0={cfg.lambda0:g}, "
          f"rel_tol={cfg.rel_tol:g}, recompute_deltas_every={cfg.recompute_deltas_every}",
          flush=True)
