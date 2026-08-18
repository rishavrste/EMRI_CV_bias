"""Configuration for the Cutler-Vallisneri EMRI/IMRI bias analysis.

Import the ready-to-use ``cfg`` object:

    from params import cfg

It exposes exactly what ``run_analysis.py`` consumes:
``signal_row``, ``mlp_points``, ``n_channels``, ``use_gpu``, ``analysis``,
and ``params_to_infer``.
"""

import numpy as np


class Config:
    """Analysis configuration.

    A parameter *row* has 19 entries in the following fixed order::

        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,   # 14 EMRI parameters
        dt, T, chi2, dev_1, dev_2       # sampling, duration, spin, deviations
    """

    # Names of the 14 EMRI parameters, in waveform order.
    PARAM_NAMES = [
        "m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS",
        "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0",
    ]

    # Parameters inferred (Fisher / bias) for each analysis type.
    INFERRED_PARAMS = {
        "0PA":     ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"],
        "1PA":     ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0"],
        "0PA_dev_simple": ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0",
                    "dev_1", "dev_2"],
        "0PA_dev_PN": ["m1", "m2", "a", "p0", "e0", "qS", "phiS", "Phi_phi0", "Phi_r0",
                    "dev_1", "dev_2"],
    }

    # Injected "truth" row and the approximate-model maximum-likelihood point.
    # DEFAULT_SIGNAL_ROW = [
    #     1000000.0, 10.0, 0.9, 7.5, 0.5, 1.0, 0.5,
    #     0.7853981633974483, 1.0, 1.0, 1.0471975511965976, 0.9, 0.5, 0.4,
    #     5.0, 1.0, 0.0, 0.0, 0.0,
    # ]

    # DEFAULT_SIGNAL_ROW = [1000000.0, 10.0, 0.9, 9.07414088, 0.2, 1.0, 1,
    #     1.04719755, 0.785398163, 0.628318531, 0.523598776, 0.1, 0.2, 0.3,
    #     10.0, 2.5, 0.95, 0.0, 0.0]
    DEFAULT_SIGNAL_ROW = [1000000.0, 10.0, 0.5, 9.97066819, 0.3, 1.0,1,
        1.04719755, 0.785398163, 0.628318531, 0.523598776, 0.1, 0.2, 0.3,
        10.0, 2.5, 0.95, 0.0, 0.0]

    # DEFAULT_MLP_POINTS = [
    #     1000381.6792690676, 10.002168560703451, 0.9002516899707742,
    #     7.4984097121426085, 0.4999493588465228, 1.0, 0.5,
    #     0.7753464299059312, 1.0095299530800796, 1.0, 1.0471975511965976,
    #     0.9247984519297087, 0.5, 0.5286647646291917,
    #     5.0, 1.0, 0.0, 0.0, 0.0,
    # ]
    # DEFAULT_MLP_POINTS =  [999996.7029121468, 10.000086186854665, 0.9000030960602112, 9.074165142663816, 0.19999631732547302, 1.0, 1,
    #     1.04654519796294, 0.782682446069237, 0.628318531, 0.523598776, 0.15561239326223392, 0.2, 0.08978856112836722,
    #     10.0, 2.5, 0.95, 0.0, 0.0]

    DEFAULT_MLP_POINTS =  [1000036.1198187952, 10.000133526144388,  0.500052021086232, 9.970417145876961, 0.2999918756916836, 1.0, 1,
       1.0453065910007755, 0.7838444158820905, 0.628318531, 0.523598776, 0.1142105583644897, 0.2, 0.25394659948792797,
        10.0, 2.5, 0.95, 0.0, 0.0]
    


    def __init__(
        self,
        analysis="0PA",
        injection_model="1PA",
        n_channels=3,
        use_gpu=True,
        signal_row_file="",
        mlp_points_file="",
    ):
        if analysis not in self.INFERRED_PARAMS:
            raise ValueError(
                f"Unsupported analysis {analysis!r}; "
                f"choose from {sorted(self.INFERRED_PARAMS)}."
            )
        if n_channels not in (2, 3):
            raise ValueError(
                f"n_channels must be 2 (A, E) or 3 (A, E, T); got {n_channels}."
            )

        self.analysis = analysis            # "0PA", "1PA", or "0PA_dev"
        self.injection_model = injection_model
        self.n_channels = n_channels
        self.use_gpu = use_gpu

        self.signal_row = self._load_row(signal_row_file, self.DEFAULT_SIGNAL_ROW)
        self.mlp_points = self._load_row(mlp_points_file, self.DEFAULT_MLP_POINTS)
        self.params_to_infer = list(self.INFERRED_PARAMS[analysis])

        self._validate_row("signal_row", self.signal_row)
        self._validate_row("mlp_points", self.mlp_points)

    @staticmethod
    def _load_row(path, default):
        """Load a parameter row from ``path`` (whitespace-separated), else default."""
        if path:
            return np.loadtxt(path).tolist()
        return list(default)

    @staticmethod
    def _validate_row(name, row):
        if len(row) != 19:
            raise ValueError(f"{name} must have 19 entries, got {len(row)}.")

    # Convenience accessors -------------------------------------------------
    @property
    def dt(self):
        return self.signal_row[14]

    @property
    def T(self):
        return self.signal_row[15]

    def to_dict(self):
        """Return a JSON-serialisable view of the public configuration."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def print_summary(self):
        """Print the active configuration."""
        print("=" * 60)
        print("CUTLER-VALLISNERI BIAS CONFIGURATION")
        print("=" * 60)
        print(f"analysis         : {self.analysis}")
        print(f"injection_model  : {self.injection_model}")
        print(f"n_channels       : {self.n_channels}")
        print(f"use_gpu          : {self.use_gpu}")
        print(f"dt / T           : {self.dt} / {self.T}")
        print(f"params_to_infer  : {self.params_to_infer}")

        print("\n--- Injected parameters (truth) ---")
        for name, value in zip(self.PARAM_NAMES, self.signal_row[:14]):
            tag = "  <-- inferred" if name in self.params_to_infer else ""
            print(f"  {name:12s} : {value:.6e}{tag}")
        chi2, dev_1, dev_2 = self.signal_row[16:19]
        print(f"  {'chi2':12s} : {chi2:.6e}")
        print(f"  {'dev_1':12s} : {dev_1:.6e}")
        print(f"  {'dev_2':12s} : {dev_2:.6e}")
        print("=" * 60)


# Default configuration imported by run_analysis.py.
cfg = Config()


if __name__ == "__main__":
    cfg.print_summary()
