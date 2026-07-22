import os
from typing import Optional
import numpy as np
import warnings
import traceback
import json
import time


class Config:

    def __init__(self, **kwargs):
    
        # Target SNR for Fisher scaling
        self.params_name = ["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
                         "Phi_phi0","Phi_theta0","Phi_r0"]
        
        self.injection_point_file = ""
        self.injection_point = np.loadtxt(self.injection_point_file) if self.injection_point_file else None
        if self.njection_point == None:
             self.injection_point =[1000000.0, 10.0, 0.9, 7.5, 0.5, 1.0, 5.0, 0.7853981633974483, 1.0,1.0,
                                 1.0471975511965976,0.9, 0.5, 0.4,5.0, 1.0, 0.0,0.0, 0.0]

        self.injection_model = "1PA"

        self.mlp_point_file = ""
        self.mlp_points = np.loadtxt(self.mlp_point_file) if self.mlp_point_file else None
        if self.mlp_points==None:
             self.mlp_points=  [1000381.6792690676, 10.002168560703451, 0.9002516899707742, 7.4984097121426085, 0.4999493588465228,1.0, 5.0,
 0.7753464299059312, 1.0095299530800796,1.0, 1.0471975511965976,0.9247984519297087, 0.5, 0.5286647646291917,5.0, 1.0, 0.0, 0.0, 0.0]


        self.analysis_model = "0PA"

        if  self.analysis_model == "0PA":
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0',"qS","phiS","Phi_phi0","Phi_r0"]
        elif self.analysis_model == "0PA_dev":
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0',"qS","phiS","Phi_phi0","Phi_r0",'dev_1','dev_2']
        else:
            self.param_names_to_infer = ['m1', 'm2', 'a', 'p0', 'e0',"qS","phiS","Phi_phi0","Phi_r0"]

        self.nchannels = 3  #Number of TDI channels to use (default 3 for A, E, T)

        self.basedir = "/scratch/e1583490/emri_5_params/"  #Base directory for saving results; can be overridden by --basedir CLI arg
        self.output_text_file = "paris_optimization_results.txt"  #File to save optimization results in text format

        self.use_gpu = True  #Whether to use GPU acceleration (default False for testing)
    
    def get_default_config(**kwargs):
        """
        Get default configuration with optional overrides.
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters to override
        Returns
        -------
        Config
            Configuration object

        Examples
        --------
        >>> cfg = get_default_config()
        >>> cfg = get_default_config(use_gpu=True, n_walkers=100)
        """
        return Config(**kwargs)

    def print_summary(self):
        """Print a detailed summary of current configuration."""

        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        # ------------------ PARAMETERS ------------------
        print("\n--- All Parameters ---")
        if len(self.params_name) != len(self.params):
            print("WARNING: params_name and params length mismatch!")

        for i, (name, value) in enumerate(zip(self.params_name, self.params)):
            tag = " (inferred)" if name in self.param_names_to_infer else ""
            print(f"[{i:02d}] {name:12s} : {value:.6e}{tag}")

        # ------------------ INFERENCE ------------------
        print("\n--- Inference Parameters ---")
        print(f"Parameters to infer ({len(self.param_names_to_infer)}):")
        for p in self.param_names_to_infer:
            print(f"  - {p}")

        # ------------------ COMPUTATION ------------------
        print("\n--- Computation Settings ---")
        print(f"dt                : {self.dt}")
        print(f"T                 : {self.T}")
        print(f"TARGET_SNR        : {self._TARGET_SNR}")
        print(f"include_noise     : {self.include_noise}")

        # ------------------ OPTIMIZER ------------------
        print("\n--- Optimizer Settings ---")
        print(f"optimizer         : {self.optimizer}")
        print(f"target_func       : {self.target_func}")
        print(f"nm_xatol          : {self.nm_xatol}")
        print(f"nm_fatol          : {self.nm_fatol}")

        # ------------------ PARIS ------------------
        print("\n--- PARIS Settings ---")
        print(f"spread_scale      : {self.spread_scale}")
        print(f"prior_sigma_range : {self.prior_sigma_range}")
        print(f"using_evec        : {self.using_evec}")
        print(f"seed_cloud        : {self.seed_cloud}")

        # ------------------ RUN SETUP ------------------
        print("\n--- Run Setup ---")
        print(f"grid_index        : {self.grid_index}")
        print(f"startingpoints    : {self.startingpoints}")
        print(f"parameter_selected: {self.parameter_selected}")
        print(f"run_type          : {self.run_type}")
        print(f"basedir           : {self.basedir}")

        # ------------------ DIAGNOSTICS ------------------
        print("\n--- Diagnostics ---")
        print(f"chi2              : {self.chi2}")
        print(f"dev_1             : {self.dev_1}")
        print(f"dev_2             : {self.dev_2}")

        print("\n" + "=" * 60)

    def to_dict(self):
        """Convert config to a serializable dictionary."""
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")  # optional: skip private vars
        }
    
    def save_results_with_config(cfg, results: dict, save_dir: str, filename_prefix: str):
        """
        Save results + config to:
        1. JSON (structured)
        2. Text file (human readable)
        """

        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # -------- JSON (structured) --------
        full_output = {
            "timestamp": timestamp,
            "config": cfg.to_dict(),
            "results": results,
        }

        json_path = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(full_output, f, indent=2)

        # -------- TEXT (human readable) --------
        text_path = os.path.join(save_dir, cfg.output_text_file)

        with open(text_path, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RUN TIMESTAMP: {timestamp}\n")

            # ---- CONFIG ----
            f.write("\n--- CONFIG ---\n")
            for k, v in cfg.to_dict().items():
                f.write(f"{k}: {v}\n")

            # ---- RESULTS ----
            f.write("\n--- RESULTS ---\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")

            f.write("=" * 80 + "\n")

        print(f"[SAVE] JSON: {json_path}")
        print(f"[SAVE] TEXT: {text_path}")

    if __name__ == "__main__":
        cfg = get_default_config()
        cfg.print_summary()


