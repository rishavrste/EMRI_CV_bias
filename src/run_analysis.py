import os
import json
from random import seed
import time
import argparse
from typing import Tuple, Optional
import re

import numpy as np
from scipy.optimize import minimize

#few and SEF imports
from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.geodesic import get_fundamental_frequencies
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from few.utils.constants import MTSUN_SI

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product,fishinv
from stableemrifisher.fisher import StableEMRIFisher
import matplotlib.pyplot as plt
from misc import compute_fft_with_windowing, _is_pos_def

from params import cfg
try:
    import cupy as cp
    xp=cp
except:
    xp=np
    print("CuPy not found, using NumPy instead.")


(m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0, dt, T, chi2, dev_1, dev_2
    ) = cfg.signal_row

mlp_points = cfg.mlp_points

noise= cfg.include_noise
nchannels = cfg.n_channels
use_gpu = cfg.use_gpu
params_to_infer = cfg.params_to_infer

sum_kwargs = dict(pad_output=True, odd_len=True)
waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform, sum_kwargs=sum_kwargs, return_list=False,use_gpu=use_gpu)

t0 = 10000.0
tdi_gen = "1st generation"
order = 20
index_lambda = 8  # phiS
index_beta = 7    # qS

channels = [A1TDISens, E1TDISens, T1TDISens]
if nchannels == 3:
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    tdi_chan = "AET"
elif nchannels == 2:
    noise_kwargs = [{"sens_fn": ch} for ch in channels[:2]]
    tdi_chan = "AE"
else:
    raise ValueError(f"Unsupported number of channels: {nchannels}. Only 2 (A,E) or 3 (A,E,T) are supported.")

waveform_response = ResponseWrapper(
        waveform_gen=waveform_model,
        Tobs=T,
        t0=t0,
        dt=dt,
        index_lambda=index_lambda,
        index_beta=index_beta,
        flip_hx=True,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend = "cuda12x" if use_gpu else "cpu",
        order=order,
        tdi=tdi_gen,
        tdi_chan=tdi_chan)

print("[INFO] Finished loading modules and building ResponseWrapper")

wave_params=[]
waveform_true = xp.array(waveform_response(*wave_params))[0:nchannels,:]  # Shape (3, N) for A, E, T channels
print("[INFO] Finished generating true waveform")

PSD_funcs = generate_PSD(
    waveform=waveform_true,
    dt=dt,
    noise_PSD=get_sensitivity,
    channels=channels,
    noise_kwargs=noise_kwargs,
    use_gpu=use_gpu,)

# Verify SNR level (grid builder normalized dist to target PA2 SNR already)
PSD_funcs_ = xp.array(PSD_funcs)

print("shape check - waveform_true:", xp.shape(waveform_true), "PSD_funcs:", xp.shape(PSD_funcs_))
snr_2 = inner_product(waveform_true, waveform_true, PSD_funcs_, dt, use_gpu=use_gpu)
#if hasattribute get u
snr = np.sqrt(snr_2.get()) if hasattr(snr_2, "get") else np.sqrt(snr_2)
print(f"[TRUE] SNR: {snr:.6f}")     
for i in range(PSD_funcs_.shape[0]):    
   #check snr of each channel
     snr_i_2 = inner_product(waveform_true[i:i+1,], waveform_true[i:i+1,], PSD_funcs_[i:i+1,:], dt, use_gpu=use_gpu)
     snr_i = np.sqrt(snr_i_2.get()) if hasattr(snr_i_2, "get") else np.sqrt(snr_i_2)
     print(f"[TRUE] SNR - Channel {i}: {snr_i:.6f}")

     
N_fiducial = len(waveform_true[0])
freq=np.fft.rfftfreq(N_fiducial, dt)
delta_f = freq[1] - freq[0]
print(f"[INFO] Frequency resolution delta_f: {delta_f:.6e} Hz, Number of frequency bins: {len(freq)}")
print(f"freq_max: {freq[-1]:.6f} Hz and Freq_min: {freq[1]:.6f} Hz")


waveform_true_fft = compute_fft_with_windowing(waveform_true, dt, N_fiducial, use_gpu=use_gpu, n_channels=nchannels)
print("[INFO] Finished preparing true waveform (GPU)")

sef = StableEMRIFisher(waveform_class=SuperKludgeWaveform,
                       waveform_class_kwargs = dict(sum_kwargs=dict(pad_output=False, odd_len=True)),
                       waveform_generator = GenerateEMRIWaveform,
                       waveform_generator_kwargs= dict(return_list=False),
                       ResponseWrapper=ResponseWrapper,
                       ResponseWrapper_kwargs = dict(Tobs=T,
                                                    t0=10000.0,
                                                    dt=dt,
                                                    index_lambda=8,
                                                    index_beta=7,
                                                    flip_hx=True,
                                                    is_ecliptic_latitude=False,
                                                    remove_garbage="zero",
                                                    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                                                    force_backend = "cuda12x" if use_gpu else "cpu",
                                                    order=20,
                                                    tdi="1st generation",
                                                    tdi_chan=tdi_chan),
                       stats_for_nerds = True, use_gpu = use_gpu,
                       deriv_type='stable',
                       noise_model=get_sensitivity,
                       noise_kwargs=noise_kwargs,
                       channels=channels,
                       T = T, dt = dt,
                       stability_plot = False,
                       der_order = 6, Ndelta = 12,
                       plunge_check=True, return_derivatives=True
                       )

emri_kwargs = {"T":T, "dt":dt}
additional_kwargs={"chi2":chi2,"evolve_1PA":False,"evolve_primary":False,"evolve_2PA":False,"deviation_included":False,"dev_1":dev_1,"dev_2":dev_2}

if cfg.analysis == "0PA":
    #use mlp params
    fisher_params = mlp_points[0:14]
    additional_kwargs['evolve_1PA']=False
    pars_list_com = list(fisher_params) + [chi2,additional_kwargs['evolve_1PA'],additional_kwargs['evolve_primary'],
     additional_kwargs['evolve_2PA'],additional_kwargs['deviation_included'],additional_kwargs['dev_1'],additional_kwargs['dev_2']]
elif cfg.analysis == "1PA":
    #use mlp params
    fisher_params = mlp_points[0:14]
    additional_kwargs['evolve_1PA']=True
    pars_list_com = list(fisher_params) + [chi2,additional_kwargs['evolve_1PA'],additional_kwargs['evolve_primary'],
     additional_kwargs['evolve_2PA'],additional_kwargs['deviation_included'],additional_kwargs['dev_1'],additional_kwargs['dev_2']]
elif cfg.analysis == "0PA_dev":
    #use mlp params
    fisher_params = mlp_points[0:14]
    additional_kwargs['evolve_1PA']=False
    additional_kwargs['deviation_included']=True
    pars_list_com = list(fisher_params) + [chi2,additional_kwargs['evolve_1PA'],additional_kwargs['evolve_primary'],
    additional_kwargs['evolve_2PA'],additional_kwargs['deviation_included'],additional_kwargs['dev_1'],additional_kwargs['dev_2']]
else:
    raise ValueError(f"Unsupported analysis type: {cfg.analysis}. Supported types are '0PA', '1PA', and '0PA_dev'.")
    
SNR = sef.SNRcalc_SEF(*pars_list_com,**emri_kwargs,use_gpu=use_gpu)
print("SNR: ", SNR)

param_dict = {
    'm1': fisher_params[0],
    'm2': fisher_params[1],
    'a': fisher_params[2],
    'p0': fisher_params[3],
    'e0': fisher_params[4],
    'xI0': fisher_params[5],
    'dist': fisher_params[6],
    'qS': fisher_params[7],
    'phiS': fisher_params[8],
    'qK': fisher_params[9],
    'phiK': fisher_params[10],
    'Phi_phi0': fisher_params[11],
    'Phi_theta0': fisher_params[12],
    'Phi_r0': fisher_params[13]}

Fisher = sef(wave_params = param_dict,param_names=params_to_infer, add_param_args=additional_kwargs,
        live_dangerously = False, stability_plot = True,der_order = 8, Ndelta = 20,)
               
try:
    F = np.asarray(Fisher[0], dtype=float)
    print(f"[FISHER] {repr(F)}")
    print(f"[FISHER_IS_PD] {_is_pos_def(F)}")
    F_inv = fishinv(param_dict['m1'], Fisher, index_of_M=0)
    print(f"[FISHER_INV] {repr(F_inv)}")
    print(f"[FISHER_INV_IS_PD] {_is_pos_def(F_inv)}")
    F_std = np.sqrt(np.diag(F_inv))
    print(f"[FISHER_STD] {repr(F_std)}")
except Exception as e:
    raise RuntimeError(f"Fisher computation failed: {e}")

waveform_tmpl = xp.array(sef.waveform)
print("shape of waveform_tmpl: ", waveform_tmpl.shape)

partial_derivs = xp.array(Fisher[1])

def cutlervallis(waveform_truth, waveform_approx, Fisher_truth, partial_approx, best_fit, PSD_func, dt, use_gpu=False):
    """
    Calculate best-fit param points using the Cutler-Vallisneri linear-bias approximation.
    params:
        waveform_truth (ndarray) : the time-series waveform from the true template at params_truth. ndarray of shape 'N x M' where N is number of LISA channels and M is the time-series length
        waveform_approx (ndarray) : the time-series waveform from the approximate template at params_truth. ndarray of shape 'N x M' where N is number of LISA channels and M is the time-series length
        partial_approx (ndarray) : the time-series partial derivative of the approximate template at params_truth. ndarray of shape 'N x M' where N is number of LISA channels and M is the time-series length
        Fisher_truth (ndarray) : the Fisher matrix (log-mass units) at params_truth. ndarray of shape 'd x d'
        params_truth (ndarray) : the array of true params. ndarray of shape 'd'
        PSD_func (ndarray) : the frequency-domain LISA noise sensitivity curve. ndarray of shape 'N x L' where L is the length of the frequency series. 

    returns:
        CV_bias (ndarray): params_truth + np.linalg.inv(Fisher_truth) @ inner_product(partial_approx, waveform_truth - waveform_approx)
    """

#Equation 29 of cutler-vallisneri paper
    if use_gpu:
        xp = cp
    else:
        xp = np
        
    #waveform_truth = padding(waveform_truth, waveform_approx, use_gpu = use_gpu) #make waveform_truth the same length as waveform_approx
    delta_wave = waveform_truth - waveform_approx
    #truth contains 1 order PN and approx contains 0 order PN with deviations

    # calculate all the column-wise inner products
    inn_prods = []
    
    for j in range(len(Fisher_truth)):
        
        print("delta_wave.shape, partial_approx.shape: ", delta_wave.shape, partial_approx[j].shape)
                      
        inn_prod_j = inner_product(partial_approx[j], delta_wave, PSD=PSD_func, dt=dt, use_gpu = use_gpu) #should be a scalar
        if not use_gpu:
            inn_prod_j = xp.asarray(inn_prod_j)
        else:
            inn_prod_j = xp.asnumpy(inn_prod_j)
            
        print(f'inner_prod at j = {j}: ', inn_prod_j)
        inn_prods.append(inn_prod_j)
        
    inn_prods = np.array(inn_prods)
    # calculate all the param shifts
    delta_param_all = []
    
    for i in range(len(Fisher_truth)):
            
        delta_param_i = np.linalg.inv(Fisher_truth)[i,:]@inn_prods  #should be a scalars
        print(f"{params_to_infer[i]} have delta {delta_param_i}")
        delta_param_all.append(delta_param_i)
        
    print('delta_param_all: ', delta_param_all)
    return best_fit + np.array(delta_param_all)