import os
import json
import signal
import time
import argparse
from typing import Tuple, Optional,Dict, Any
from scipy.signal.windows import tukey
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    xp=cp
except:
    xp=np
    print("CuPy not found, using NumPy instead.")


def compute_fft_with_windowing(waveform, dt, N,type=None,use_gpu=False,n_channels=3):
    if use_gpu:
        try:
            xp = cp
        except ImportError:
            print("[WARN] CuPy not available, falling back to NumPy")
            xp = np
    else:
        xp = np
    if type == 'tukey':
        window = xp.asarray(tukey(N, 0.01))
        waveform_windowed = waveform * window
        waveform_f = xp.asarray([xp.fft.rfft(waveform_windowed[i]) * dt for i in range(n_channels)])[:,1:]
    else:
        waveform_f = xp.asarray([xp.fft.rfft(waveform[i]) * dt for i in range(n_channels)])[:,1:]
    return waveform_f

def _is_pos_def(mat: np.ndarray) -> bool:
    """Return True if matrix is positive-definite via Cholesky test."""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    return 4 * delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj() / PSD))

def inner_prod_without_phase(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    return 4 * delta_f * xp.abs(xp.sum(signal_1_f * signal_2_f.conj() / PSD))


def fishinv(M, Fisher, index_of_M=0):
    J = np.eye(len(Fisher))
    J[index_of_M, index_of_M] = M

    Fisher_lnM = J.T @ Fisher @ J
    Fisher_lnM_inv = np.linalg.inv(Fisher_lnM)  # Jacobian for Covariance = partial new/partial old, going from lnM -> M
    Fisher_inv = J.T @ Fisher_lnM_inv @ J
    return Fisher_inv