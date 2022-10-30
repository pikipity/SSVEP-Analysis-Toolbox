# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal

import numpy as np

from .algsupport import gen_ref_sin, floor

def ref_sig_fun(dataself, sig_len: float, N: int, phases: List[float], srate: float):
    """
    Because downsampling, the sampling rate of reference signals also should be changed
    """
    L = floor(sig_len * dataself.srate)
    ref_sig = [gen_ref_sin(freq, dataself.srate, L, N, phase) for freq, phase in zip(dataself.stim_info['freqs'], phases)]
    ref_sig = [signal.resample_poly(sig, 1000*srate, 1000*dataself.srate, axis = 1) for sig in ref_sig]
    return ref_sig

def suggested_weights_filterbank(num_subbands : int = 1) -> List[float]:
    if num_subbands == 1:
        return [1 for _ in range(num_subbands)]
    else:
        return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def suggested_ch() -> List[int]:
    """
    Provide suggested channels for openBMI

    Returns
    -------
    ch_used: List
        Suggested channels
    """
    return list(range(23-1,32))

def preprocess(dataself,
               X: ndarray,
               downsample_srate: float) -> ndarray:
    """
    Suggested preprocessing function for openBMI

    resample to 100 Hz
    """

    preprocess_X = signal.resample_poly(X, 1000*downsample_srate, 1000*dataself.srate, axis = 1)

    band = [0.5, 40]
    bpB, bpA = signal.butter(5, [band_val/(downsample_srate/2) for band_val in band], 'bandpass')
    tmp = signal.filtfilt(bpB, bpA, preprocess_X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
    tmp = signal.detrend(tmp, axis = 1)
    tmp = tmp - np.mean(tmp, axis = 1, keepdims = True)
    tmp_std = np.std(tmp, axis = 1, keepdims = True, ddof=1)
    preprocess_X = tmp / tmp_std
    
    return preprocess_X

def filterbank(dataself,
               X: ndarray,
               srate: float,
               num_subbands: int = 1) -> ndarray:
    """
    Suggested filterbank function for openBMI
    """

    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))

    # band = [0.5, 40]
    # bpB, bpA = signal.butter(5, [band_val/(srate/2) for band_val in band], 'bandpass')
    # tmp = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
    # tmp = signal.detrend(tmp, axis = 1)
    # tmp = tmp - np.mean(tmp, axis = 1, keepdims = True)
    # tmp_std = np.std(tmp, axis = 1, keepdims = True, ddof=1)
    # tmp = tmp / tmp_std
    # filterbank_X[0,:,:] = tmp.copy()
    
    for k in range(1, num_subbands+1, 1):
        if k == 1:
            filterbank_X[k-1,:,:] = X.copy()
        else:
            band = [60/11*k, 40]
            bpB, bpA = signal.butter(5, [band_val/(srate/2) for band_val in band], 'bandpass')
            tmp = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
            tmp = signal.detrend(tmp, axis = 1)
            tmp = tmp - np.mean(tmp, axis = 1, keepdims = True)
            tmp_std = np.std(tmp, axis = 1, keepdims = True, ddof=1)
            filterbank_X[k-1,:,:] = tmp / tmp_std
            # Wp = [(60/11*k)/(srate/2), 80/(srate/2)]
            # Ws = [(60/11*k-1)/(srate/2), 90/(srate/2)]
            # N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
            
            # bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
            # tmp = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
            # filterbank_X[k-1,:,:] = signal.detrend(tmp, axis = 1)
        
    return filterbank_X

