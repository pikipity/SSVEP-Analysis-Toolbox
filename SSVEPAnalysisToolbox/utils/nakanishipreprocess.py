# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal

import numpy as np

def suggested_weights_filterbank(num_subbands: Optional[int] = 1) -> List[float]:
    """
    Provide suggested weights of filterbank for Nakanishi 2015

    Returns
    -------
    weights_filterbank : List[float]
        Suggested weights of filterbank
    """
    return [1 for _ in range(num_subbands)]

def suggested_ch() -> List[int]:
    """
    Provide suggested channels for Nakanishi 2015

    Returns
    -------
    ch_used: List
        Suggested channels
    """
    return [i for i in range(8)]

def preprocess(X: ndarray,
               srate: Optional[float] = 256) -> ndarray:
    """
    Suggested preprocessing function for Nakanishi 2015
    """
    
    # notch filter at 60 Hz
    # f0 = 60
    # Q = 35
    # notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=srate)
    # preprocess_X = signal.filtfilt(notchB, notchA, X, axis = 1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))

    preprocess_X = signal.detrend(X, axis = 1)
    
    return preprocess_X

def filterbank(X: ndarray,
               srate: Optional[float] = 256,
               num_subbands: Optional[int] = 1) -> ndarray:
    """
    Suggested filterbank function for Nakanishi 2015
    """
    
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
    
    for k in range(1, num_subbands+1, 1):
        Wp = [(6*k)/(srate/2), 80/(srate/2)]
        Ws = [(6*k-2)/(srate/2), 90/(srate/2)]
        N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
        
        bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')

        filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
        
    return filterbank_X

