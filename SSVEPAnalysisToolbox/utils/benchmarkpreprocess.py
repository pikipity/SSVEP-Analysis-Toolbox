# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal

import numpy as np

def suggested_ch() -> List:
    """
    Provide suggested channels for benchmark dataset

    Returns
    -------
    ch_used: List
        Suggested channels
    """
    return [i-1 for i in [48, 54, 55, 56, 57, 58, 61, 62, 63]]

def preprocess(X: ndarray,
               srate: Optional[float] = 250) -> ndarray:
    """
    Suggested preprocessing function for benchmark dataset
    
    notch filter at 50 Hz
    """
    
    # notch filter at 50 Hz
    f0 = 50
    Q = 35
    notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=srate)
    preprocess_X = signal.filtfilt(notchB, notchA, X, axis = 1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))
    
    return preprocess_X
    
def filterbank(X: ndarray,
               srate: Optional[float] = 250,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
    
    for k in range(1, num_subbands+1, 1):
        Wp = [(8*k)/(srate/2), 90/(srate/2)]
        Ws = [(8*k-2)/(srate/2), 100/(srate/2)]
        N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
        
        bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')

        filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
        
    return filterbank_X