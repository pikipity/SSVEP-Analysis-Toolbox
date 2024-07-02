# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal

import numpy as np

def suggested_weights_filterbank(num_subbands: Optional[int] = 5) -> List[float]:
    """
    Provide suggested weights of filterbank for benchmark dataset

    Returns
    -------
    weights_filterbank : List[float]
        Suggested weights of filterbank
    """
    return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def suggested_ch() -> List[int]:
    """
    Provide suggested channels for benchmark dataset

    Returns
    -------
    ch_used: List
        Suggested channels (PZ, PO5, PO3, POz, PO4, PO6, O1, Oz, O2)
    """
    return [i-1 for i in [48, 54, 55, 56, 57, 58, 61, 62, 63]]

def occipital_19_ch() -> List[int]:
    """
    Provide 19 channels around occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        19 channels in occipital regions (P7, P5, P3, P1, PZ, P2, P4, P6,
        P8, PO7, PO5, PO3, POz, PO4, PO6, PO8, O1, Oz, O2)
    """
    return [i-1 for i in [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63]]

def center_occipital_26_ch() -> List[int]:
    """
    Provide 26 channels from center region to occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        26 channels from center region to occipital region 
        (CP5, CP3, CP1, CPZ, CP2, CP4, CP6,
         P7, P5, P3, P1, PZ, P2, P4, P6, P8,
         PO7, PO5, PO3, POz, PO4, PO6, PO8
         O1, Oz, O2)
    """
    return [i-1 for i in [35, 36, 37, 38, 39, 40, 41,
                          44, 45, 46, 47, 48, 49, 50, 51, 52,
                          53, 54, 55, 56, 57, 58, 59,
                          61, 62, 63]]

def center_occipital_33_ch() -> List[int]:
    """
    Provide 33 channels from center region to occipital region for benchmark dataset

    Returns
    -------
    ch_used: List
        33 channels from center region to occipital region 
        (C5, C3, C1, Cz, C2, C4, C6,
         CP5, CP3, CP1, CPZ, CP2, CP4, CP6,
         P7, P5, P3, P1, PZ, P2, P4, P6, P8,
         PO7, PO5, PO3, POz, PO4, PO6, PO8
         O1, Oz, O2)
    """
    return [i-1 for i in [25, 26, 27, 28, 29, 30, 31,
                          35, 36, 37, 38, 39, 40, 41,
                          44, 45, 46, 47, 48, 49, 50, 51, 52,
                          53, 54, 55, 56, 57, 58, 59,
                          61, 62, 63]]

def preprocess(dataself,
               X: ndarray) -> ndarray:
    """
    Suggested preprocessing function for benchmark dataset
    
    notch filter at 50 Hz
    """
    srate = dataself.srate

    # notch filter at 50 Hz
    f0 = 50
    Q = 35
    notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=srate)
    preprocess_X = signal.filtfilt(notchB, notchA, X, axis = 1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))
    
    return preprocess_X
    
def filterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = dataself.srate
    
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
    
    for k in range(1, num_subbands+1, 1):
        Wp = [(8*k)/(srate/2), 90/(srate/2)]
        Ws = [(8*k-2)/(srate/2), 100/(srate/2)]

        gstop = 40
        while gstop>=20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
                filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
                break
            except:
                gstop -= 1
        if gstop<20:
            raise ValueError("""
Filterbank cannot be processed. You may try longer signal lengths.
Filterbank order: {:n}
gstop: {:n}
bpB: {:s}
bpA: {:s}
Required signal length: {:n}
Signal length: {:n}""".format(k,
                                gstop,
                                str(bpB),
                                str(bpA),
                                3*(max(len(bpB),len(bpA))-1),
                                X.shape[1]))
        
        
    return filterbank_X