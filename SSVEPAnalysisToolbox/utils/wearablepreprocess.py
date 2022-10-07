# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple
from scipy import signal

import numpy as np

def subj_idx_highperformance(subj_no: int = 49):
    subj_idx_from_matlab = [4,99,84,87,69,79,24,39,91,7,54,28,66,30,38,98,72,53,19,11,92,65,3,5,29,62,50,23,83,16,18,52,9,17,25,56,46,78,82,61,81,80,33,40,93,12,75,100,20,26,55,86,70,51,74,89,90,88,41,68,97,36,49,27,31,8,42,95,22,76,37,71,44,59,47,14,6,96,58,77,85,60,94,45,67,64,43,21,57,2,102,48,35,63,13,1,73,34,15,32,10,101]
    subj_idx_for_python = [i-1 for i in subj_idx_from_matlab]
    return subj_idx_for_python[0:subj_no]

def suggested_weights_filterbank(num_subbands: int = 5,
                                 data_type: str = 'wet',
                                 method_type:str = 'cca'):
    """
    Provide weights of filterbank
    """
    # if method_type.lower() == 'cca':
    #     if data_type.lower() == 'wet':
    #         return [i**(-1.25)+0 for i in range(1,num_subbands+1,1)]
    #     else:
    #         return [i**(-2)+0.25 for i in range(1,num_subbands+1,1)]
    # else:
    #     if data_type.lower() == 'wet':
    #         return [i**(-1.75)+0.5 for i in range(1,num_subbands+1,1)]
    #     else:
    #         return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]
    return [i**(-1.25)+0.25 for i in range(1,num_subbands+1,1)]

def suggested_ch() -> List[int]:
    """
    Provide suggested channels

    Returns
    -------
    ch_used: List
        Suggested channels
    """
    return list(range(8))

def preprocess(dataself,
               X: ndarray) -> ndarray:
    """
    Suggested preprocessing function
    
    notch filter at 50 Hz
    """
    srate = dataself.srate

    # notch filter at 50 Hz
    # f0 = 50
    # Q = 35
    # notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=srate)
    # preprocess_X = signal.filtfilt(notchB, notchA, X, axis = 1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))

    b1, a1 = signal.cheby1(4,2,[47.0/(srate/2), 53.0/(srate/2)], 'bandstop')
    preprocess_X = signal.filtfilt(b1, a1, X, axis = 1, padtype='odd', padlen=3*(max(len(b1),len(a1))-1))
    
    return preprocess_X
    
def filterbank(dataself,
               X: ndarray,
               num_subbands: Optional[int] = 5) -> ndarray:
    """
    Suggested filterbank function
    """
    srate = dataself.srate
    
    filterbank_X = np.zeros((num_subbands, X.shape[0], X.shape[1]))
    
    for k in range(1, num_subbands+1, 1):
        b2, a2 = signal.cheby1(4,1,[(9.0*k)/(srate/2), 90.0/(srate/2)],'bandpass')
        tmp =  signal.filtfilt(b2, a2, X, axis = 1, padtype='odd', padlen=3*(max(len(b2),len(a2))-1))
        tmp = signal.detrend(tmp, axis = 1)
        tmp = tmp - np.mean(tmp, axis = 1, keepdims = True)
        tmp_std = np.std(tmp, axis = 1, keepdims = True, ddof=1)
        tmp = tmp / tmp_std
        filterbank_X[k-1,:,:] = tmp.copy()
        
#         Wp = [(9.25*k)/(srate/2), 90/(srate/2)]
#         Ws = [(9.25*k-2)/(srate/2), 100/(srate/2)]

#         gstop = 40
#         while gstop>=20:
#             try:
#                 N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
#                 bpB, bpA = signal.cheby1(N, 0.5, Wn, btype = 'bandpass')
#                 filterbank_X[k-1,:,:] = signal.filtfilt(bpB, bpA, X, axis = 1, padtype='odd', padlen=3*(max(len(bpB),len(bpA))-1))
#                 break
#             except:
#                 gstop -= 1
#         if gstop<20:
#             raise ValueError("""
# Filterbank cannot be processed. You may try longer signal lengths.
# Filterbank order: {:n}
# gstop: {:n}
# bpB: {:s}
# bpA: {:s}
# Required signal length: {:n}
# Signal length: {:n}""".format(k,
#                                 gstop,
#                                 str(bpB),
#                                 str(bpA),
#                                 3*(max(len(bpB),len(bpA))-1),
#                                 X.shape[1]))
        
        
    return filterbank_X