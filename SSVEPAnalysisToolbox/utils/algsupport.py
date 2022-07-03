# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, linspace, pi, sin, cos, expand_dims, concatenate

def gen_ref_sin(freq: float,
                srate: int,
                L: int,
                N: int,
                phase: float) -> ndarray:
    """
    Generate sine-cosine-based reference signals of one stimulus

    Parameters
    ----------
    freq : float
        Stimulus frequency
    srate : int
        Sampling rate
    L : int
        Signal length
    N : int
        Number of harmonics
    phase : float
        Stimulus phase

    Returns
    -------
    ref_sig: ndarray
        Sine-cosine-based reference signal
        N * L
    """
    
    t = linspace(0, (L-1)/srate, L)
    t = expand_dims(t,0)
    
    y = []
    for n in range(1,N+1,1):
        y.append(sin( 2*pi*n*freq*t + n*phase ))
        y.append(cos( 2*pi*n*freq*t + n*phase ))
    y = concatenate(y, axis = 0)
        
    return y