# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, linspace, pi, sin, cos, expand_dims, concatenate
import numpy as np
from scipy import signal

def sine_snr(X : ndarray,
             ref : ndarray):
    """
    Calculate SNR using reference
    """
    ref = ref/ref.shape[1]
    ref_mul = ref.T @ ref
    X_ref = np.trace(X @ ref.T @ ref @ X.T)
    X_non_ref = np.trace(X @ (np.eye(ref.shape[1]) - ref_mul) @ X.T)
    return 10*np.log10(X_ref/X_non_ref)

def freqs_snr(X : ndarray,
              target_fre : float,
              srate : float,
              Nh : int,
              detrend_flag : bool = True,
              NFFT : Optional[int] = None):
    """
    Calculate FFT and then calculate SNR
    """
    freq, fft_res = fft(X, srate, detrend_flag = detrend_flag, NFFT = NFFT)
    abs_fft_res = np.abs(fft_res)

    stim_amp = 0
    for n in range(Nh):
        freLoc = np.argmin(np.abs(freq - (target_fre*(n+1))))
        stim_amp += abs_fft_res[0,freLoc]
    snr = 10*np.log10(stim_amp/(np.sum(abs_fft_res)-stim_amp))
    return snr

def freqs_phase(X : ndarray,
                target_fre : float,
                target_phase : float,
                srate : float,
                detrend_flag : bool = True,
                NFFT : Optional[int] = None):
    """
    Calculate FFT and then calculate phase
    """
    freq, fft_res = fft(X, srate, detrend_flag = detrend_flag, NFFT = NFFT)
    angle_fft_res = np.angle(fft_res)
    freLoc = np.argmin(np.abs(freq - target_fre))
    stim_phase = angle_fft_res[0,freLoc]
    if stim_phase != target_phase:
        k1 = np.floor((target_phase - stim_phase)/(2*np.pi))
        k2 = -k1
        k3 = np.floor((stim_phase - target_phase)/(2*np.pi))
        k4 = -k3
        k = np.array([k1,k2,k3,k4])
        k_loc = np.argmin(np.abs(stim_phase + 2*np.pi*k - target_phase))
        stim_phase = stim_phase + 2*np.pi*k[k_loc]
    return stim_phase


def nextpow2(n):
    '''
    Retrun the first P such that 2 ** P >= abs(n).
    '''
    return np.ceil(np.log2(np.abs(n)))

def fft(X : ndarray,
        fs : float,
        detrend_flag : bool = True,
        NFFT : Optional[int] = None):
    """
    Calculate FFT

    Parameters
    -----------
    X : ndarray
        Input signal. The shape is (1*N) where N is the sampling number.
    fs : float
        Sampling freqeuncy.
    detrend_flag : bool
        Whether detrend. If True, X will be detrended first. Default is True.
    NFFT : Optional[int]
        Number of FFT. If None, NFFT is equal to 2^nextpow2(X.shape[1]). Default is None.

    Returns
    -------------
    freqs : ndarray
        Corresponding frequencies
    fft_res : ndarray
        FFT result
    """
    X_raw, X_col = X.shape
    if X_raw!=1:
        raise ValueError('The row number of the input signal for the FFT must be 1.')
    if X_col==1:
        raise ValueError('The column number of the input signal for the FFT cannot be 1.')
    if NFFT is None:
        NFFT = 2 ** nextpow2(X_col)
    if type(NFFT) is not int:
        NFFT = int(NFFT)
    
    if detrend_flag:
        X = signal.detrend(X, axis = 1)

    fft_res = np.fft.fft(X, NFFT, axis = 1)
    freqs = np.fft.fftfreq(NFFT, 1/fs)
    freqs = np.expand_dims(freqs,0)
    if NFFT & 0x1:
        fft_res = fft_res[:,:int((NFFT+1)/2)]
        freqs = freqs[:,:int((NFFT+1)/2)]
    else:
        fft_res = fft_res[:,:int(NFFT/2)]
        freqs = freqs[:,:int(NFFT/2)]
    # fft_res = fft_res/X_col
    
    return freqs, fft_res

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
        2N * L
    """
    
    t = linspace(0, (L-1)/srate, L)
    t = expand_dims(t,0)
    
    y = []
    for n in range(1,N+1,1):
        y.append(sin( 2*pi*n*freq*t + n*phase ))
        y.append(cos( 2*pi*n*freq*t + n*phase ))
    y = concatenate(y, axis = 0)
        
    return y

def floor(x: float) -> int:
    """
    Floor operation. Convert np.floor result to int

    Parameters
    ----------
    x : float

    Returns
    -------
    int

    """
    
    return int(np.floor(x))