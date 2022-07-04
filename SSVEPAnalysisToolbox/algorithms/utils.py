# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

import scipy.linalg as slin
import numpy as np

def qr_inverse(Q: ndarray, 
               R: ndarray,
               P: ndarray) -> ndarray:
    """
    Inverse QR decomposition

    Parameters
    ----------
    Q : ndarray
        (M * K)
    R : ndarray
        (K * N)
    P : ndarray
        (N,)

    Returns
    -------
    X : ndarray
        (M * N)
    """
    tmp = Q @ R
    X = np.zeros(tmp.shape)
    for i in range(X.shape[1]):
        X[:,P[i]] = tmp[:,i]
    return X

def qr_remove_mean(X: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Remove column mean and QR decomposition 

    Parameters
    ----------
    X : ndarray
        (M * N)

    Returns
    -------
    Q : ndarray
        (M * K)
    R : ndarray
        (K * N)
    P : ndarray
        (N,)
    """
    
    X_remove_mean = X - np.mean(X,0)
    
    Q, R, P = slin.qr(X_remove_mean, mode = 'economic', pivoting = True)
    
    return Q, R, P

def mldivide(A: ndarray,
             B: ndarray) -> ndarray:
    """
    A\B, Solve Ax = B

    Parameters
    ----------
    A : ndarray
    B : ndarray

    Returns
    -------
    x: ndarray
    """
    
    return slin.pinv(A) @ B