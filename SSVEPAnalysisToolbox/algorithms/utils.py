# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

import scipy.linalg as slin
import numpy as np

def canoncorr(X: ndarray, 
              Y: ndarray,
              force_output_UV: Optional[bool] = False) -> Union[Tuple[ndarray, ndarray, ndarray], ndarray]:
    """
    Canonical correlation analysis following matlab

    Parameters
    ----------
    X : ndarray
    Y : ndarray
    force_output_UV : Optional[bool]
        whether calculate and output A and B
    
    Returns
    -------
    A : ndarray
        if force_output_UV, return A
    B : ndarray
        if force_output_UV, return B
    r : ndarray
    """
    n, p1 = X.shape
    _, p2 = Y.shape
    
    Q1, T11, perm1 = qr_remove_mean(X)
    Q2, T22, perm2 = qr_remove_mean(Y)
    
    svd_X = Q1.T @ Q2
    if svd_X.shape[0]>svd_X.shape[1]:
        full_matrices=False
    else:
        full_matrices=True
        
    L, D, M = slin.svd(svd_X,
                     full_matrices=full_matrices,
                     check_finite=False,
                     lapack_driver='gesvd')
    M = M.T
    
    r = D
    
    if force_output_UV:
        A = mldivide(T11, L) * np.sqrt(n - 1)
        B = mldivide(T22, M) * np.sqrt(n - 1)
        A_r = np.zeros(A.shape)
        for i in range(A.shape[0]):
            A_r[perm1[i],:] = A[i,:]
        B_r = np.zeros(B.shape)
        for i in range(B.shape[0]):
            B_r[perm2[i],:] = B[i,:]
            
        return A_r, B_r, r
    else:
        return r

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