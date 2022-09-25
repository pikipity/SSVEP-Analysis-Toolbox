# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

import scipy.linalg as slin
import numpy as np

def sum_list(X: list) -> ndarray:
    """
    Calculate sum of a list

    Parameters
    ------------
    X: list

    Returns
    -------------
    sum_X: ndarray
    """
    sum_X = None
    for x in X:
        if type(x) is list:
            x = sum_list(x)
        if sum_X is None:
            sum_X = x
        else:
            sum_X = sum_X + x
    return sum_X

def mean_list(X: list) -> ndarray:
    """
    Calculate mean of a list

    Parameters
    -----------
    X: list

    Returns
    ----------
    mean_X: ndarray
    """
    tmp = []
    for X_single_trial in X:
        if type(X_single_trial) is list:
            X_single_trial = mean_list(X_single_trial)
        tmp.append(np.expand_dims(X_single_trial, axis = 0))
    tmp = np.concatenate(tmp, axis = 0)
    return np.mean(tmp, axis=0)

def blkmat(X: ndarray):
    """
    Build the block diag matrix by using X

    Parameters
    -----------
    X : ndarray
        Matrix used for building the block diag matrix
        (trial_num, channel_num, signal_len)
    """
    trial_num, channel_num, signal_len = X.shape
    blkmatrix = np.ndarray([])
    for trial_idx in range(trial_num):
        if len(blkmatrix.shape)==0:
            blkmatrix = X[trial_idx,:,:]
        else:
            A1 = np.concatenate((blkmatrix, np.zeros((blkmatrix.shape[0], signal_len))), axis = 1)
            A2 = np.concatenate((np.zeros((channel_num, blkmatrix.shape[1])), X[trial_idx,:,:]), axis = 1)
            blkmatrix = np.concatenate((A1, A2), axis = 0)
    return blkmatrix

def blkrep(X: ndarray,
           N: int):
    """
    Build the block diag matrix by repeat X with N times

    Parameters
    -----------
    X : ndarray
        Matrix used for building the block diag matrix
    N : int
        Number of X in the diag line
    """
    blkmatrix = np.ndarray([])
    for _ in range(N):
        if len(blkmatrix.shape)==0:
            blkmatrix = X
        else:
            A1 = np.concatenate((blkmatrix, np.zeros((blkmatrix.shape[0], X.shape[1]))), axis = 1)
            A2 = np.concatenate((np.zeros((X.shape[0], blkmatrix.shape[1])), X), axis = 1)
            blkmatrix = np.concatenate((A1, A2), axis = 0)
    return blkmatrix


def sort(X: list) -> Tuple[list, list, list]:
    """
    Sort given list

    Parameters
    -----------
    X : list

    Returns
    -----------
    sorted_X : list
        Sorted X
    sort_idx : list
        Indices that is applied to transfer list from X to sorted_X
    return_idx : list
        Indices that is applied to transfer list from sorted_X to X
    """
    sort_idx = list(np.argsort(X))
    sorted_X = [X[i] for i in sort_idx]
    return_idx = [None] * len(sort_idx)
    for loc, idx in enumerate(sort_idx):
        return_idx[idx] = loc
    return sorted_X, sort_idx, return_idx

def separate_trainSig(X: List[ndarray],
                      Y: List[int]) -> List[ndarray]:
    """
    Separate training signals

    Parameters
    ----------
    X : List[ndarray]
        Training data
        List shape: (trial_num,)
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[int]
        Training label
        List shape: (trial_num,)

    Returns
    -------
    template_sig : List[ndarray]
        Template signal
        List of shape: (stimulus_num,)
        Template shape: (trial_num, filterbank_num, channel_num, signal_len)
    """
    # Get possible stimulus class
    unique_Y = list(set(Y))
    unique_Y.sort()
    # 
    template_sig = []
    for i in unique_Y:
        # i-th class trial index
        target_idx = [k for k in range(len(Y)) if Y[k] == unique_Y[i]]
        # Get i-th class training data
        template_sig_single = [np.expand_dims(X[k], axis=0) for k in target_idx]
        template_sig_single = np.concatenate(template_sig_single, axis=0)
        # Store i-th class template
        template_sig.append(template_sig_single)
    return template_sig

def gen_template(X: List[ndarray],
                 Y: List[int]) -> List[ndarray]:
    """
    Average training data according to training label to generate template signal

    Parameters
    ----------
    X : List[ndarray]
        Training data
        List shape: (trial_num,)
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[int]
        Training label
        List shape: (trial_num,)

    Returns
    -------
    template_sig : List[ndarray]
        Template signal
        List of shape: (stimulus_num,)
        Template shape: (filterbank_num, channel_num, signal_len)
    """
    # Get possible stimulus class
    unique_Y = list(set(Y))
    unique_Y.sort()
    # 
    template_sig = []
    for i in unique_Y:
        # i-th class trial index
        target_idx = [k for k in range(len(Y)) if Y[k] == unique_Y[i]]
        # Get i-th class training data
        template_sig_single = [np.expand_dims(X[k], axis=0) for k in target_idx]
        template_sig_single = np.concatenate(template_sig_single, axis=0)
        # Average all i-th class training data
        template_sig_single = np.mean(template_sig_single, axis=0)
        # Store i-th class template
        template_sig.append(template_sig_single)
    return template_sig

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
        (M * K) - reference
        (filterbank_num * M * K) - template
    R : ndarray
        (K * N) - reference
        (filterbank_num * K * N) - template
    P : ndarray
        (N,) - reference
        (filterbank_num * N) - template

    Returns
    -------
    X : ndarray
        (M * N) - reference
        (filterbank_num * M * N) - template
    """
    if len(Q.shape)==2: # reference
        tmp = Q @ R
        X = np.zeros(tmp.shape)
        for i in range(X.shape[1]):
            X[:,P[i]] = tmp[:,i]
    elif len(Q.shape)==3: # template
        X = [np.expand_dims(qr_inverse(Q[i,:,:], R[i,:,:], P[i,:]), axis=0) for i in range(Q.shape[0])]
        X = np.concatenate(X, axis=0)
    else:
        raise ValueError('Unknown data type')
    return X

def qr_list(X : List[ndarray]) -> Tuple[List[ndarray], List[ndarray], List[ndarray]]:
    """
    QR decomposition of list X
    Note: Elements in X will be transposed first and then decomposed

    Parameters
    ----------
    X : List[ndarray]

    Returns
    -------
    Q : List[ndarray]
    R : List[ndarray]
    P : List[ndarray]
    """
    Q = []
    R = []
    P = []
    for el in X:
        if len(el.shape) == 2: # reference signal
            Q_tmp, R_tmp, P_tmp = qr_remove_mean(el.T)
            Q.append(Q_tmp)
            R.append(R_tmp)
            P.append(P_tmp)
        elif len(el.shape) == 3: # template signal
            Q_tmp = []
            R_tmp = []
            P_tmp = []
            for k in range(el.shape[0]):
                Q_tmp_tmp, R_tmp_tmp, P_tmp_tmp = qr_remove_mean(el[k,:,:].T)
                Q_tmp.append(np.expand_dims(Q_tmp_tmp, axis=0))
                R_tmp.append(np.expand_dims(R_tmp_tmp, axis=0))
                P_tmp.append(np.expand_dims(P_tmp_tmp, axis=0))
            Q.append(np.concatenate(Q_tmp,axis=0))
            R.append(np.concatenate(R_tmp,axis=0))
            P.append(np.concatenate(P_tmp,axis=0))
        else:
            raise ValueError('Unknown data type')
    return Q, R, P

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