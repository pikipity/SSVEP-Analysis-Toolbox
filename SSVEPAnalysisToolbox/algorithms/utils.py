# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import iscomplex, ndarray

import scipy.linalg as slin
import numpy as np
import numpy.linalg as nplin
import numpy.matlib as npmat

def get_depend_column_ind(X: ndarray):
    """
    Obtain depend column index

    Parameters
    ------------
    X : ndarray
        Input matrix
    """
    X_rank = nplin.matrix_rank(X)
    _, X_r = nplin.qr(X)
    X_r_diag = np.diag(X_r)
    X_r_diag_abssort_ind = np.argsort(np.abs(X_r_diag))[::-1]
    return X_r_diag_abssort_ind[X_rank:]


def remove_mean_all_trial(X : ndarray,
                          axis : int = 0, 
                          remove_val : Optional[ndarray] = None):
    """
    Remove mean of all trials

    Parameters
    ------------
    X : ndarray
        All trial signals
        Shape: [trials, samples, ch]
    axis : int
        Remove which axis's mean
    """
    Nt, Np, Nc = X.shape
    X_remove_mean = np.zeros_like(X)
    for i in range(Nt):
        if remove_val is None:
            X_remove_mean[i,:,:] = X[i,:,:] - np.mean(X[i,:,:], axis)
        else:
            X_remove_mean[i,:,:] = X[i,:,:] - remove_val
    return X_remove_mean

def column_eyes(number_of_eyes : int, 
                size_of_eyes : int):
    return repmat(np.eye(size_of_eyes),number_of_eyes,1)

def inv(X : ndarray):
    return nplin.inv(X)

def repmat(X : ndarray,
           rep_x : int,
           rep_y : int):
    return npmat.repmat(X, rep_x, rep_y)

def svd(X : ndarray,
        full_matrices : bool,
        compute_uv : bool):
    if compute_uv:
        L, D, M = slin.svd(X,
                            full_matrices=full_matrices,
                            compute_uv=compute_uv,
                            check_finite=False,
                            lapack_driver='gesvd')
        return L, D, M
    else:
        D = slin.svd(X,
                        full_matrices=full_matrices,
                        compute_uv=compute_uv,
                        check_finite=False,
                        lapack_driver='gesvd')
        return D

def cholesky(M : ndarray):
    """
    Calculate cholesky decomposition of M. If M is not positive definite matrix, the nearest positive definite matrix for M will be created.

    ref: https://github.com/Cysu/open-reid/commit/61f9c4a4da95d0afc3634180eee3b65e38c54a14

    Find the nearest positive definite matrix for M. Modified from
    http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    Might take several minutes
    """
    M = (M + M.T) * 0.5
    k = 0
    I = np.eye(M.shape[0])
    Ki = None
    while k<=1000:
        try:
            Ki = nplin.cholesky(M)
            break
        except nplin.LinAlgError:
            k += 1
            v = eigvec(M)
            min_eig = v.min()
            M += (-min_eig * k * k + np.spacing(min_eig)) * I
    if Ki is None:
        raise ValueError('Cannot calculate cholesky decomposition')
    else:
        return Ki

def norm_direction(V : ndarray,
                   V_norm : Optional[List[float]] = None):
    """
    Normalize directions of colums 
    """
    _, n_col = V.shape
    if V_norm is None:
        if n_col == 1:
            V_norm = [1]
        else:
            V_norm = []
            for col_idx in range(n_col):
                v1 = V[:,0]/nplin.norm(V[:,0])
                v2 = V[:,col_idx]/nplin.norm(V[:,col_idx])
                v_dot = np.dot(v1, v2)
                if v_dot > 1:
                    v_dot = 1
                if v_dot < -1:
                    v_dot = -1
                v_angle = np.arccos(v_dot)/np.pi
                if v_angle > 0.5:
                    V[:,col_idx] = -1 * V[:,col_idx]
                    V_norm.append(-1)
                else:
                    V_norm.append(1)
    else:
        for col_idx in range(n_col):
            V[:,col_idx] = V_norm[col_idx] * V[:,col_idx]
    return V, V_norm

def eigvec(X : ndarray,
           Y : Optional[ndarray] = None):
    """
    Calculate eigenvectors

    Parameters
    -----------------
    X : ndarray
        A complex or real matrix whose eigenvalues and eigenvectors will be computed.
    Y : ndarray
        If Y is given, eig(Y\X), or say  eig(X, Y), will be computed

    Returns
    ---------------
    eig_vec : ndarray
        Eigenvectors. The order follows the corresponding eigenvalues (from high to low values)
    """
    if Y is None:
        eig_d1, eig_v1 = slin.eig(X) #eig(X)
    else:
        eig_d1, eig_v1 = slin.eig(X, Y) #eig(Y\X)

    if len(eig_d1.shape) == 2:
        eig_d1 = np.diagonal(eig_d1)

    sort_idx = np.argsort(eig_d1)[::-1]
    eig_vec = eig_v1[:,sort_idx]

    if Y is not None:
        square_val = np.diag(eig_vec.T @ Y @ eig_vec)
        norm_v = np.sqrt(square_val)
        eig_vec = eig_vec/norm_v

    if np.iscomplex(eig_vec).any():
        eig_vec = np.real(eig_vec)

    return eig_vec

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
    X : ndarray or list
        Matrix used for building the block diag matrix
        (trial_num, channel_num, signal_len) or [(channel_num, signal_len)]
    """
    blkmatrix = np.ndarray([])
    if type(X) == np.ndarray:
        trial_num, channel_num, signal_len = X.shape
        for trial_idx in range(trial_num):
            if len(blkmatrix.shape)==0:
                blkmatrix = X[trial_idx,:,:]
            else:
            #     A1 = np.concatenate((blkmatrix, np.zeros((blkmatrix.shape[0], signal_len))), axis = 1)
            #     A2 = np.concatenate((np.zeros((channel_num, blkmatrix.shape[1])), X[trial_idx,:,:]), axis = 1)
            #     blkmatrix = np.concatenate((A1, A2), axis = 0)
                blkmatrix = slin.block_diag(blkmatrix, X[trial_idx,:,:])
    elif type(X) == list:
        for tmp in X:
            if len(blkmatrix.shape)==0:
                blkmatrix = tmp
            else:
                blkmatrix = slin.block_diag(blkmatrix, tmp)
    else:
        raise ValueError('Unknown input type in blkmat')
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
            blkmatrix = X.copy()
        else:
        #     A1 = np.concatenate((blkmatrix, np.zeros((blkmatrix.shape[0], X.shape[1]))), axis = 1)
        #     A2 = np.concatenate((np.zeros((X.shape[0], blkmatrix.shape[1])), X), axis = 1)
        #     blkmatrix = np.concatenate((A1, A2), axis = 0)
            blkmatrix = slin.block_diag(blkmatrix, X)
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