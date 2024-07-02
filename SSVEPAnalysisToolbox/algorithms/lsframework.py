# -*- coding: utf-8 -*-
"""
LS Framework
"""

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial
from copy import deepcopy
import warnings

import numpy as np
import numpy.linalg as nplin
import scipy.linalg as slin

# from .basemodel import BaseModel
from .cca import (
    _r_cca_canoncorr_withUV, _r_cca_qr_withUV
)
from .utils_ls import (
    canoncorr_ls, lsframe, 
    get_lsconfig,
    ssvep_lsframe,
    combine_name
)
from .utils import (
    qr_inverse, svd, qr_list, gen_template, sort, separate_trainSig, 
    blkmat, blkrep, column_eyes, remove_mean_all_trial, repmat
)
from .cca import (
    MsetCCAwithR,
    MsetCCA, 
    MSCCA,
    ECCA, ITCCA,
    SCCA_canoncorr, SCCA_qr
)
from .trca import (
    TRCA, ETRCA,
    TRCAwithR, ETRCAwithR,
    MSETRCA
)
from .tdca import (
    TDCA, _gen_delay_X
)

def _r_cca_canoncorr_ls(X: ndarray,
                     Y: List[ndarray],
                     n_component: int,
                     force_output_UV: Optional[bool] = False,
                     LSconfig: Optional[dict] = None) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
    """
    Calculate correlation of CCA based on canoncorr for single trial data 

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
    n_component : int
        Number of eigvectors for spatial filters.
    force_output_UV : Optional[bool]
        Whether return spatial filter 'U' and weights of harmonics 'V'

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)
    """
    filterbank_num, channel_num, signal_len = X.shape
    if len(Y[0].shape)==2:
        harmonic_num = Y[0].shape[0]
    elif len(Y[0].shape)==3:
        harmonic_num = Y[0].shape[1]
    else:
        raise ValueError('Unknown data type')
    stimulus_num = len(Y)
    
    R = np.zeros((filterbank_num, stimulus_num))
    U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
    V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
                
            if n_component == 0 and force_output_UV is False:
                D = canoncorr_ls(tmp.T, Y_tmp.T, False, LSconfig = LSconfig)
                r = D[0]
            else:
                A_r, B_r, D = canoncorr_ls(tmp.T, Y_tmp.T, True, LSconfig = LSconfig)
                
                a = A_r[:channel_num, :n_component].T @ tmp
                b = B_r[:harmonic_num, :n_component].T @ Y_tmp
                a = np.reshape(a, (-1))
                b = np.reshape(b, (-1))
                
                # r = stats.pearsonr(a, b)[0]
                r = np.corrcoef(a, b)[0,1]
                U[k,i,:,:] = A_r[:channel_num, :n_component]
                V[k,i,:,:] = B_r[:harmonic_num, :n_component]
                
            R[k,i] = r
    if force_output_UV:
        return R, U, V
    else:
        return R

def _r_cca_qr_ls(X: ndarray,
           Y_Q: List[ndarray],
           Y_R: List[ndarray],
           Y_P: List[ndarray],
           n_component: int,
           force_output_UV: Optional[bool] = False,
           LSconfig: Optional[dict] = None) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
    """
    Calculate correlation of CCA based on QR decomposition for single trial data 

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y_Q : List[ndarray]
        Q of reference signals
    Y_R: List[ndarray]
        R of reference signals
    Y_P: List[ndarray]
        P of reference signals
    n_component : int
        Number of eigvectors for spatial filters.
    force_output_UV : Optional[bool]
        Whether return spatial filter 'U' and weights of harmonics 'V'

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)
    """
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)

    filterbank_num, channel_num, signal_len = X.shape
    harmonic_num = Y_R[0].shape[-1]
    stimulus_num = len(Y_Q)
    
    Y = [qr_inverse(Y_Q[i],Y_R[i],Y_P[i]) for i in range(len(Y_Q))]
    if len(Y[0].shape)==2: # reference
        Y = [Y_tmp.T for Y_tmp in Y]
    elif len(Y[0].shape)==3: # template
        Y = [np.transpose(Y_tmp, (0,2,1)) for Y_tmp in Y]
    else:
        raise ValueError('Unknown data type')
    
    R = np.zeros((filterbank_num, stimulus_num))
    U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
    V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        Z = tmp.T - np.mean(tmp.T,0)
        K = np.eye(Z.shape[0])
        _, Dx, Vx = svd(K.T @ Z, False, True)
        Dx = np.diag(Dx)
        VDV = Vx.T @ nplin.inv(Dx) @ Vx
        Z = K.T @ Z
        for i in range(stimulus_num):
            if len(Y_Q[i].shape)==2: # reference
                Y_Q_tmp = Y_Q[i]
                # Y_R_tmp = Y_R[i]
                # Y_P_tmp = Y_P[i]
                Y_tmp = Y[i]
            elif len(Y_Q[i].shape)==3: # template
                Y_Q_tmp = Y_Q[i][k,:,:]
                # Y_R_tmp = Y_R[i][k,:,:]
                # Y_P_tmp = Y_P[i][k,:]
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
            P = Y_Q_tmp @ Y_Q_tmp.T
            PZ = P.T @ Z
            X_rank = nplin.matrix_rank(PZ)
            if X_rank!=min(PZ.shape) and check_full_rank:
                applied_max_n = 1
            else:
                applied_max_n = max_n
            W, _, _, _, _ = lsframe(PZ, 
                                    PZ @ VDV, max_n = applied_max_n,
                                    LSMethod = LSMethod,
                                    displayConvergWarn = displayConvergWarn,
                                    M_error_threshold = M_error_threshold,
                                    W_error_threshold = W_error_threshold,
                                    l1_alpha = l1_alpha,
                                    l2_alpha = l2_alpha,
                                    alpha = alpha,
                                    reg_iter = reg_iter,
                                    reg_tol = reg_tol,
                                    kernel_fun = kernel_fun)
            A_r = W[-1].copy()
            B_r, _, _, _ = nplin.lstsq(Y_tmp.T, tmp.T @ A_r, rcond=None)
            # A_r, residuals_2, _, _ = nplin.lstsq(tmp.T, Y_tmp.T @ B_r, rcond=None)
            # A_r_diff = np.divide(A_r,A_r_tmp)
            if X_rank!=min(PZ.shape) and check_full_rank:
                A_r, _, _, _ = nplin.lstsq(tmp.T, Y_tmp.T @ B_r, rcond=None)
            
            a = A_r[:channel_num, :n_component].T @ tmp
            # a_tmp =  A_r_tmp[:channel_num, :n_component].T @ tmp
            # a_diff = np.divide(a,a_tmp)
            b = B_r[:harmonic_num, :n_component].T @ Y_tmp
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            U[k,i,:,:] = A_r[:channel_num, :n_component]
            V[k,i,:,:] = B_r[:harmonic_num, :n_component]
            r = np.corrcoef(a, b)[0,1]
            R[k,i] = r
    if force_output_UV:
        return R, U, V
    else:
        return R


class SCCA_ls(SCCA_canoncorr):
    """
    Standard CCA based on LS framework
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: bool = False,
                 update_UV: bool = True,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        """
        Special Parameters
        ----------
        force_output_UV : Optional[bool] 
            Whether store U and V. Default is False
        update_UV: Optional[bool]
            Whether update U and V in next time of applying "predict" 
            If false, and U and V have not been stored, they will be stored
            Default is True
        """
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank,
                         force_output_UV = force_output_UV,
                         update_UV = update_UV)
        self.ID = 'sCCA (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
        
    def __copy__(self):
        copy_model = SCCA_ls(n_component = self.n_component,
                             n_jobs = self.n_jobs,
                             weights_filterbank = self.model['weights_filterbank'],
                             force_output_UV = self.force_output_UV,
                             update_UV = self.update_UV)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
        n_component = self.n_component
        Y = self.model['ref_sig']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                if self.n_jobs is not None:
                    r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_ls, n_component=n_component, Y=Y, force_output_UV=True, LSconfig = self.LSconfig))(a) for a in X))
                else:
                    r = []
                    U = []
                    V = []
                    for a in X:
                        r_temp, U_temp, V_temp = _r_cca_canoncorr_ls(a, n_component=n_component, Y=Y, force_output_UV=True, LSconfig = self.LSconfig)
                        r.append(r_temp)
                        U.append(U_temp)
                        V.append(V_temp)
                self.model['U'] = U
                self.model['V'] = V
            else:
                if self.n_jobs is not None:
                    r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_ls, n_component=n_component, Y=Y, force_output_UV=False, LSconfig = self.LSconfig))(a) for a in X)
                else:
                    r = []
                    for a in X:
                        r.append(
                            _r_cca_canoncorr_ls(a, n_component=n_component, Y=Y, force_output_UV=False, LSconfig = self.LSconfig)
                        )
        else:
            U = self.model['U']
            V = self.model['V']
            if self.n_jobs is not None:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_withUV, Y=Y))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
            else:
                r = []
                for a, u, v in zip(X,U,V):
                    r.append(
                        _r_cca_canoncorr_withUV(X=a, U=u, V=v, Y=Y)
                    )
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred, r

class SCCA_ls_qr(SCCA_qr):
    """
    Standard CCA based on LS framework and qr decomposition
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: bool = False,
                 update_UV: bool = True,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        """
        Special Parameters
        ----------
        force_output_UV : Optional[bool] 
            Whether store U and V. Default is False
        update_UV: Optional[bool]
            Whether update U and V in next time of applying "predict" 
            If false, and U and V have not been stored, they will be stored
            Default is True
        """
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank,
                         force_output_UV = force_output_UV,
                         update_UV = update_UV)
        self.ID = 'sCCA (ls-qr)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
        
    def __copy__(self):
        copy_model = SCCA_ls_qr(n_component = self.n_component,
                                    n_jobs = self.n_jobs,
                                    weights_filterbank = self.model['weights_filterbank'],
                                    force_output_UV = self.force_output_UV,
                                    update_UV = self.update_UV)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
        n_component = self.n_component
        Y_Q = self.model['ref_sig_Q']
        Y_R = self.model['ref_sig_R']
        Y_P = self.model['ref_sig_P']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                if self.n_jobs is not None:
                    r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_ls, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True, LSconfig = self.LSconfig))(a) for a in X))
                else:
                    r = []
                    U = []
                    V = []
                    for a in X:
                        r_temp, U_temp, V_temp = _r_cca_qr_ls(a, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True, LSconfig = self.LSconfig)
                        r.append(r_temp)
                        U.append(U_temp)
                        V.append(V_temp)
                self.model['U'] = U
                self.model['V'] = V
            else:
                if self.n_jobs is not None:
                    r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_ls, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False, LSconfig = self.LSconfig))(a) for a in X)
                else:
                    r = []
                    for a in X:
                        r.append(
                            _r_cca_qr_ls(a, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False, LSconfig = self.LSconfig)
                        )
        else:
            U = self.model['U']
            V = self.model['V']
            if self.n_jobs is not None:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
            else:
                r = []
                for a, u, v in zip(X,U,V):
                    r.append(
                        _r_cca_qr_withUV(X=a, U=u, V=v, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P)
                    )
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred, r

class ITCCA_ls(ITCCA):
    """
    itCCA based on LS framework
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: bool = False,
                 update_UV: bool = True,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank,
                         force_output_UV = force_output_UV,
                         update_UV = update_UV)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
        
    def __copy__(self):
        copy_model = ITCCA_ls(n_component = self.n_component,
                              n_jobs = self.n_jobs,
                              weights_filterbank = self.model['weights_filterbank'],
                              force_output_UV = self.force_output_UV,
                              update_UV = self.update_UV)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def predict(self,
                X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
            
        n_component = self.n_component
        Y_Q = self.model['template_sig_Q']
        Y_R = self.model['template_sig_R']
        Y_P = self.model['template_sig_P']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                if self.n_jobs is not None:
                    r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_ls, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True, LSconfig = self.LSconfig))(a) for a in X))
                else:
                    r = []
                    U = []
                    V = []
                    for a in X:
                        r_temp, U_temp, V_temp = _r_cca_qr_ls(a, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True, LSconfig = self.LSconfig)
                        r.append(r_temp)
                        U.append(U_temp)
                        V.append(V_temp)
                self.model['U'] = U
                self.model['V'] = V
            else:
                if self.n_jobs is not None:
                    r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_ls, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False, LSconfig = self.LSconfig))(a) for a in X)
                else:
                    r = []
                    for a in X:
                        r.append(
                            _r_cca_qr_ls(a, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False, LSconfig = self.LSconfig)
                        )
        else:
            U = self.model['U']
            V = self.model['V']
            if self.n_jobs is not None:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
            else:
                r = []
                for a, u, v in zip(X,U,V):
                    r.append(
                        _r_cca_qr_withUV(X=a, U=u, V=v, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P)
                    )
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred, r

def _ECCA3_ls_calW(trainSig, Q, template_sig, ref_sig, LSconfig):
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    #
    template_sig_mean = np.mean(template_sig,0)
    template_sig = template_sig - template_sig_mean
    ref_sig = ref_sig - np.mean(ref_sig, 0)
    Nt, Np, Nc = trainSig.shape
    trainSig = remove_mean_all_trial(trainSig, remove_val=template_sig_mean)
    Z = blkmat(trainSig) @ column_eyes(Nt, Nc)
    Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    Px = blkrep(Q, Nt) @ blkrep(Q, Nt).T
    Ly = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    Py = np.eye(Nt * Np)
    W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Z, Lx, Px, Ly, Py,
                                                               LSMethod = LSMethod,
                                                               displayConvergWarn = displayConvergWarn,
                                                               max_n = max_n,
                                                               M_error_threshold = M_error_threshold,
                                                               W_error_threshold = W_error_threshold,
                                                               largest_n = max_n,
                                                               check_rank = check_full_rank,
                                                               l1_alpha = l1_alpha,
                                                               l2_alpha = l2_alpha,
                                                               alpha = alpha,
                                                               reg_iter = reg_iter,
                                                               reg_tol = reg_tol,
                                                               kernel_fun = kernel_fun)
    W = W[-1].copy()
    V, _, _, _ = nplin.lstsq(ref_sig, template_sig @ W, rcond=None)
    if not full_rank_check:
        W, _, _, _ = nplin.lstsq(template_sig, ref_sig @ V, rcond=None)
    return W, V

class ECCA_ls(ECCA):
    """
    eCCA based on LS framework
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 # force_output_UV: Optional[bool] = False,
                 update_UV: bool = True,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        """
        Special Parameters
        ----------
        update_UV: Optional[bool]
            Whether update U and V in next time of applying "predict" 
            If false, and U and V have not been stored, they will be stored
            Default is True
        """
        super().__init__(update_UV = update_UV,
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
        
    def __copy__(self):
        copy_model = ECCA_ls(n_component = self.n_component,
                            n_jobs = self.n_jobs,
                            weights_filterbank = self.model['weights_filterbank'],
                            update_UV = self.update_UV)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model
        
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if ref_sig is None:
            raise ValueError('eCCA requires sine-cosine-based reference signal')
        if Y is None:
            raise ValueError('eCCA requires training label')
        if X is None:
            raise ValueError('eCCA requires training data')

        separated_trainSig = separate_trainSig(X, Y)
            
        # generate reference realted QR
        ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig) # List of shape: (stimulus_num,);
                                                           # Template shape: (harmonic_num, signal_len)
        self.model['ref_sig_Q'] = ref_sig_Q # List of shape: (stimulus_num,);
        self.model['ref_sig_R'] = ref_sig_R
        self.model['ref_sig_P'] = ref_sig_P
        
        # generate template related QR
        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                           # Template shape: (filterbank_num, channel_num, signal_len)
        template_sig_Q, template_sig_R, template_sig_P = qr_list(template_sig)
        self.model['template_sig_Q'] = template_sig_Q # List of shape: (stimulus_num,);
        self.model['template_sig_R'] = template_sig_R
        self.model['template_sig_P'] = template_sig_P
        
        # spatial filters of template and reference: U3 and V3
        #   U3: (filterbank_num * stimulus_num * channel_num * n_component)
        #   V3: (filterbank_num * stimulus_num * harmonic_num * n_component)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        harmonic_num = ref_sig[0].shape[0]
        n_component = self.n_component
        U3 = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        V3 = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
        for filterbank_idx in range(filterbank_num):
            if self.n_jobs is not None:
                U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(_ECCA3_ls_calW)(trainSig = np.swapaxes(trainSig_single[:,filterbank_idx,:,:],1,2), 
                                                                                 Q = Q_single, 
                                                                                 template_sig = template_sig_single[filterbank_idx,:,:].T, 
                                                                                 ref_sig = ref_sig_single.T,
                                                                                 LSconfig = self.LSconfig) 
                                                            for trainSig_single, Q_single, template_sig_single, ref_sig_single in zip(separated_trainSig,ref_sig_Q,template_sig,ref_sig)))
            else:
                U = []
                V = []
                for trainSig_single, Q_single, template_sig_single, ref_sig_single in zip(separated_trainSig, ref_sig_Q, template_sig, ref_sig):
                    U_temp, V_temp = _ECCA3_ls_calW(trainSig = np.swapaxes(trainSig_single[:,filterbank_idx,:,:],1,2), 
                                                    Q = Q_single, 
                                                    template_sig = template_sig_single[filterbank_idx,:,:].T, 
                                                    ref_sig = ref_sig_single.T, 
                                                    LSconfig = self.LSconfig)
                    U.append(U_temp)
                    V.append(V_temp)
            for stim_idx, (u, v) in enumerate(zip(U,V)):
                U3[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
                V3[filterbank_idx, stim_idx, :, :] = v[:harmonic_num,:n_component]
        self.model['U3'] = U3
        self.model['V3'] = V3
            
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        if type(weights_filterbank) is list:
            weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        else:
            if len(weights_filterbank.shape) != 2:
                raise ValueError("'weights_filterbank' has wrong shape")
            if weights_filterbank.shape[0] != 1:
                weights_filterbank = weights_filterbank.T
        if weights_filterbank.shape[0] != 1:
            raise ValueError("'weights_filterbank' has wrong shape")
        n_component = self.n_component
        update_UV = self.update_UV
        
        ref_sig_Q = self.model['ref_sig_Q']
        ref_sig_R = self.model['ref_sig_R']
        ref_sig_P = self.model['ref_sig_P']
        
        template_sig_Q = self.model['template_sig_Q'] 
        template_sig_R = self.model['template_sig_R'] 
        template_sig_P = self.model['template_sig_P'] 
        
        U3 = self.model['U3'] 
        # V3 = self.model['V3'] 
        
        # r1
        if update_UV or self.model['U1'] is None or self.model['V1'] is None:
            if self.n_jobs is not None:
                r1, U1, V1 = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_ls, n_component=n_component, Y_Q=ref_sig_Q, Y_R=ref_sig_R, Y_P=ref_sig_P, force_output_UV=True, LSconfig = self.LSconfig))(a) for a in X))
            else:
                r1 = []
                U1 = []
                V1 = []
                for a in X:
                    r1_temp, U1_temp, V1_temp = _r_cca_qr_ls(a, n_component=n_component, Y_Q=ref_sig_Q, Y_R=ref_sig_R, Y_P=ref_sig_P, force_output_UV=True, LSconfig = self.LSconfig)
                    r1.append(r1_temp)
                    U1.append(U1_temp)
                    V1.append(V1_temp)
            self.model['U1'] = U1
            self.model['V1'] = V1
        else:
            U1 = self.model['U1']
            V1 = self.model['V1']
            if self.n_jobs is not None:
                r1 = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=ref_sig_Q, Y_R=ref_sig_R, Y_P=ref_sig_P))(X=a, U=u, V=v) for a, u, v in zip(X,U1,V1))
            else:
                r1 = []
                for a, u, v in zip(X,U1,V1):
                    r1.append(
                        _r_cca_qr_withUV(X=a, U=u, V=v, Y_Q=ref_sig_Q, Y_R=ref_sig_R, Y_P=ref_sig_P)
                    )
        
        # r2
        if update_UV or self.model['U2'] is None:
            if self.n_jobs is not None:
                _, U2, _ = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_ls, n_component=n_component, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P, force_output_UV=True, LSconfig = self.LSconfig))(a) for a in X))
            else:
                U2 = []
                for a in X:
                    _, U2_temp, _ = _r_cca_qr_ls(a, n_component=n_component, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P, force_output_UV=True, LSconfig = self.LSconfig)
                    U2.append(U2_temp)
            self.model['U2'] = U2
        
        if self.n_jobs is not None:
            r2 = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P))(X=a, U=u, V=v) for a, u, v in zip(X,U2,U2))
            
            # r3
            r3 = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P))(X=a, U=u, V=v) for a, u, v in zip(X,U1,U1))
            
            # r4
            r4 = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P, U=U3, V=U3))(X=a) for a in X)
        else:
            r2 = []
            for a, u, v in zip(X,U2,U2):
                r2.append(
                    _r_cca_qr_withUV(X=a, U=u, V=v, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P)
                )
            r3 = []
            for a, u, v in zip(X,U1,U1):
                r3.append(
                    _r_cca_qr_withUV(X=a, U=u, V=v, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P)
                )
            r4 = []
            for a in X:
                r4.append(
                    _r_cca_qr_withUV(X=a, Y_Q=template_sig_Q, Y_R=template_sig_R, Y_P=template_sig_P, U=U3, V=U3)
                )
        
        
        Y_pred = [int( np.argmax( weights_filterbank @ (np.sign(r1_single) * np.square(r1_single) + 
                                                        np.sign(r2_single) * np.square(r2_single) +
                                                        np.sign(r3_single) * np.square(r3_single) +
                                                        np.sign(r4_single) * np.square(r4_single)))) for r1_single, r2_single, r3_single, r4_single in zip(r1, r2, r3, r4)]
        r = [(np.sign(r1_single) * np.square(r1_single) + 
              np.sign(r2_single) * np.square(r2_single) +
              np.sign(r3_single) * np.square(r3_single) +
              np.sign(r4_single) * np.square(r4_single)) for r1_single, r2_single, r3_single, r4_single in zip(r1, r2, r3, r4)]
        
        return Y_pred, r

def _MSCCA_ls_calW(trainSig, Q, template_sig, ref_sig, LSconfig):
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    template_sig_mean = np.mean(template_sig,0)
    template_sig = template_sig - template_sig_mean
    ref_sig_mean = np.mean(ref_sig,0)
    ref_sig = ref_sig - ref_sig_mean
    Z = []
    # Lx = np.ndarray([])
    Px_half = []
    # Nt_total = 0
    for trainSig_single_freq, Q_single_freq in zip(trainSig, Q):
        Nt, Np, Nc = trainSig_single_freq.shape
        # Q_single_freq, _, _ = slin.qr(ref_sig[Np_total:(Np_total + Np),:], mode = 'economic', pivoting = True)
        # Np_total += Np
        # _, Nh = Q_single_freq.shape
        # Nt_total += Nt
        trainSig_single_freq = remove_mean_all_trial(trainSig_single_freq, remove_val=template_sig_mean)
        for _ in range(Nt):
            Z.append(
                np.sum(trainSig_single_freq, axis = 0)
            )
        # if len(Z.shape)==0:
        #     Z = blkmat(trainSig_single_freq)
            # Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
            # Px_half = blkrep(Q_single_freq, Nt)
        # else:
        #     Z = slin.block_diag(Z, blkmat(trainSig_single_freq))
            # Lx = slin.block_diag(Lx, column_eyes(Nt, Np) @ column_eyes(Nt, Np).T)
            # Px_half = slin.block_diag(Px_half, blkrep(Q_single_freq, Nt))
        for _ in range(Nt):
            Px_half.append(Q_single_freq - ref_sig_mean)
    # Z = Z @ column_eyes(Nt_total, Nc)
    Z = np.concatenate(Z, axis = 0)
    # Px_half = Px_half @ column_eyes(Nt_total, Nh)
    Px_half = np.concatenate(Px_half, axis = 0)
    Px_half, _, _ = slin.qr(Px_half, mode = 'economic', pivoting = True)
    X = Px_half @ (Px_half.T @ Z)
    # Ly = Lx.copy()
    # Py = np.eye(Nt_total * Np)
    W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(X, Z,
                                                               LSMethod = LSMethod,
                                                               displayConvergWarn = displayConvergWarn,
                                                               max_n = max_n,
                                                               M_error_threshold = M_error_threshold,
                                                               W_error_threshold = W_error_threshold,
                                                               largest_n = max_n,
                                                               check_rank = check_full_rank,
                                                               l1_alpha = l1_alpha,
                                                               l2_alpha = l2_alpha,
                                                               alpha = alpha,
                                                               reg_iter = reg_iter,
                                                               reg_tol = reg_tol,
                                                               kernel_fun = kernel_fun)
    W = W[-1].copy()
    V, _, _, _ = nplin.lstsq(ref_sig, template_sig @ W, rcond=None)
    if not full_rank_check:
        W, _, _, _ = nplin.lstsq(template_sig, ref_sig @ V, rcond=None)
    return W, V

class MSCCA_ls(MSCCA):
    """
    ms-CCA based on LS framework
    """
    def __init__(self,
                 n_neighbor: int = 12,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        """
        Special parameter
        ------------------
        n_neighbor: int
            Number of neighbors considered for computing spatical filter
        """
        super().__init__(n_neighbor = n_neighbor,
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
        
    def __copy__(self):
        copy_model = MSCCA_ls(n_neighbor = self.n_neighbor,
                            n_component = self.n_component,
                            n_jobs = self.n_jobs,
                            weights_filterbank = self.model['weights_filterbank'])
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model
        
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            freqs: Optional[List[float]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        freqs : Optional[List[float]], optional
            List of stimulus frequencies. The default is None.
            List shape: (trial_num,)
        """
        if freqs is None:
            raise ValueError('ms-CCA requires the list of stimulus frequencies')
        if ref_sig is None:
            raise ValueError('ms-CCA requires sine-cosine-based reference signal')
        if Y is None:
            raise ValueError('ms-CCA requires training label')
        if X is None:
            raise ValueError('ms-CCA requires training data')

        separated_trainSig = separate_trainSig(X, Y)
        
        # save reference
        self.model['ref_sig'] = ref_sig

        # generate template 
        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                           # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # spatial filters of template and reference: U3 and V3
        #   U3: (filterbank_num * stimulus_num * channel_num * n_component)
        #   V3: (filterbank_num * stimulus_num * harmonic_num * n_component)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        harmonic_num = ref_sig[0].shape[0]
        n_component = self.n_component
        n_neighbor = self.n_neighbor
        # construct reference and template signals for ms-cca
        d0 = int(np.floor(n_neighbor/2))
        U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
        _, freqs_idx, return_freqs_idx = sort(freqs)
        ref_sig_sort = [ref_sig[i] for i in freqs_idx]
        template_sig_sort = [template_sig[i] for i in freqs_idx]
        separated_trainSig_sort = [separated_trainSig[i] for i in freqs_idx]
        ref_sig_Q_sort = [ref_sig[i] for i in freqs_idx]
        ref_sig_mscca = []
        template_sig_mscca = []
        trainSig_mscca = []
        Q_mscca = []
        for class_idx in range(1,stimulus_num+1):
            if class_idx <= d0:
                start_idx = 0
                end_idx = n_neighbor
            elif class_idx > d0 and class_idx < (stimulus_num-d0+1):
                start_idx = class_idx - d0 - 1
                end_idx = class_idx + (n_neighbor-d0-1)
            else:
                start_idx = stimulus_num - n_neighbor
                end_idx = stimulus_num
            ref_sig_tmp = [ref_sig_sort[i] for i in range(start_idx, end_idx)]
            ref_sig_mscca.append(np.concatenate(ref_sig_tmp, axis = -1))
            template_sig_tmp = [template_sig_sort[i] for i in range(start_idx, end_idx)]
            template_sig_mscca.append(np.concatenate(template_sig_tmp, axis = -1))
            trainSig_tmp = [separated_trainSig_sort[i] for i in range(start_idx, end_idx)]
            trainSig_mscca.append(trainSig_tmp)
            Q_tmp = [ref_sig_Q_sort[i] for i in range(start_idx, end_idx)]
            Q_mscca.append(Q_tmp)
        for filterbank_idx in range(filterbank_num):
            if self.n_jobs is not None:
                U_tmp, V_tmp = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_MSCCA_ls_calW, LSconfig = self.LSconfig))(trainSig = [np.swapaxes(tmp[:,filterbank_idx,:,:],1,2) for tmp in trainSig_single], 
                                                                                            Q = [tmp.T for tmp in Q_single], 
                                                                                            template_sig=template_sig_single[filterbank_idx,:,:].T, 
                                                                                            ref_sig=ref_sig_single.T) 
                                                            for trainSig_single, Q_single, template_sig_single, ref_sig_single in zip(trainSig_mscca,Q_mscca,template_sig_mscca,ref_sig_mscca)))
            else:
                U_tmp = []
                V_tmp = []
                for trainSig_single, Q_single, template_sig_single, ref_sig_single in zip(trainSig_mscca,Q_mscca,template_sig_mscca,ref_sig_mscca):
                    U_temp_temp, V_temp_temp = _MSCCA_ls_calW(trainSig = [np.swapaxes(tmp[:,filterbank_idx,:,:],1,2) for tmp in trainSig_single], 
                                                                 Q = [tmp.T for tmp in Q_single], 
                                                                 template_sig=template_sig_single[filterbank_idx,:,:].T, 
                                                                 ref_sig=ref_sig_single.T,
                                                                 LSconfig = self.LSconfig)
                    # U_temp_temp1, V_temp_temp1, _ = canoncorr_ls(X=template_sig_single[filterbank_idx,:,:].T, Y=ref_sig_single.T, force_output_UV = True)
                    # diff = np.divide(U_temp_temp1, U_temp_temp)
                    U_tmp.append(U_temp_temp)
                    V_tmp.append(V_temp_temp)
            for stim_idx, (u, v) in enumerate(zip(U_tmp,V_tmp)):
                U[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
                V[filterbank_idx, stim_idx, :, :] = v[:harmonic_num,:n_component]
        self.model['U'] = U[:, return_freqs_idx, :, :]
        self.model['V'] = V[:, return_freqs_idx, :, :]

def _TRCA_ls_calW(trainSig, LSconfig):
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    Nt, Np, Nc = trainSig.shape
    Z = blkmat(trainSig) @ column_eyes(Nt, Nc)
    Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    Px = np.eye(Nt * Np)
    Ly = np.eye(Nt * Np)
    Py = np.eye(Nt * Np)
    W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Z, Lx, Px, Ly, Py,
                                                               LSMethod = LSMethod,
                                                               displayConvergWarn = displayConvergWarn,
                                                               max_n = max_n,
                                                               M_error_threshold = M_error_threshold,
                                                               W_error_threshold = W_error_threshold,
                                                               largest_n = max_n,
                                                               check_rank = check_full_rank,
                                                               l1_alpha = l1_alpha,
                                                               l2_alpha = l2_alpha,
                                                               alpha = alpha,
                                                               reg_iter = reg_iter,
                                                               reg_tol = reg_tol,
                                                               kernel_fun = kernel_fun)
    W = W[-1].copy()
    return W

class TRCA_ls(TRCA):
    """
    TRCA method based on LS framework
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)' 
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)

    def __copy__(self):
        copy_model = TRCA_ls(n_component = self.n_component,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'])
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        if Y is None:
            raise ValueError('TRCA requires training label')
        if X is None:
            raise ValueError('TRCA requires training data')

        separated_trainSig = separate_trainSig(X, Y)
           
        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        n_component = self.n_component
        U_trca = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        # possible_class = list(set(Y))
        # possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            # X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(partial(_TRCA_ls_calW, LSconfig = self.LSconfig))(trainSig = np.swapaxes(trainSig_single[:,filterbank_idx,:,:],1,2)) 
                                                                                                             for trainSig_single in separated_trainSig)
            else:
                U = []
                for trainSig_single in separated_trainSig:
                    U.append(
                        _TRCA_ls_calW(trainSig = np.swapaxes(trainSig_single[:,filterbank_idx,:,:],1,2), LSconfig = self.LSconfig)
                    )
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
        self.model['U'] = U_trca

class ETRCA_ls(ETRCA):
    """
    eTRCA method based on LS framework
    """
    def __init__(self,
                 n_component: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)

    def __copy__(self):
        copy_model = ETRCA_ls(n_component = None,
                              n_jobs = self.n_jobs,
                              weights_filterbank = self.model['weights_filterbank'])
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        if Y is None:
            raise ValueError('eTRCA requires training label')
        if X is None:
            raise ValueError('eTRCA requires training data')
        
        separated_trainSig = separate_trainSig(X, Y)

        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        # n_component = 1
        U_trca = np.zeros((filterbank_num, 1, channel_num, stimulus_num))
        # possible_class = list(set(Y))
        # possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            # X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(partial(_TRCA_ls_calW, LSconfig = self.LSconfig))(trainSig = np.swapaxes(trainSig_single[:,filterbank_idx,:,:],1,2)) 
                                                                                                             for trainSig_single in separated_trainSig)
            else:
                U = []
                for trainSig_single in separated_trainSig:
                    U.append(
                        _TRCA_ls_calW(trainSig = np.swapaxes(trainSig_single[:,filterbank_idx,:,:],1,2), LSconfig = self.LSconfig)
                    )
            # U = []
            # for X_single_class in X_train:
            #     U_element = _trca_U(X = X_single_class)
            #     U.append(U_element)
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, 0, :, stim_idx] = u[:channel_num,0]
        U_trca = np.repeat(U_trca, repeats = stimulus_num, axis = 1)

        self.model['U'] = U_trca

def _msetcca_cal_template_U_ls_1212(X_single_stimulus : ndarray,
                                    I : ndarray,
                                    ref_sig_single : Optional[ndarray] = None,
                                    n_component : int = 1,
                                    LSconfig : Optional[dict] = None):
    """
    Calculate templates and trials' spatial filters in multi-set CCA based on LS framework
    """
    trial_num, filterbank_num, channel_num, signal_len = X_single_stimulus.shape
    # prepare center matrix
    Px_single = I @ I.T
    if ref_sig_single is not None:
        # ref_sig_single = repmat(ref_sig_single.T, trial_num, 1)
        ref_sig_single = np.expand_dims(ref_sig_single.T,axis=0)
        ref_sig_single = np.repeat(ref_sig_single, trial_num, 0)
        ref_sig_single = blkmat(ref_sig_single)
    # calculate templates and spatial filters of each filterbank
    U_trial = []
    CCA_template = []
    for filterbank_idx in range(filterbank_num):
        X_single_stimulus_single_filterbank = X_single_stimulus[:,filterbank_idx,:,:]
        # template = blkmat(X_single_stimulus_single_filterbank)
        W, template, full_rank_check = _MsetCCA_ls_calW(np.swapaxes(X_single_stimulus_single_filterbank, 1,2), Px_single, LSconfig)
        eig_vec = W[:,:n_component]
        if ref_sig_single is not None and not full_rank_check:
            v_tmp, _, _, _ = nplin.lstsq(ref_sig_single, template @ eig_vec, rcond=None)
            eig_vec, _, _, _ = nplin.lstsq(template, ref_sig_single @ v_tmp, rcond=None)
        U_trial.append(np.expand_dims(eig_vec, axis = 0))
        # calculate template
        template = []
        for trial_idx in range(trial_num):
            template_temp = eig_vec[(trial_idx*channel_num):((trial_idx+1)*channel_num),:n_component].T @ X_single_stimulus_single_filterbank[trial_idx,:,:]
            template.append(template_temp)
        template = np.concatenate(template, axis = 0)
        CCA_template.append(np.expand_dims(template, axis = 0))
    U_trial = np.concatenate(U_trial, axis = 0)
    CCA_template = np.concatenate(CCA_template, axis = 0)
    return U_trial, CCA_template

def _MsetCCA_ls_calW(trainSig, Px_single, LSconfig):
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    Nt, Np, Nc = trainSig.shape
    # Z = blkmat(trainSig)
    # Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    Z_ori = blkmat(trainSig)
    Z = column_eyes(Nt, Np).T @ Z_ori
    Z = repmat(Z, Nt, 1)
    # tmp = Z - Lx @ Z_ori
    Px = blkrep(Px_single, Nt)
    # Ly = 1
    # Py = 1
    W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Px @ Z, Z_ori,
                                                               LSMethod = LSMethod,
                                                               displayConvergWarn = displayConvergWarn,
                                                               max_n = max_n,
                                                               M_error_threshold = M_error_threshold,
                                                               W_error_threshold = W_error_threshold,
                                                               largest_n = max_n,
                                                               check_rank = check_full_rank,
                                                               l1_alpha = l1_alpha,
                                                               l2_alpha = l2_alpha,
                                                               alpha = alpha,
                                                               reg_iter = reg_iter,
                                                               reg_tol = reg_tol,
                                                               kernel_fun = kernel_fun)
    # W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Z, Lx, Px, Ly, Py)
    W = W[-1].copy()
    return W, Z_ori, full_rank_check

class MsetCCA_ls(MsetCCA):
    """
    Multi-set CCA (LS framework)
    """

    def __init__(self,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_component: int = 1,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
    
    def __copy__(self):
        copy_model = MsetCCA_ls(n_jobs = self.n_jobs,
                                weights_filterbank = self.model['weights_filterbank'],
                                n_component = self.n_component)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            *argv, **kwargs):
        """
        Parameters
        ----------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        """
        if Y is None:
            raise ValueError('Multi-set CCA requires training label')
        if X is None:
            raise ValueError('Multi-set CCA training data')

        separated_trainSig = separate_trainSig(X, Y)

        if self.n_jobs is not None:
            U_all_stimuli, template_all_stimuli = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_msetcca_cal_template_U_ls_1212, n_component = self.n_component,
                                                                                                                                     LSconfig = self.LSconfig))
                                                                                   (a, np.eye(a.shape[-1])) 
                                                                                   for a in separated_trainSig))
        else:
            U_all_stimuli = []
            template_all_stimuli = []
            for a in separated_trainSig:
                U_temp, template_temp = _msetcca_cal_template_U_ls_1212(a, np.eye(a.shape[-1]), n_component = self.n_component, LSconfig = self.LSconfig)
                # U_temp, template_temp = _msetcca_cal_template_U_ls(a, I = np.eye(X[0].shape[-1]))
                # diff = np.divide(U_temp, U_temp1)
                U_all_stimuli.append(U_temp)
                template_all_stimuli.append(template_temp)

        self.model['U_trial'] = U_all_stimuli
        self.model['template'] = template_all_stimuli
        # generate template related QR
        template_sig_Q, template_sig_R, template_sig_P = qr_list(template_all_stimuli)
        self.model['template_sig_Q'] = template_sig_Q # List of shape: (stimulus_num,);
        self.model['template_sig_R'] = template_sig_R
        self.model['template_sig_P'] = template_sig_P

def _trcaR_cal_template_U_ls_1212(X_single_stimulus : ndarray,
                                  I : ndarray,
                                  n_component : int,
                                  ref_sig_single : Optional[ndarray] = None,
                                  LSconfig : Optional[dict] = None):
    """
    Calculate templates and trials' spatial filters in TRCA-R
    """
    trial_num, filterbank_num, channel_num, signal_len = X_single_stimulus.shape
    # prepare center matrix
    Px_single = I @ I.T
    # I = repmat(I, trial_num, 1)
    # P = I @ I.T
    if ref_sig_single is not None:
        # ref_sig_single = repmat(ref_sig_single.T, trial_num, 1)
        ref_sig_single = np.expand_dims(ref_sig_single.T,axis=0)
        ref_sig_single = np.repeat(ref_sig_single, trial_num, 0)
        ref_sig_single = blkmat(ref_sig_single)
    # calculate spatial filters of each filterbank
    U_trial = []
    for filterbank_idx in range(filterbank_num):
        X_single_stimulus_single_filterbank = X_single_stimulus[:,filterbank_idx,:,:]
        # template = []
        # for trial_idx in range(trial_num):
        #     template.append(X_single_stimulus_single_filterbank[trial_idx,:,:])
        # template = np.concatenate(template, axis = 1)
        # # calculate spatial filters of trials
        # Z = template.T
        # _, Dx, Vx = svd(Z, False, True)
        # Dx = np.diag(Dx)
        # VDV = Vx.T @ nplin.inv(Dx) @ Vx
        # PZ = P.T @ Z
        # X_rank = nplin.matrix_rank(PZ)
        # if ref_sig_single is not None and X_rank!=min(PZ.shape):
        #     max_n = 1
        # else:
        #     max_n = 100
        # W, _, _, _, _ = lsframe(PZ, 
        #                         PZ @ VDV, max_n = max_n)
        # W1 = W[-1].copy()
        # #
        # trainSig = np.swapaxes(X_single_stimulus_single_filterbank, 1,2)
        # Nt, Np, Nc = trainSig.shape
        # Z = blkmat(trainSig) @ column_eyes(Nt, Nc)
        # Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
        # Px = blkrep(Px_single, Nt)
        # Ly = np.eye(Nt * Np)
        # Py = np.eye(Nt * Np)
        # _, Dx, Vx = svd(Ly @ Py @ Z, False, True)
        # Dx = np.diag(Dx)
        # diff1 = PZ - Lx @ Px @ Z
        # diff2 = PZ @ VDV - Lx @ Px @ Z @ Vx.T @ nplin.inv(Dx) @ Vx
        # W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Z, Lx, Px, Ly, Py)
        # W2 = W[-1].copy()
        # #
        W, template, full_rank_check = _TRCAwithR_calW(np.swapaxes(X_single_stimulus_single_filterbank, 1,2),
                                                       Px_single,
                                                       LSconfig)
        eig_vec = W[:channel_num,:n_component]
        if ref_sig_single is not None and not full_rank_check:
            v_tmp, _, _, _ = nplin.lstsq(ref_sig_single, template @ eig_vec, rcond=None)
            eig_vec, _, _, _ = nplin.lstsq(template, ref_sig_single @ v_tmp, rcond=None)
        U_trial.append(np.expand_dims(eig_vec, axis = 0))
    U_trial = np.concatenate(U_trial, axis = 0)
    return U_trial

def _TRCAwithR_calW(trainSig, Px_single, LSconfig):
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    Nt, Np, Nc = trainSig.shape
    Z_ori = blkmat(trainSig) @ column_eyes(Nt, Nc)
    # Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    Z = repmat(np.sum(trainSig, axis = 0), Nt, 1)
    # diff = Z - Lx @ Z_ori
    Px = blkrep(Px_single, Nt)
    # Ly = np.eye(Nt * Np)
    # Py = np.eye(Nt * Np)
    # W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Lx @ Px @ Z_ori, Z_ori)
    W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Px @ Z, Z_ori,
                                                               LSMethod = LSMethod,
                                                               displayConvergWarn = displayConvergWarn,
                                                               max_n = max_n,
                                                               M_error_threshold = M_error_threshold,
                                                               W_error_threshold = W_error_threshold,
                                                               largest_n = max_n,
                                                               check_rank = check_full_rank,
                                                               l1_alpha = l1_alpha,
                                                               l2_alpha = l2_alpha,
                                                               alpha = alpha,
                                                               reg_iter = reg_iter,
                                                               reg_tol = reg_tol,
                                                               kernel_fun = kernel_fun)
    W = W[-1].copy()
    return W, Z_ori, full_rank_check

class TRCAwithR_ls(TRCAwithR):
    """
    TRCA method with reference signals based on LS framework
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)

    def __copy__(self):
        copy_model = TRCAwithR_ls(n_component = self.n_component,
                                  n_jobs = self.n_jobs,
                                  weights_filterbank = self.model['weights_filterbank'])
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if Y is None:
            raise ValueError('TRCA with reference signals requires training label')
        if X is None:
            raise ValueError('TRCA with reference signals training data')
        if ref_sig is None:
            raise ValueError('TRCA with reference signals requires sine-cosine-based reference signal')

        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        separated_trainSig = separate_trainSig(X, Y)
        ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig)

        if self.n_jobs is not None:
            U_all_stimuli = Parallel(n_jobs=self.n_jobs)(delayed(partial(_trcaR_cal_template_U_ls_1212, n_component = self.n_component,
                                                                                                        LSconfig = self.LSconfig))
                                                                        (X_single_stimulus = a, I = Q, ref_sig_single = ref_sig_single) 
                                                                        for a, Q, ref_sig_single in zip(separated_trainSig, ref_sig_Q, ref_sig))
        else:
            U_all_stimuli = []
            for a, Q, ref_sig_single in zip(separated_trainSig, ref_sig_Q, ref_sig):
                # tmp1 = _trcaR_cal_template_U_ls(X_single_stimulus = a, I = Q, n_component = self.n_component, ref_sig_single = ref_sig_single)
                # tmp2 = _trcaR_cal_template_U_ls_1212(X_single_stimulus = a, I = Q, n_component = self.n_component, ref_sig_single = ref_sig_single)
                # diff = np.abs(np.divide(tmp1, tmp2))
                U_all_stimuli.append(
                    _trcaR_cal_template_U_ls_1212(X_single_stimulus = a, I = Q, n_component = self.n_component, ref_sig_single = ref_sig_single, LSconfig = self.LSconfig)
                )
        U_trca = [np.expand_dims(u, axis=1) for u in U_all_stimuli]
        U_trca = np.concatenate(U_trca, axis = 1)
        self.model['U'] = U_trca

class ETRCAwithR_ls(ETRCAwithR):
    """
    eTRCA method with reference signals based on LS framework
    """
    def __init__(self,
                 n_component: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)

    def __copy__(self):
        copy_model = ETRCAwithR_ls(n_component = None,
                                   n_jobs = self.n_jobs,
                                   weights_filterbank = self.model['weights_filterbank'])
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if Y is None:
            raise ValueError('eTRCA with reference signals requires training label')
        if X is None:
            raise ValueError('eTRCA with reference signals training data')
        if ref_sig is None:
            raise ValueError('eTRCA with reference signals requires sine-cosine-based reference signal')

        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        separated_trainSig = separate_trainSig(X, Y)
        ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig)

        if self.n_jobs is not None:
            U_all_stimuli = Parallel(n_jobs=self.n_jobs)(delayed(partial(_trcaR_cal_template_U_ls_1212, n_component = self.n_component,
                                                                                                        LSconfig = self.LSconfig))
                                                                (X_single_stimulus = a, I = Q, ref_sig_single = ref_sig_single) 
                                                                for a, Q, ref_sig_single in zip(separated_trainSig, ref_sig_Q, ref_sig))
        else:
            U_all_stimuli = []
            for a, Q, ref_sig_single in zip(separated_trainSig, ref_sig_Q, ref_sig):
                U_all_stimuli.append(
                    _trcaR_cal_template_U_ls_1212(X_single_stimulus = a, I = Q, n_component = self.n_component, ref_sig_single = ref_sig_single, LSconfig = self.LSconfig)
                )
        # U_trca = [u for u in U_all_stimuli]
        U_trca = np.concatenate(U_all_stimuli, axis = 2)
        U_trca = np.expand_dims(U_trca, axis = 1)
        U_trca = np.repeat(U_trca, repeats = len(U_all_stimuli), axis = 1)
        self.model['U'] = U_trca

class MsetCCAwithR_ls(MsetCCAwithR):
    """
    Multi-set CCA with reference signals based on LS framework
    """

    def __init__(self,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_component: int = 1,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank,
                         n_component = n_component)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)
    
    def __copy__(self):
        copy_model = MsetCCAwithR_ls(n_jobs = self.n_jobs,
                                    weights_filterbank = self.model['weights_filterbank'],
                                    n_component = self.n_component)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargs):
        """
        Parameters
        ----------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
        """
        if Y is None:
            raise ValueError('Multi-set CCA with reference signals requires training label')
        if X is None:
            raise ValueError('Multi-set CCA with reference signals training data')
        if ref_sig is None:
            raise ValueError('Multi-set CCA with reference signals requires sine-cosine-based reference signal')
        

        separated_trainSig = separate_trainSig(X, Y)
        ref_sig_Q, ref_sig_R, ref_sig_P = qr_list(ref_sig)

        if self.n_jobs is not None:
            U_all_stimuli, template_all_stimuli = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_msetcca_cal_template_U_ls_1212, n_component = self.n_component,
                                                                                                                                     LSconfig = self.LSconfig))
                                                                                   (a, Q, ref_sig_single) 
                                                                                   for a, Q, ref_sig_single in zip(separated_trainSig, ref_sig_Q, ref_sig)))
        else:
            U_all_stimuli = []
            template_all_stimuli = []
            for a, Q, ref_sig_single in zip(separated_trainSig, ref_sig_Q, ref_sig):
                U_temp, template_temp = _msetcca_cal_template_U_ls_1212(a, Q, ref_sig_single, n_component = self.n_component, LSconfig = self.LSconfig)
                # U_temp, template_temp = _msetcca_cal_template_U_ls(X_single_stimulus = a, I = Q, ref_sig_single = ref_sig_single)
                # diff = np.divide(U_temp, U_temp1)
                U_all_stimuli.append(U_temp)
                template_all_stimuli.append(template_temp)

        self.model['U_trial'] = U_all_stimuli
        self.model['template'] = template_all_stimuli
        # generate template related QR
        template_sig_Q, template_sig_R, template_sig_P = qr_list(template_all_stimuli)
        self.model['template_sig_Q'] = template_sig_Q # List of shape: (stimulus_num,);
        self.model['template_sig_R'] = template_sig_R
        self.model['template_sig_P'] = template_sig_P

def _MSTRCA_ls_calW(trainSig, LSconfig):
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    #
    X = []
    for trainSig_single_freq in trainSig:
        Nt, Np, Nc = trainSig_single_freq.shape
        for _ in range(Nt):
            X.append(np.sum(trainSig_single_freq, axis = 0))
    X = np.concatenate(X, axis = 0)
    Y = []
    for trainSig_single_freq in trainSig:
        Nt, Np, Nc = trainSig_single_freq.shape
        for n in range(Nt):
            Y.append(trainSig_single_freq[n,:,:])
    Y = np.concatenate(Y, axis = 0)
    # Z = np.ndarray([])
    # Lx = np.ndarray([])
    # Nt_total = 0
    # for trainSig_single_freq in trainSig:
    #     Nt, Np, Nc = trainSig_single_freq.shape
    #     # Q_single_freq, _, _ = slin.qr(ref_sig[Np_total:(Np_total + Np),:], mode = 'economic', pivoting = True)
    #     # Np_total += Np
    #     # _, Nh = Q_single_freq.shape
    #     Nt_total += Nt
    #     if len(Z.shape)==0:
    #         Z = blkmat(trainSig_single_freq)
    #         Lx = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    #     else:
    #         Z = slin.block_diag(Z, blkmat(trainSig_single_freq))
    #         Lx = slin.block_diag(Lx, column_eyes(Nt, Np) @ column_eyes(Nt, Np).T)
    # Z = Z @ column_eyes(Nt_total, Nc)
    # Px = np.eye(Nt_total * Np)
    # Ly = np.eye(Nt_total * Np)
    # Py = np.eye(Nt_total * Np)
    # W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(Z, Lx, Px, Ly, Py)
    W, M, M_error, W_error, n, full_rank_check = ssvep_lsframe(X, Y,
                                                               LSMethod = LSMethod,
                                                               displayConvergWarn = displayConvergWarn,
                                                               max_n = max_n,
                                                               M_error_threshold = M_error_threshold,
                                                               W_error_threshold = W_error_threshold,
                                                               largest_n = max_n,
                                                               check_rank = check_full_rank,
                                                               l1_alpha = l1_alpha,
                                                               l2_alpha = l2_alpha,
                                                               alpha = alpha,
                                                               reg_iter = reg_iter,
                                                               reg_tol = reg_tol,
                                                               kernel_fun = kernel_fun)
    W = W[-1].copy()
    return W

class MSETRCA_ls(MSETRCA):
    """
    ms-eTRCA method based on LS framework
    """
    def __init__(self,
                 n_neighbor: int = 2,
                 n_component: Optional[int] = None,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        super().__init__(n_neighbor = n_neighbor,
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)

    def __copy__(self):
        copy_model = MSETRCA_ls(n_neighbor = self.n_neighbor,
                                n_component = None,
                                n_jobs = self.n_jobs,
                                weights_filterbank = self.model['weights_filterbank'])
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            freqs: Optional[List[float]] = None,
            *argv, **kwargs):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        freqs : Optional[List[float]], optional
            List of stimulus frequencies. The default is None.
            List shape: (trial_num,)
        """
        if freqs is None:
            raise ValueError('ms-eTRCA requires the list of stimulus frequencies')
        if Y is None:
            raise ValueError('ms-eTRCA requires training label')
        if X is None:
            raise ValueError('ms-eTRCA requires training data')

        separated_trainSig = separate_trainSig(X, Y)
           
        template_sig = gen_template(X, Y) # List of shape: (stimulus_num,); 
                                          # Template shape: (filterbank_num, channel_num, signal_len)
        self.model['template_sig'] = template_sig

        # spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X: (filterbank_num, channel_num, signal_len)
        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        n_neighbor = self.n_neighbor
        # n_component = 1
        d0 = int(np.floor(n_neighbor/2))
        _, freqs_idx, return_freqs_idx = sort(freqs)
        separated_trainSig = [separated_trainSig[i] for i in freqs_idx]
        U_trca = np.zeros((filterbank_num, 1, channel_num, stimulus_num))
        # possible_class = list(set(Y))
        # possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            # X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            # X_train = [X_train[i] for i in freqs_idx]

            # nTrial_list = []
            # for a in X_train:
            #     nTrial_list.append(len(a))

            # if self.n_jobs is not None:
            #     trca_X1, trca_X2 = zip(*Parallel(n_jobs=self.n_jobs)(delayed(_trca_U_1)(a) for a in X_train))
            # else:
            #     trca_X1 = []
            #     trca_X2 = []
            #     for a in X_train:
            #         trca_X1_temp, trca_X2_temp = _trca_U_1(a)
            #         trca_X1.append(trca_X1_temp)
            #         trca_X2.append(trca_X2_temp)

            # nTrial_mstrca = []
            # trca_X1_mstrca = []
            # trca_X2_mstrca = []
            trainSig_mstrca = []
            for class_idx in range(1,stimulus_num+1):
                if class_idx <= d0:
                    start_idx = 0
                    end_idx = n_neighbor
                elif class_idx > d0 and class_idx < (stimulus_num-d0+1):
                    start_idx = class_idx - d0 - 1
                    end_idx = class_idx + (n_neighbor-d0-1)
                else:
                    start_idx = stimulus_num - n_neighbor
                    end_idx = stimulus_num
                trainSig_mstrca_tmp = [np.swapaxes(separated_trainSig[i][:,filterbank_idx,:,:], 1,2) for i in range(start_idx, end_idx)]
                trainSig_mstrca.append(trainSig_mstrca_tmp)
                # nTrial_mstrca_tmp = [nTrial_list[i] for i in range(start_idx, end_idx)]
                # nTrial_mstrca.append(sum(nTrial_mstrca_tmp))
                # trca_X1_mstrca_tmp = [trca_X1[i] for i in range(start_idx, end_idx)]
                # trca_X1_mstrca.append(np.concatenate(trca_X1_mstrca_tmp, axis=-1))
                # trca_X2_mstrca_tmp = [trca_X2[i].T for i in range(start_idx, end_idx)]
                # trca_X2_mstrca.append(np.concatenate(trca_X2_mstrca_tmp, axis=-1))

            if self.n_jobs is not None:
                U = Parallel(n_jobs = self.n_jobs)(delayed(partial(_MSTRCA_ls_calW, LSconfig = self.LSconfig))(trainSig_single_class) for trainSig_single_class in trainSig_mstrca)
            else:
                U = []
                for trainSig_single_class in trainSig_mstrca:
                    U_tmp = _MSTRCA_ls_calW(trainSig_single_class, LSconfig = self.LSconfig)
                    U.append(
                        U_tmp
                    )
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, 0, :, stim_idx] = u[:channel_num,0]
        U_trca = np.repeat(U_trca[:,:,:,return_freqs_idx], repeats = stimulus_num, axis = 1)

        self.model['U'] = U_trca

def _TDCA_ls_calW(trainSig, n_jobs = None, LSconfig = None):
    def X_single_freq(trainSig_single_freq, All_train_sum):
        Nt, Np, Nc = trainSig_single_freq.shape
        X = []
        for _ in range(Nt):
            X_tmp = np.sum(trainSig_single_freq, axis = 0) - 1/Nf * All_train_sum
            X.append(X_tmp)
        X = np.concatenate(X, axis = 0)
        return X
    def Y_single_freq(trainSig_single_freq):
        Nt, Np, Nc = trainSig_single_freq.shape
        Y = []
        for n in range(Nt):
            Y_tmp = trainSig_single_freq[n,:,:] - 1/Nt * np.sum(trainSig_single_freq, axis = 0)
            Y.append(Y_tmp)
        Y = np.concatenate(Y, axis = 0)
        return Y
    #
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    #
    Nf = len(trainSig)
    All_train_sum = None
    for trainSig_single_freq in trainSig:
        if All_train_sum is None:
            All_train_sum = np.sum(trainSig_single_freq, axis = 0)
        else:
            All_train_sum = All_train_sum + np.sum(trainSig_single_freq, axis = 0)
    if n_jobs is None:
        X = []
        for trainSig_single_freq in trainSig:
            X.append(
                X_single_freq(trainSig_single_freq, All_train_sum)
            )
            # Nt, Np, Nc = trainSig_single_freq.shape
            # for n in range(Nt):
            #     X_tmp = np.sum(trainSig_single_freq, axis = 0) - 1/Nf * All_train_sum
            #     X.append(X_tmp)
    else:
        X = Parallel(n_jobs=n_jobs)(delayed(partial(X_single_freq, All_train_sum = All_train_sum))
                                                   (trainSig_single_freq = trainSig_single_freq)
                                                   for trainSig_single_freq in trainSig)
    X = np.concatenate(X, axis = 0)
    if n_jobs is None:
        Y = []
        for trainSig_single_freq in trainSig:
            Y.append(
                Y_single_freq(trainSig_single_freq)
            )
            # Nt, Np, Nc = trainSig_single_freq.shape
            # for n in range(Nt):
            #     Y_tmp = trainSig_single_freq[n,:,:] - 1/Nt * np.sum(trainSig_single_freq, axis = 0)
            #     Y.append(Y_tmp)
    else:
        Y = Parallel(n_jobs=n_jobs)(delayed(Y_single_freq)
                                            (trainSig_single_freq = trainSig_single_freq)
                                            for trainSig_single_freq in trainSig)
    Y = np.concatenate(Y, axis = 0)
    W, M, M_error, W_error, ls_n, full_rank_check = ssvep_lsframe(X, Y,
                                                                  LSMethod = LSMethod,
                                                                  displayConvergWarn = displayConvergWarn,
                                                                  max_n = max_n,
                                                                  M_error_threshold = M_error_threshold,
                                                                  W_error_threshold = W_error_threshold,
                                                                  largest_n = max_n,
                                                                  check_rank = check_full_rank,
                                                                  l1_alpha = l1_alpha,
                                                                  l2_alpha = l2_alpha,
                                                                  alpha = alpha,
                                                                  reg_iter = reg_iter,
                                                                  reg_tol = reg_tol,
                                                                  kernel_fun = kernel_fun)
    #
    # Nf = len(trainSig)
    # Z = np.ndarray([])
    # Lx_first = np.ndarray([])
    # # Px = np.ndarray([])
    # Nt_total = 0
    # for trainSig_single_freq in trainSig:
    #     Nt, Np, Nc = trainSig_single_freq.shape
    #     # Q_single_freq, _, _ = slin.qr(ref_sig[Np_total:(Np_total + Np),:], mode = 'economic', pivoting = True)
    #     # Np_total += Np
    #     # _, Nh = Q_single_freq.shape
    #     Nt_total += Nt
    #     if len(Z.shape)==0:
    #         Z = blkmat(trainSig_single_freq)
    #         Lx_first = column_eyes(Nt, Np) @ column_eyes(Nt, Np).T
    #         # Px = blkrep(np.concatenate([np.eye(Np),P_single_freq],axis=1), Nt)
    #     else:
    #         Z = slin.block_diag(Z, blkmat(trainSig_single_freq))
    #         Lx_first = slin.block_diag(Lx_first, column_eyes(Nt, Np) @ column_eyes(Nt, Np).T)
    #         # Px = slin.block_diag(Px, blkrep(np.concatenate([np.eye(Np),P_single_freq],axis=1), Nt))
    # Z = Z @ column_eyes(Nt_total, Nc)
    # Lx = Lx_first - 1/Nf * (column_eyes(Nt_total, Np) @ column_eyes(Nt_total, Np).T)
    # Px = np.eye(Nt_total * Np)
    # Ly = np.eye(Nt_total * Np) - 1/Nt * Lx_first
    # W, M, M_error, W_error, ls_n, full_rank_check = ssvep_lsframe(Z, Lx, Px, Ly, Px)
    #
    W = W[-1].copy()
    return W

class TDCA_ls(TDCA):
    """
    TDCA method based on LS framework
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_delay: int = 0,
                 LSMethod = 'lstsq',
                 displayConvergWarn: bool = False,
                 max_n: int = 100,
                 M_error_threshold: float = 1e-6,
                 W_error_threshold: float = 1e-6,
                 l1_alpha = None,
                 l2_alpha = None,
                 alpha = None,
                 reg_iter = 1000,
                 reg_tol = 1e-4):
        """
        Special parameter
        -----------------
        n_delay: int
            Number of delayed signals
            Default is 0 (no delay)
        """
        super().__init__(n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank,
                         n_delay = n_delay)
        self.ID = self.ID + ' (ls)'
        self.LSconfig = {
            'LSMethod': LSMethod,
            'displayConvergWarn': displayConvergWarn,
            'max_n': max_n,
            'M_error_threshold': M_error_threshold,
            'W_error_threshold': W_error_threshold,
            'l1_alpha': l1_alpha,
            'l2_alpha': l2_alpha,
            'alpha': alpha,
            'reg_iter': reg_iter,
            'reg_tol': reg_tol
        }
        self.ID = combine_name(self)

    def __copy__(self):
        copy_model = TDCA_ls(n_component = self.n_component,
                             n_jobs = self.n_jobs,
                             weights_filterbank = self.model['weights_filterbank'],
                             n_delay = self.n_delay)
        copy_model.LSconfig = deepcopy(self.LSconfig)
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None,
            *argv, **kwargv):
        """
        Parameters
        -------------
        X : Optional[List[ndarray]], optional
            List of training EEG data. The default is None.
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)
        Y : Optional[List[int]], optional
            List of labels (stimulus indices). The default is None.
            List shape: (trial_num,)
        ref_sig : Optional[List[ndarray]], optional
            Sine-cosine-based reference signals. The default is None.
            List of shape: (stimulus_num,)
            Reference signal shape: (harmonic_num, signal_len)
        """
        if Y is None:
            raise ValueError('TDCA requires training label')
        if X is None:
            raise ValueError('TDCA requires training data')
        if ref_sig is None:
            raise ValueError("TDCA requires reference signals")

        ref_sig_Q, _, _ = qr_list(ref_sig)
        ref_sig_P = [Q @ Q.T for Q in ref_sig_Q]
        self.model['ref_sig_P'] = ref_sig_P

        n_delay = self.n_delay
        X_P = _gen_delay_X(X, n_delay)
        separated_trainSig = separate_trainSig(X_P, Y)
        separated_trainSig = [np.concatenate([trainSig_single_freq, trainSig_single_freq @ P_single_freq], axis=-1)
                              for trainSig_single_freq, P_single_freq in zip(separated_trainSig, ref_sig_P)]

        # template signals and spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X or template_sig: (filterbank_num, channel_num, signal_len)
        filterbank_num, channel_num, signal_len = X_P[0].shape
        stimulus_num = len(ref_sig)
        n_component = self.n_component
        # possible_class = list(set(Y))
        # possible_class.sort(reverse = False)
        # template_sig = [np.zeros((filterbank_num, channel_num, signal_len)) for _ in range(stimulus_num)]
        U_tdca = np.zeros((filterbank_num, 1, channel_num, n_component))
        for filterbank_idx in range(filterbank_num):
            # X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            # trial_num = len(X_train[0])

            # if self.n_jobs is not None:
            #     X_train_delay = Parallel(n_jobs = self.n_jobs)(delayed(partial(_gen_delay_X, n_delay = n_delay))(X = X_single_class) for X_single_class in X_train)
            #     P_combine_X_train = Parallel(n_jobs = self.n_jobs)(delayed(_gen_P_combine_X)(X = X_single_class, P = P_single_class) for X_single_class, P_single_class in zip(X_train_delay, ref_sig_P))
            # else:
            #     X_train_delay = []
            #     for X_single_class in X_train:
            #         X_train_delay.append(
            #             _gen_delay_X(X = X_single_class, n_delay = n_delay)
            #         )
            #     P_combine_X_train = []
            #     for X_single_class, P_single_class in zip(X_train_delay, ref_sig_P):
            #         P_combine_X_train.append(
            #             _gen_P_combine_X(X = X_single_class, P = P_single_class)
            #         )
            # # Calculate template
            # if self.n_jobs is not None:
            #     P_combine_X_train_mean = Parallel(n_jobs=self.n_jobs)(delayed(mean_list)(X = P_combine_X_train_single_class) for P_combine_X_train_single_class in P_combine_X_train)
            # else:
            #     P_combine_X_train_mean = []
            #     for P_combine_X_train_single_class in P_combine_X_train:
            #         P_combine_X_train_mean.append(
            #             mean_list(X = P_combine_X_train_single_class)
            #         )
            # for stim_idx, P_combine_X_train_mean_single_class in enumerate(P_combine_X_train_mean):
            #     template_sig[stim_idx][filterbank_idx,:,:] = P_combine_X_train_mean_single_class
            # # Calulcate spatial filter
            # P_combine_X_train_all_mean = mean_list(P_combine_X_train_mean)
            # X_tmp = []
            # X_mean = []
            # for P_combine_X_train_single_class, P_combine_X_train_mean_single_class in zip(P_combine_X_train, P_combine_X_train_mean):
            #     for X_tmp_tmp in P_combine_X_train_single_class:
            #         X_tmp.append(X_tmp_tmp)
            #         X_mean.append(P_combine_X_train_mean_single_class)

            # if self.n_jobs is not None:
            #     Sw_list = Parallel(n_jobs=self.n_jobs)(delayed(partial(_covariance_tdca, num = trial_num,
            #                                                                             division_num = trial_num))(X = X_tmp_tmp, X_mean = X_mean_tmp)
            #                                                                             for X_tmp_tmp, X_mean_tmp in zip(X_tmp, X_mean))
            #     Sb_list = Parallel(n_jobs=self.n_jobs)(delayed(partial(_covariance_tdca, X_mean = P_combine_X_train_all_mean,
            #                                                                             num = stimulus_num,
            #                                                                             division_num = stimulus_num))(X = P_combine_X_train_mean_single_class)
            #                                                                             for P_combine_X_train_mean_single_class in P_combine_X_train_mean)
            # else:
            #     Sw_list = []
            #     for X_tmp_tmp, X_mean_tmp in zip(X_tmp, X_mean):
            #         Sw_list.append(
            #             _covariance_tdca(X = X_tmp_tmp, X_mean = X_mean_tmp, num = trial_num, division_num = trial_num)
            #         )
            #     Sb_list = []
            #     for P_combine_X_train_mean_single_class in P_combine_X_train_mean:
            #         Sb_list.append(
            #             _covariance_tdca(X = P_combine_X_train_mean_single_class, X_mean = P_combine_X_train_all_mean, num = stimulus_num, division_num = stimulus_num)
            #         )

            # Sw = sum_list(Sw_list)
            # Sb = sum_list(Sb_list)
            # eig_vec = eigvec(Sb, Sw)

            # Calulcate spatial filter
            W = _TDCA_ls_calW([np.swapaxes(trainSig[:,filterbank_idx,:,:], 1,2) for trainSig in separated_trainSig],
                              self.n_jobs,
                              LSconfig = self.LSconfig)
            U_tdca[filterbank_idx,0,:,:] = W[:,:n_component]
        # Calculate template
        template_sig = [np.mean(tmp, 0) for tmp in separated_trainSig]
        # for stim_idx in range(stimulus_num):
        #     trainSig_single_freq = separated_trainSig[stim_idx]
        #     P_single_freq = ref_sig_P[stim_idx]
        #     tmp = np.concatenate([trainSig_single_freq, trainSig_single_freq @ P_single_freq], axis=-1)
        #     template_sig[stim_idx] = np.mean(tmp, 0)
        U_tdca = np.repeat(U_tdca, repeats = stimulus_num, axis = 1)
        self.model['U'] = U_tdca
        self.model['template_sig'] = template_sig
