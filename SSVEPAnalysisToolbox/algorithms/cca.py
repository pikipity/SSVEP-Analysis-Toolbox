# -*- coding: utf-8 -*-
"""
CCA based recognition methods
"""

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial

import numpy as np
import scipy.linalg as slin
import scipy.stats as stats

from .basemodel import BaseModel
from .utils import qr_remove_mean, qr_inverse, mldivide, canoncorr

def _r_cca_canoncorr_withUV(X: ndarray,
                            Y: List[ndarray],
                            U: ndarray,
                            V: ndarray) -> ndarray:
    """
    Calculate correlation of CCA based on canoncorr for single trial data using existing U and V

    Parameters
    ----------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of reference signals
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    """
    filterbank_num, channel_num, signal_len = X.shape
    harmonic_num = Y[0].shape[0]
    stimulus_num = len(Y)
    
    R = np.zeros((filterbank_num, stimulus_num))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            A_r = U[k,i,:,:]
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp
            b = B_r.T @ Y[i]
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            # r2 = stats.pearsonr(a, b)[0]
            r = stats.pearsonr(a, b)[0]
            R[k,i] = r
    return R

def _r_cca_qr_withUV(X: ndarray,
                  Y_Q: List[ndarray],
                  Y_R: List[ndarray],
                  Y_P: List[ndarray],
                  U: ndarray,
                  V: ndarray) -> ndarray:
    """
    Calculate correlation of CCA based on qr decomposition for single trial data using existing U and V

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
    U : ndarray
        Spatial filter
        shape: (filterbank_num * stimulus_num * channel_num * n_component)
    V : ndarray
        Weights of harmonics
        shape: (filterbank_num * stimulus_num * harmonic_num * n_component)

    Returns
    -------
    R : ndarray
        Correlation
        shape: (filterbank_num * stimulus_num)
    """
    filterbank_num, channel_num, signal_len = X.shape
    harmonic_num = Y_R[0].shape[1]
    stimulus_num = len(Y_Q)
    
    Y = [qr_inverse(Y_Q[i],Y_R[i],Y_P[i]).T for i in range(len(Y_Q))]
    
    R = np.zeros((filterbank_num, stimulus_num))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        X_Q, X_R, X_P = qr_remove_mean(tmp.T)
        for i in range(stimulus_num):
            A_r = U[k,i,:,:]
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp
            b = B_r.T @ Y[i]
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            # r2 = stats.pearsonr(a, b)[0]
            r = stats.pearsonr(a, b)[0]
            R[k,i] = r
    return R
    
def _r_cca_canoncorr(X: ndarray,
                     Y: List[ndarray],
                     n_component: int,
                     force_output_UV: Optional[bool] = False) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
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
    harmonic_num = Y[0].shape[0]
    stimulus_num = len(Y)
    
    # R1 = np.zeros((filterbank_num,stimulus_num))
    # R2 = np.zeros((filterbank_num,stimulus_num))
    R = np.zeros((filterbank_num, stimulus_num))
    U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
    V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            
            if n_component == 0 and force_output_UV is False:
                D = canoncorr(tmp.T, Y[i].T, False)
                r = D[0]
            else:
                A_r, B_r, D = canoncorr(tmp.T, Y[i].T, True)
                r = D[0]
                U[k,i,:,:] = A_r[:channel_num, :n_component]
                V[k,i,:,:] = B_r[:harmonic_num, :n_component]
                
            # R1[k,i] = r1
            # R2[k,i] = r2
            R[k,i] = r
    if force_output_UV:
        return R, U, V
    else:
        return R

def _r_cca_qr(X: ndarray,
           Y_Q: List[ndarray],
           Y_R: List[ndarray],
           Y_P: List[ndarray],
           n_component: int,
           force_output_UV: Optional[bool] = False) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
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
    filterbank_num, channel_num, signal_len = X.shape
    harmonic_num = Y_R[0].shape[1]
    stimulus_num = len(Y_Q)
    
    Y = [qr_inverse(Y_Q[i],Y_R[i],Y_P[i]).T for i in range(len(Y_Q))]
    
    # R1 = np.zeros((filterbank_num,stimulus_num))
    # R2 = np.zeros((filterbank_num,stimulus_num))
    R = np.zeros((filterbank_num, stimulus_num))
    U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
    V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        X_Q, X_R, X_P = qr_remove_mean(tmp.T)
        for i in range(stimulus_num):
            svd_X = X_Q.T @ Y_Q[i]
            if svd_X.shape[0]>svd_X.shape[1]:
                full_matrices=False
            else:
                full_matrices=True
            
            if n_component == 0 and force_output_UV is False:
                D = slin.svd(svd_X,
                             full_matrices=full_matrices,
                             compute_uv=False,
                             check_finite=False,
                             lapack_driver='gesvd')
                # r1 = D[0]
                r = D[0]
            else:
                L, D, M = slin.svd(svd_X,
                                 full_matrices=full_matrices,
                                 check_finite=False,
                                 lapack_driver='gesvd')
                M = M.T
                A = mldivide(X_R, L) * np.sqrt(signal_len - 1)
                B = mldivide(Y_R[i], M) * np.sqrt(signal_len - 1)
                A_r = np.zeros(A.shape)
                for n in range(A.shape[0]):
                    A_r[X_P[n],:] = A[n,:]
                B_r = np.zeros(B.shape)
                for n in range(B.shape[0]):
                    B_r[Y_P[i][n],:] = B[n,:]
                
                # a = A_r[:channel_num, :n_component].T @ tmp
                # b = B_r[:harmonic_num, :n_component].T @ Y[i]
                # a = np.reshape(a, (-1))
                # b = np.reshape(b, (-1))
                
                # r2 = stats.pearsonr(a, b)[0]
                r = D[0]
                U[k,i,:,:] = A_r[:channel_num, :n_component]
                V[k,i,:,:] = B_r[:harmonic_num, :n_component]
                
            # R1[k,i] = r1
            # R2[k,i] = r2
            R[k,i] = r
    if force_output_UV:
        return R, U, V
    else:
        return R
   

class SCCA_canoncorr(BaseModel):
    """
    Standard CCA based on canoncorr
    
    Computational time - Long
    Required memory - Small
    """
    def __init__(self,
                 n_component: Optional[int] = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: Optional[bool] = False,
                 update_UV: Optional[bool] = True):
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
        super().__init__(ID = 'sCCA_canoncorr',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.force_output_UV = force_output_UV
        self.update_UV = update_UV
        
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None):
        if ref_sig is None:
            raise ValueError('sCCA requires sine-cosine-based reference signal')
           
            
        self.model['ref_sig'] = ref_sig
        
        self.model['U'] = None # Spatial filter of EEG
        self.model['V'] = None # Weights of harmonics
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        n_component = self.n_component
        Y = self.model['ref_sig']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, n_component=n_component, Y=Y, force_output_UV=True))(a) for a in X))
                self.model['U'] = U
                self.model['V'] = V
            else:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr, n_component=n_component, Y=Y, force_output_UV=False))(a) for a in X)
        else:
            U = self.model['U']
            V = self.model['V']
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_canoncorr_withUV, Y=Y))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred
     

class SCCA_qr(BaseModel):
    """
    Standard CCA based on qr decomposition
    
    Computational time - Short
    Required memory - Large
    """
    def __init__(self,
                 n_component: Optional[int] = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 force_output_UV: Optional[bool] = False,
                 update_UV: Optional[bool] = True):
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
        super().__init__(ID = 'sCCA_qr',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.force_output_UV = force_output_UV
        self.update_UV = update_UV
        
    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None):
        if ref_sig is None:
            raise ValueError('sCCA requires sine-cosine-based reference signal')
            
        ref_sig_Q = []
        ref_sig_R = []
        ref_sig_P = []
        for ref_sig_single in ref_sig:
            Q, R, P = qr_remove_mean(ref_sig_single.T)
            ref_sig_Q.append(Q)
            ref_sig_R.append(R)
            ref_sig_P.append(P)
            
        self.model['ref_sig_Q'] = ref_sig_Q
        self.model['ref_sig_R'] = ref_sig_R
        self.model['ref_sig_P'] = ref_sig_P
        
        self.model['U'] = None # Spatial filter of EEG
        self.model['V'] = None # Weights of harmonics
        
    def predict(self,
                X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T
        n_component = self.n_component
        Y_Q = self.model['ref_sig_Q']
        Y_R = self.model['ref_sig_R']
        Y_P = self.model['ref_sig_P']
        force_output_UV = self.force_output_UV
        update_UV = self.update_UV
        
        if update_UV or self.model['U'] is None or self.model['V'] is None:
            if force_output_UV or not update_UV:
                r, U, V = zip(*Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=True))(a) for a in X))
                self.model['U'] = U
                self.model['V'] = V
            else:
                r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr, n_component=n_component, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P, force_output_UV=False))(a) for a in X)
        else:
            U = self.model['U']
            V = self.model['V']
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_cca_qr_withUV, Y_Q=Y_Q, Y_R=Y_R, Y_P=Y_P))(X=a, U=u, V=v) for a, u, v in zip(X,U,V))
        
        Y_pred = [int(np.argmax(weights_filterbank @ r_single, axis = 1)) for r_single in r]
        
        return Y_pred
    

