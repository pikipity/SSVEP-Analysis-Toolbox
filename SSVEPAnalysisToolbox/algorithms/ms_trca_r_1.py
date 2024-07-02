# -*- coding: utf-8 -*-
"""
MS-TRCA-R-1
"""

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial
from copy import deepcopy

import numpy as np

from .lsframework import TDCA_ls
from .utils_ls import (
    get_lsconfig,
    ssvep_lsframe
)
from .utils import (
    qr_list, sort, separate_trainSig
)
from .tdca import (
    _gen_delay_X
)

class TRCAwithR_multi_f(TDCA_ls):
    def __init__(self,
                 n_neighbor: int = 2,
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
        super().__init__(n_component, n_jobs, weights_filterbank, n_delay, LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol)
        self.ID = 'TRCAwithR_multi_f'
        self.n_neighbor = n_neighbor

    def __copy__(self):
        copy_model = TRCAwithR_multi_f(n_neighbor = self.n_neighbor,
                                    n_component = self.n_component,
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
            freqs: Optional[List[float]] = None,
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
        if freqs is None:
            raise ValueError('ms-eTRCA requires the list of stimulus frequencies')
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
        separated_trainSig_1 = separate_trainSig(X_P, Y)
        separated_trainSig = [np.concatenate([trainSig_single_freq @ P_single_freq], axis=-1)
                              for trainSig_single_freq, P_single_freq in zip(separated_trainSig_1, ref_sig_P)]

        # template signals and spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X or template_sig: (filterbank_num, channel_num, signal_len)
        filterbank_num, channel_num, signal_len = X_P[0].shape
        stimulus_num = len(ref_sig)
        n_component = self.n_component
        n_neighbor = self.n_neighbor
        d0 = int(np.floor(n_neighbor/2))
        _, freqs_idx, return_freqs_idx = sort(freqs)
        separated_trainSig = [separated_trainSig[i] for i in freqs_idx]
        separated_trainSig_1 = [separated_trainSig_1[i] for i in freqs_idx]
        # possible_class = list(set(Y))
        # possible_class.sort(reverse = False)
        # template_sig = [np.zeros((filterbank_num, channel_num, signal_len)) for _ in range(stimulus_num)]
        U_tdca = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        for filterbank_idx in range(filterbank_num):
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
                separated_trainSig_tmp = [separated_trainSig[i] for i in range(start_idx, end_idx)]
                # Calulcate spatial filter
                W = _TDCA_ls_calW([np.swapaxes(trainSig[:,filterbank_idx,:,:], 1,2) for trainSig in separated_trainSig_tmp],
                                self.n_jobs,
                                LSconfig = self.LSconfig)
                U_tdca[filterbank_idx,class_idx-1,:,:] = W[:,:n_component]
        # Calculate template
        separated_trainSig_1 = [separated_trainSig_1[i] for i in return_freqs_idx]
        template_sig = [np.mean(tmp, 0) for tmp in separated_trainSig_1]
        # Generate spatial filters for all stimuli
        # U_tdca = np.repeat(U_tdca[:,:,:,return_freqs_idx], repeats = stimulus_num, axis = 1)
        self.model['U'] = U_tdca[:,return_freqs_idx,:,:]
        self.model['template_sig'] = template_sig
    
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
        n_delay = self.n_delay

        X_delay = _gen_delay_X(X, n_delay)

        template_sig = self.model['template_sig']
        U = self.model['U'] 
        ref_sig_P = self.model['ref_sig_P']

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_tdca_canoncorr_withUV, Y=template_sig, P=ref_sig_P, U=U, V=U))(X=a) for a in X_delay)
        else:
            r = []
            for a in X_delay:
                r.append(
                    _r_tdca_canoncorr_withUV(X=a, Y=template_sig, P=ref_sig_P, U=U, V=U)
                )

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]
        
        return Y_pred, r
    

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

def _r_tdca_canoncorr_withUV(X: ndarray,
                            Y: List[ndarray],
                            P: List[ndarray],
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
    P : List[ndarray]
        List of P
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
    if len(Y[0].shape)==2:
        harmonic_num = Y[0].shape[0]
    elif len(Y[0].shape)==3:
        harmonic_num = Y[0].shape[1]
    else:
        raise ValueError('Unknown data type')
    stimulus_num = len(Y)
    
    R = np.zeros((filterbank_num, stimulus_num))
    
    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            # tmp = np.concatenate([tmp_X, tmp_X @ P[i]], axis=-1)
            if len(Y[i].shape)==2:
                Y_tmp = Y[i]
            elif len(Y[i].shape)==3:
                Y_tmp = Y[i][k,:,:]
            else:
                raise ValueError('Unknown data type')
            
            A_r = U[k,i,:,:]
            B_r = V[k,i,:,:]
            
            a = A_r.T @ tmp
            b = B_r.T @ Y_tmp
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            
            # r2 = stats.pearsonr(a, b)[0]
            # r = stats.pearsonr(a, b)[0]
            r = np.corrcoef(a, b)[0,1]
            R[k,i] = r
    return R


class eTRCAwithR_multi_f(TDCA_ls):
    def __init__(self,
                 n_neighbor: int = 2,
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
        super().__init__(n_component, n_jobs, weights_filterbank, n_delay, LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol)
        self.ID = 'eTRCAwithR_multi_f'
        self.n_neighbor = n_neighbor

    def __copy__(self):
        copy_model = eTRCAwithR_multi_f(n_neighbor = self.n_neighbor,
                                    n_component = self.n_component,
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
            freqs: Optional[List[float]] = None,
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
        if freqs is None:
            raise ValueError('ms-eTRCA requires the list of stimulus frequencies')
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
        separated_trainSig_1 = separate_trainSig(X_P, Y)
        separated_trainSig = [np.concatenate([trainSig_single_freq @ P_single_freq], axis=-1)
                              for trainSig_single_freq, P_single_freq in zip(separated_trainSig_1, ref_sig_P)]

        # template signals and spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X or template_sig: (filterbank_num, channel_num, signal_len)
        filterbank_num, channel_num, signal_len = X_P[0].shape
        stimulus_num = len(ref_sig)
        n_component = self.n_component
        n_neighbor = min(self.n_neighbor,stimulus_num)
        d0 = int(np.floor(n_neighbor/2))
        _, freqs_idx, return_freqs_idx = sort(freqs)
        separated_trainSig = [separated_trainSig[i] for i in freqs_idx]
        separated_trainSig_1 = [separated_trainSig_1[i] for i in freqs_idx]
        # possible_class = list(set(Y))
        # possible_class.sort(reverse = False)
        # template_sig = [np.zeros((filterbank_num, channel_num, signal_len)) for _ in range(stimulus_num)]
        U_tdca = np.zeros((filterbank_num, 1, channel_num, stimulus_num))
        for filterbank_idx in range(filterbank_num):
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
                separated_trainSig_tmp = [separated_trainSig[i] for i in range(start_idx, end_idx)]
                # Calulcate spatial filter
                W = _TDCA_ls_calW([np.swapaxes(trainSig[:,filterbank_idx,:,:], 1,2) for trainSig in separated_trainSig_tmp],
                                self.n_jobs,
                                LSconfig = self.LSconfig)
                U_tdca[filterbank_idx,0,:,(class_idx-1):class_idx] = W[:,:n_component]
        # Calculate template
        separated_trainSig_1 = [separated_trainSig_1[i] for i in return_freqs_idx]
        template_sig = [np.mean(tmp, 0) for tmp in separated_trainSig_1]
        # Generate spatial filters for all stimuli
        # U_tdca = np.repeat(U_tdca[:,:,:,return_freqs_idx], repeats = stimulus_num, axis = 1)
        self.model['U'] = np.repeat(U_tdca, repeats = stimulus_num, axis = 1)
        self.model['template_sig'] = template_sig
    
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
        n_delay = self.n_delay

        X_delay = _gen_delay_X(X, n_delay)

        template_sig = self.model['template_sig']
        U = self.model['U'] 
        ref_sig_P = self.model['ref_sig_P']

        if self.n_jobs is not None:
            r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_tdca_canoncorr_withUV, Y=template_sig, P=ref_sig_P, U=U, V=U))(X=a) for a in X_delay)
        else:
            r = []
            for a in X_delay:
                r.append(
                    _r_tdca_canoncorr_withUV(X=a, Y=template_sig, P=ref_sig_P, U=U, V=U)
                )

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]
        
        return Y_pred, r