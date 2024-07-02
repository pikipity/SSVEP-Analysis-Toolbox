# -*- coding: utf-8 -*-
"""
MS-MsetCCA-R-3
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
    qr_list, sort, separate_trainSig, 
    blkmat
)
from .tdca import (
    _gen_delay_X, _r_tdca_canoncorr_withUV
)

def _TDCA_ls_calW(trainSig, n_jobs = None, LSconfig = None):
    def X_single_freq(trainSig_single_freq, All_train_sum):
        Nt, Np, Nc = trainSig_single_freq.shape
        X = []
        for n in range(Nt):
            X_tmp = trainSig_single_freq[n,:,:]
            X.append(X_tmp)
        X = np.concatenate(X, axis = 1)
        X = np.tile(X, (Nt,1))
        return X - 1/Nf * All_train_sum
    def Y_single_freq(trainSig_single_freq):
        Nt, Np, Nc = trainSig_single_freq.shape
        Y = []
        for n in range(Nt):
            Y_tmp = trainSig_single_freq[n,:,:] - 1/Nt * np.sum(trainSig_single_freq, axis = 0)
            Y.append(Y_tmp)
        Y = blkmat(Y)
        return Y
    #
    LSMethod, displayConvergWarn, max_n, M_error_threshold, W_error_threshold, l1_alpha, l2_alpha, alpha, reg_iter, reg_tol, check_full_rank, kernel_fun = get_lsconfig(LSconfig)
    # 
    Nf = len(trainSig)
    All_train_sum = None
    for trainSig_single_freq in trainSig:
        if All_train_sum is None:
            All_train_sum = X_single_freq(trainSig_single_freq, 0)
        else:
            All_train_sum = All_train_sum + X_single_freq(trainSig_single_freq, 0)
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
        X = Parallel(n_jobs=n_jobs)(delayed(X_single_freq)
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
    W, _, _, _, _, _ = ssvep_lsframe(X, Y,
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



class MsetCCA_multi_f(TDCA_ls):
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
        self.ID = 'MsetCCA_multi_f'
        self.n_neighbor = n_neighbor

    def __copy__(self):
        copy_model = MsetCCA_multi_f(n_neighbor = self.n_neighbor,
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
        if freqs is None:
            raise ValueError('ms-eTRCA requires the list of stimulus frequencies')
        if Y is None:
            raise ValueError('MsetCCA_multi_f requires training label')
        if X is None:
            raise ValueError('MsetCCA_multi_f requires training data')
        if ref_sig is None:
            raise ValueError("MsetCCA_multi_f requires reference signals")
        
        ref_sig_Q, _, _ = qr_list(ref_sig)
        ref_sig_P = [Q @ Q.T for Q in ref_sig_Q]
        self.model['ref_sig_P'] = ref_sig_P

        n_delay = self.n_delay
        X_P = _gen_delay_X(X, n_delay)
        separated_trainSig = separate_trainSig(X_P, Y)
        separated_trainSig = [np.concatenate([trainSig_single_freq, trainSig_single_freq @ P_single_freq], axis=-1)
                              for trainSig_single_freq, P_single_freq in zip(separated_trainSig, ref_sig_P)]
        
        filterbank_num, channel_num, signal_len = X_P[0].shape
        stimulus_num = len(ref_sig)
        n_component = self.n_component
        n_neighbor = self.n_neighbor
        d0 = int(np.floor(n_neighbor/2))
        _, freqs_idx, return_freqs_idx = sort(freqs)
        separated_trainSig = [separated_trainSig[i] for i in freqs_idx]

        # template_all_band = []
        trial_num = separated_trainSig[0].shape[0]
        U_tdca = np.zeros((filterbank_num, stimulus_num, channel_num, trial_num))

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
                W = _TDCA_ls_calW([np.swapaxes(trainSig[:,filterbank_idx,:,:], 1,2) for trainSig in separated_trainSig_tmp],
                                self.n_jobs,
                                LSconfig = self.LSconfig)
                W_tmp = []
                for trial_idx in range(trial_num):
                    W_tmp.append(W[(trial_idx*channel_num):((trial_idx+1)*channel_num),:n_component])
                U_tdca[filterbank_idx,class_idx-1,:,:] = np.concatenate(W_tmp, axis = 1)
        
        # calculate template
        separated_trainSig = [separated_trainSig[i] for i in return_freqs_idx]
        template_sig = [np.mean(tmp, 0) for tmp in separated_trainSig]
        #     template = []
        #     for trainSig in separated_trainSig:
        #         template_tmp = []
        #         trial_num = trainSig.shape[0]
        #         for trial_idx in range(trial_num):
        #             template_single_trial = eig_vec[(trial_idx*channel_num):((trial_idx+1)*channel_num),:n_component].T @ trainSig[trial_idx,filterbank_idx,:,:]
        #             template_tmp.append(template_single_trial)
        #         template_tmp = np.concatenate(template_tmp , axis=0)
        #         template.append(template_tmp)
        #     template_all_band.append(template)

        # template_all_stimuli = []
        # for freq_i in range(stimulus_num):
        #     template_single_freq = []
        #     for filterbank_idx in range(filterbank_num):
        #         template_single_freq.append(np.expand_dims(template_all_band[filterbank_idx][freq_i], axis = 0))
        #     template_single_freq = np.concatenate(template_single_freq, axis = 0)
        #     template_all_stimuli.append(template_single_freq)
        
        # self.model['template'] = template_all_stimuli
        # template_sig_Q, template_sig_R, template_sig_P = qr_list(template_all_stimuli)
        # self.model['template_sig_Q'] = template_sig_Q # List of shape: (stimulus_num,);
        # self.model['template_sig_R'] = template_sig_R
        # self.model['template_sig_P'] = template_sig_P

        # U_tdca = np.repeat(U_tdca, repeats = stimulus_num, axis = 1)
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
    
# def _r_cca_qr_withP(X: ndarray,
#            Y_Q: List[ndarray],
#            Y_R: List[ndarray],
#            Y_P: List[ndarray],
#            P: List[ndarray],
#            n_component: int,
#            force_output_UV: Optional[bool] = False) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
#     """
#     Calculate correlation of CCA based on QR decomposition for single trial data 

#     Parameters
#     ----------
#     X : ndarray
#         Single trial EEG data
#         EEG shape: (filterbank_num, channel_num, signal_len)
#     Y_Q : List[ndarray]
#         Q of reference signals
#     Y_R: List[ndarray]
#         R of reference signals
#     Y_P: List[ndarray]
#         P of reference signals
#     n_component : int
#         Number of eigvectors for spatial filters.
#     force_output_UV : Optional[bool]
#         Whether return spatial filter 'U' and weights of harmonics 'V'

#     Returns
#     -------
#     R : ndarray
#         Correlation
#         shape: (filterbank_num * stimulus_num)
#     U : ndarray
#         Spatial filter
#         shape: (filterbank_num * stimulus_num * channel_num * n_component)
#     V : ndarray
#         Weights of harmonics
#         shape: (filterbank_num * stimulus_num * harmonic_num * n_component)
#     """
#     filterbank_num, channel_num, signal_len = X.shape
#     harmonic_num = Y_R[0].shape[-1]
#     stimulus_num = len(Y_Q)
    
#     Y = [qr_inverse(Y_Q[i],Y_R[i],Y_P[i]) for i in range(len(Y_Q))]
#     if len(Y[0].shape)==2: # reference
#         Y = [Y_tmp.T for Y_tmp in Y]
#     elif len(Y[0].shape)==3: # template
#         Y = [np.transpose(Y_tmp, (0,2,1)) for Y_tmp in Y]
#     else:
#         raise ValueError('Unknown data type')
    
#     # R1 = np.zeros((filterbank_num,stimulus_num))
#     # R2 = np.zeros((filterbank_num,stimulus_num))
#     R = np.zeros((filterbank_num, stimulus_num))
#     U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
#     V = np.zeros((filterbank_num, stimulus_num, harmonic_num, n_component))
    
#     for k in range(filterbank_num):
#         tmp_X = X[k,:,:]
#         for i in range(stimulus_num):
#             tmp = np.concatenate([tmp_X, tmp_X @ P[i]], axis = -1)
#             X_Q, X_R, X_P = qr_remove_mean(tmp.T)
#             if len(Y_Q[i].shape)==2: # reference
#                 Y_Q_tmp = Y_Q[i]
#                 Y_R_tmp = Y_R[i]
#                 Y_P_tmp = Y_P[i]
#                 Y_tmp = Y[i]
#             elif len(Y_Q[i].shape)==3: # template
#                 Y_Q_tmp = Y_Q[i][k,:,:]
#                 Y_R_tmp = Y_R[i][k,:,:]
#                 Y_P_tmp = Y_P[i][k,:]
#                 Y_tmp = Y[i][k,:,:]
#             else:
#                 raise ValueError('Unknown data type')
#             svd_X = X_Q.T @ Y_Q_tmp
#             if svd_X.shape[0]>svd_X.shape[1]:
#                 full_matrices=False
#             else:
#                 full_matrices=True
            
#             if n_component == 0 and force_output_UV is False:
#                 D = svd(svd_X, full_matrices, False)
#                 r = D[0]
#             else:
#                 L, D, M = svd(svd_X, full_matrices, True)
#                 M = M.T
#                 try:
#                     A = mldivide(X_R, L) * np.sqrt(signal_len - 1)
#                     B = mldivide(Y_R_tmp, M) * np.sqrt(signal_len - 1)
#                 except:
#                     print('Error')
#                 A_r = np.zeros(A.shape)
#                 for n in range(A.shape[0]):
#                     A_r[X_P[n],:] = A[n,:]
#                 B_r = np.zeros(B.shape)
#                 for n in range(B.shape[0]):
#                     B_r[Y_P_tmp[n],:] = B[n,:]
                
#                 a = A_r[:channel_num, :n_component].T @ tmp
#                 b = B_r[:harmonic_num, :n_component].T @ Y_tmp
#                 a = np.reshape(a, (-1))
#                 b = np.reshape(b, (-1))
                
#                 # r2 = stats.pearsonr(a, b)[0]
#                 # r = stats.pearsonr(a, b)[0]
#                 r = np.corrcoef(a, b)[0,1]
#                 U[k,i,:,:] = A_r[:channel_num, :n_component]
#                 V[k,i,:,:] = B_r[:harmonic_num, :n_component]
                
#             # R1[k,i] = r1
#             # R2[k,i] = r2
#             R[k,i] = r
#     if force_output_UV:
#         return R, U, V
#     else:
#         return R

        



class eMsetCCA_multi_f(TDCA_ls):
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
        self.ID = 'eMsetCCA_multi_f'
        self.n_neighbor = n_neighbor

    def __copy__(self):
        copy_model = eMsetCCA_multi_f(n_neighbor = self.n_neighbor,
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
        if freqs is None:
            raise ValueError('ms-eTRCA requires the list of stimulus frequencies')
        if Y is None:
            raise ValueError('MsetCCA_multi_f requires training label')
        if X is None:
            raise ValueError('MsetCCA_multi_f requires training data')
        if ref_sig is None:
            raise ValueError("MsetCCA_multi_f requires reference signals")
        
        ref_sig_Q, _, _ = qr_list(ref_sig)
        ref_sig_P = [Q @ Q.T for Q in ref_sig_Q]
        self.model['ref_sig_P'] = ref_sig_P

        n_delay = self.n_delay
        X_P = _gen_delay_X(X, n_delay)
        separated_trainSig = separate_trainSig(X_P, Y)
        separated_trainSig = [np.concatenate([trainSig_single_freq, trainSig_single_freq @ P_single_freq], axis=-1)
                              for trainSig_single_freq, P_single_freq in zip(separated_trainSig, ref_sig_P)]
        
        filterbank_num, channel_num, signal_len = X_P[0].shape
        stimulus_num = len(ref_sig)
        n_component = self.n_component
        n_neighbor = min(self.n_neighbor,stimulus_num)
        d0 = int(np.floor(n_neighbor/2))
        _, freqs_idx, return_freqs_idx = sort(freqs)
        separated_trainSig = [separated_trainSig[i] for i in freqs_idx]

        # template_all_band = []
        trial_num = separated_trainSig[0].shape[0]
        U_tdca = np.zeros((filterbank_num, 1, channel_num, trial_num*stimulus_num))

        for filterbank_idx in range(filterbank_num):
            W_tmp = []
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
                W = _TDCA_ls_calW([np.swapaxes(trainSig[:,filterbank_idx,:,:], 1,2) for trainSig in separated_trainSig_tmp],
                                self.n_jobs,
                                LSconfig = self.LSconfig)
                for trial_idx in range(trial_num):
                    W_tmp.append(W[(trial_idx*channel_num):((trial_idx+1)*channel_num),:n_component])
            U_tdca[filterbank_idx,0,:,:] = np.concatenate(W_tmp, axis = 1)
        
        # calculate template
        separated_trainSig = [separated_trainSig[i] for i in return_freqs_idx]
        template_sig = [np.mean(tmp, 0) for tmp in separated_trainSig]
        #     template = []
        #     for trainSig in separated_trainSig:
        #         template_tmp = []
        #         trial_num = trainSig.shape[0]
        #         for trial_idx in range(trial_num):
        #             template_single_trial = eig_vec[(trial_idx*channel_num):((trial_idx+1)*channel_num),:n_component].T @ trainSig[trial_idx,filterbank_idx,:,:]
        #             template_tmp.append(template_single_trial)
        #         template_tmp = np.concatenate(template_tmp , axis=0)
        #         template.append(template_tmp)
        #     template_all_band.append(template)

        # template_all_stimuli = []
        # for freq_i in range(stimulus_num):
        #     template_single_freq = []
        #     for filterbank_idx in range(filterbank_num):
        #         template_single_freq.append(np.expand_dims(template_all_band[filterbank_idx][freq_i], axis = 0))
        #     template_single_freq = np.concatenate(template_single_freq, axis = 0)
        #     template_all_stimuli.append(template_single_freq)
        
        # self.model['template'] = template_all_stimuli
        # template_sig_Q, template_sig_R, template_sig_P = qr_list(template_all_stimuli)
        # self.model['template_sig_Q'] = template_sig_Q # List of shape: (stimulus_num,);
        # self.model['template_sig_R'] = template_sig_R
        # self.model['template_sig_P'] = template_sig_P

        U_tdca = np.repeat(U_tdca, repeats = stimulus_num, axis = 1)
        self.model['U'] = U_tdca
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
