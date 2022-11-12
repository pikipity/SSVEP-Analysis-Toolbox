"""
TDCA based recognition methods
"""

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial
from copy import deepcopy
import warnings

import numpy as np

from .basemodel import BaseModel
from .utils import qr_list, mean_list, sum_list, eigvec

def _covariance_tdca(X: ndarray, 
                     X_mean: ndarray, 
                     num: int, 
                     division_num: int) -> ndarray:
    if num == 1:
        X_tmp = X
    else:
        X_tmp = X - X_mean
    return X_tmp @ X_tmp.T / division_num

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
        tmp_X = X[k,:,:]
        for i in range(stimulus_num):
            tmp = np.concatenate([tmp_X, tmp_X @ P[i]], axis=-1)
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

def _gen_delay_X(X: List[ndarray],
                 n_delay: int) -> List[ndarray]:
    """
    Generate delayed signal

    Parameters
    -----------
    X: List[ndarray]
        Original EEG signals
    n_delay: int
        Number of delayed signals
        0 means no delay

    Returns
    -------------
    X_delay: List[ndarray]
        Combine original signals and delayed signals along channel axis
    """
    X_delay = []
    for X_single_trial in X:
        if len(X_single_trial.shape) == 2:
            ch_num, sig_len = X_single_trial.shape
            X_delay_single_trial = [np.concatenate([X_single_trial[:,dn:sig_len],np.zeros((ch_num,dn))],axis=-1)
                                    for dn in range(n_delay)]
            X_delay.append(np.concatenate(X_delay_single_trial,axis=0))
        elif len(X_single_trial.shape) == 3:
            filterbank_num, ch_num, sig_len = X_single_trial.shape
            X_delay_single_trial = []
            for filterbank_idx in range(filterbank_num):
                tmp = [np.concatenate([X_single_trial[filterbank_idx,:,dn:sig_len],np.zeros((ch_num,dn))],axis=-1)
                       for dn in range(n_delay)]
                tmp = np.concatenate(tmp,axis=0)
                X_delay_single_trial.append(np.expand_dims(tmp, axis=0))
            X_delay_single_trial = np.concatenate(X_delay_single_trial, axis=0)
            X_delay.append(X_delay_single_trial)
        else:
            raise ValueError("Shapes of X have error")
    return X_delay

def _gen_P_combine_X(X: List[ndarray],
                     P: ndarray) -> List[ndarray]:
    """
    Combine signal and signal * P

    Parameters
    --------------
    X: List[ndarray]
        Original signal
    P: ndarray
        P = Q @ Q.T

    Returns
    -------------
    P_combine_X: List[ndarray]
        Combine X and X @ P along time axis
    """
    return [np.concatenate([X_single_trial, X_single_trial @ P], axis=-1) for X_single_trial in X]

class TDCA(BaseModel):
    """
    TDCA method
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None,
                 n_delay: int = 0):
        """
        Special parameter
        -----------------
        n_delay: int
            Number of delayed signals
            Default is 0 (no delay)
        """
        super().__init__(ID = 'TDCA',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.n_delay = n_delay
        self.model['U'] = None # Spatial filter of EEG

    def __copy__(self):
        copy_model = TDCA(n_component = self.n_component,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'],
                          n_delay = self.n_delay)
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

        # template signals and spatial filters
        #   U: (filterbank_num * stimulus_num * channel_num * n_component)
        #   X or template_sig: (filterbank_num, channel_num, signal_len)
        filterbank_num, channel_num, signal_len = X[0].shape
        stimulus_num = len(ref_sig)
        n_component = self.n_component
        n_delay = self.n_delay
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        template_sig = [np.zeros((filterbank_num, channel_num * n_delay, signal_len + ref_sig_Q[0].shape[0])) for _ in range(stimulus_num)]
        U_tdca = np.zeros((filterbank_num, 1, channel_num * n_delay, n_component))
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            trial_num = len(X_train[0])

            if self.n_jobs is not None:
                X_train_delay = Parallel(n_jobs = self.n_jobs)(delayed(partial(_gen_delay_X, n_delay = n_delay))(X = X_single_class) for X_single_class in X_train)
                P_combine_X_train = Parallel(n_jobs = self.n_jobs)(delayed(_gen_P_combine_X)(X = X_single_class, P = P_single_class) for X_single_class, P_single_class in zip(X_train_delay, ref_sig_P))
            else:
                X_train_delay = []
                for X_single_class in X_train:
                    X_train_delay.append(
                        _gen_delay_X(X = X_single_class, n_delay = n_delay)
                    )
                P_combine_X_train = []
                for X_single_class, P_single_class in zip(X_train_delay, ref_sig_P):
                    P_combine_X_train.append(
                        _gen_P_combine_X(X = X_single_class, P = P_single_class)
                    )
            # Calculate template
            if self.n_jobs is not None:
                P_combine_X_train_mean = Parallel(n_jobs=self.n_jobs)(delayed(mean_list)(X = P_combine_X_train_single_class) for P_combine_X_train_single_class in P_combine_X_train)
            else:
                P_combine_X_train_mean = []
                for P_combine_X_train_single_class in P_combine_X_train:
                    P_combine_X_train_mean.append(
                        mean_list(X = P_combine_X_train_single_class)
                    )
            for stim_idx, P_combine_X_train_mean_single_class in enumerate(P_combine_X_train_mean):
                template_sig[stim_idx][filterbank_idx,:,:] = P_combine_X_train_mean_single_class
            # Calulcate spatial filter
            P_combine_X_train_all_mean = mean_list(P_combine_X_train_mean)
            X_tmp = []
            X_mean = []
            for P_combine_X_train_single_class, P_combine_X_train_mean_single_class in zip(P_combine_X_train, P_combine_X_train_mean):
                for X_tmp_tmp in P_combine_X_train_single_class:
                    X_tmp.append(X_tmp_tmp)
                    X_mean.append(P_combine_X_train_mean_single_class)

            if self.n_jobs is not None:
                Sw_list = Parallel(n_jobs=self.n_jobs)(delayed(partial(_covariance_tdca, num = trial_num,
                                                                                        division_num = trial_num))(X = X_tmp_tmp, X_mean = X_mean_tmp)
                                                                                        for X_tmp_tmp, X_mean_tmp in zip(X_tmp, X_mean))
                Sb_list = Parallel(n_jobs=self.n_jobs)(delayed(partial(_covariance_tdca, X_mean = P_combine_X_train_all_mean,
                                                                                        num = stimulus_num,
                                                                                        division_num = stimulus_num))(X = P_combine_X_train_mean_single_class)
                                                                                        for P_combine_X_train_mean_single_class in P_combine_X_train_mean)
            else:
                Sw_list = []
                for X_tmp_tmp, X_mean_tmp in zip(X_tmp, X_mean):
                    Sw_list.append(
                        _covariance_tdca(X = X_tmp_tmp, X_mean = X_mean_tmp, num = trial_num, division_num = trial_num)
                    )
                Sb_list = []
                for P_combine_X_train_mean_single_class in P_combine_X_train_mean:
                    Sb_list.append(
                        _covariance_tdca(X = P_combine_X_train_mean_single_class, X_mean = P_combine_X_train_all_mean, num = stimulus_num, division_num = stimulus_num)
                    )

            Sw = sum_list(Sw_list)
            Sb = sum_list(Sb_list)
            eig_vec = eigvec(Sb, Sw)
            U_tdca[filterbank_idx,0,:,:] = eig_vec[:,:n_component]
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