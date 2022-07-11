"""
TRCA based recognition methods
"""

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray
from joblib import Parallel, delayed
from functools import partial
from copy import deepcopy

import numpy as np
import scipy.linalg as slin
import scipy.stats as stats

from .basemodel import BaseModel
from .utils import gen_template


def _trca_U(X: list,
            n_component: int) -> ndarray:
    """
    Calculate spatial filters of trca

    Parameters
    ------------
    X : list
        List of EEG data
        Length: (trial_num,)
        Shape of EEG: (channel_num, signal_len)

    Returns
    -----------
    U : ndarray
        Spatial filter
        shape: (channel_num * n_component)
    """
    trca_X1 = np.zeros(X[0].shape)
    trca_X2 = []
    for X0 in X:
        trca_X1 = trca_X1 + X0
        trca_X2.append(X0.T)
    trca_X2 = np.concatenate(trca_X2, axis = 0)
    S=trca_X1 @ trca_X1.T - trca_X2.T @ trca_X2
    Q=trca_X2.T @ trca_X2
    eig_d1, eig_v1 = slin.eig(S, Q)
    sort_idx = np.argsort(eig_d1)[::-1]
    eig_vec=eig_v1[:,sort_idx]

    return eig_vec

def _r_corr(X: ndarray,
            Y: List[ndarray],
            U: ndarray) -> ndarray:
    """
    Calculate correlation

    Parameters
    ------------
    X : ndarray
        Single trial EEG data
        EEG shape: (filterbank_num, channel_num, signal_len)
    Y : List[ndarray]
        List of template signals
    U : ndarray
        Shape: (filterbank_num * stimulus_num * channel_num * n_component)
    """
    filterbank_num, channel_num, signal_len = X.shape
    stimulus_num = len(Y)
    R = np.zeros((filterbank_num, stimulus_num))

    for k in range(filterbank_num):
        tmp = X[k,:,:]
        for i in range(stimulus_num):
            Y_tmp = Y[i][k,:,:]
            a = U[k,i,:,:].T @ tmp
            b = U[k,i,:,:].T @ Y_tmp
            a = np.reshape(a, (-1))
            b = np.reshape(b, (-1))
            r = stats.pearsonr(a, b)[0]
            R[k,i] = r
    
    return R


class TRCA(BaseModel):
    """
    TRCA method
    """
    def __init__(self,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        super().__init__(ID = 'TRCA',
                         n_component = n_component,
                         n_jobs = n_jobs,
                         weights_filterbank = weights_filterbank)
        self.model['U'] = None # Spatial filter of EEG

    def __copy__(self):
        copy_model = TRCA(n_component = self.n_component,
                          n_jobs = self.n_jobs,
                          weights_filterbank = self.model['weights_filterbank'])
        copy_model.model = deepcopy(self.model)
        return copy_model

    def fit(self,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None):
        if Y is None:
            raise ValueError('TRCA requires training label')
        if X is None:
            raise ValueError('TRCA requires training data')
           
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
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        for filterbank_idx in range(filterbank_num):
            X_train = [[X[i][filterbank_idx,:,:] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
            U = Parallel(n_jobs = self.n_jobs)(delayed(partial(_trca_U, n_component = n_component))(X = X_single_class) for X_single_class in X_train)
            for stim_idx, u in enumerate(U):
                U_trca[filterbank_idx, stim_idx, :, :] = u[:channel_num,:n_component]
        self.model['U'] = U_trca

    def predict(self,
            X: List[ndarray]) -> List[int]:
        weights_filterbank = self.model['weights_filterbank']
        if weights_filterbank is None:
            weights_filterbank = [1 for _ in range(X[0].shape[0])]
        weights_filterbank = np.expand_dims(np.array(weights_filterbank),1).T

        template_sig = self.model['template_sig']
        U = self.model['U'] 

        r = Parallel(n_jobs=self.n_jobs)(delayed(partial(_r_corr, Y=template_sig, U=U))(X=a) for a in X)

        Y_pred = [int( np.argmax( weights_filterbank @ r_tmp)) for r_tmp in r]
        
        return Y_pred 
