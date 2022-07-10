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

        filterbank_num = template_sig[0].shape[0]
        stimulus_num = len(template_sig)
        channel_num = template_sig[0].shape[1]
        n_component = self.n_component
        U = np.zeros((filterbank_num, stimulus_num, channel_num, n_component))
        # X shape: (filterbank_num, channel_num, signal_len)
        possible_class = list(set(Y))
        possible_class.sort(reverse = False)
        X_train = [[X[i] for i in np.where(np.array(Y) == class_val)[0]] for class_val in possible_class]
