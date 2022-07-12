# -*- coding: utf-8 -*-
"""
Base class of SSVEP recognition method
"""

import abc
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

class BaseModel(metaclass=abc.ABCMeta):
    """
    BaseModel
    """
    def __init__(self,
                 ID: str,
                 n_component: int = 1,
                 n_jobs: Optional[int] = None,
                 weights_filterbank: Optional[List[float]] = None):
        """
        Parameters for all SSVEP recognition methods

        Parameters
        ----------
        ID : str
            Unique identifier for method
        n_component : Optional[int], optional
            Number of eigvectors for spatial filters. The default is 1.
        n_jobs : Optional[int], optional
            Number of CPU for computing different trials. The default is None.
        weights_filterbank : Optional[List[float]], optional
            Weights of spatial filters. 
            If None, all weights are 1.
            The default is None.
        """
        if n_component < 0:
            raise ValueError('n_component must be larger than 0')
        
        self.ID = ID
        self.n_component = n_component
        self.n_jobs = n_jobs
        
        self.model = {}
        self.model['weights_filterbank'] = weights_filterbank
        
    @abc.abstractclassmethod
    def fit(self,
            freqs: Optional[List[float]] = None,
            X: Optional[List[ndarray]] = None,
            Y: Optional[List[int]] = None,
            ref_sig: Optional[List[ndarray]] = None):
        """
        Training function

        Parameters
        ----------
        freqs : Optional[List[float]], optional
            List of stimulus frequencies. The default is None.
            List shape: (trial_num,)
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

        Spatial filters and other parameters will be stored in self.model
        """
        pass
    
    @abc.abstractclassmethod
    def predict(self,
                X: List[ndarray]) -> List[int]:
        """
        Testing function

        Parameters
        ----------
        X : List[ndarray]
            List of testing EEG data
            List shape: (trial_num,)
            EEG shape: (filterbank_num, channel_num, signal_len)

        Returns
        -------
        Y : List[int]
            List of recognition labels (stimulus indices)
            List shape: (trial_num,)
        """
        pass
    
    @abc.abstractclassmethod
    def __copy__(self):
        """
        copy itself
        """
        pass