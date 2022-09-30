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
    def fit(self, *argv, **kwargs):
        """
        Training function

        Different methods may require different parameters.

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