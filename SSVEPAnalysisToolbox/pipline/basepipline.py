# -*- coding: utf-8 -*-
"""
Base class of pipline
"""

import abc
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

class PerformanceContainer():
    """
    Container of performance, containing:
        - True labels
        - Predicted labels
        - Training times
        - Testing times
    """
    def __init__(self, 
                 method_ID: str):
        self.method_ID = method_ID
        self.clear()
        
    def clear(self):
        self.container = {}
        self.container['true-labels']=[]
        self.container['predict-labels']=[]
        self.container['train-times']=[]
        self.container['test-times']=[]
        
    def add(self, key, value):
        self.container[key].append(value)

class BasePipline(metaclass=abc.ABCMeta):
    """
    BasePipline
    """
    def __init__(self, 
                 ch_used: list,
                 harmonic_num: list,
                 model_container: list,
                 dataset_container: list,
                 save_model: Optional[bool] = False,
                 disp_processbar: Optional[bool] = True,
                 shuffle_trials: Optional[bool] = False):
        """
        Parameters for all piplines

        Parameters
        ----------
        ch_used : list
            List of channels
            Shape: (dataset_num,)
            Shape of element: (ch_num,)
        harmonic_num : list
            List of harmonic numbers
            Shape: (dataset_num,)
        model_container : list
            List of all models for evaluation
        dataset_container : list
            List of all datasets
        save_model : Optional[bool]
            Whether save models
        disp_processbar : Optional[bool]
            Whether display process bar
        shuffle_trials: Optional[bool]
            Whether shuffle trials
        """
        if type(model_container) is not list:
            model_container = [model_container]
        if type(dataset_container) is not list:
            dataset_container = [dataset_container]
        if len(ch_used) != len(dataset_container):
            raise ValueError("Channels of all datasets should be provided")
        if len(harmonic_num) != len(dataset_container):
            raise ValueError("Harmonic numbers of all datasets should be provided")
        
        self.disp_processbar = disp_processbar
        self.shuffle_trials = shuffle_trials
        self.harmonic_num = harmonic_num
        self.ch_used = ch_used
        self.model_container = model_container
        self.dataset_container = dataset_container
        self.save_model = save_model
        
        self.trained_model_container = []
        self.performance_container = []
        
    @abc.abstractclassmethod
    def run(self, 
            n_jobs: Optional[int] = None):
        """
        Run pipline
        
        Parameters
        ----------
        n_jobs : Optional[int]
            Number of CPUs
        """
        pass
        

