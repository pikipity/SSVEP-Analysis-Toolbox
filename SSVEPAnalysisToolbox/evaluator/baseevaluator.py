# -*- coding: utf-8 -*-
"""
Base class of evaluator
"""

import abc
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

from tqdm import tqdm
import time

def create_pbar(loop_list_num: List[int],
                desc: str = ''):
    """
    Create process bar

    Parameters
    ----------
    loop_list_num : List[int]
        List of following loop numbers
    desc : str
        Prefix for the progressbar

    Returns
    -------
    pbar : tqdm
        processbar
    """
    total_num = 1
    for loop_num in loop_list_num:
        total_num = total_num * loop_num
    pbar = tqdm(total = total_num,
                desc = desc,
                bar_format = '{desc}{percentage:3.3f}%|{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}<{remaining}]')
    return pbar

class PerformanceContainer:
    """
    Container of performance, containing:
        - true_label_train: True labels for train
        - pred_label_train: Predicted labels for train
        - true_label_test: True labels for test
        - pred_label_test: Preducted labels for test
        - train_time: Training times
        - test_time_train: Testing times for train
        - test_time_test: Testing times for test
    """
    def __init__(self, 
                 method_ID: str):
        self.method_ID = method_ID
        self.clear()
        
    def add_true_label_train(self,true_label:list):
        self.true_label_train.extend(true_label)
    def add_pred_label_train(self,pred_label:list):
        self.pred_label_train.extend(pred_label)
    def add_true_label_test(self,true_label:list):
        self.true_label_test.extend(true_label)
    def add_pred_label_test(self,pred_label:list):
        self.pred_label_test.extend(pred_label)
    def add_train_time(self,train_time:list):
        self.train_time.append(train_time)
    def add_test_time_train(self,test_time:list):
        self.test_time_train.append(test_time)
    def add_test_time_test(self,test_time:list):
        self.test_time_test.append(test_time)
    def clear(self):
        self.true_label_train = []
        self.pred_label_train = []
        self.true_label_test = []
        self.pred_label_test = []
        self.train_time = []
        self.test_time_train = []
        self.test_time_test = []

class TrialInfo:
    """
    TrialInfo
    
    Store training and testing trial information
    
    If there are more than two datasets, sine-cone-based reference signals will 
    be generated based on the first dataset information
    """
    def __init__(self):
        """
        Parameters
        ----------
        dataset_idx : list
            List of dataset index
        sub_idx : list
            List of subjects
            Length is equal to dataset_idx
            Each element is a list of subjext index for one dataset
        block_idx : list
            List of blocks
            Length is equal to dataset_idx
            Each element is a list of block index for one dataset
        trial_idx : list
            List of trials
            Length is equal to dataset_idx
            Each element is a list of trial index for one dataset
        ch_idx : list
            List of channels
            Length is equal to dataset_idx
            Each element is a list of channel index for one dataset
        t_latency ; list
            List of latency time
            Length  is equal to dataset_idx
            Each element is a list of latency time for one dataset
        harmonic_num : int
            Number of harmonics for sine-cosine-based references
        tw : float
            Signal window length
        shuffle : List[bool]
            Whether shuffle trials
            Length is equal to dataset_idx
            Each element is the number of shuffle flag for one dataset
        """
        
        self.dataset_idx = []
        self.sub_idx = []
        self.block_idx = []
        self.trial_idx = []
        self.ch_idx = []
        self.harmonic_num = []
        self.tw = []
        self.t_latency = []
        self.shuffle = []
                
    def add_dataset(self,
                    dataset_idx : int,
                    sub_idx : List[int],
                    block_idx : List[int],
                    trial_idx : List[int],
                    ch_idx : List[int],
                    harmonic_num : List[int],
                    tw : float,
                    t_latency : Optional[float] = None,
                    shuffle : bool = False):
        self.dataset_idx.append(dataset_idx)
        self.sub_idx.append(sub_idx)
        self.block_idx.append(block_idx)
        self.trial_idx.append(trial_idx)
        self.ch_idx.append(ch_idx)
        self.t_latency.append(t_latency)
        self.harmonic_num = harmonic_num
        self.tw = tw
        self.shuffle.append(shuffle)
        return self
    
    def get_data(self,
                 dataset_container: list) -> Tuple[list, list, list]:
        """
        Get data based trian information

        Parameters
        ----------
        dataset_container : list
            List of datasets

        Returns
        -------
        X: list
            Data
        Y: list
            Labels
        ref_sig: list
            Reference signal

        """
        dataset = dataset_container[self.dataset_idx[0]]
        ref_sig = dataset.get_ref_sig(self.tw,self.harmonic_num)
        X = []
        Y = []
        for (dataset_idx,
            sub_idx,
            block_idx,
            trial_idx,
            ch_idx,
            t_latency,
            shuffle) in zip(self.dataset_idx, 
                           self.sub_idx,
                           self.block_idx,
                           self.trial_idx,
                           self.ch_idx,
                           self.t_latency,
                           self.shuffle):
            dataset = dataset_container[dataset_idx]
            X_tmp, Y_tmp = dataset.get_data(sub_idx = sub_idx,
                                            blocks = block_idx,
                                            trials = trial_idx,
                                            channels = ch_idx,
                                            sig_len = self.tw,
                                            t_latency = t_latency,
                                            shuffle = shuffle)
        X.extend(X_tmp)
        Y.extend(Y_tmp)
        return X, Y, ref_sig
        

class BaseEvaluator:
    """
    BaseEvaluator
    """
    def __init__(self,
                 dataset_container : list,
                 model_container : list,
                 trial_container : list,
                 save_model: bool = False,
                 disp_processbar: bool = True):
        """
        Parameters for all evaluators

        Parameters
        ----------
        dataset_container : list
            List of datasets
        model_container : list
            List of models
        trial_container : list
            List of trians
            Each element is a list of two elements that are training and testing trials
        save_model : Optional[bool], optional
            Whether save models
        disp_processbar : Optional[bool], optional
            Whether display process bar
        """
        self.dataset_container = dataset_container
        self.model_container = model_container
        self.trial_container = trial_container
        self.save_model = save_model
        self.disp_processbar = disp_processbar
        
        self.performance_container = []
        self.trained_model_container = []
        
    def run(self,
            n_jobs : Optional[int] = None,
            eval_train : bool = False):
        """
        Run evaluator

        Parameters
        ----------
        n_jobs : Optional[int], optional
            Number of CPUs. The default is None.
        eval_train : bool
            Whether evaluate train performance
        """
        if n_jobs is not None:
            for i in range(len(self.model_container)):
                self.model_container[i].n_jobs = n_jobs
                
        if self.disp_processbar:
            print('========================\n   Start\n========================\n')
            pbar = create_pbar([len(self.trial_container)])
            
        for trial_idx, trial in enumerate(self.trial_container):
            # Create performance for one trial
            performance_one_trial = [PerformanceContainer(model.ID) for model in self.model_container]
            
            # Get train data
            train_trial_info = trial[0]
            # if self.disp_processbar:
            #     print('-------train info------------')
            #     print(train_trial_info.__dict__)
            if len(train_trial_info.dataset_idx) == 0:
                raise ValueError('Train trial {:d} information is empty'.format(trial_idx))
            X, Y, ref_sig = train_trial_info.get_data(self.dataset_container)
            
            # Train models 
            model_one_trial = []
            for train_model_idx, model_tmp in enumerate(self.model_container):
                trained_model = model_tmp.__copy__()
                tic = time.time()
                trained_model.fit(X=X, Y=Y, ref_sig=ref_sig) 
                performance_one_trial[train_model_idx].add_train_time(time.time()-tic)
                model_one_trial.append(trained_model)
            if eval_train:
                for test_model_idx, model_tmp in enumerate(model_one_trial):
                    tic = time.time()
                    pred_label = model_tmp.predict(X)
                    performance_one_trial[test_model_idx].add_test_time_train(time.time()-tic)
                    performance_one_trial[test_model_idx].add_pred_label_train(pred_label)
                    performance_one_trial[test_model_idx].add_true_label_train(Y)
                
            # Get test data
            test_trial_info = trial[1]
            # if self.disp_processbar:
            #     print('-------test info------------')
            #     print(test_trial_info.__dict__)
            if len(test_trial_info.dataset_idx) == 0:
                raise ValueError('Test trial {:d} information is empty'.format(trial_idx))
            X, Y, ref_sig = test_trial_info.get_data(self.dataset_container)
                
            # Test models
            for test_model_idx, model_tmp in enumerate(model_one_trial):
                tic = time.time()
                pred_label = model_tmp.predict(X)
                performance_one_trial[test_model_idx].add_test_time_test(time.time()-tic)
                performance_one_trial[test_model_idx].add_pred_label_test(pred_label)
                performance_one_trial[test_model_idx].add_true_label_test(Y)
                
            self.performance_container.append(performance_one_trial)
            if self.save_model:
                self.trained_model_container.append(model_one_trial)
                
            if self.disp_processbar:
                pbar.update(1)
        
        if self.disp_processbar:
            pbar.close()
            print('========================\n   End\n========================\n')     
            
    def search_trial_idx(self,
                         train_or_test: str,
                         trial_info: dict) -> list:
        if train_or_test.lower() == "train":
            idx = 0
        elif train_or_test.lower() == "test":
            idx = 1
        else:
            raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
        res = []
        for trial_idx, trial in enumerate(self.trial_container):
            store_flag = True
            for search_key, search_value in trial_info.items():
                store_flag = store_flag and (trial[idx].__dict__[search_key] == search_value)
            if store_flag:
                res.append(trial_idx)
                
        return res
            

        
                