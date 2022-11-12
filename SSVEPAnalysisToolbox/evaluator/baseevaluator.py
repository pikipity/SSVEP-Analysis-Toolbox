# -*- coding: utf-8 -*-
"""
Base class of evaluator
"""

import abc
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

from tqdm import tqdm
import time
import numpy as np
import os
import pickle
import copy

from joblib import Parallel, delayed
from functools import partial

import warnings

def gen_trials_onedataset_individual_online(dataset_idx: int,
                                         tw_seq: List[float],
                                         dataset_container: list,
                                         harmonic_num: int,
                                         repeat_num: int,
                                         trials: List[int],
                                         ch_used: List[int],
                                         t_latency: Optional[float] = None,
                                         shuffle: bool = False) -> list:
    """
    Generate evaluation trials for one dataset
    Evaluations will be carried out on each subject and each signal length
    Training datasets are the 1st block
    Testing datasets are the all blocks

    Parameters
    ----------
    dataset_idx : int
        dataset index of dataset_container
    tw : List[float]
        signal length
    sub_idx : int
        Subject index
    dataset_container : list
        List of datasets
    harmonic_num : int
        Number of harmonics
    repeat_num : int
        Number of randon times
    trials: List[int]
        List of trial index
    ch_used : List[int]
        List of channels
    t_latency : Optional[float]
        Latency time
        If None, default latency time of dataset will be used
    shuffle : bool
        Whether shuffle

    Returns
    -------
    trial_container : list
        List of trial information

    """
    sub_num = len(dataset_container[dataset_idx].subjects)
    trial_container = []
    for tw in tw_seq:
        for sub_idx in range(sub_num):
            for _ in range(repeat_num):
                train_block = [0]
                test_block = [block_idx for block_idx in range(dataset_container[dataset_idx].block_num)]
                train_trial = TrialInfo().add_dataset(dataset_idx = dataset_idx,
                                                        sub_idx = sub_idx,
                                                        block_idx = train_block,
                                                        trial_idx = trials,
                                                        ch_idx = ch_used,
                                                        harmonic_num = harmonic_num,
                                                        tw = tw,
                                                        t_latency = t_latency,
                                                        shuffle = shuffle)
                test_trial = TrialInfo().add_dataset(dataset_idx = dataset_idx,
                                                        sub_idx = sub_idx,
                                                        block_idx = test_block,
                                                        trial_idx = trials,
                                                        ch_idx = ch_used,
                                                        harmonic_num = harmonic_num,
                                                        tw = tw,
                                                        t_latency = t_latency,
                                                        shuffle = shuffle)
                trial_container.append([train_trial, test_trial])
    return trial_container

def gen_trials_onedataset_individual_diffsiglen(dataset_idx: int,
                                         tw_seq: List[float],
                                         dataset_container: list,
                                         harmonic_num: int,
                                         trials: List[int],
                                         ch_used: List[int],
                                         subjects: Optional[List[int]] = None,
                                         t_latency: Optional[float] = None,
                                         shuffle: bool = False) -> list:
    """
    Generate evaluation trials for one dataset
    Evaluations will be carried out on each subject and each signal length
    Training and testing datasets are separated based on the leave-one-block-out rule

    Parameters
    ----------
    dataset_idx : int
        dataset index of dataset_container
    tw_seq : List[float]
        List of signal length
    dataset_container : list
        List of datasets
    harmonic_num : int
        Number of harmonics
    trials: List[int]
        List of trial index
    ch_used : List[int]
        List of channels
    subjects : Optional[List[int]]
        List of subject indices
        If None, all subjects will be included
    t_latency : Optional[float]
        Latency time
        If None, default latency time of dataset will be used
    shuffle : bool
        Whether shuffle

    Returns
    -------
    trial_container : list
        List of trial information

    """
    if subjects is None:
        sub_num = len(dataset_container[dataset_idx].subjects)
        subjects = list(range(sub_num))
    trial_container = []
    for tw in tw_seq:
        for sub_idx in subjects:
            for block_idx in range(dataset_container[dataset_idx].block_num):
                test_block, train_block = dataset_container[dataset_idx].leave_one_block_out(block_idx)
                train_trial = TrialInfo().add_dataset(dataset_idx = dataset_idx,
                                                      sub_idx = sub_idx,
                                                      block_idx = train_block,
                                                      trial_idx = trials,
                                                      ch_idx = ch_used,
                                                      harmonic_num = harmonic_num,
                                                      tw = tw,
                                                      t_latency = t_latency,
                                                      shuffle = shuffle)
                test_trial = TrialInfo().add_dataset(dataset_idx = dataset_idx,
                                                      sub_idx = sub_idx,
                                                      block_idx = test_block,
                                                      trial_idx = trials,
                                                      ch_idx = ch_used,
                                                      harmonic_num = harmonic_num,
                                                      tw = tw,
                                                      t_latency = t_latency,
                                                      shuffle = shuffle)
                trial_container.append([train_trial, test_trial])
    return trial_container

class pbarParallel(Parallel):
    def __init__(self, 
                 loop_list_num: List[int],
                 use_tqdm : bool = True,
                 desc: str = '',
                 *args, **kwargs):
        self.loop_list_num = loop_list_num
        self.use_tqdm = use_tqdm
        self.desc = desc
        super().__init__(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        with create_pbar(self.use_tqdm, self.loop_list_num, self.desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)
    def print_progress(self):
        # self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def create_pbar(loop_list_num: List[int],
                use_tqdm : bool = True,
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
    pbar = tqdm(disable=not use_tqdm,
                total = total_num,
                desc = desc,
                bar_format = '{desc}{percentage:3.3f}%|{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}<{remaining}]',
                dynamic_ncols = True)
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
    def add_pred_r_train(self,pred_r:list):
        self.pred_r_train.extend(pred_r)
    def add_true_label_test(self,true_label:list):
        self.true_label_test.extend(true_label)
    def add_pred_label_test(self,pred_label:list):
        self.pred_label_test.extend(pred_label)
    def add_pred_r_test(self,pred_r:list):
        self.pred_r_test.extend(pred_r)
    def add_train_time(self,train_time:list):
        self.train_time.append(train_time)
    def add_test_time_train(self,test_time:list):
        self.test_time_train.append(test_time)
    def add_test_time_test(self,test_time:list):
        self.test_time_test.append(test_time)
    def clear(self):
        self.true_label_train = []
        self.pred_label_train = []
        self.pred_r_train = []
        self.true_label_test = []
        self.pred_label_test = []
        self.pred_r_test = []
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
                 dataset_container: list,
                 ignore_stim_phase: bool) -> Tuple[list, list, list, list]:
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
        freqs: list
            List of stimulus frequencies corresponding to reference signal
        """
        dataset = dataset_container[self.dataset_idx[0]]
        ref_sig = dataset.get_ref_sig(self.tw,self.harmonic_num, ignore_stim_phase)
        freqs = dataset.stim_info['freqs']
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
        return X, Y, ref_sig, freqs
        

def _run_loop(trial_idx, trial_container, 
              model_container, dataset_container, ignore_stim_phase,
              eval_train,
              save_model):
    trial = trial_container[trial_idx]

    # Create performance for one trial
    performance_one_trial = [PerformanceContainer(model.ID) for model in model_container]
    
    # Get train data
    train_trial_info = trial[0]
    # if self.disp_processbar:
    #     print('-------train info------------')
    #     print(train_trial_info.__dict__)
    if len(train_trial_info.dataset_idx) == 0:
        raise ValueError('Train trial {:d} information is empty'.format(trial_idx))
    X, Y, ref_sig, freqs = train_trial_info.get_data(dataset_container, ignore_stim_phase)
    
    # Train models 
    model_one_trial = []
    for train_model_idx, model_tmp in enumerate(model_container):
        trained_model = model_tmp.__copy__()
        tic = time.time()
        trained_model.fit(X=X, Y=Y, ref_sig=ref_sig, freqs=freqs) 
        performance_one_trial[train_model_idx].add_train_time(time.time()-tic)
        model_one_trial.append(trained_model)
        # if disp_processbar:
        #     pbar.update(pbar_update_val)
    if eval_train:
        for test_model_idx, model_tmp in enumerate(model_one_trial):
            tic = time.time()
            pred_label, pred_r = model_tmp.predict(X)
            performance_one_trial[test_model_idx].add_test_time_train(time.time()-tic)
            performance_one_trial[test_model_idx].add_pred_label_train(pred_label)
            performance_one_trial[test_model_idx].add_pred_r_train(pred_r)
            performance_one_trial[test_model_idx].add_true_label_train(Y)
            # if disp_processbar:
            #     pbar.update(pbar_update_val)

    # Get test data
    test_trial_info = trial[1]
    # if self.disp_processbar:
    #     print('-------test info------------')
    #     print(test_trial_info.__dict__)
    if len(test_trial_info.dataset_idx) == 0:
        raise ValueError('Test trial {:d} information is empty'.format(trial_idx))
    X, Y, ref_sig, _ = test_trial_info.get_data(dataset_container, ignore_stim_phase)
        
    # Test models
    for test_model_idx, model_tmp in enumerate(model_one_trial):
        tic = time.time()
        pred_label, pred_r = model_tmp.predict(X)
        # print(np.array(Y)-np.array(pred_label))
        performance_one_trial[test_model_idx].add_test_time_test(time.time()-tic)
        performance_one_trial[test_model_idx].add_pred_label_test(pred_label)
        performance_one_trial[test_model_idx].add_pred_r_test(pred_r)
        performance_one_trial[test_model_idx].add_true_label_test(Y)
        # if disp_processbar:
        #     pbar.update(pbar_update_val)

    if save_model:
        return performance_one_trial, model_one_trial
    else:
        return performance_one_trial, None
        
    # self.performance_container.append(performance_one_trial)
    # if self.save_model:
    #     self.trained_model_container.append(model_one_trial)

class BaseEvaluator:
    """
    BaseEvaluator
    """
    def __init__(self,
                 dataset_container : Optional[list] = None,
                 model_container : Optional[list] = None,
                 trial_container : Optional[list] = None,
                 save_model: bool = False,
                 disp_processbar: bool = True,
                 ignore_stim_phase: bool = False):
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
        ignore_stim_phase: bool
            Whether ignore stimulus phases when generating reference signals
        """
        self.dataset_container = dataset_container
        self.model_container = model_container
        self.trial_container = trial_container
        self.save_model = save_model
        self.disp_processbar = disp_processbar
        self.ignore_stim_phase = ignore_stim_phase
        
        self.performance_container = []
        self.trained_model_container = []

    def save(self,
             file: str):
        desertation_dir = os.path.dirname(file)
        if not os.path.exists(desertation_dir):
            os.makedirs(desertation_dir)
        if os.path.isfile(file):
            os.remove(file)
        with open(file,'wb') as file_:
            try:
                pickle.dump(self, file_, pickle.HIGHEST_PROTOCOL)
            except:
                warnings.warn("Cannot save whole evaluator. So remove 'dataset_container' and try to save it again.")
                saved_self = copy.deepcopy(self)
                try:
                    saved_self.dataset_container = None
                    pickle.dump(saved_self, file_, pickle.HIGHEST_PROTOCOL)
                except:
                    warnings.warn("Remove 'dataset_container' still cannot be saved. So remove 'model_container' and 'trained_model_container' and try to save it again.")
                    try:
                        saved_self.model_container = None
                        saved_self.trained_model_container = None
                        pickle.dump(saved_self, file_, pickle.HIGHEST_PROTOCOL)
                    except:
                        warnings.warn("Remove 'model_container' and 'trained_model_container' still cannot be saved. So only save 'performance_container'.")
                        pickle.dump(self.performance_container, file_, pickle.HIGHEST_PROTOCOL)

    def load(self,
             file: str):
        if not os.path.isfile(file):
            raise ValueError('{:s} does not exist!!'.format(file))
        with open(file,'rb') as file_:
            self_load = pickle.load(file_)
        if type(self_load) is not list and type(self_load) is not tuple:
            self.dataset_container = copy.deepcopy(self_load.dataset_container)
            self.model_container = copy.deepcopy(self_load.model_container)
            self.trial_container = copy.deepcopy(self_load.trial_container)
            self.save_model = copy.deepcopy(self_load.save_model)
            self.disp_processbar = copy.deepcopy(self_load.disp_processbar)
            self.ignore_stim_phase = copy.deepcopy(self_load.ignore_stim_phase)
            self.performance_container = copy.deepcopy(self_load.performance_container)
            self.trained_model_container = copy.deepcopy(self_load.trained_model_container)
        else:
            self.performance_container = copy.deepcopy(self_load)
            warnings.warn("'{:s}' only contains a list or a tuple. So only 'performance_container' is reloaded.".format(file))
        
    def run(self,
            n_jobs : Optional[int] = None,
            timeout : Optional[int] = None,
            eval_train : bool = False):
        """
        Run evaluator

        Parameters
        ----------
        n_jobs : Optional[int], optional
            Number of CPUs. The default is None.
        eval_train : bool
            Whether evaluate train performance
        save_model_after_evaluate : bool
            Whether save the model after the evaluation. To evaluate the online method, this flag is set as True
        """
        if self.dataset_container is None or self.model_container is None or self.trial_container is None:
            raise ValueError("Please check 'dataset_container', 'model_container', and 'trial_container'.")
        # if n_jobs is not None:
        #     for i in range(len(self.model_container)):
        #         self.model_container[i].n_jobs = n_jobs
        for i in range(len(self.model_container)):
            self.model_container[i].n_jobs = None
                
        if self.disp_processbar:
            print('\n========================\n   Start\n========================\n')
            # pbar = create_pbar([len(self.trial_container)])
            # if eval_train:
            #     pbar = create_pbar([len(self.trial_container), len(self.model_container)*3])
            # else:
            #     pbar = create_pbar([len(self.trial_container), len(self.model_container)*2])
            # pbar_update_val = 1
                
        self.performance_container, self.trained_model_container = zip(*pbarParallel(n_jobs=n_jobs,timeout=timeout,loop_list_num=[len(self.trial_container)],use_tqdm=self.disp_processbar)
                                                                                               (delayed(partial(_run_loop, model_container = self.model_container,
                                                                                                                           trial_container = self.trial_container,
                                                                                                                           dataset_container = self.dataset_container,
                                                                                                                           ignore_stim_phase = self.ignore_stim_phase,
                                                                                                                           eval_train = eval_train,
                                                                                                                           save_model = self.save_model
                                                                                                                           ))(trial_idx = trial_idx) 
                                                                                                                           for trial_idx in range(len(self.trial_container))))
        
        if self.disp_processbar:
            # pbar.close()
            print('\n========================\n   End\n========================\n')     
            
    def search_trial_idx(self,
                         train_or_test: str,
                         trial_info: dict) -> list:
        """
        According to given trial information, search trials

        Parameters
        ----------
        train_or_test : str
            Given trial information related to training or testing trials
        trial_info : dict
            Trial information. It can contain following items:
                - dataset_idx : list
                    List of dataset index
                sub_idx : list
                    List of subjects
                block_idx : list
                    List of blocks
                trial_idx : list
                    List of trials
                ch_idx : list
                    List of channels
                t_latency ; list
                    List of latency time
                harmonic_num : int
                    Number of harmonics for sine-cosine-based references
                tw : float
                    Signal window length
                shuffle : List[bool]
                    Whether shuffle trials

        Returns
        -------
        trial_idx: list
            List of indices of trials that satify the given information

        """
        if self.dataset_container is None or self.model_container is None or self.trial_container is None:
            raise ValueError("Please check 'dataset_container', 'model_container', and 'trial_container'.")
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
            

        
                