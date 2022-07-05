# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray

import time

from .basepipline import BasePipline, PerformanceContainer
from .utils import create_pbar


def gen_train_test_blocks_leave_one_block_out(dataset_container: list) -> Tuple[list, list]:
    """
    Generate Lists of training blocks and testing blocks for pipline
    Leave one block out rule

    Parameters
    ----------
    dataset_container : list
        List of datasets

    Returns
    -------
    testing_blocks : list
    training_blocks : list
    """
    training_blocks = []
    testing_blocks = []
    for dataset in dataset_container:
        tmp1 = []
        tmp2 = []
        for block_idx in range(dataset.block_num):
            tmp_test_block, tmp_train_block = dataset.leave_one_block_out(block_idx)
            tmp1.append(tmp_test_block)
            tmp2.append(tmp_train_block)
        testing_blocks.append(tmp1)
        training_blocks.append(tmp2)
    return testing_blocks, training_blocks

def gen_train_test_trials_all_trials(dataset_container: list) -> Tuple[list, list]:
    """
    Generate Lists of training and testing trials for pipline
    including all trials

    Parameters
    ----------
    dataset_container : list
        List of datasets

    Returns
    -------
    training_trials: list,
    testing_trials: list
    """
    training_trials = []
    testing_trials = []
    for dataset in dataset_container:
        tmp1 = []
        tmp2 = []
        for block_idx in range(dataset.block_num):
            tmp1.append([i for i in range(dataset.stim_info['stim_num'])])
            tmp2.append([i for i in range(dataset.stim_info['stim_num'])])
        testing_trials.append(tmp1)
        training_trials.append(tmp2)
    return testing_trials, training_trials



class PiplineIndividual(BasePipline):
    """
    Pipline for individual training
    
    Performance will be evaluated for each provided time signal length and each subject
    """
    def __init__(self, 
                ch_used: list,
                harmonic_num: list,
                model_container: list,
                dataset_container: list,
                tw_seq: List[float],
                training_blocks: list,
                testing_blocks: list,
                training_trials: list,
                testing_trials: list,
                save_model: Optional[bool] = False,
                disp_processbar: Optional[bool] = True,
                shuffle_trials: Optional[bool] = False):
        """
        Special parameters
        
        tw_seq: List[float]
            List of signal length
        training_blocks: list
            List of training blocks
            Shape: (dataset_num,)
            Shape of element: (iteration_num,)
            Shape of sub-element: (train_block_num,)
        testing_blocks: list
            List of testing blocks
            shape: (dataset_num)
            Shape of element: (iteration_num,)
            Shape of sub-element: (testing_block_num,)
        training_trials: list
            List of training trials
            shape: (dataset_num)
            Shape of element: (iteration_num,)
            Shape of sub-element: (train_trial_num,)
        testing_trials: list
            List of testing trials
            shape: (dataset_num)
            Shape of element: (iteration_num,)
            Shape of sub-element: (test_trial_num,)
        """
        super().__init__(ch_used = ch_used,
                         harmonic_num = harmonic_num,
                        model_container = model_container,
                        dataset_container = dataset_container,
                        save_model = save_model,
                        disp_processbar = disp_processbar,
                        shuffle_trials = shuffle_trials)
        
        if len(training_blocks) != len(self.dataset_container):
            raise ValueError("Training blocks should contain all datasets' training blocks")
        if len(testing_blocks) != len(self.dataset_container):
            raise ValueError("Testing blocks should contain all datasets' testing blocks")
            
        for train_block, test_block in zip(training_blocks, testing_blocks):
            if len(train_block) != len(test_block):
                raise ValueError("Training and testing blocks should contain all ierations' training and testing blocks")
        
        self.tw_seq = tw_seq
        self.training_blocks = training_blocks
        self.testing_blocks = testing_blocks
        self.training_trials = training_trials
        self.testing_trials = testing_trials
        
        
    def run(self, 
            n_jobs: Optional[int] = None):
        if n_jobs is not None:
            for i in range(len(self.model_container)):
                self.model_container[i].n_jobs = n_jobs
        
        self.trained_model_container = []
        self.performance_container = []
        pbar = None
        
        if self.disp_processbar:
            print('Start')
        
        for dataset_idx, (dataset, 
                          dataset_training_blocks,
                          dataset_testing_blocks,
                          dataset_used_ch,
                          dataset_harmonic_num,
                          dataset_train_trials,
                          dataset_test_trials) in enumerate(zip(self.dataset_container, 
                                                                    self.training_blocks,
                                                                    self.testing_blocks,
                                                                    self.ch_used,
                                                                    self.harmonic_num,
                                                                    self.training_trials,
                                                                    self.testing_trials)):
            
            PerformanceContainer_one_dataset = []
            model_container_one_dataset = []
            
            if self.disp_processbar:
                if pbar is not None:
                    pbar.close()
                print('Dataset {:d}: {:s}'.format(dataset_idx, dataset.ID))
                pbar = create_pbar([len(self.tw_seq), len(dataset.subjects), len(dataset_training_blocks)],
                                   'T: {:d}/{:d}, Sub: {:d}/{:d}, Itr: {:d}/{:d}, Train: {:d}/{:d}, Test: {:d}/{:d}'.format(0,len(self.tw_seq),
                                                                                                                     0,len(dataset.subjects),
                                                                                                                     0,len(dataset_training_blocks),
                                                                                                                     0,len(self.model_container),
                                                                                                                     0,len(self.model_container)))
            
            for tw_idx, tw in enumerate(self.tw_seq):
                PerformanceContainer_one_tw = []
                model_container_one_tw = []
                
                for sub_idx in range(len(dataset.subjects)):
                    PerformanceContainer_one_sub = []
                    model_container_one_sub = []
                    
                    for iteration_idx, (training_block_one_iteration, 
                                        testing_block_one_iteration,
                                        train_trial_one_iteration,
                                        testing_trial_one_iteration) in enumerate(zip(dataset_training_blocks, 
                                                                                      dataset_testing_blocks,
                                                                                      dataset_train_trials,
                                                                                      dataset_test_trials)):                                                                                      
                        PerformanceContainer_one_iteration = [PerformanceContainer(model_tmp.ID) for model_tmp in self.model_container]
                        
                        # Training
                        ref_sig = dataset.get_ref_sig(tw,dataset_harmonic_num)
                        X, Y = dataset.get_data(sub_idx = sub_idx,
                                                blocks = training_block_one_iteration,
                                                trials = train_trial_one_iteration,
                                                channels = dataset_used_ch,
                                                sig_len = tw,
                                                shuffle = self.shuffle_trials)
                        
                        model_container_one_iteration = []
                        for train_model_idx, model_tmp in enumerate(self.model_container):
                            trained_model = model_tmp.__copy__()
                            tic = time.time()
                            trained_model.fit(X=X, Y=Y, ref_sig=ref_sig) 
                            PerformanceContainer_one_iteration[train_model_idx].add('train-times',time.time()-tic)
                            model_container_one_iteration.append(trained_model)
                            if self.disp_processbar:
                                pbar.set_description('T: {:d}/{:d}, Sub: {:d}/{:d}, Itr: {:d}/{:d}, Train: {:d}/{:d}, Test: {:d}/{:d}'.format(tw_idx+1,len(self.tw_seq),
                                                                                                                                       sub_idx+1,len(dataset.subjects),
                                                                                                                                       iteration_idx+1,len(dataset_training_blocks),
                                                                                                                                       train_model_idx+1,len(self.model_container),
                                                                                                                                       0,len(self.model_container)))
                        
                        # Testing
                        X, Y = dataset.get_data(sub_idx = sub_idx,
                                                blocks = testing_block_one_iteration,
                                                trials = testing_trial_one_iteration,
                                                channels = dataset_used_ch,
                                                sig_len = tw,
                                                shuffle = self.shuffle_trials)
                        
                        for test_model_idx, model_tmp in enumerate(model_container_one_iteration):
                            tic = time.time()
                            pred_label = model_tmp.predict(X)
                            PerformanceContainer_one_iteration[test_model_idx].add('test-times', time.time()-tic)
                            PerformanceContainer_one_iteration[test_model_idx].add('predict-labels', pred_label)
                            PerformanceContainer_one_iteration[test_model_idx].add('true-labels', Y)
                            if self.disp_processbar:
                                pbar.set_description('T: {:d}/{:d}, Sub: {:d}/{:d}, Itr: {:d}/{:d}, Train: {:d}/{:d}, Test: {:d}/{:d}'.format(tw_idx+1,len(self.tw_seq),
                                                                                                                                       sub_idx+1,len(dataset.subjects),
                                                                                                                                       iteration_idx+1,len(dataset_training_blocks),
                                                                                                                                       train_model_idx+1,len(self.model_container),
                                                                                                                                       test_model_idx+1,len(self.model_container)))
                            
                        PerformanceContainer_one_sub.append(PerformanceContainer_one_iteration)
                        if self.save_model:
                            model_container_one_sub.append(model_container_one_iteration)
                        
                        if self.disp_processbar:
                            pbar.update(1)
                            pbar.set_description('T: {:d}/{:d}, Sub: {:d}/{:d}, Itr: {:d}/{:d}, Train: {:d}/{:d}, Test: {:d}/{:d}'.format(tw_idx+1,len(self.tw_seq),
                                                                                                                                   sub_idx+1,len(dataset.subjects),
                                                                                                                                   iteration_idx+1,len(dataset_training_blocks),
                                                                                                                                   train_model_idx+1,len(self.model_container),
                                                                                                                                   test_model_idx+1,len(self.model_container)))
                        
                    PerformanceContainer_one_tw.append(PerformanceContainer_one_sub)
                    if self.save_model:
                        model_container_one_tw.append(model_container_one_sub)
                PerformanceContainer_one_dataset.append(PerformanceContainer_one_tw)
                if self.save_model:
                    model_container_one_dataset.append(model_container_one_tw)
            self.trained_model_container.append(PerformanceContainer_one_dataset)
            if self.save_model:
                self.performance_container.append(model_container_one_dataset)

        if self.disp_processbar:
            print('Finish')
        
                    
                        
        
        
        

