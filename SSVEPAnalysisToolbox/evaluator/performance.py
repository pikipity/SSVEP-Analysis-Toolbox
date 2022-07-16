# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

import numpy as np

def cal_confusionmatrix_onedataset_individual_diffsiglen(evaluator: object,
                                                        dataset_idx: int,
                                                        tw_seq: List[float],
                                                        train_or_test: str) -> Tuple[ndarray, ndarray]:
    """
    Calculate confusion matrix for one dataset
    Evaluations will be carried out on each subject and each signal length
    Under each subject and each signal length, there may be several trials. Acc and itr values of all these trials will be averaged.

    Parameters
    ----------
    evaluator : object
        evaluator
    dataset_idx : int
        dataset index
    tw_seq : List[float]
        List of signal length
    train_or_test : str
        Calculate performance of training or testing performance

    Returns
    -------
    confusion_matrix: ndarray
        Classification accuracy
        Shape: (method_num, subject_num, signal_len_num, true_class_num, pred_class_num)
    """
    if train_or_test.lower() == "train":
        idx = 0
    elif train_or_test.lower() == "test":
        idx = 1
    else:
        raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
    dataset_container = evaluator.dataset_container
    model_container = evaluator.model_container
    sub_num = len(dataset_container[dataset_idx].subjects)
    N = dataset_container[dataset_idx].stim_info['stim_num']

    confusion_matrix = np.zeros((len(model_container),sub_num, len(tw_seq), N, N))

    for sub_idx in range(sub_num):
        for tw_idx, tw in enumerate(tw_seq):
            trial_info = {'dataset_idx':[dataset_idx],
                          'sub_idx':[sub_idx],
                          'tw':tw}
            trial_idx = evaluator.search_trial_idx(train_or_test, trial_info)
            
            for method_idx in range(len(model_container)):
                for i in trial_idx:
                    if idx == 0:
                        true_label_list = evaluator.performance_container[i][method_idx].true_label_train
                        pred_label_list = evaluator.performance_container[i][method_idx].pred_label_train
                    else:
                        true_label_list = evaluator.performance_container[i][method_idx].true_label_test
                        pred_label_list = evaluator.performance_container[i][method_idx].pred_label_test
                    for true_label, pred_label in zip(true_label_list, pred_label_list):
                        confusion_matrix[method_idx, sub_idx, tw_idx, true_label, pred_label] = confusion_matrix[method_idx, sub_idx, tw_idx, true_label, pred_label] + 1
    return confusion_matrix

def cal_performance_onedataset_individual_diffsiglen(evaluator: object,
                                                     dataset_idx: int,
                                                     tw_seq: List[float],
                                                     train_or_test: str) -> Tuple[ndarray, ndarray]:
    """
    Calculate acc and ITR for one dataset
    Evaluations will be carried out on each subject and each signal length
    Under each subject and each signal length, there may be several trials. Acc and itr values of all these trials will be averaged.

    Parameters
    ----------
    evaluator : object
        evaluator
    dataset_idx : int
        dataset index
    tw_seq : List[float]
        List of signal length
    train_or_test : str
        Calculate performance of training or testing performance

    Returns
    -------
    acc_store: ndarray
        Classification accuracy
        Shape: (method_num, subject_num, signal_len_num)
    itr_store: ndarray
        Classification ITR
        Shape: (method_num, subject_num, signal_len_num)
    """
    if train_or_test.lower() == "train":
        idx = 0
    elif train_or_test.lower() == "test":
        idx = 1
    else:
        raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
    dataset_container = evaluator.dataset_container
    model_container = evaluator.model_container
    sub_num = len(dataset_container[dataset_idx].subjects)
    t_break = dataset_container[dataset_idx].t_break
    
    acc_store = np.zeros((len(model_container),sub_num, len(tw_seq)))
    itr_store = np.zeros((len(model_container),sub_num, len(tw_seq)))
    for sub_idx in range(sub_num):
        for tw_idx, tw in enumerate(tw_seq):
            trial_info = {'dataset_idx':[dataset_idx],
                          'sub_idx':[sub_idx],
                          'tw':tw}
            trial_idx = evaluator.search_trial_idx(train_or_test, trial_info)
            
            for method_idx in range(len(model_container)):
                tmp_acc = []
                tmp_itr = []
                for i in trial_idx:
                    performance_container = [evaluator.performance_container[i][method_idx]]
                    tmp_acc.append(cal_acc_trials(train_or_test, performance_container))
                    N = len(set(performance_container[0].true_label_train))
                    t_latency = evaluator.trial_container[i][idx].t_latency[dataset_idx]
                    tw = evaluator.trial_container[i][idx].tw
                    if t_latency is None:
                        t_latency = dataset_container[dataset_idx].default_t_latency
                    tmp_itr.append(cal_itr_trials(train_or_test, performance_container, tw, t_break, t_latency, N))
                acc_store[method_idx,sub_idx,tw_idx] = mean(tmp_acc)
                itr_store[method_idx,sub_idx,tw_idx] = mean(tmp_itr)
    return acc_store, itr_store

def cal_itr_trials(train_or_test: str,
                   performance_container: list,
                   tw,t_break,t_latency,N) -> float:
    """
    Calculate itr of trials

    Parameters
    ----------
    train_or_test : str
        Test "train" or "test"
    performance_container : list
        List of performance

    Returns
    -------
    itr: float
    """
    list_acc = []
    for performance in performance_container:
        if train_or_test.lower() == "train":
            acc = cal_acc(performance.true_label_train, performance.pred_label_train)
            t_comp = sum(performance.test_time_train)/len(performance.true_label_train)
            list_acc.append(cal_itr(tw,t_break,t_latency,t_comp,N, acc))
        elif train_or_test.lower() == "test":
            acc = cal_acc(performance.true_label_test, performance.pred_label_test)
            t_comp = sum(performance.test_time_test)/len(performance.true_label_test)
            list_acc.append(cal_itr(tw,t_break,t_latency,t_comp,N, acc))
        else:
            raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
            
    return mean(list_acc)

def cal_acc_trials(train_or_test: str,
                   performance_container: list) -> float:
    """
    Calculate acc of trials

    Parameters
    ----------
    train_or_test : str
        Test "train" or "test"
    performance_container : list
        List of performance

    Returns
    -------
    acc: float
    """
    list_acc = []
    for performance in performance_container:
        if train_or_test.lower() == "train":
            list_acc.append(cal_acc(performance.true_label_train, performance.pred_label_train))
        elif train_or_test.lower() == "test":
            list_acc.append(cal_acc(performance.true_label_test, performance.pred_label_test))
        else:
            raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
            
    return mean(list_acc)
    

def mean(X:list) -> float:
    """
    Calculate mean of a list

    Parameters
    ----------
    X : list

    Returns
    -------
    mean : float

    """    
    return sum(X) / len(X)

def cal_acc(Y_true: List[int],
            Y_pred: List[int]) -> float:
    """
    Calculate accuracy

    Parameters
    ----------
    Y_true : List[int]
        True labels
    Y_pred : List[int]
        Predicted labels

    Returns
    -------
    acc : float
        Accuracy
    """
    if len(Y_true) != len(Y_pred):
        raise ValueError('Lengths of true labels and predicted labels should be same')
    true_detect = [1 for i in range(len(Y_true)) if int(Y_true[i])==int(Y_pred[i])]
    true_detect_count = sum(true_detect)
    acc = true_detect_count/len(Y_true)
    return acc

def cal_itr(tw: Union[int, float],
            t_break: Union[int, float],
            t_latency: Union[int, float],
            t_comp: Union[int, float],
            N: int,
            acc: float) -> float:
    """
    Calculate ITR

    Parameters
    ----------
    tw : Union[int, float]
        Signal length (in second)
    t_break : Union[int, float]
        Time required for shifting visual attention
    t_latency : Union[int, float]
        Latency time
    t_comp : Union[int, float]
        Computational time of each trial
    N : int
        Number of classes
    acc : float
        Accuracy

    Returns
    -------
    ITR: float
        bits/min
    """
    total_t = tw + t_break + t_latency + t_comp
    if acc == 1:
        itr = 60/total_t * (np.log2(N) + acc * np.log2(acc))
    elif acc < 1/N:
        itr = 0
    else:
        itr = 60/total_t * (np.log2(N) + acc * np.log2(acc) + (1-acc) * np.log2( (1-acc)/(N-1) ))
    return float(itr)
        