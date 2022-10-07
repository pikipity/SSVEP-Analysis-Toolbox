# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

import numpy as np

def cal_confusionmatrix_onedataset_individual_online(evaluator: object,
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
        Shape: (subject_num, signal_len_num, method_num, trial_num, sub_trial_num, true_class_num, pred_class_num)
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

    confusion_matrix = []
    for sub_idx in range(sub_num):
        tw_matrix = []
        for tw in tw_seq:

            trial_info = {'dataset_idx':[dataset_idx],
                          'sub_idx':[sub_idx],
                          'tw':tw}
            trial_idx = evaluator.search_trial_idx(train_or_test, trial_info)
            
            method_matrix = []
            for method_idx in range(len(model_container)):
                trial_matrix = []
                for i in trial_idx:
                    if idx == 0:
                        true_label_list = evaluator.performance_container[i][method_idx].true_label_train
                        pred_label_list = evaluator.performance_container[i][method_idx].pred_label_train
                    else:
                        true_label_list = evaluator.performance_container[i][method_idx].true_label_test
                        pred_label_list = evaluator.performance_container[i][method_idx].pred_label_test
                    sub_trial_matrix = []
                    for j in range(len(true_label_list)):
                        confusion_matrix_tmp = np.zeros((N,N))
                        for true_label, pred_label in zip(true_label_list[:(i+1)], pred_label_list[:(i+1)]):
                            confusion_matrix_tmp[true_label, pred_label] = confusion_matrix_tmp[true_label, pred_label] + 1
                        confusion_matrix_tmp = np.expand_dims(confusion_matrix_tmp, 0)
                        sub_trial_matrix.append(confusion_matrix_tmp)
                    sub_trial_matrix = np.concatenate(sub_trial_matrix, 0)
                    sub_trial_matrix = np.expand_dims(sub_trial_matrix, 0)
                    trial_matrix.append(sub_trial_matrix)
                trial_matrix = np.concatenate(trial_matrix, 0)
                trial_matrix = np.expand_dims(trial_matrix, 0)
                method_matrix.append(trial_matrix)
            method_matrix = np.concatenate(method_matrix, 0)
            method_matrix = np.expand_dims(method_matrix, 0)
            tw_matrix.append(method_matrix)
        tw_matrix = np.concatenate(tw_matrix, 0)
        tw_matrix = np.expand_dims(tw_matrix, 0)
        confusion_matrix.append(tw_matrix)
    confusion_matrix = np.concatenate(confusion_matrix, 0)
                    
    return confusion_matrix

def cal_performance_onedataset_individual_online(evaluator: object,
                                                     dataset_idx: int,
                                                     tw_seq: List[float],
                                                     train_or_test: str) -> Tuple[ndarray, ndarray]:
    """
    Calculate acc and ITR for one dataset
    Evaluations will be carried out on each subject and each signal length
    Under each subject and each signal length, there may be several trials. Acc and itr values of all these trials will be stored

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
        Shape: (subject_num, signal_len_num, method_num, trial_num, sub_trial_num)
    itr_store: ndarray
        Classification ITR
        Shape: (subject_num, signal_len_num, method_num, trial_num, sub_trial_num)
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

    # acc_store = np.zeros((len(model_container),sub_num, len(tw_seq)))
    # itr_store = np.zeros((len(model_container),sub_num, len(tw_seq)))
    acc_store = []
    itr_store = []

    for sub_idx in range(sub_num):
        acc_tw = []
        itr_tw = []
        for tw in tw_seq:
            trial_info = {'dataset_idx':[dataset_idx],
                          'sub_idx':[sub_idx],
                          'tw':tw}
            trial_idx = evaluator.search_trial_idx(train_or_test, trial_info)
            
            acc_method = []
            itr_method = []
            for method_idx in range(len(model_container)):
                tmp_acc = []
                tmp_itr = []
                for i in trial_idx:
                    performance_container = [evaluator.performance_container[i][method_idx]]
                    tmp_acc.extend(cal_acc_trials_onebyone(train_or_test, performance_container))
                    t_latency = evaluator.trial_container[i][idx].t_latency[dataset_idx]
                    tw = evaluator.trial_container[i][idx].tw
                    if t_latency is None:
                        t_latency = dataset_container[dataset_idx].default_t_latency
                    tmp_itr.extend(cal_itr_trials_onebyone(train_or_test, performance_container, tw, t_break, t_latency))
                tmp_acc = np.expand_dims(tmp_acc, 0)
                acc_method.append(tmp_acc)
                tmp_itr = np.expand_dims(tmp_itr, 0)
                itr_method.append(tmp_itr)
            acc_method = np.concatenate(acc_method, 0)
            acc_method = np.expand_dims(acc_method, 0)
            itr_method = np.concatenate(itr_method, 0)
            itr_method = np.expand_dims(itr_method, 0)
            acc_tw.append(acc_method)
            itr_tw.append(itr_method)
        acc_tw = np.concatenate(acc_tw, 0)
        acc_tw = np.expand_dims(acc_tw, 0)
        itr_tw = np.concatenate(itr_tw, 0)
        itr_tw = np.expand_dims(itr_tw, 0)
        acc_store.append(acc_tw)
        itr_store.append(itr_tw)
    acc_store = np.concatenate(acc_store, 0)
    itr_store = np.concatenate(itr_store, 0)

    return acc_store, itr_store

def cal_confusionmatrix_onedataset_individual_diffsiglen(evaluator: object,
                                                        dataset_idx: int,
                                                        tw_seq: List[float],
                                                        train_or_test: str,
                                                        subj_seq: Optional[List[int]] = None) -> Tuple[ndarray, ndarray]:
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
    subj_seq : Optional[List[int]]
        List of subject indices
        If None, all subjects will be included

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
    # sub_num = len(dataset_container[dataset_idx].subjects)
    N = dataset_container[dataset_idx].stim_info['stim_num']
    if subj_seq is None:
        subj_seq = list(range(len(dataset_container[dataset_idx].subjects)))
    sub_num = len(subj_seq)

    confusion_matrix = np.zeros((len(model_container),sub_num, len(tw_seq), N, N))

    for sub_i, sub_idx in enumerate(subj_seq):
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
                        confusion_matrix[method_idx, sub_i, tw_idx, true_label, pred_label] = confusion_matrix[method_idx, sub_i, tw_idx, true_label, pred_label] + 1
    return confusion_matrix

def cal_performance_onedataset_individual_diffsiglen(evaluator: object,
                                                     dataset_idx: int,
                                                     tw_seq: List[float],
                                                     train_or_test: str,
                                                     subj_seq: Optional[List[int]] = None) -> Tuple[ndarray, ndarray]:
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
    subj_seq : Optional[List[int]]
        List of subject indices
        If None, all subjects will be included

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
    # sub_num = len(dataset_container[dataset_idx].subjects)
    t_break = dataset_container[dataset_idx].t_break
    if subj_seq is None:
        subj_seq = list(range(len(dataset_container[dataset_idx].subjects)))
    sub_num = len(subj_seq)
    
    acc_store = np.zeros((len(model_container),sub_num, len(tw_seq)))
    itr_store = np.zeros((len(model_container),sub_num, len(tw_seq)))
    for sub_i, sub_idx in enumerate(subj_seq):
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
                    t_latency = evaluator.trial_container[i][idx].t_latency[dataset_idx]
                    tw = evaluator.trial_container[i][idx].tw
                    if t_latency is None:
                        t_latency = dataset_container[dataset_idx].default_t_latency
                    tmp_itr.append(cal_itr_trials(train_or_test, performance_container, tw, t_break, t_latency))
                acc_store[method_idx,sub_i,tw_idx] = mean(tmp_acc)
                itr_store[method_idx,sub_i,tw_idx] = mean(tmp_itr)
    return acc_store, itr_store

def cal_itr_trials_onebyone(train_or_test: str,
                            performance_container: list,
                            tw,t_break,t_latency) -> List[float]:
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
    list_itr: List[float]
        List of ITR
    """
    list_acc = []
    for performance in performance_container:
        if train_or_test.lower() == "train":
            for i in range(len(performance.true_label_train)):
                acc = cal_acc(performance.true_label_train[:(i+1)], performance.pred_label_train[:(i+1)])
                t_comp = sum(performance.test_time_train[:(i+1)])/len(performance.true_label_train[:(i+1)])
                N = len(set(performance.true_label_train[:(i+1)]))
                list_acc.append(cal_itr(tw,t_break,t_latency,t_comp,N, acc))
        elif train_or_test.lower() == "test":
            for i in range(len(performance.true_label_test)):
                acc = cal_acc(performance.true_label_test[:(i+1)], performance.pred_label_test[:(i+1)])
                t_comp = sum(performance.test_time_test[:(i+1)])/len(performance.true_label_test[:(i+1)])
                N = len(set(performance.true_label_test[:(i+1)]))
                list_acc.append(cal_itr(tw,t_break,t_latency,t_comp,N, acc))
        else:
            raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
    return list_acc

def cal_itr_trials(train_or_test: str,
                   performance_container: list,
                   tw,t_break,t_latency) -> float:
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
            N = len(set(performance.true_label_train))
            list_acc.append(cal_itr(tw,t_break,t_latency,t_comp,N, acc))
        elif train_or_test.lower() == "test":
            acc = cal_acc(performance.true_label_test, performance.pred_label_test)
            t_comp = sum(performance.test_time_test)/len(performance.true_label_test)
            N = len(set(performance.true_label_test))
            list_acc.append(cal_itr(tw,t_break,t_latency,t_comp,N, acc))
        else:
            raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
            
    return mean(list_acc)

def cal_acc_trials_onebyone(train_or_test: str,
                            performance_container: list) -> List[float]:
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
    list_acc: List[float]
        List of accuracy
    """
    list_acc = []
    for performance in performance_container:
        if train_or_test.lower() == "train":
            for i in range(len(performance.true_label_train)):
                list_acc.append(cal_acc(performance.true_label_train[:(i+1)], performance.pred_label_train[:(i+1)]))
        elif train_or_test.lower() == "test":
            for i in range(len(performance.true_label_test)):
                list_acc.append(cal_acc(performance.true_label_test[:(i+1)], performance.pred_label_test[:(i+1)]))
        else:
            raise ValueError("Unknown train_or_test type. It must be 'train' or 'test'")
    return list_acc

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
        