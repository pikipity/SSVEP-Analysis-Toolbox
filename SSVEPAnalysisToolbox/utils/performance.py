# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable

import numpy as np

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
        