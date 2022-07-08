# -*- coding: utf-8 -*-
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

import numpy as np
import matplotlib.pyplot as plt

def bar_plot_with_errorbar(Y: ndarray,
             bar_sep: float = 0.25,
             x_label: Optional[str] = None,
             y_label: Optional[str] = None,
             x_ticks: Optional[List[str]] = None,
             legend: Optional[List[str]] = None,
             errorbar_type: str = 'std'):
    """
    Plot bars

    For each group, a set of bars will be ploted on all variables
    Bar heights are equal to the mean of all observations
    Error bar are calculated across observations
    (x axis: observation; y axis: variable)

    Parameters
    -----------
    Y: ndarray
        Plot data
        Shape: (group_num, observation_num, variable_num)
    bar_sep: Optional[float]
        Separation between two variables
        Default is 0.25
    x_label: str
        Label of x axis
        Default is None
    y_label: str
        Label of y axis
        Default is None
    x_ticks: List[str]
        Ticks of x axis
        Default is None
    legend: List[str]
        Legend of groups
        Default is None
    errorbar_type: str
        Method of calculating error, including:
            - 'std': Standard derivation
            - '95ci': 95% confidence interval
        Default is 'std'
    """
    if len(Y.shape) != 3:
        raise ValueError("Plot data must have 3 dimentions")
    group_num, observation_num, variable_num = Y.shape
    if x_ticks is not None:
        if len(x_ticks) != variable_num:
            raise ValueError("Length of 'x_ticks' should be equal to 3rd dimention of data")
    if legend is not None:
        if len(legend) != group_num:
            raise ValueError("Length of 'legend' should be equal to 1st dimention of data")

    x_center = np.arange(1, variable_num+1, 1)
    width = (1-bar_sep)/group_num
    x = x_center - 0.5 + bar_sep/2 + width/2
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    for group_idx in range(group_num):
        Y_tmp = Y[group_idx,:,:]
        Y_mean = np.mean(Y_tmp,0)
        if errorbar_type.lower() == 'std':
            Y_error = np.std(Y_tmp, 0)
        ax.bar(x, Y_mean, width = width)
        x = x + width

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if x_ticks is not None:
        ax.set_xticks(x_center, x_ticks)
    if legend is not None:
        ax.legend(labels=legend)