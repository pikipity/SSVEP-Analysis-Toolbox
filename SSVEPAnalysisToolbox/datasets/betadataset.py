# -*- coding: utf-8 -*-

import os
import numpy as np
import py7zr

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, transpose

from .basedataset import BaseDataset
from .subjectinfo import SubInfo
from ..utils.download import download_single_file
from ..utils.io import loadmat

class BETADataset(BaseDataset):
    """
    BETA Dataset
    
    
    
    Paper: 
    B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “BETA: A large benchmark database toward SSVEP-BCI application,” Front. Neurosci., vol. 14, p. 627, 2020.
    """

    _CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6',
        'F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5',
        'C3','C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ',
        'CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6',
        'P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ]
    
    _FREQS = [
        8.6, 8.8, 
        9, 9.2, 9.4, 9.6, 9.8,
        10, 10.2, 10.4, 10.6, 10.8, 
        11, 11.2, 11.4, 11.6, 11.8,
        12, 12.2, 12.4, 12.6, 12.8,
        13, 13.2, 13.4, 13.6, 13.8,
        14, 14.2, 14.4, 14.6, 14.8, 
        15, 15.2, 15.4, 15.6, 15.8, 
        8, 8.2, 8.4
    ]