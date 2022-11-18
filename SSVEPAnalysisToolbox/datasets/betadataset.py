# -*- coding: utf-8 -*-

import os
import numpy as np
import tarfile

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, transpose

from .basedataset import BaseDataset
from .subjectinfo import SubInfo
from ..utils.download import download_single_file
from ..utils.io import loadmat

class BETADataset(BaseDataset):
    """
    BETA Dataset
    
    EEG data after preprocessing are store as a 4-way tensor, with a dimension of channel x time point x block x condition. 
    Each trial comprises 0.5-s data before the event onset and 0.5-s data after the time window of 2 s or 3 s. 
    For S1-S15, the time window is 2 s and the trial length is 3 s, whereas for S16-S70 the time window is 3 s and the trial length is 4 s. 
    Additional details about the channel and condition information can be found in the following supplementary information.

    Total: around 4.91 GB
    
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

    _PHASES = [
        1.5, 0,
        0.5, 1, 1.5, 0, 0.5,
        1, 1.5, 0, 0.5, 1,
        1.5, 0, 0.5, 1, 1.5,
        0, 0.5, 1, 1.5, 0,
        0.5, 1, 1.5, 0, 0.5,
        1, 1.5, 0, 0.5, 1,
        1.5, 0, 0.5, 1, 1.5,
        0, 0.5, 1
    ]

    _SUBJECTS = [SubInfo(ID = 'S{:d}'.format(sub_idx)) for sub_idx in range(1,70+1,1)]

    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'BETA Dataset', 
                         url = 'http://bci.med.tsinghua.edu.cn/upload/liubingchuan/', 
                         paths = path, 
                         channels = self._CHANNELS, 
                         srate = 250, 
                         block_num = 4, 
                         trial_num = len(self._FREQS),
                         trial_len = 3, 
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = ['note.pdf',
                                          'description.pdf'],
                         path_support_file = path_support_file,
                         t_prestim = 0.5,
                         t_break = 0.5,
                         default_t_latency = 0.13)
    
    def download_single_subject(self,
                                subject: SubInfo):
        data_file = os.path.join(subject.path, subject.ID + '.mat')

        sub_idx = int(subject.ID[1:])
        if sub_idx <= 10:
            file_name = 'S1-S10.tar.gz'
        elif sub_idx <= 20:
            file_name = 'S11-S20.tar.gz'
        elif sub_idx <= 30:
            file_name = 'S21-S30.tar.gz'
        elif sub_idx <= 40:
            file_name = 'S31-S40.tar.gz'
        elif sub_idx <= 50:
            file_name = 'S41-S50.tar.gz'
        elif sub_idx <= 60:
            file_name = 'S51-S60.tar.gz'
        elif sub_idx <= 70:
            file_name = 'S61-S70.tar.gz'
        source_url = self.url + file_name
        desertation = os.path.join(subject.path, file_name)

        download_flag = True
        
        if not os.path.isfile(data_file):
            try:
                download_single_file(source_url, desertation)
            
                with tarfile.open(desertation,'r') as archive:
                    archive.extractall(subject.path)
                    
                os.remove(desertation)
            except:
                download_flag = False

        return download_flag, source_url, desertation

    def download_file(self,
                      file_name: str):
        source_url = self.url + file_name
        desertation = os.path.join(self.path_support_file, file_name)

        download_flag = True
        
        if not os.path.isfile(desertation):
            try:
                download_single_file(source_url, desertation)
            except:
                download_flag = False

        return download_flag, source_url, desertation

    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))
        
        sub_info = self.subjects[sub_idx]
        file_path = os.path.join(sub_info.path, sub_info.ID + '.mat')
        
        mat_data = loadmat(file_path)
        data = mat_data['data']['EEG']
        data = transpose(data, (2,3,0,1)) # block_num * stimulus_num * ch_num * whole_trial_samples

        return data

    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return trial_idx