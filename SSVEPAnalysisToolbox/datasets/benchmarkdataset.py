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

class BenchmarkDataset(BaseDataset):
    """
    Benchmark Dataset
    
    This dataset gathered SSVEP-BCI recordings of 35 healthy subjects (17 females, aged 17-34 years, mean age: 22 years) focusing on 40 characters flickering at different frequencies (8-15.8 Hz with an interval of 0.2 Hz).
    For each subject, the experiment consisted of 6 blocks. Each block contained 40 trials corresponding to all 40 characters indicated in a random order. Each trial started with a visual cue (a red square) indicating a target stimulus. The cue appeared for 0.5 s on the screen.
    Following the cue offset, all stimuli started to flicker on the screen concurrently and lasted 5 s.
    After stimulus offset, the screen was blank for 0.5 s before the next trial began, which allowed the subjects to have short breaks between consecutive trials.
    Each trial lasted a total of 6 s.

    Total: around 3.45 GB
    
    Paper:
    Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for SSVEP-based braincomputer interfaces,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 10, pp. 17461752, 2017.
    """
 
    _CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2',
        'F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
        'FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','M1','TP7',
        'CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5',
        'P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4',
        'PO6','PO8','CB1','O1','Oz','O2','CB2'
    ]

    _FREQS = [
        8, 9, 10, 11, 12, 13, 14, 15, 
        8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 
        8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
        8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
        8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
    ]

    _PHASES = [
        0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
        0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0,
        1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5,
        1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
        0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5
    ]
    
    _SUBJECTS = [SubInfo(ID = 'S{:d}'.format(sub_idx)) for sub_idx in range(1,35+1,1)]
    
    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'Benchmark Dataset', 
                         url = 'http://bci.med.tsinghua.edu.cn/upload/yijun/', 
                         paths = path, 
                         channels = self._CHANNELS, 
                         srate = 250, 
                         block_num = 6, 
                         trial_num = len(self._FREQS),
                         trial_len = 6, 
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = ['Readme.txt',
                                          'Sub_info.txt',
                                          '64-channels.loc',
                                          'Freq_Phase.mat'],
                         path_support_file = path_support_file,
                         t_prestim = 0.5,
                         t_break = 0.5,
                         default_t_latency = 0.14)
    
    def download_single_subject(self,
                                subject: SubInfo):
        source_url = self.url + subject.ID + '.mat.7z'
        desertation = os.path.join(subject.path, subject.ID + '.mat.7z')
        
        data_file = os.path.join(subject.path, subject.ID + '.mat')

        download_flag = True
        
        if not os.path.isfile(data_file):
            try:
                download_single_file(source_url, desertation)
            
                with py7zr.SevenZipFile(desertation,'r') as archive:
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
        data = mat_data['data']
        data = transpose(data, (3,2,0,1)) # block_num * stimulus_num * ch_num * whole_trial_samples
        
        return data
    
    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return trial_idx
        