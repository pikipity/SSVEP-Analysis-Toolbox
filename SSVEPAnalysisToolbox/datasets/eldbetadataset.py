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

class ELDBETADataset(BaseDataset):
    """
    eldBETA Dataset

    For the BCI users, there was an associated epoched record that is stored in ".mat" structure array from MATLAB. 
    
    The structure array in each record was composed of the EEG data ("EEG") and its associated supplementary information ("Suppl_info") as its fields. In the "EEG" field of the record, two types of EEG data, i.e., EEG epochs and raw EEG were provided for researchers to facilitate diverse research purposes. 
    
    The EEG epochs were the EEG data with the data processing and stored as 4-dimensional matrices (channel x time point x condition x block). The names and locations of the channel dimension were given in the supplementary information. 
    
    For the dimension of time point, the epochs had a length of 6 s, which included 0.5 s before the stimulus onset, 5 s during the stimulation (SSVEPs) and 0.5 s after the stimulus offset. 
    
    Different from the epoched data, the raw EEG provided continuous EEG that were converted by EEGLAB. The raw EEG were stored as cell arrays, each of which contained a block of EEG data. The "Suppl_info" field of the record provided a basic information about personal statistics and experimental protocol. The personal statistics included the aged, gender, BCIQ and SNR with respect to each subject. The experimental protocol included channel location ("Channel), stimulus frequency ("Frequency"), stimulus initial phase ("Phase") and sampling rate ("Srate"). The channel location was represented by a 64x4 cell arrays. The first column and the fourth column denoted the channel index and channel name, respectively. The second column and the third column denoted the channel location in polar coordinates, i.e., degree and radius, respectively. The stimulus initial phase was given in radius. The sampling rate of the epoch data was denoted by "Srate". 

    Total: around 20.0 GB

    Paper:
    Liu, B., Wang, Y., Gao, X. and Chen, X., 2022. eldBETA: A Large Eldercare-oriented Benchmark Database of SSVEP-BCI for the Aging Population. Scientific Data, 9(1), pp.1-12.

    -----------------Corrigendum------------------------------------------
    - 1. In page 4, "perspective" should be changed to "prospective".
    - 2. In figure 6 and the main text, the method name "temporally local multivariate synchronization index (tMSI)" should be changed to "an extension to MSI (EMSI)", in which we chose the best competing algorithm (i.e., EMSI) and the names were confused. The green line in figure 6 denotes EMSI in "Zhang, Y., Guo, D., Yao, D. and Xu, P., 2017. The extension of multivariate synchronization index method for SSVEP-based BCI. Neurocomputing, 269, pp.226-231".
    """

    _CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6',
        'F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5',
        'C3','C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ',
        'CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6',
        'P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ]

    _FREQS = [
        8.0, 9.5, 11,
        8.5, 10, 11.5,
        9, 10.5, 12
    ]

    _PHASES = [
        0, 1.5, 1,
        0.5, 0, 1.5,
        1, 0.5, 0
    ]

    _SUBJECTS = [SubInfo(ID = 'S{:d}'.format(sub_idx)) for sub_idx in range(1,100+1,1)]

    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'eldBETA Dataset', 
                         url = 'http://bci.med.tsinghua.edu.cn/upload/liubingchuan_eldBETA_database/', 
                         paths = path, 
                         channels = self._CHANNELS, 
                         srate = 250, 
                         block_num = 7, 
                         trial_num = len(self._FREQS),
                         trial_len = 6, 
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = ['Description.tar.gz'],
                         path_support_file = path_support_file,
                         t_prestim = 0.5,
                         t_break = 0.5,
                         default_t_latency = 0.14)

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
        elif sub_idx <= 80:
            file_name = 'S71-S80.tar.gz'
        elif sub_idx <= 90:
            file_name = 'S81-S90.tar.gz'
        elif sub_idx <= 100:
            file_name = 'S91-S100.tar.gz'
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

        if download_flag:
            # load age and gender
            mat_data = loadmat(data_file)['data']['Suppl_info']
            age = mat_data['Age']
            gender = mat_data['Gender']
            if gender.lower()=='male':
                gender = 'M'
            else:
                gender = 'F'
            
            for sub_idx, sub_info in enumerate(self.subjects):
                if subject.ID == sub_info.ID:
                    self.subjects[sub_idx].age = age
                    self.subjects[sub_idx].gender = gender
                    break

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
        data = mat_data['data']['EEG']['Epoch']
        data = transpose(data, (3,2,0,1)) # block_num * stimulus_num * ch_num * whole_trial_samples

        return data

    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return trial_idx
        