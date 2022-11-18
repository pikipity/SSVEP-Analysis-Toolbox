# -*- coding: utf-8 -*-

import os
import numpy as np
import zipfile
import shutil

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, transpose

from .basedataset import BaseDataset
from .subjectinfo import SubInfo
from ..utils.download import download_single_file
from ..utils.io import loadmat

class NakanishiDataset(BaseDataset):
    """
    Nakanishi 2015

    Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject. 
    Please see the reference paper for the detail.

    size(eeg) = [Num. of targets, Num. of channels, Num. of sampling points, Num. of trials]
    Num. of Targets 	: 12
    Num. of Channels 	: 8
    Num. of sampling points : 1114
    Num. of trials 		: 15
    Sampling rate 		: 256 Hz
    * The order of the stimulus frequencies in the EEG data: 
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz
    (e.g., eeg(1,:,:,:) and eeg(5,:,:,:) are the EEG data while a subject was gazing at the visual stimuli flickering at 9.25 Hz and 11.75Hz, respectively.)
    * The onset of visual stimulation is at 39th sample point.

    Reference:
    Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,
    "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials,"
    PLoS One, vol.10, no.10, e140703, 2015.
    """

    _CHANNELS = [
        'PO7','PO3','POz','PO4','PO8',
        'O1','Oz','O2'
    ]

    _FREQS = [
        9.25, 11.25, 13.25, 
        9.75, 11.75, 13.75, 
        10.25, 12.25, 14.25, 
        10.75, 12.75, 14.75
    ]

    _PHASES = [
        0, 0, 0,
        0.5, 0.5, 0.5,
        1, 1, 1,
        1.5, 1.5, 1.5
    ]

    _SUBJECTS = [SubInfo(ID = 's{:d}'.format(sub_idx)) for sub_idx in range(1,10+1,1)]

    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'Nakanishi2015', 
                         url = 'ftp://sccn.ucsd.edu/pub/cca_ssvep.zip', 
                         paths = path, 
                         channels = self._CHANNELS, 
                         srate = 256, 
                         block_num = 15, 
                         trial_num = len(self._FREQS),
                         trial_len = 4, 
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = [],
                         path_support_file = path_support_file,
                         t_prestim = 39/256,
                         t_break = 1,
                         default_t_latency = 0.135)

    def download_single_subject(self,
                                subject: SubInfo):
        source_url = self.url
        desertation = os.path.join(subject.path, 'cca_ssvep.zip')
        
        data_file = os.path.join(subject.path, subject.ID + '.mat')

        download_flag = True
        
        if not os.path.isfile(data_file):
            try:
                download_single_file(source_url, desertation, progressbar = False)
            
                with zipfile.ZipFile(desertation,'r') as archive:
                    archive.extractall(subject.path)
                    
                os.remove(desertation)

                for (dirpath, dirnames, filenames) in os.walk(os.path.join(subject.path, 'cca_ssvep')):
                    for filename in filenames:
                        src = os.path.join(dirpath, filename)
                        dst = os.path.join(subject.path, filename)
                        shutil.copyfile(src, dst)
                shutil.rmtree(os.path.join(subject.path, 'cca_ssvep'))
            except:
                download_flag = False

        return download_flag, source_url, desertation

    def download_file(self,
                      file_name: str):
        return True, None, None

    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))
        
        sub_info = self.subjects[sub_idx]
        file_path = os.path.join(sub_info.path, sub_info.ID + '.mat')
        
        mat_data = loadmat(file_path)
        data = mat_data['eeg']
        data = transpose(data, (3,0,1,2)) # block_num * stimulus_num * ch_num * whole_trial_samples

        return data

    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return trial_idx