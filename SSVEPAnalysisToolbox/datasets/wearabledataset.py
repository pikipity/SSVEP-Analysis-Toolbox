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

import warnings

class WearableDataset_wet(BaseDataset):
    """
    Wearable SSVEP Dataset (wet)

    This study relied on the BCI Brain-Controlled Robot Contest at the 2020 World Robot Contest to recruit participants.

    One hundred and two healthy subjects (64 males and 38 females, with an average age of 30.03 ± 0.79 years ranging from 8 to 52 years) with normal or corrected-to-normal eyesight participated in the experiment. total, 53 subjects wore the dry-electrode headband first and 49 subjects wore the wet-electrode headband first. 

    This research designed an online BCI system with a 12-target speller as a virtual keypad of a phone.

    An 8-channel NeuSenW (Neuracle, Ltd. Changzhou, China) wireless EEG acquisition system was used to record the SSVEPs in this study.

    Each block included 12 trials, and each trial corresponded to each target.

    EEG data were recorded using Neuracle EEG Recorder NeuSen W (Neuracle, Ltd.), a wireless EEG acquisition system with a sampling rate of 1000 Hz. Eight electrodes (POz, PO3, PO4, PO5, PO6, Oz, O1 and O2, sorted by channel index in the dataset) were placed at the parietal and occipital regions on the basis of the international 10 to 20 system to record SSVEPs and two electrodes were placed at the forehead as the reference and ground, respectively.

    In accordance with the stimulus onsets recorded in the event channel of the continuous EEG data, data epochs could be extracted. The length of each data epoch was 2.84 s, including 0.5 s before the stimulus onset, 0.14 s for visual response delay, 2 s for stimulus, and 0.2 s after stimulus. With the purpose of reducing the storage and computation costs, all data were down sampled to 250 Hz.

    The electrode impedances recorded before each block were provided in the data matrix of ‘Impedance.mat’ with dimensions of [8, 10, 2, 102]. The channel index are corresponding to POz, PO3, PO4, PO5, PO6, Oz, O1, O2. The numbers in the four dimensions represent the number of channels, blocks, headband types (1: wet, 2: dry) and subjects respectively. The impedance information can be used to study the relationship be tween impedance and BCI performance.

    The “Subjects_information.mat” file lists the information of all 102 subjects together with aquestionnaire on the comfort level and preference of the two headbands after the experiment. For each participant, there are 10 columns of parameters (factors). The first 4 colu mns are the subjects’ personal information including “subject index”, “gender”, “age”, and “dominant hand”. The 6 columns(5th 10th) are listed as results in questionnaires, which are “Comfort of dry electrode headband”, “Wearing time of dry electrode when pain occurs”, “Comfort of wet electrode headband”, “Wearing time of wet electrode when pain occurs”, “Only consider comfort, headband preference” and “comprehensively consider comfort and convenience (need assistance from others, conductive paste, shampoo, etc.), headband preference". The last column shows the order of wearing the two headbands.

    The “stimulation_information.pdf” file lists the stimulation parameters of the 12 characters, including frequency and phase information of each character.

    Total: around 929 MB

    Reference:
    F. Zhu, L. Jiang, G. Dong, X. Gao, and Y. Wang, “An Open Dataset for Wearable SSVEP-Based Brain-Computer Interfaces,” Sensors, vol. 21, no. 4, p. 1256, 2021. DOI: 10.3390/s21041256
    https://www.mdpi.com/1424-8220/21/4/1256
    """

    _CHANNELS = [
        'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2'
    ]

    _FREQS = [
        9.25, 11.25, 13.25, 9.75, 
        11.75, 13.75, 10.25, 12.25, 
        14.25, 10.75, 12.75, 14.75
    ]

    _PHASES = [
        0, 0, 0, 0.5, 
        0.5, 0.5, 1, 1, 
        1, 1.5, 1.5, 1.5
    ]

    _SUBJECTS = [SubInfo(ID = 'S{:d}'.format(sub_idx)) for sub_idx in range(1,102+1,1)]

    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None,
                 ID = 'Wearable (wet)'):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = ID, 
                         url = 'http://bci.med.tsinghua.edu.cn/upload/zhufangkun/', 
                         paths = path, 
                         channels = self._CHANNELS, 
                         srate = 250, 
                         block_num = 10, 
                         trial_num = len(self._FREQS),
                         trial_len = 2.84, 
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = ['Readme.pdf',
                                          'stimulation_information.pdf',
                                          'Subjects_Information.mat',
                                          'Impedance.mat'],
                         path_support_file = path_support_file,
                         t_prestim = 0.5,
                         t_break = 0.2,
                         default_t_latency = 0.14)
    
    def download_single_subject(self,
                                subject: SubInfo):
        sub_idx = int(subject.ID[1:])
        if sub_idx < 10:
            file_name = 'S00{:n}.mat'.format(sub_idx)
        elif sub_idx < 100:
            file_name = 'S0{:n}.mat'.format(sub_idx)
        else:
            file_name = 'S{:n}.mat'.format(sub_idx)
        data_file = os.path.join(subject.path, file_name)

        sub_idx = int(subject.ID[1:])
        if sub_idx <= 10:
            file_name = 'S001-S010.zip'
        elif sub_idx <= 20:
            file_name = 'S011-S020.zip'
        elif sub_idx <= 30:
            file_name = 'S021-S030.zip'
        elif sub_idx <= 40:
            file_name = 'S031-S040.zip'
        elif sub_idx <= 50:
            file_name = 'S041-S050.zip'
        elif sub_idx <= 60:
            file_name = 'S051-S060.zip'
        elif sub_idx <= 70:
            file_name = 'S061-S070.zip'
        elif sub_idx <= 80:
            file_name = 'S071-S080.zip'
        elif sub_idx <= 90:
            file_name = 'S081-S090.zip'
        else:
            file_name = 'S091-S102.zip'
        source_url = self.url + file_name
        desertation = os.path.join(subject.path, file_name)

        download_flag = True

        if not os.path.isfile(data_file):
            try:
                download_single_file(source_url, desertation)

                with zipfile.ZipFile(desertation,'r') as archive:
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

    def download_support_files(self, total_retry_time = 10):
        """
        Download all support files
        """
        if self.support_files is not None and self.path_support_file is not None:
            for file_name in self.support_files:
                download_try_count = 0
                download_flag = False
                while (not download_flag) and (download_try_count < total_retry_time):
                    if download_try_count>0:
                        warnings.warn("There is an error when donwloading '{:s}'. So retry ({:n})".format(file_name, download_try_count))
                    download_flag, source_url, desertation = self.download_file(file_name)
                    if (not download_flag):
                        if os.path.isfile(desertation):
                            os.remove(desertation)
                    download_try_count += 1
                # load subject information
                if download_flag:
                    if file_name == 'Subjects_Information.mat':
                        sub_mat = np.array(loadmat(desertation)['Subjects_Information'], dtype=object)
                        for sub_idx in range(len(self.subjects)):
                            self.subjects[sub_idx].age = sub_mat[sub_idx+1, 2]
                            if sub_mat[sub_idx+1, 1] == 'Male':
                                self.subjects[sub_idx].gender = 'M'
                            else:
                                self.subjects[sub_idx].gender = 'F'
                else:
                    raise ValueError("Cannot download '{:s}'.".format(file_name))

    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))

        sub_info = self.subjects[sub_idx]
        sub_idx = int(sub_info.ID[1:])
        if sub_idx < 10:
            file_name = 'S00{:n}.mat'.format(sub_idx)
        elif sub_idx < 100:
            file_name = 'S0{:n}.mat'.format(sub_idx)
        else:
            file_name = 'S{:n}.mat'.format(sub_idx)
        file_path = os.path.join(sub_info.path, file_name)

        mat_data = loadmat(file_path)
        data = mat_data['data'][:,:,1,:,:] # Only wet
        data = transpose(data, (2,3,0,1))

        return data

    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        return trial_idx

class WearableDataset_dry(WearableDataset_wet):
    """
    Wearable SSVEP Dataset (wet)
    """
    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(path = path,
                        path_support_file = path_support_file,
                        ID = 'Wearable (dry)')
    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        if sub_idx < 0:
            raise ValueError('Subject index cannot be negative')
        if sub_idx > len(self.subjects)-1:
            raise ValueError('Subject index should be smaller than {:d}'.format(len(self.subjects)))

        sub_info = self.subjects[sub_idx]
        sub_idx = int(sub_info.ID[1:])
        if sub_idx < 10:
            file_name = 'S00{:n}.mat'.format(sub_idx)
        elif sub_idx < 100:
            file_name = 'S0{:n}.mat'.format(sub_idx)
        else:
            file_name = 'S{:n}.mat'.format(sub_idx)
        file_path = os.path.join(sub_info.path, file_name)

        mat_data = loadmat(file_path)
        data = mat_data['data'][:,:,0,:,:] # Only dry
        data = transpose(data, (2,3,0,1))

        return data