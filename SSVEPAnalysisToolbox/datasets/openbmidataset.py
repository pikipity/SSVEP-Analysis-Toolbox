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

class openBMIDataset(BaseDataset):
    """
    openBMI Dataset

    Fifty-four healthy subjects (ages 24-35, 25 females) participated in the experiment. Thirty-eight subjects were naive BCI users. The others had previous experience with BCI experiments. None of the participants had a history of neurological, psychiatric, or any other pertinent disease that otherwise might have affected the experimental results.

    EEG signals were recorded with a sampling rate of 1000 Hz and collected with 62 Ag/AgCl electrodes.

    Four target SSVEP stimuli were designed to flicker at 5.45, 6.67, 8.57, and 12 Hz and were presented in four positions (down, right, left, and up, respectively) on a monitor. The designed paradigm followed the conventional types of SSVEP-based BCI systems that require four-direction movements [40]. Participants were asked to fixate the center of a black screen and then to gaze in the direction where the target stimulus was highlighted in a different color (see Figure 2-C). Each SSVEP stimulus was presented for 4 s with an ISI of 6 s. Each target frequency was presented 25 times. Therefore, the corrected EEG data had 100 trials (4 classes × 25 trials) in the offline training phase and another 100 trials in the online test phase. Visual feedback was presented in the test phase; the estimated target frequency was highlighted for one second with a red border at the end of each trial.

    Total: around 55.6 GB

    Paper:
    M.-H. Lee, O.-Y. Kwon, Y.-J. Kim, H.-K. Kim, Y.-E. Lee, J. Williamson, S. Fazli, and S.-W. Lee, “EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy,” GigaScience, vol. 8, no. 5, p. giz002, 2019. DOI: 10.1093/gigascience/giz002.
    https://doi.org/10.1093/gigascience/giz002

    Data:
    Lee M; Kwon O; Kim Y; Kim H; Lee Y; Williamson J; Fazli S; Lee S (2019): Supporting data for "EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy" GigaScience Database. DOI: 10.5524/100542.
    http://dx.doi.org/10.5524/100542
    """

    _CHANNELS = [
        'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7',
        'C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3',
        'Pz','P4','P8','PO9','O1','Oz','O2','PO10','FC3','FC4','C5','C1',
        'C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9','FTT9h','TTP7h',
        'TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9','F10',
        'AF7','AF3','AF4','AF8','PO3','PO4'
    ]

    _FREQS = [
        60.0/5.0, 60.0/7.0, 60.0/9.0, 60.0/11.0
    ]

    _PHASES = [
        0, 0, 0, 0
    ]

    _SUBJECTS = [SubInfo(ID = 'S{:d}'.format(sub_idx)) for sub_idx in range(1,54+1,1)]

    def __init__(self, 
                 path: Optional[str] = None,
                 path_support_file: Optional[str] = None):
        super().__init__(subjects = self._SUBJECTS, 
                         ID = 'openBMI', 
                         url = 'ftp://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100542/', 
                         paths = path, 
                         channels = self._CHANNELS, 
                         srate = 1000, 
                         block_num = 4, # 2 condition * 2 sessions
                         trial_num = 100,
                         trial_len = 4, 
                         stim_info = {'stim_num': len(self._FREQS),
                                      'freqs': self._FREQS,
                                      'phases': [i * np.pi for i in self._PHASES]},
                         support_files = ['Questionnaire_results.csv',
                                          'readme.txt',
                                          'random_cell_order.mat',
                                          'OpenBMI-master.zip'],
                         path_support_file = path_support_file,
                         t_prestim = 0,
                         t_break = 0,
                         default_t_latency = 0)
    
    def download_single_subject(self,
                                subject: SubInfo):
        download_flag_store=[]
        source_url_store=[]
        desertation_store=[]
        for sess_idx in range(2):
            data_file = os.path.join(subject.path, 'sess{:n}-{:s}.mat'.format(sess_idx+1, subject.ID))
            sub_idx = int(subject.ID[1:])
            if sub_idx<10:
                file_name = 'session{:n}/s{:n}/sess0{:n}_subj0{:n}_EEG_SSVEP.mat'.format(sess_idx+1, sub_idx, sess_idx+1, sub_idx)
            else:
                file_name = 'session{:n}/s{:n}/sess0{:n}_subj{:n}_EEG_SSVEP.mat'.format(sess_idx+1, sub_idx, sess_idx+1, sub_idx)
            source_url = self.url + file_name
            desertation = data_file

            download_flag = True

            if not os.path.isfile(data_file):
                try:
                    download_single_file(source_url, desertation)
                except:
                    download_flag = False
            
            download_flag_store.append(download_flag)
            source_url_store.append(source_url)
            desertation_store.append(desertation)

        return download_flag_store, source_url_store, desertation_store
    
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
        sub_ID = sub_info.ID

        struct_info = ['EEG_SSVEP_test', 'EEG_SSVEP_train']

        sig_len = int(np.floor(self.trial_len * self.srate))

        data = np.zeros((self.block_num, 100, len(self.channels), sig_len)) # block_num * stimulus_num * ch_num * whole_trial_samples
                                                                          # block sequence: sess0-test, sess0-train, sess1-test, sess1-train
        for sess_idx in range(2):
            file_path = os.path.join(sub_info.path, 'sess{:n}-{:s}.mat'.format(sess_idx+1, sub_ID))
            mat_data = loadmat(file_path)

            for cond_idx in range(len(struct_info)):
                cond_data = mat_data[struct_info[cond_idx]]
                x = cond_data['x']
                t = cond_data['t']
                for t_idx, start_t in enumerate(t):
                    end_t = np.floor(start_t + sig_len)
                    data[sess_idx*2+cond_idx, t_idx, :, :] = x[int(start_t):int(end_t),:].T
        
        return data

    def get_label_trial(self,
                        sub_idx : int,
                        block_idx : int,
                        trials : List[int]):
        """
        Redefine get_label_trial
        """
        if block_idx<2:
            sess_idx = 0
        else:
            sess_idx = 1
        
        sub_info = self.subjects[sub_idx]
        sub_ID = sub_info.ID

        struct_info = ['EEG_SSVEP_test', 'EEG_SSVEP_train']

        file_path = os.path.join(sub_info.path, 'sess{:n}-{:s}.mat'.format(sess_idx+1, sub_ID))
        mat_data = loadmat(file_path)

        cond_idx = block_idx%2
        cond_data = mat_data[struct_info[cond_idx]]
        y_dec = cond_data['y_dec'] 

        return [y_dec[trial_idx]-1 for trial_idx in trials]
        
    
    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               trial_idx: int) -> int:
        if block_idx<2:
            sess_idx = 0
        else:
            sess_idx = 1
        
        sub_info = self.subjects[sub_idx]
        sub_ID = sub_info.ID

        struct_info = ['EEG_SSVEP_test', 'EEG_SSVEP_train']

        file_path = os.path.join(sub_info.path, 'sess{:n}-{:s}.mat'.format(sess_idx+1, sub_ID))
        mat_data = loadmat(file_path)

        cond_idx = block_idx%2
        cond_data = mat_data[struct_info[cond_idx]]
        y_dec = cond_data['y_dec'] 

        return y_dec[trial_idx]-1
                        



