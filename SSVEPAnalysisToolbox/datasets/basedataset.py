# -*- coding: utf-8 -*-
"""
Base class of SSVEP datasets
"""

import abc
from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray, expand_dims

import random
random.seed()

from .subjectinfo import SubInfo
from ..utils.algsupport import gen_ref_sin, floor

class BaseDataset(metaclass=abc.ABCMeta):
    """
    BaseDataset
    """
    def __init__(self, 
                 subjects: List[SubInfo],
                 ID: str,
                 url: str,
                 paths: Union[str, List[str]],
                 channels: List[str],
                 srate: int,
                 block_num: int,
                 trial_len: float,
                 stim_info: Dict[str, Union[int, List[float]]],
                 t_prestim: float,
                 t_break: float,
                 support_files: Optional[List[str]] = None,
                 path_support_file: Optional[str] = None,
                 default_t_latency: float = 0):
        """
        Parameters required for all datasets
        
        Parameters
        -------------------
        subjects: List[SubInfo]
            List of subjects (elements stores subject information)
            
        ID: str
            Unique identifier for dataset
            
        url: str
            Download url
            
        paths: Union[str, List[str]]
            Local paths for storing data of all subjects
            
        channels: list[str]
            List of channel names
            
        srate: int
            Sampling rate
            
        block_num: int
            Number of blocks
            
        trial_len: float
            Signal length of single trial (in second)
            
        stim_info: Dict[int, List[float]], List[float]]
            Stimulus information. the format is
            {
                stim_num: int
                freqs: List[float]
                phases: List[float]
            }
            stim_num: Stimulus number
            freqs: Stimulus Frequencies
            phases: Stimulus phases
            
        t_prestim: float
            Pre-stimulus time
            
        t_break: float
            Time required for shifting visual attention
            
        support_files: Optional[List[str]]
            List of support file names
            
        path_support_file: Optional[str]
            Local path for storing support files
            
        default_t_latency: Optional[float]
            Default latency time
        """
        if type(paths) is str:
            paths = [paths for _ in range(len(subjects))]
        if len(subjects) != len(paths):
            raise ValueError('Lengths of subjects and paths are not equal. ')
        
        self.subjects = subjects
        self.ID = ID
        self.url = url
        self.channels = [ch.upper() for ch in channels]
        self.srate = srate
        self.block_num = block_num
        self.stim_info = stim_info
        self.trial_len = trial_len
        self.support_files = support_files
        self.path_support_file = path_support_file
        self.default_t_latency = default_t_latency
        self.t_prestim = t_prestim
        
        # Set paths for subjects
        for sub_idx, path in enumerate(paths):
            subjects[sub_idx].path = path
        
        # download dataset
        self.download_all()
        
        # download support files
        self.download_support_files()
        
        # regist default preprocess and filterbank functions
        def default_preprocess(X: ndarray) -> ndarray:
            """
            default preprocess (do nothing)
            """
            return X
        
        def default_filterbank(X: ndarray) -> ndarray:
            """
            default filterbank (1 filterbank contains original signal)
            """
            return expand_dims(X,0)
        
        self.regist_preprocess(default_preprocess)
        self.regist_filterbank(default_filterbank)
    
    def download_all(self):
        """
        Download all subjects' data
        """
        for subject in self.subjects:
            self.download_single_subject(subject)
            
    def download_support_files(self):
        """
        Download all support files
        """
        if self.support_files is not None and self.path_support_file is not None:
            for file_name in self.support_files:
                self.download_file(file_name)
    
    def leave_one_block_out(self,
                            block_idx: int) -> Tuple[List[int], List[int]]:
        """
        Generate testing and training blocks for specific block based on leave-one-out rule

        Parameters
        ----------
        block_idx : int
            Specific block index

        Returns
        -------
        test_block: List[int]
            Testing block
        train_block: List[int]
            Training block
        """
        if block_idx < 0:
            raise ValueError('Block index cannot be negative')
        if block_idx > self.block_num-1:
            raise ValueError('Block index should be smaller than {:d}'.format(self.block_num))
            
        test_block = [block_idx]
        train_block = [i for i in range(self.block_num)]
        train_block.remove(block_idx)
        
        return test_block, train_block
    
    def get_data(self,
                 sub_idx: int,
                 blocks: List[int],
                 trials: List[int],
                 channels: List[int],
                 sig_len: float,
                 t_latency: Optional[float] = None,
                 shuffle: Optional[bool] = False) -> Tuple[List[ndarray], List[int]]:
        """
        Construct data, corresponding labels, and sine-cosine-based reference signals 
        from one subject (specific stimuli)

        Parameters
        ----------
        sub_idx : int
            Subject index
        blocks:
            List of block indx
            Note - The index of the 1st block is 0
        trials : List[int]
            List of trial indx
            Note - The index of the 1st trial is 0
        channels: List[int]
            List of channels
        sig_len : float
            signal length (in second)
        t_latency : Optional[float]
            latency time (in second)
        shuffle : Optional[bool]
            Whether shuffle trials

        Returns
        -------
        X: List[ndarray]
            List of single trial data
        Y: List[int]
            List of corresponding label (stimulus idx)
            Note - The index of the 1st stimulus is 0
        """
        if t_latency is None:
            t_latency = self.default_t_latency
        if type(blocks) is not list:
            blocks = [blocks]
        if type(trials) is not list:
            trials = [trials]
            
        sub_data = self.get_sub_data(sub_idx)
        
        X = [self.get_data_single_trial(sub_data, block_idx, stim_idx, channels, sig_len, t_latency) for block_idx in blocks for stim_idx in trials]
        Y = [self.get_label_single_trial(sub_idx,block_idx, stim_idx) for block_idx in blocks for stim_idx in trials]
        
        trial_seq = [i for i in range(len(X))]
        if shuffle:
            random.shuffle(trial_seq)
        
        return [X[i] for i in trial_seq], [Y[i] for i in trial_seq]
    
    def get_data_all_stim(self,
                          sub_idx: int,
                          blocks: List[int],
                          channels: List[int],
                          sig_len: float,
                          t_latency: Optional[float] = None,
                          shuffle: Optional[bool] = False) -> Tuple[List[ndarray], List[int]]:
        """
        Construct data, corresponding labels, and sine-cosine-based reference signals 
        from one subject (all stimuli)

        Parameters
        ----------
        sub_idx : int
            Subject index
        blocks:
            List of block indx
            Note - The index of the 1st block is 0
        channels: List[int]
            List of channels
        sig_len : float
            signal length (in second)
        t_latency : Optional[float]
            latency time (in second)
        shuffle : Optional[bool]
            Whether shuffle trials

        Returns
        -------
        X: List[ndarray]
            List of single trial data
        Y: List[int]
            List of corresponding label (stimulus idx)
            Note - The index of the 1st stimulus is 0
        """
        if type(blocks) is not list:
            blocks = [blocks]
        trials = [i for i in range(self.stim_info['stim_num'])]
            
        X, Y = self.get_data(sub_idx, blocks, trials, channels, sig_len, t_latency)
        
        trial_seq = [i for i in range(len(X))]
        if shuffle:
            random.shuffle(trial_seq)
        
        return [X[i] for i in trial_seq], [Y[i] for i in trial_seq]
    
    def get_data_single_trial(self,
                             sub_data: ndarray,
                             block_idx: int,
                             stim_idx: int,
                             channels: List[int],
                             sig_len: float,
                             t_latency: float) -> ndarray:
        """
        Get single trial data

        Parameters
        ----------
        sub_data : ndarray
            Subject data
            block_num * stimulus_num * ch_num * whole_trial_samples
        block_idx : int
            Block index
        stim_idx : int
            Stimulus index
        channels: List[int]
            List of channels
        sig_len : float
            signal length (in second)
        t_latency : float
            latency time (in second)

        Returns
        -------
        single_trial_data: ndarray
            filterbank_num * ch_num * (sig_len * self.srate)

        """
        if block_idx < 0:
            raise ValueError('Block index cannot be negative')
        if block_idx > self.block_num-1:
            raise ValueError('Block index should be smaller than {:d}'.format(self.block_num))
            
        if stim_idx < 0:
            raise ValueError('Stimulus index cannot be negative')
        if stim_idx > self.stim_info['stim_num']-1:
            raise ValueError('Stimulus index should be smaller than {:d}'.format(self.stim_info['stim_num']))
        
        if type(channels) is not list:
            channels = [channels]
        min_ch_idx = min(channels)
        if min_ch_idx < 0:
            raise ValueError('Channel index cannot be negative')
        max_ch_idx = max(channels)
        if max_ch_idx > len(self.channels)-1:
            raise ValueError('Channel index should be smaller than {:d}'.format(len(self.channels)))
            
        if sig_len < 0 or t_latency < 0:
            raise ValueError('Time cannot be negative')
        if sig_len + t_latency > self.trial_len:
            raise ValueError('Total time length cannot be larger than single trial time')
            
        t_pre = floor((self.t_prestim + t_latency) * self.srate)
        t_prestim = floor(self.t_prestim * self.srate)
        t_latency = floor(t_latency * self.srate)
        sig_len = floor(sig_len * self.srate)
        
        start_t_idx = t_prestim
        end_t_idx = t_pre + sig_len
        eeg_total = sub_data[block_idx,stim_idx,channels,start_t_idx:end_t_idx]
        # end_t_idx = eeg_total.shape[-1]
        eeg_total = self.preprocess_fun(eeg_total)
        eeg_total = self.filterbank_fun(eeg_total)
        
        start_t_idx = t_latency
        end_t_idx = t_latency + sig_len
        eeg_trial = eeg_total[:,:,start_t_idx:end_t_idx]
        
        return eeg_trial
    
    def get_ref_sig(self,
                    sig_len: float,
                    N: int) -> List[ndarray]:
        """
        Construct sine-cosine-based reference signals  for all stimuli
        
        Parameters
        ----------
        sig_len : float
            signal length (in second)
        N : int
            Number of harmonics

        Returns
        -------
        ref_sig : List[ndarray]
            List of reference signals
        """
        L = floor(sig_len * self.srate)
        ref_sig = [gen_ref_sin(freq, self.srate, L, N, phase) for freq, phase in zip(self.stim_info['freqs'], self.stim_info['phases'])]
        return ref_sig
    
    def regist_preprocess(self,
                          preprocess_fun: Callable[[ndarray],ndarray]):
        """
        Regist preprocess function

        Parameters
        ----------
        preprocess_fun : Callable[[ndarray],ndarray]
            preprocess function
            
            Parameters
            ----------
            X : ndarray
                EEG signal (including latency time and desired signal window, but pre-stimulus time has been removed)
                ch_num * signal_length
                
            Returns
            ----------
            preprocess_X: ndarray
        """
        self.preprocess_fun = preprocess_fun
    
    def regist_filterbank(self,
                          filterbank_fun: Callable[[ndarray],ndarray]):
        """
        Regist filterbank function

        Parameters
        ----------
        filterbank_fun : Callable[[ndarray],ndarray]
            filterbank function
            
            Parameters
            ----------
            X : ndarray
                EEG signal after preprocess (including latency time and desired signal window, but pre-stimulus time has been removed)
                ch_num * signal_length

            Returns
            -------
            filterbank_X: ndarray
                filterbanks
        """
        self.filterbank_fun = filterbank_fun
        
            
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        desc = """Dataset: {:s}:\n   Subjects: {:d}\n   Srate: {:d}\n   Channels: {:d}\n   Blocks: {:d}\n   Stimuli: {:d}\n   Signal length: {:.3f}\n""".format(
            self.ID,
            len(self.subjects),
            self.srate,
            len(self.channels),
            self.block_num,
            self.stim_info['stim_num'],
            self.trial_len
            )
        return desc
    
    @abc.abstractclassmethod
    def download_single_subject(self,
                                subject: SubInfo):
        """
        Download single subject's data
        
        parameters
        --------------
        subject: SubInfo
            Subject information

        """
        pass
    
    @abc.abstractclassmethod
    def download_file(self,
                      file_name: str):
        """
        Download specific file

        Parameters
        ----------
        file_name : str
            File name
        """
        pass
    
    
    @abc.abstractclassmethod
    def get_sub_data(self, 
                     sub_idx: int) -> ndarray:
        """
        Get single subject data

        Parameters
        ----------
        sub_idx : int
            Subject index

        Returns
        -------
        sub_data : ndarray
            Subject data
            block_num * stimulus_num * ch_num * whole_trial_samples
        """
        pass
    
    @abc.abstractclassmethod
    def get_label_single_trial(self,
                               sub_idx: int,
                               block_idx: int,
                               stim_idx: int) -> int:
        """
        Get the label of single trial

        Parameters
        ----------
        sub_idx : int
            Subject index
        block_idx : int
            Block index
        stim_idx : int
            Trial index

        Returns
        -------
        label: int
            Label of stimulus
        """
        pass
    
    
    