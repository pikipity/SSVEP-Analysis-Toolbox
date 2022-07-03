# -*- coding: utf-8 -*-
"""
Base class of SSVEP datasets
"""

import abc
from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray, floor

from .subjectinfo import SubInfo
from ..utils.algsupport import gen_ref_sin

class BaseDataset(metaclass=abc.ABCMeta):
    """
    BaseDataset
    """
    def __init__(self, 
                 subjects: List[SubInfo],
                 ID: str,
                 url: str,
                 paths: Union(str, List[str]),
                 channels: List[str],
                 srate: int,
                 block_num: int,
                 trial_len: float,
                 stim_info: Dict[int, List[float], List[float]],
                 support_files: Optional[List[str]] = None,
                 path_support_file: Optional[str] = None,
                 default_t_latency: Optional[float] = 0):
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
            
        paths: Union(str, List[str])
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
            
        support_files: Optional[List[str]]
            List of support file names
            
        path_support_file: Optional[str]
            Local path for storing support files
            
        default_t_latency: Optional[float]
            Default latency time
            
        Raises
        -----------------------
        ValueError
            raise error if lengths of subjects and paths are not equal
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
        
        # Set paths for subjects
        for sub_idx, path in enumerate(paths):
            subjects[sub_idx].path = path
        
        # download dataset
        self.download_all()
        
        # download support files
        self.download_support_files()
    
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
            
    def get_data_all_stim(self,
                          sub_idx: int,
                          blocks: List[int],
                          sig_len: float,
                          t_latency: Optional[float] = None) -> Tuple[List[ndarray], List[int]]:
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
        sig_len : float
            signal length (in second)
        t_latency : Optional[float]
            latency time (in second)

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
            
        sub_data = self.get_sub_data(sub_idx)
        
        X = [self.get_data_single_trial(sub_data, block_idx, stim_idx, sig_len, t_latency) for block_idx in blocks for stim_idx in range(self.stim_info['stim_num'])]
        Y = [stim_idx for block_idx in blocks for stim_idx in range(self.stim_info['stim_num'])]
        
        return X, Y
    
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
        sig_len : float
            signal length (in second)
        t_latency : float
            latency time (in second)

        Returns
        -------
        single_trial_data: ndarray
            ch_num * (sig_len * self.srate)

        """
        if block_idx < 0:
            raise ValueError('Block index cannot be negative')
        if block_idx > self.block_num-1:
            raise ValueError('Block index should be smaller than {:d}'.format(self.block_num))
            
        if stim_idx < 0:
            raise ValueError('Stimulus index cannot be negative')
        if stim_idx > self.stim_info['stim_num']-1:
            raise ValueError('Stimulus index should be smaller than {:d}'.format(self.stim_info['stim_num']))
            
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
            
        t_latency = floor(t_latency * self.srate)
        sig_len = floor(sig_len * self.srate)
        start_t_idx = t_latency + 1
        end_t_idx = t_latency + sig_len
        
        return sub_data[block_idx,stim_idx,channels,start_t_idx:end_t_idx]
    
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
    
    
    
    