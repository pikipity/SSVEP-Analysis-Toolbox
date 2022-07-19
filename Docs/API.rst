User API
=======================

.. contents:: Table of Contents

.. role::  raw-html(raw)
    :format: html

Datasets
--------------------

Built-in dataset initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.datasets.benchmarkdataset.BenchmarkDataset

    Initialize the benchmark dataset.

    This dataset gathered SSVEP-BCI recordings of 35 healthy subjects (17 females, aged 17-34 years, mean age: 22 years) focusing on 40 characters flickering at different frequencies (8-15.8 Hz with an interval of 0.2 Hz).

    For each subject, the experiment consisted of 6 blocks. Each block contained 40 trials corresponding to all 40 characters indicated in a random order. Each trial started with a visual cue (a red square) indicating a target stimulus. The cue appeared for 0.5 s on the screen.

    Following the cue offset, all stimuli started to flicker on the screen concurrently and lasted 5 s.

    After stimulus offset, the screen was blank for 0.5 s before the next trial began, which allowed the subjects to have short breaks between consecutive trials.
    Each trial lasted a total of 6 s.

    Total: around 3.45 GB.

    Paper: Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for SSVEP-based braincomputer interfaces,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 10, pp. 17461752, 2017. DOI: `10.1109/TNSRE.2016.2627556 <https://doi.org/10.1109/TNSRE.2016.2627556>`_. 

    URL: `http://bci.med.tsinghua.edu.cn/ <http://bci.med.tsinghua.edu.cn/>`_.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`Benchmark Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`Readme.txt`, :file:`Sub_info.txt`, :file:`64-channels.loc`, and :file:`Freq_Phase.mat`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

.. py:function:: SSVEPAnalysisToolbox.datasets.betadataset.BETADataset

    Initialize the BETA dataset.

    EEG data after preprocessing are store as a 4-way tensor, with a dimension of channel x time point x block x condition. 

    Each trial comprises 0.5-s data before the event onset and 0.5-s data after the time window of 2 s or 3 s. 

    For S1-S15, the time window is 2 s and the trial length is 3 s, whereas for S16-S70 the time window is 3 s and the trial length is 4 s. 

    Additional details about the channel and condition information can be found in the following supplementary information.

    Total: around 4.91 GB.
    
    Paper: B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “BETA: A large benchmark database toward SSVEP-BCI application,” Front. Neurosci., vol. 14, p. 627, 2020. DOI: `10.1109/TNSRE.2016.2627556 <https://doi.org/10.1109/TNSRE.2016.2627556>`_.

    URL: `http://bci.med.tsinghua.edu.cn/ <http://bci.med.tsinghua.edu.cn/>`_.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`BETA Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`note.pdf`, and :file:`description.pdf`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

Parameters of datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All datasets have these parameters. Parameters in different datasets have different values.

:subjects: A list of subject information. Each element is a ``SubInfo`` instance, which contains following parameters:

    :ID: Unique identifier of subject.

    :path: Path of corresponding EEG data file.

    :name: Name of subject.

    :age: Age of subject.

    :gender: Gender of subject. ``M`` for male. ``F`` for female.

:ID: Name/ID of the dataset.

:url: Download URL.

:paths: A list of EEG data path. Each subject has a individual data path.

:channels: A list of channel names

:srate: Sampling rate (Hz)

:block_num: Number of blocks

:trial_len: Signal length (in second) of single trial. If different trials have different siganl length, the shorted signal length is stored. 

:stim_info: A dictionary storing stimulus information, which contains following keys:

    :stim_num: Number of stimuli.

    :freqs: A list of stimulus frequencies.

    :phases: A list of stimulus phases.

:t_prestim: Pre-stimulus time (in second).

:t_break: Time for shifting visual attention (in second).

:support_files: A list of supported files.

:path_support_file: Path of supported files

:default_t_latency: Default/suggested latency time (in second).

Functions of datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All datasets have these functions.

.. py:function:: download_all

    Download all subjects' data file. Because all data files will be donwloaded automatically when a dataset is initialized, this function normally does not need to be run manually.

.. py:function:: download_support_files

    Download all supported files. Because all supported files will be downloaded automatically when a dataset is initialized, this function normally does not need to be run manually.

.. py:function:: reset_preprocess

    Set the preprocess function as the default preprocess function. The default preprocess function is empty. It will directly return the original EEG signals without any preprocessing.

.. py:function:: reset_filterbank

    Set the filterbank function as the default filterbank function. In the default filterbank function, the original EEG signals will be considered as one filterbank. If the original EEG signal is a 2D signal (channels :raw-html:`&#215;` samples), one more dimention will be expanded (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples). If the original EEG signal is a 3D signal, original signal will be returned without any processing. 

.. py:function:: leave_one_block_out

    According to the given testing block index, generate lists of testing and training block indices following the leave-one-block-out rule.  

    .. tip::

        Leave-one-block-out rule: One block works as the testing block. All other blocks work as the training blocks.

    :param block_idx: Given testing block index. 

.. py:function:: get_data

    



