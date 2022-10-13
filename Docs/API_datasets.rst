.. role::  raw-html(raw)
    :format: html

Datasets
--------------------

Built-in dataset initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmark Dataset
=====================

This dataset gathered SSVEP-BCI recordings of 35 healthy subjects (17 females, aged 17-34 years, mean age: 22 years) focusing on 40 characters flickering at different frequencies (8-15.8 Hz with an interval of 0.2 Hz).

For each subject, the experiment consisted of 6 blocks. Each block contained 40 trials corresponding to all 40 characters indicated in a random order. Each trial started with a visual cue (a red square) indicating a target stimulus. The cue appeared for 0.5 s on the screen.

Following the cue offset, all stimuli started to flicker on the screen concurrently and lasted 5 s.

After stimulus offset, the screen was blank for 0.5 s before the next trial began, which allowed the subjects to have short breaks between consecutive trials.
Each trial lasted a total of 6 s.

.. py:function:: SSVEPAnalysisToolbox.datasets.BenchmarkDataset

    Initialize the benchmark dataset.

    =============================   =======
    Num. of Subjects                35
    -----------------------------   -------
    Num. of Targets                 40
    -----------------------------   -------
    Num. of Channels                64
    -----------------------------   -------
    Num. of Blocks                  6
    -----------------------------   -------
    Num. of trials in each blcok    40
    -----------------------------   -------
    Pre-stimulus Time               0.5 s
    -----------------------------   -------
    Break Time                      0.5 s
    -----------------------------   -------
    Whole Trial Length              6 s
    -----------------------------   -------
    Default Latency Time            0.14 s
    -----------------------------   -------
    Sampling rate                   250 Hz
    =============================   =======

    .. image:: ./_static/benchmarkdataset.png

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

BETA Dataset
================

EEG data after preprocessing are store as a 4-way tensor, with a dimension of channel x time point x block x condition. 

Each trial comprises 0.5-s data before the event onset and 0.5-s data after the time window of 2 s or 3 s. 

For S1-S15, the time window is 2 s and the trial length is 3 s, whereas for S16-S70 the time window is 3 s and the trial length is 4 s. 

Additional details about the channel and condition information can be found in the supplementary information.

.. py:function:: SSVEPAnalysisToolbox.datasets.BETADataset

    Initialize the BETA dataset.

    =============================   =======
    Num. of Subjects                70
    -----------------------------   -------
    Num. of Targets                 40
    -----------------------------   -------
    Num. of Channels                64
    -----------------------------   -------
    Num. of Blocks                  4
    -----------------------------   -------
    Num. of trials in each blcok    40
    -----------------------------   -------
    Pre-stimulus Time               0.5 s
    -----------------------------   -------
    Break Time                      0.5 s
    -----------------------------   -------
    Whole Trial Length              2 s
    -----------------------------   -------
    Default Latency Time            0.13 s
    -----------------------------   -------
    Sampling rate                   250 Hz
    =============================   =======

    .. image:: ./_static/beta.png

    Total: around 4.91 GB.
    
    Paper: B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “BETA: A large benchmark database toward SSVEP-BCI application,” Front. Neurosci., vol. 14, p. 627, 2020. DOI: `10.3389/fnins.2020.00627 <https://doi.org/10.3389/fnins.2020.00627>`_.

    URL: `http://bci.med.tsinghua.edu.cn/ <http://bci.med.tsinghua.edu.cn/>`_.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`BETA Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`note.pdf`, and :file:`description.pdf`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

Nakanishi 2015 Dataset
============================

Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject. 
Please see the reference paper for the detail.

size(eeg) = [Num. of targets, Num. of channels, Num. of sampling points, Num. of trials]

+ The order of the stimulus frequencies in the EEG data: [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz (e.g., eeg(1,:,:,:) and eeg(5,:,:,:) are the EEG data while a subject was gazing at the visual stimuli flickering at 9.25 Hz and 11.75Hz, respectively.)

+ The onset of visual stimulation is at 39th sample point.

.. py:function:: SSVEPAnalysisToolbox.datasets.NakanishiDataset

    Initialize the Nakanishi2015 dataset.

    =============================   ====================
    Num. of Subjects                10
    -----------------------------   --------------------
    Num. of Targets                 12
    -----------------------------   --------------------
    Num. of Channels                8
    -----------------------------   --------------------
    Num. of Blocks                  15
    -----------------------------   --------------------
    Num. of trials in each blcok    12
    -----------------------------   --------------------
    Pre-stimulus Time               0.1523 (39/256) s
    -----------------------------   --------------------
    Break Time                      1 s
    -----------------------------   --------------------
    Whole Trial Length              4.3516 (1114/256) s
    -----------------------------   --------------------
    Default Latency Time            0.135 s
    -----------------------------   --------------------
    Sampling rate                   256 Hz
    =============================   ====================

    .. image:: ./_static/Nakanishi2015.png

    Total: around 148 MB.
    
    Paper: M. Nakanishi, Y. Wang, Y.-T. Wang, T.-P. Jung, "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," *PLoS ONE*, vol. 10, p. e0140703, 2015. DOI: `10.1371/journal.pone.0140703 <https://doi.org/10.1371/journal.pone.0140703>`_.

    URL: ``ftp://sccn.ucsd.edu/pub/cca_ssvep.zip <ftp://sccn.ucsd.edu/pub/cca_ssvep.zip``.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`Nakanishi2015 Dataset` in the working path. 

eldBETA Dataset
=====================

For the BCI users, there was an associated epoched record that is stored in ".mat" structure array from MATLAB. 
    
The structure array in each record was composed of the EEG data ("EEG") and its associated supplementary information ("Suppl_info") as its fields. In the "EEG" field of the record, two types of EEG data, i.e., EEG epochs and raw EEG were provided for researchers to facilitate diverse research purposes. 

The EEG epochs were the EEG data with the data processing and stored as 4-dimensional matrices (channel x time point x condition x block). The names and locations of the channel dimension were given in the supplementary information. 

For the dimension of time point, the epochs had a length of 6 s, which included 0.5 s before the stimulus onset, 5 s during the stimulation (SSVEPs) and 0.5 s after the stimulus offset. 

Different from the epoched data, the raw EEG provided continuous EEG that were converted by EEGLAB. The raw EEG were stored as cell arrays, each of which contained a block of EEG data. The "Suppl_info" field of the record provided a basic information about personal statistics and experimental protocol. The personal statistics included the aged, gender, BCIQ and SNR with respect to each subject. The experimental protocol included channel location ("Channel), stimulus frequency ("Frequency"), stimulus initial phase ("Phase") and sampling rate ("Srate"). The channel location was represented by a 64x4 cell arrays. The first column and the fourth column denoted the channel index and channel name, respectively. The second column and the third column denoted the channel location in polar coordinates, i.e., degree and radius, respectively. The stimulus initial phase was given in radius. The sampling rate of the epoch data was denoted by "Srate". 

.. py:function:: SSVEPAnalysisToolbox.datasets.ELDBETADataset

    Initialize the eldBETA dataset.

    =============================   ====================
    Num. of Subjects                100
    -----------------------------   --------------------
    Num. of Targets                 9
    -----------------------------   --------------------
    Num. of Channels                64
    -----------------------------   --------------------
    Num. of Blocks                  7
    -----------------------------   --------------------
    Num. of trials in each blcok    9
    -----------------------------   --------------------
    Pre-stimulus Time               0.5 s
    -----------------------------   --------------------
    Break Time                      0.5 s
    -----------------------------   --------------------
    Whole Trial Length              6 s
    -----------------------------   --------------------
    Default Latency Time            0.14 s
    -----------------------------   --------------------
    Sampling rate                   250 Hz
    =============================   ====================

    .. image:: ./_static/eldBETA.png

    Total: around 20.0 GB

    Paper: B. Liu, Y. Wang, X. Gao, and X. Chen, "eldBETA: A Large eldercare-oriented benchmark database of SSVEP-BCI for the aging population," Scientific Data, vol. 9, no. 1, pp.1-12, 2022. DOI: `10.1038/s41597-022-01372-9 <https://www.nature.com/articles/s41597-022-01372-9>`_. 

    URL: `http://bci.med.tsinghua.edu.cn/ <http://bci.med.tsinghua.edu.cn/>`_.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`BETA Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`note.pdf`, and :file:`description.pdf`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

openBMI Dataset
=====================

Fifty-four healthy subjects (ages 24-35, 25 females) participated in the experiment. Thirty-eight subjects were naive BCI users. The others had previous experience with BCI experiments. None of the participants had a history of neurological, psychiatric, or any other pertinent disease that otherwise might have affected the experimental results.

EEG signals were recorded with a sampling rate of 1000 Hz and collected with 62 Ag/AgCl electrodes.

Four target SSVEP stimuli were designed to flicker at 5.45, 6.67, 8.57, and 12 Hz and were presented in four positions (down, right, left, and up, respectively) on a monitor. The designed paradigm followed the conventional types of SSVEP-based BCI systems that require four-direction movements. Participants were asked to fixate the center of a black screen and then to gaze in the direction where the target stimulus was highlighted in a different color. Each SSVEP stimulus was presented for 4 s with an ISI of 6 s. Each target frequency was presented 25 times. Therefore, the corrected EEG data had 100 trials (4 classes × 25 trials) in the offline training phase and another 100 trials in the online test phase. Visual feedback was presented in the test phase; the estimated target frequency was highlighted for one second with a red border at the end of each trial.

.. py:function:: SSVEPAnalysisToolbox.datasets.openBMIDataset

    Initialize the openBMI dataset.

    =============================   ===============================================
    Num. of Subjects                54
    -----------------------------   -----------------------------------------------
    Num. of Targets                 4
    -----------------------------   -----------------------------------------------
    Num. of Channels                62
    -----------------------------   -----------------------------------------------
    Num. of Blocks                  4 (2 sessions * (online part + offline part))
    -----------------------------   -----------------------------------------------
    Num. of trials in each blcok    100
    -----------------------------   -----------------------------------------------
    Pre-stimulus Time               0 s
    -----------------------------   -----------------------------------------------
    Break Time                      0 s
    -----------------------------   -----------------------------------------------
    Whole Trial Length              4 s
    -----------------------------   -----------------------------------------------
    Default Latency Time            0 s
    -----------------------------   -----------------------------------------------
    Sampling rate                   1000 Hz
    =============================   ===============================================

    .. image:: ./_static/openbmi.png

    Total: around 55.6 GB

    Paper:
    M.-H. Lee, O.-Y. Kwon, Y.-J. Kim, H.-K. Kim, Y.-E. Lee, J. Williamson, S. Fazli, and S.-W. Lee, "EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy," GigaScience, vol. 8, no. 5, p. giz002, 2019. DOI: `10.1093/gigascience/giz002 <https://doi.org/10.1093/gigascience/giz002>`_.

    Data:
    M. Lee, O. Kwon, Y. Kim, H. Kim, Y. Lee, J. Williamson, S. Fazli, S. Lee, "Supporting data for 'EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy'," GigaScience Database, 2019. DOI: `10.5524/100542 <http://dx.doi.org/10.5524/100542>`_.

    URL: ``ftp://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100542/``.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`BETA Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`note.pdf`, and :file:`description.pdf`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

Wearable SSVEP Dataset
==========================

This dataset relied on the BCI Brain-Controlled Robot Contest at the 2020 World Robot Contest to recruit participants.

One hundred and two healthy subjects (64 males and 38 females, with an average age of 30.03 ± 0.79 years ranging from 8 to 52 years) with normal or corrected-to-normal eyesight participated in the experiment. total, 53 subjects wore the dry-electrode headband first and 49 subjects wore the wet-electrode headband first. 

This research designed an online BCI system with a 12-target speller as a virtual keypad of a phone.

An 8-channel NeuSenW (Neuracle, Ltd. Changzhou, China) wireless EEG acquisition system was used to record the SSVEPs in this study.

Each block included 12 trials, and each trial corresponded to each target.

EEG data were recorded using Neuracle EEG Recorder NeuSen W (Neuracle, Ltd.), a wireless EEG acquisition system with a sampling rate of 1000 Hz. Eight electrodes (POz, PO3, PO4, PO5, PO6, Oz, O1 and O2, sorted by channel index in the dataset) were placed at the parietal and occipital regions on the basis of the international 10 to 20 system to record SSVEPs and two electrodes were placed at the forehead as the reference and ground, respectively.

In accordance with the stimulus onsets recorded in the event channel of the continuous EEG data, data epochs could be extracted. The length of each data epoch was 2.84 s, including 0.5 s before the stimulus onset, 0.14 s for visual response delay, 2 s for stimulus, and 0.2 s after stimulus. With the purpose of reducing the storage and computation costs, all data were down sampled to 250 Hz.

The electrode impedances recorded before each block were provided in the data matrix of ‘Impedance.mat’ with dimensions of [8, 10, 2, 102]. The channel index are corresponding to POz, PO3, PO4, PO5, PO6, Oz, O1, O2. The numbers in the four dimensions represent the number of channels, blocks, headband types (1: wet, 2: dry) and subjects respectively. The impedance information can be used to study the relationship be tween impedance and BCI performance.

The “Subjects_information.mat” file lists the information of all 102 subjects together with aquestionnaire on the comfort level and preference of the two headbands after the experiment. For each participant, there are 10 columns of parameters (factors). The first 4 colu mns are the subjects’ personal information including “subject index”, “gender”, “age”, and “dominant hand”. The 6 columns(5th 10th) are listed as results in questionnaires, which are “Comfort of dry electrode headband”, “Wearing time of dry electrode when pain occurs”, “Comfort of wet electrode headband”, “Wearing time of wet electrode when pain occurs”, “Only consider comfort, headband preference” and “comprehensively consider comfort and convenience (need assistance from others, conductive paste, shampoo, etc.), headband preference". The last column shows the order of wearing the two headbands.

The “stimulation_information.pdf” file lists the stimulation parameters of the 12 characters, including frequency and phase information of each character.

.. py:function:: SSVEPAnalysisToolbox.datasets.WearableDataset_wet

    Initialize the wearable dataset (wet electrodes).

    =============================   ===============================================
    Num. of Subjects                102
    -----------------------------   -----------------------------------------------
    Num. of Targets                 12
    -----------------------------   -----------------------------------------------
    Num. of Channels                8
    -----------------------------   -----------------------------------------------
    Num. of Blocks                  10
    -----------------------------   -----------------------------------------------
    Num. of trials in each blcok    12
    -----------------------------   -----------------------------------------------
    Pre-stimulus Time               0.5 s
    -----------------------------   -----------------------------------------------
    Break Time                      0.2 s
    -----------------------------   -----------------------------------------------
    Whole Trial Length              2.84 s
    -----------------------------   -----------------------------------------------
    Default Latency Time            0.14 s
    -----------------------------   -----------------------------------------------
    Sampling rate                   250 Hz
    =============================   ===============================================

    .. image:: ./_static/wearable.png

    Total: around 929 MB

    Paper:
    F. Zhu, L. Jiang, G. Dong, X. Gao, and Y. Wang, “An Open Dataset for Wearable SSVEP-Based Brain-Computer Interfaces,” Sensors, vol. 21, no. 4, p. 1256, 2021. DOI: `10.3390/s21041256 <https://www.mdpi.com/1424-8220/21/4/1256>`_.

    URL: `http://bci.med.tsinghua.edu.cn/ <http://bci.med.tsinghua.edu.cn/>`_..

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`BETA Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`note.pdf`, and :file:`description.pdf`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

.. py:function:: SSVEPAnalysisToolbox.datasets.WearableDataset_dry

    Initialize the wearable dataset (dry electrodes).

    =============================   ===============================================
    Num. of Subjects                102
    -----------------------------   -----------------------------------------------
    Num. of Targets                 12
    -----------------------------   -----------------------------------------------
    Num. of Channels                8
    -----------------------------   -----------------------------------------------
    Num. of Blocks                  10
    -----------------------------   -----------------------------------------------
    Num. of trials in each blcok    12
    -----------------------------   -----------------------------------------------
    Pre-stimulus Time               0.5 s
    -----------------------------   -----------------------------------------------
    Break Time                      0.2 s
    -----------------------------   -----------------------------------------------
    Whole Trial Length              2.84 s
    -----------------------------   -----------------------------------------------
    Default Latency Time            0.14 s
    -----------------------------   -----------------------------------------------
    Sampling rate                   250 Hz
    =============================   ===============================================

    .. image:: ./_static/wearable.png

    Total: around 929 MB

    Paper:
    F. Zhu, L. Jiang, G. Dong, X. Gao, and Y. Wang, “An Open Dataset for Wearable SSVEP-Based Brain-Computer Interfaces,” Sensors, vol. 21, no. 4, p. 1256, 2021. DOI: `10.3390/s21041256 <https://www.mdpi.com/1424-8220/21/4/1256>`_.

    URL: `http://bci.med.tsinghua.edu.cn/ <http://bci.med.tsinghua.edu.cn/>`_..

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`BETA Dataset` in the working path. 

    :param path_support_file: Path of supported files, i.e., :file:`note.pdf`, and :file:`description.pdf`. 
    
        The missing supported files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
        
        Default path is same as data path ``path``.

Attributes of datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All datasets have these attributes. Attributes in different datasets have different values.

:subjects: A list of subject information. Each element is a ``SubInfo`` instance, which contains following attributes:

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

:trial_num: Number of trials in each block

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

Methods of datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all datasets, the toolbox will the unified APIs to hook the proprocessing and filterbank functions and output signals. The unified APIs are listed here:

.. py:function:: download_all

    Download all subjects' data file. Because all data files will be donwloaded automatically when a dataset is initialized, this function normally does not need to be run manually.

.. py:function:: download_support_files

    Download all supported files. Because all supported files will be downloaded automatically when a dataset is initialized, this function normally does not need to be run manually.

.. py:function:: reset_preprocess

    Set the preprocess function as the default preprocess function. The default preprocess function is empty. It will directly return the original EEG signals without any preprocessing.

.. py:function:: regist_preprocess

    Hook the user-defined preprocessing function. 

    :param preprocess_fun: User-defined preprocessing function.

    .. note::

        The given ``preprocess_fun`` should be a callable function name (only name). This callable function should only have two input parameter ``dataself`` and ``X``. 
        
        + ``dataself`` is the data istance. If you need to use parameters in the data module, you can directly use them from ``dataself``. 
        + ``X`` is a 2D EEG signal (channels :raw-html:`&#215;` samples). The pre-stimulus time has been removed from the EEG signal. The latency time is maintained in the EEG signal. The detailed data extraction procedures please refer to `"get_data" function <#get_data>`_.
        
        If your preprocess function needs other input parameters, you may use `lambda function <https://www.w3schools.com/python/python_lambda.asp>`_. Check demos to get more hints.

        You may refer the following default preprocess function to define your own function.

    .. code-block:: python
        :linenos:

        def default_preprocess(dataself, X: ndarray) -> ndarray:
            return X

.. py:function:: reset_filterbank

    Set the filterbank function as the default filterbank function. In the default filterbank function, the original EEG signals will be considered as one filterbank. If the original EEG signal is a 2D signal (channels :raw-html:`&#215;` samples), one more dimention will be expanded (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples). If the original EEG signal is a 3D signal, original signal will be returned without any processing. 

.. py:function:: regist_filterbank

    Hook the user-defined filterbank function.

    :param filterbank_fun: User-defined filterbank function.

    .. note::

        The given ``filterbank_fun`` should be a callable function name (only name). This callable function should only have two input parameter ``dataself`` and ``X``. 
        
        + ``dataself`` is the data istance. If you need to use parameters in the data module, you can directly use them from ``dataself``.
        + ``X`` is a 2D EEG signal (channels :raw-html:`&#215;` samples). The pre-stimulus time has been removed from the EEG signal. The latency time is maintained in the EEG signal. The detailed data extraction procedures please refer to `"get_data" function <#get_data>`_.

        The output of the given ``filterbank_fun`` should be a 3D EEG signal (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples). The bandpass filtered EEG signals of filterbanks should be stored in the first dimension. 

        If your filterbank function needs other input parameters, you may use `lambda function <https://www.w3schools.com/python/python_lambda.asp>`_. Check demos to get more hints.

        You may refer the following default preprocess function to define your own function.

    .. code-block:: python
        :linenos:

        def default_filterbank(dataself, X: ndarray) -> ndarray:
            """
            default filterbank (1 filterbank contains original signal)
            """
            if len(X.shape) == 2:
                return expand_dims(X,0)
            elif len(X.shape) == 3:
                return X
            else:
                raise ValueError("The shapes of EEG signals are not correct")

.. py:function:: leave_one_block_out

    According to the given testing block index, generate lists of testing and training block indices following the leave-one-block-out rule.  

    .. tip::

        Leave-one-block-out rule: One block works as the testing block. All other blocks work as the training blocks.

    :param block_idx: Given testing block index. 
    :return: 

        + ``test_block``: List of one testing block index
        + ``train_block``: List of training block indices

.. py:function:: get_data

    Extract EEG signals and corresponding labels from the dataset

    :param sub_idx: Subject index.
    :param blocks: List of block indices.
    :param trials: List of trial indices.
    :param channels: List of channel indices.
    :param sig_len: Signal length (in second).
    :param t_latency: Latency time (in second). Default is the default/suggested latency time of the dataset.
    :param shuffle: If ``True``, the order of trials will be shuffled. Otherwise, the order of trials will follow the given ``blocks`` and ``trials``.

    :return:

        + ``X``: List of single trial EEG signals.
        + ``Y``: List of labels.

    .. note::

        The preprocess and filterbanks are applied to windowed signals (not whole trial signal), which is close to the real online situation. The extraction will follow these steps:

        1. Cut the signal according to given ``sig_len``. The pre-stimulus time ``t_prestim`` will be removed. The latency time is maintained.
        2. Apply the hooked preprocessing function.
        3. Apply the bandpass filters of filterbanks.
        4. Remove the latency time. 

        The extraction process follows the below figure.

    .. image:: _static/dataset-processing.png

.. py:function:: get_data_all_trials

    Extract EEG signals of all trials in given blocks and corresponding labels from the dataset. This function is similar as ``get_data`` but it does not need ``trials`` and will extract all trials of given blocks.

    :param sub_idx: Subject index.
    :param blocks: List of block indices.
    :param channels: List of channel indices.
    :param sig_len: Signal length (in second).
    :param t_latency: Latency time (in second). Default is the default/suggested latency time of the dataset.
    :param shuffle: If ``True``, the order of trials will be shuffled. Otherwise, the order of trials will follow the given ``blocks`` and ``trials``.

    :return:

        + ``X``: List of single trial EEG signals.
        + ``Y``: List of labels.

.. py:function:: reset_ref_sig_fun

    Set the reference signal generation function as the default sine-cosine reference generation function. The default sine-cosine reference generation function uses the sampling frequency of the original signal (recoded in the dataset) to generate the reference signals. The reference signals of :math:`i\text{-th}` stimulus can be presented as

    .. math::

        \mathbf{Y}_i(t) = \left[ \begin{array}{c}
                            \sin(2\pi f_i t + \theta_i)\\
                            \cos(2\pi f_i t + \theta_i)\\
                            \vdots\\
                            \sin(2\pi N_h f_i t + N_h \theta_i)\\
                            \cos(2\pi N_h f_i t + N_h \theta_i)
                        \end{array} \right]

    where :math:`f_i` and :math:`\theta_i` denote the stimulus frequency and phase of the :math:`i\text{-th}` stimulus, and :math:`N_h` denotes the total number of harmonic components.

.. py:function:: regist_ref_sig_fun

    Hook the user-defined reference generation function. 

    :param ref_sig_fun: User-defined reference generation function.

    .. note::

        The given ``preprocess_fun`` should be a callable function name (only name). This callable function should only have four input parameter:
        
        + ``dataself`` is the data istance. If you need to use parameters in the data module, you can directly use them from ``dataself``. 
        + ``sig_len`` is the signal length (in second).
        + ``N`` is the total number of harmonic components.
        + ``phases`` is the phases of stimuli.

        The frequencies of stimuli can be obtained from ``dataself``.
        
        If your reference generation function needs other input parameters, you may use `lambda function <https://www.w3schools.com/python/python_lambda.asp>`_. Check demos to get more hints.

        Normally, you do not need to define your own reference signal generation function. But, when you change the sampling rate (upsampling or downsampling in the preprocess), you must define your own reference signal generation function using the new sampling rate. You may refer the following default reference signal generation function to define your own function.

    .. code-block:: python
        :linenos:

        def default_ref_sig_fun(dataself, sig_len: float, N: int, phases: List[float]):
            L = floor(sig_len * dataself.srate)
            ref_sig = [gen_ref_sin(freq, dataself.srate, L, N, phase) for freq, phase in zip(dataself.stim_info['freqs'], phases)]
            return ref_sig
    
    .. tip::

        ``gen_ref_sin`` is in :ref:`computation-fun`.

.. py:function:: get_ref_sig

    Generate sine-cosine-based reference signals by using the registed reference generation function.

    :param sig_len: Signal length (in second). It should be same as the signal length of extracted EEG signals.
    :param N: Total number of harmonic components.
    :param ignore_stim_phase: If ``True``, all stimulus phases will be set as 0. Otherwise, the stimulus phases stored in the dataset will be applied.

    :return: 

        + ``ref_sig``: List of reference signals. Each stimulus have one set of reference signals.

.. _define-own-dataset:

How to define your own dataset class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the abstract class ``SSVEPAnalysisToolbox.datasets.BaseDataset`` as the father class to define your own dataset class. In your own dataset class, the following functions should be defined:

1. ``__init__``: Except ``path`` and ``path_support_file``, other attributes mentioned in `Section "Attributes of datasets" <#attributes-of-datasets>`_ normally have been defined in the dataset. Therefore, the initialization function should be re-defined. You may ask for ``__init__`` of the father class ``SSVEPAnalysisToolbox.datasets.BaseDataset`` to store these attributes in class.
2.  Following abstract functions in ``SSVEPAnalysisToolbox.datasets.BaseDataset`` are empty and should be defined in your own dataset class:

    .. py:function:: download_single_subject

        Donwload one subject's data file. 

        :param subject: One ``SubInfo`` instance stored in ``subjects`` mentioned in `Section "Attributes of datasets" <#attributes-of-datasets>`_.

    .. py:function:: download_file

        Download one supported file.

        :param file_name: File name that will be downloaded.

    .. tip::

        You may use `"download_single_file" function <#SSVEPAnalysisToolbox.utils.download.download_single_file>`_ to download the required file. You also may need `"tarfile" <https://docs.python.org/3/library/tarfile.html>`_ or `"py7zr" <https://github.com/miurahr/py7zr>`_ to uncompress data files.

    .. py:function:: get_sub_data

        Read one subject data from the local data file. 

        :param sub_idx: Subject index.

        :return:

            + ``data``: The provided data should be a 4D data (blocks :raw-html:`&#215;` trials :raw-html:`&#215;` channels :raw-html:`&#215;` samples). Each trial should contain the whole trial data including pre-stimulus time, and latency time.

    .. note::

        The ``data`` provided by `"get_sub_data" function <#get_sub_data>`_ must be 4D. The order of dimentions should be exactly (blocks :raw-html:`&#215;` trials :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

    .. py:function:: get_label_single_trial

        Generate the label of one specific trial.

        :param sub_idx: Subject index.

        :param block_idx: Block index.

        :param stim_idx: Trial index.

        :return:

            + ``label``: Label of the specific trial. The label should be one integer number.

3. According to your requirements, you may re-define existed functions listed in `Functions of datasets <#functions-of-datasets>`_.