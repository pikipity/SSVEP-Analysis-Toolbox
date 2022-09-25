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

.. py:function:: SSVEPAnalysisToolbox.datasets.betadataset.NakanishiDataset

    Initialize the Nakanishi2015 dataset.

    Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject. 
    Please see the reference paper for the detail.

    size(eeg) = [Num. of targets, Num. of channels, Num. of sampling points, Num. of trials]

    =======================   =======
    Num. of Targets           12
    -----------------------   -------
    Num. of Channels          8
    -----------------------   -------
    Num. of sampling points   1114
    -----------------------   -------
    Num. of trials            15
    -----------------------   -------
    Sampling rate             56 Hz
    =======================   =======

    + The order of the stimulus frequencies in the EEG data: [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz (e.g., eeg(1,:,:,:) and eeg(5,:,:,:) are the EEG data while a subject was gazing at the visual stimuli flickering at 9.25 Hz and 11.75Hz, respectively.)
    
    + The onset of visual stimulation is at 39th sample point.

    Total: around 148 MB.
    
    Paper: M. Nakanishi, Y. Wang, Y.-T. Wang, T.-P. Jung, "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," *PLoS ONE*, vol. 10, p. e0140703, 2015. DOI: `10.1371/journal.pone.0140703 <https://doi.org/10.1371/journal.pone.0140703>`_.

    URL: `ftp://sccn.ucsd.edu/pub/cca_ssvep.zip <ftp://sccn.ucsd.edu/pub/cca_ssvep.zip>`_.

    :param path: Path of storing EEG data. 
    
        The missing subjects' data files will be downloaded when the dataset is initialized. 
        
        If the provided path is not existed, the provided path will be created. 
    
        Default path is a folder :file:`Nakanishi2015 Dataset` in the working path. 

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

.. py:function:: regist_preprocess

    Hook the user-defined preprocessing function. 

    :param preprocess_fun: User-defined preprocessing function.

    .. note::

        The given ``preprocess_fun`` should be a callable function name (only name). This callable function should only have one input parameter ``X``. ``X`` is a 2D EEG signal (channels :raw-html:`&#215;` samples). The pre-stimulus time has been removed from the EEG signal. The latency time is maintained in the EEG signal. The detailed data extraction procedures please refer to `"get_data" function <#get_data>`_.
        
        If your preprocess function needs other input parameters, you may use `lambda function <https://www.w3schools.com/python/python_lambda.asp>`_. Check demos to get more hints.

.. py:function:: reset_filterbank

    Set the filterbank function as the default filterbank function. In the default filterbank function, the original EEG signals will be considered as one filterbank. If the original EEG signal is a 2D signal (channels :raw-html:`&#215;` samples), one more dimention will be expanded (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples). If the original EEG signal is a 3D signal, original signal will be returned without any processing. 

.. py:function:: regist_filterbank

    Hook the user-defined filterbank function.

    :param filterbank_fun: User-defined filterbank function.

    .. note::

        The given ``filterbank_fun`` should be a callable function name (only name). This callable function should only have one input parameter ``X``. ``X`` is a 2D EEG signal (channels :raw-html:`&#215;` samples). The pre-stimulus time has been removed from the EEG signal. The latency time is maintained in the EEG signal. The detailed data extraction procedures please refer to `"get_data" function <#get_data>`_.

        The output of the given ``filterbank_fun`` should be a 3D EEG signal (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples). The bandpass filtered EEG signals of filterbanks should be stored in the first dimension. 

        If your filterbank function needs other input parameters, you may use `lambda function <https://www.w3schools.com/python/python_lambda.asp>`_. Check demos to get more hints.

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

.. py:function:: get_data_all_stim

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

.. py:function:: get_ref_sig

    Generate sine-cosine-based reference signals. The reference signals of :math:`i\text{-th}` stimulus can be presented as

    .. math::

        \mathbf{Y}_i(t) = \left[ \begin{array}{c}
                            \sin(2\pi f_i t + \theta_i)\\
                            \cos(2\pi f_i t + \theta_i)\\
                            \vdots\\
                            \sin(2\pi N_h f_i t + N_h \theta_i)\\
                            \cos(2\pi N_h f_i t + N_h \theta_i)
                        \end{array} \right]

    where :math:`f_i` and :math:`\theta_i` denote the stimulus frequency and phase of the :math:`i\text{-th}` stimulus, and :math:`N_h` denotes the total number of harmonic components.

    :param sig_len: Signal length (in second). It should be same as the signal length of extracted EEG signals.
    :param N: Total number of harmonic components.
    :param ignore_stim_phase: If ``True``, all stimulus phases will be set as 0. Otherwise, the stimulus phases stored in the dataset will be applied.

    :return: 

        + ``ref_sig``: List of reference signals. Each stimulus have one set of reference signals.

How to define your own dataset class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use the abstract class ``SSVEPAnalysisToolbox.basedataset.BaseDataset`` as the father class to define your own dataset class. In your own dataset class, the following functions should be defined:

1. ``__init__``: Except ``path`` and ``path_support_file``, other parameters mentioned in `Section "Parameters of datasets" <#parameters-of-datasets>`_ normally have been defined in the dataset. Therefore, the initialization function should be re-defined. You may ask for ``__init__`` of the father class ``SSVEPAnalysisToolbox.basedataset.BaseDataset`` to store these parameters in class.
2.  Following abstract functions in ``SSVEPAnalysisToolbox.basedataset.BaseDataset`` are empty and should be defined in your own dataset class:

    .. py:function:: download_single_subject

        Donwload one subject's data file. 

        :param subject: One ``SubInfo`` instance stored in ``subjects`` mentioned in `Section "Parameters of datasets" <#parameters-of-datasets>`_.

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

Utility Functions
------------------------------

Download functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.utils.download.download_single_file

    Download one file. 

    :param source_url: Source URL.

    :param desertation: Local path for storing the downloaded file. The path should be an absolute path with the file name.

    :param known_hash: Hash code of the downloaded file. Set ``None`` if the hash code is unknown. 

IO functions
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.utils.io.savedata

    Save a dictionary data.

    :param file: Path of saving file including the absolute path and file name.

    :param data: Dictionary data that will be saved.

    :save_type: There are two options of the saving data type: 

        + ``'mat'``: Save data as a matlab ``.mat`` file. The varaible names are the key values of the dictionary. The variable values are the values of the dictionary. If use this option, this function will work like `"scipy.io.savemat" <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html>`_.

        + ``'np'``: Save data as a numpy ``.npy`` binary file. If use this option, this function will work like `"numpy.save" <https://numpy.org/doc/stable/reference/generated/numpy.save.html>`_.

.. py:function:: SSVEPAnalysisToolbox.utils.io.loaddata

    Load a local data file.

    :param file: Local data path including the absolute path and file name.

    :param save_type: There are two options of the local data type:

        + ``'mat'``: Local data is a matlab ``.mat`` file. The varaible names are the key values of the dictionary. The variable values are the values of the dictionary. If use this option, this function will work like `"scipy.io.loadmat" <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html>`_ or `"mat73.loadmat" <https://github.com/skjerns/mat7.3>`_.

        + ``'np'``: Local data is a numpy ``.npy`` binary file. If use this option, this function will work like `"numpy.load" <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_.

    :return:

        + ``data``: Loaded dictionary data.

Computation functions
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.utils.algsupport.gen_ref_sin

    Generate sine-cosine-based reference signal of one stimulus. This function is similar as `"get_ref_sig" function <#get_ref_sig>`_ in dataset class. But this function is more flexible, requires more input parameters, and is only for one stimulus.

    :param freq: One stimulus frequency.

    :param srate: Sampling rate.

    :param L: Signal length (in samples). 

    :param N: Total number of harmonic components.

    :param phase: One stimulus phase.

    :return:

        + ``ref_sig``: Reference signals of one stimulus. The dimention is (2N :raw-html:`&#215;` L).

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.sum_list

    Iteratively sum all values in a list. If the input list contains lists, these contained lists will be summed first. 

    :param X: List that will be sumed. 

    :return:

        + ``sum_X``: Summation result.

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.mean_list

    Iteratively calculate average value of a list. If the input list contains lists, these contained lists will be averaged first.

    :param X: List that will be averaged.

    :return:

        + ``mean_X``: Average result.

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.sort

    Sort the given list

    :param X: List that will be sorted.

    :return:

        + ``sorted_X``: Sorted ``X``.
        + ``sort_idx``: List of indices that can transfer ``X`` to ``sorted_X``.
        + ``return_idx``: List of indices that can transfer ``sorted_X`` to ``X``.

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.gen_template

    Generate averaged templates. For each stimulus, EEG signals of all trials are averaged as the template signals.

    :param X: List of EEG signals. Each element is one single trial EEG signal. The dimentions of EEG signals should be (filterbanks :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

    :param Y: List of labels. Each element is one single trial label. The labels should be integer numbers.

    :return:

        + ``template_sig``: List of template signals. Each element is one class template signals. The dimentions of template signals are (filterbanks :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.canoncorr

    Calculate canoncial correlation of two matrices following `"canoncorr" in MATLAB <https://www.mathworks.com/help/stats/canoncorr.html>`_.

    :param X: First input matrix. The rows correspond to observations, and the columns correspond to variables.

    :param Y: Second input matrix. The rows correspond to observations, and the columns correspond to variables.

    :param force_output_UV: If ``True``, canonical coefficients will be calculated and provided. Otherwise, only the correlations are computed and provided.

    :return:
        + ``A``: Canonical coefficients of ``X``. If ``force_output_UV == True``, this value will be returned.
        + ``B``: Canonical coefficients of ``Y``. If ``force_output_UV == True``, this value will be returned.
        + ``r``: Canonical correlations.

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.qr_inverse

    Inverse QR decomposition.

    :param Q: Orthogonal factor obtained from the QR decomposition.

    :param R: Upper-triangular factor obtained from the QR decomposition.

    :param P: Permutation information obtained from the QR decomposition.

    :return:

        + ``X``: Results of the inverse QR decomposition. :math:`\mathbf{X}=\mathbf{Q}\times\mathbf{R}`. The column order of ``X`` has been adjusted according to ``P``.

.. note::

    In `"qr_inverse" function <#SSVEPAnalysisToolbox.algorithms.utils.qr_inverse>`_, the inputs ``Q``, ``R`` and ``P`` can be 2D or 3D. If the dimension is 2D, it is the conventional inverse QR decomposition. If the dimension is 3D, the conventional inverse QR decomposition will be applied along the first dimension. 

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.qr_remove_mean

    QR decomposition. Before the QR decomposition, the column means are firstly removed from the input matrix.

    :param X: Input matrix.

    :return:

        + ``Q``: Orthogonal factor.
        + ``R``: Upper-triangular factor.
        + ``P``: Permutation information.

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.qr_list

    Apply `"qr_remove_mean" function <#SSVEPAnalysisToolbox.algorithms.utils.qr_remove_mean>`_ to each element in the given list.

    :param X: List of input matrices for the QR decomposition.

    :return:

        + ``Q``: List of orthogonal factors.
        + ``R``: List of upper-triangular factors.
        + ``P``: List of permutation information.

.. note::

    In `"qr_list" function <#SSVEPAnalysisToolbox.algorithms.utils.qr_list>`_, elements of the input list can be 2D or 3D. If 2D, `"qr_remove_mean" function <#SSVEPAnalysisToolbox.algorithms.utils.qr_remove_mean>`_ is directly applied to each element. If 3D, `"qr_remove_mean" function <#SSVEPAnalysisToolbox.algorithms.utils.qr_remove_mean>`_ is applied to each element along the first dimension. 

.. py:function:: SSVEPAnalysisToolbox.algorithms.utils.mldivide

    Calculate A\\B. The minimum norm least-squares solution of solving :math:`\mathbf{A}\times \mathbf{x} = \mathbf{B}` for :math:`\mathbf{x}`. 

    :param A: First input matrix.

    :param B: Second input matrix.

    :return:

        + ``x``: Minimum norm least-squares solution. :math:`\mathbf{x} = \mathbf{A}^{-1}\times\mathbf{B}`. The inverse of the matrix ``A`` is performed by the `pseudo-inverse <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html>`_. 



Benchmark and BETA datasets related functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions are related to suggested filterbanks, channels, preprocessing function, and weights of filterbanks for the benchmark and BETA datasets. They also can be regarded as demos of preparing your own related functions. Values are refered to the following two papers:

+ Y. Wang, X. Chen, X. Gao, and S. Gao, "A benchmark dataset for SSVEP-based braincomputer interfaces," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, no. 10, pp. 1746-1752, 2017. DOI: `10.1109/TNSRE.2016.2627556 <https://doi.org/10.1109/TNSRE.2016.2627556>`_.
+ B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "BETA: A large benchmark database toward SSVEP-BCI application," *Front. Neurosci.*, vol. 14, p. 627, 2020. DOI: `10.3389/fnins.2020.00627 <https://doi.org/10.3389/fnins.2020.00627>`_.

.. py:function:: filterbank

    Suggested filterbank function. It contains five filterbanks. Each filterbank is a `Chebyshev type I bandpass filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html>`_ where ``N`` and ``Wn`` are generated by `"cheb1ord" <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheb1ord.html#scipy.signal.cheb1ord>`_ with ``gpass=3`` and ``gstop=40``, and ``rp=0.5``. The passband of the :math:`i\text{-th}` filterbank is from :math:`8i` Hz to :math:`90` Hz. The stopband of the :math:`i\text{-th}` filterbank is from :math:`(8i-2)` Hz to :math:`100` Hz.

    :param X: EEG signal following `"regist_filterbank" function <#regist_filterbank>`_.

    :param srate: Sampling frequency (Hz).

.. note::

    The `"filterbank" function <#filterbank>`_ needs one more input parameter ``srate`` compared to requriements of the `"regist_filterbank" function <#regist_filterbank>`_. If your dataset instance is ``dataset``, you can hook this filterbank function by ``dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))``.

.. py:function:: suggested_weights_filterbank

    Generate suggested weights of filterbanks. The weight of :math:`i\text{-th}` filterbank is :math:`(i^{-1.25}+0.25)`.

.. py:function:: suggested_ch

    Generate a list of suggested channel indices. 

.. py:function:: preprocess

    Suggested preprocess function. Only one notch filter at 50 Hz is applied. This filter is a `IIR notching digital comb filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iircomb.html>`_ where ``w0`` is 50, ``Q`` is 35, ``fs`` is the input parameter ``srate``.

    :param X: EEG signal following `"regist_preprocess" function <#regist_preprocess>`_.

    :param srate: Sampling frequency.

.. note::

    The `"preprocess" function <#preprocess>`_ needs one more input parameter ``srate`` compared to requriements of the `"regist_preprocess" function <#regist_preprocess>`_. If your dataset instance is ``dataset``, you can hook this filterbank function by ``dataset.regist_preprocess(lambda X: preprocess(X, dataset.srate))``.

Performance Evalution
------------------------

This toolbox provides a ``BaseEvaluator`` class for evaluating recognition performance. Users can use this class as the father class to write your own evaluator or use the above given functions or classes to write your own evaluation process. 

The ``BaseEvaluator`` class is a trial based evaluator. Evaluator contains several evaluation trials and evaluate performance trial by trial. Each trial contains several training and testing trials. In each trial, the ``BaseEvaluator`` uses the given training trials to train all models one by one and then tests their performance in testing trials. The training time, evaluation time, ture labels and predicted labels will be stored. The recognition accuracies and ITRs can be further computed. 

Evaluator
^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.evaluator.baseevaluator.BaseEvaluator

    Initialize the evaluator.

    :param dataset_container: A list of datasets. Each element is a instance of one dataset class introduced in `"Datasets" <#datasets>`_.

    :param model_container: A list of recognition models/methods. Each element is a instance of one recognition model/method class introduced in `"Recognition algorithms" <#recognition-algorithms>`_.

    :param trial_container: A list of trials. The format is 

        .. code-block:: python

            [[train_trial_info, test_trial_info],
             [train_trial_info, test_trial_info],
             ...,
             [train_trial_info, test_trial_info]]

        where ``train_trial_info`` and ``test_trial_info`` are instances of the ``TrialInfo`` class. 

    :param save_model: If ``True``, trained models in all trials will be stored in ``trained_model_container``. The format of ``trained_model_container`` is

        .. code-block:: python

            [[trained_model_method_1, trained_model_method_2, ...],
             [trained_model_method_1, trained_model_method_2, ...],
             ...,
             [trained_model_method_1, trained_model_method_2, ...]]

        where ``trained_model_method_1``, ``trained_model_method_2``, ... are instances of recognition model/method classes, which order is same as ``model_container``.

        If ``False``, ``trained_model_container`` is an empty list. 

        Default is ``False``.

    :param disp_processbar: If ``True``, a progress bar will be shown in console to illustrate the evaluation process. Otherwise, the progress bar will be shown. Default is ``True``.

    :param ignore_stim_phase: If ``True``, stimulus phases of generating reference signals will be set as 0 during the evalution. Otherwise, stimulus phases will use the dataset information. Default is ``False``.

.. note::

    Saving models by setting ``save_model`` as ``True`` may occupy large memory.  

.. py:function:: run
    :module: BaseEvaluator

    Run the evaluation process. Performance will be stored in ``performance_container``. The format of ``performance_container`` is 

    .. code-block:: python

        [[performance_method_1, performance_method_2, ...],
         [performance_method_1, performance_method_2, ...],
         ...,
         [performance_method_1, performance_method_2, ...]]

    where ``performance_method_1``, ``performance_method_2``, ... are instances of the ``PerformanceContainer`` class for different recognition models/methods. The order follows ``model_container``.

    :param n_jobs: Number of threadings using for recognition methods. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. The evaluator will reset ``n_jobs`` in recognition methods.

    :param eval_train: *Please ignore this parameter and leave this parameter as the default value. The function related to this parameter is under development.* 

Trial Information
^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.evaluator.baseevaluator.TrialInfo

    The instances of this class are the basic elements of ``trial_container`` in ``BaseEvaluator``. 

    It contains following parameters:

    + ``dataset_idx``: A list of dataset indeices.
    + ``sub_idx``: A list of all datasets' subject index list. The format is
      
      .. code-block:: python

        [[sub_idx_1, sub_idx_2, ...],
         [sub_idx_1, sub_idx_2, ...],
         ...,
         [sub_idx_1, sub_idx_2, ...]]

      where ``sub_idx_1``, ``sub_idx_2``, ... are subject indices for different datasets. The order follows ``dataset_idx``.

    + ``block_idx``: A list of all datasets' block index list. The format is same as ``sub_idx`` but the integer numbers in lists are block indices.
    + ``trial_idx``: A list of all datasets' trial index list. The format is same as ``sub_idx`` but the integer numbers in lists are trial indices.
    + ``ch_idx``: A list of all datasets' channel index list. The format is same as ``sub_idx`` but the integer numbers in lists are channel indices.
    + ``harmonic_num``: The harmonic number is used to generate reference signals. One integer number. 
    + ``tw``: The signal length (in second). One float number.
    + ``t_latency``: A list of latency times of datasets. Each element is a float number denoting a latency time of one dataset.
    + ``shuffle``: A list of shuffle flag. Each element is a bool value denoting whether shuffle trials.

.. py:function:: add_dataset
    :module: TrialInfo

    Push one dataset information into the trial information

    :param dataset_idx: dataset index. One integer number.
    :param sub_idx: List of subject indices. A list of integer numbers.
    :param block_idx: List of block indices. A list of integer numbers.
    :param trial_idx: List of trial indices. A list of integer numbers.
    :param ch_idx: List of channel indices. A list of integer numbers.
    :param harmonic_num: The harmonic number is used to generate reference signals. This input parameter will update ``harmonic_num`` of the trial information. One integer number.
    :param tw: The signal length (in second). This input parameter will update ``tw`` of the trial information. One float number.
    :param t_latency: Latency time (in second). A float number.
    :param shuffle: If ``True``, the order of trials will be shuffled.

    :return: The instance itself.

.. py:function:: get_data

    Based on the trial information, get all data, labels, and reference signals.

    :param dataset_container: List of datasets.

    :return:

        + ``X``: List of all EEG trials.
        + ``Y``: List of all labels.
        + ``ref_sig``: This function will use the first dataset in ``dataset_idx`` to generate reference signals. 
        + ``freqs``: List of stimulus frequencies corresponding to generated reference signals.

.. note::

    This ``TrialInfo`` will only use the first dataset to generate reference signals. If datasets have different stimuli, please separate them into different trials. The more safety way is that one ``TrialInfo`` cotains only one dataset.

Performance Container
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.evaluator.baseevaluator.PerformanceContainer

    The instances of this class are the element of ``performance_container`` in ``BaseEvaluator``. 

    It contains following parameters:

    + ``true_label_train``: After training, to evaluate the training performance, the list of true labels of training trials is stored in this parameter. The format is 

      .. code-block:: python

        [[true_label_1, true_label_2, ...],
         [true_label_1, true_label_2, ...],
         ...,
         [true_label_1, true_label_2, ...]]

      where ``true_label_1``, ``true_label_2``, ... are true labels of different evaluation trials.
    
    + ``pred_label_train``: After training, to evaluate the training performance, the list of predicted labels of training trials is stored in this parameter. The format is same as ``true_label_train``.
    + ``true_label_test``: The list of true labels of testing trials is stored in this parameter. The format is same as ``true_label_train``.
    + ``pred_label_test``: The list of predicted labels of testing trials is stored in this parameter. The format is same as ``true_label_train``.
    + ``train_time``: A list of storing time of training the model. Each element in the list is one training time of one evaluation trial.
    + ``test_time_train``: A list of storing time of using the training trials to testing the model. Each element in the list is one testing time of one evaluation trial.
    + ``test_time_test``: A list of storing time of using the testing trials to test the model. Each element in the list is one testing time of one evaluation trial. 

Supplementary functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.evaluator.baseevaluator.gen_trials_onedataset_individual_diffsiglen

    Generate ``trial_container`` for ``BaseEvaluator``. These evaluation trials only use one dataset. One block is used for testing. Other blocks for training. All blocks will be tested one by one. All subjects will be evaluated one by one for each signal length.

    :param dataset_idx: Dataset index. One integer number.
    :param tw_seq: List of signal lengths (in second). A list of float numbers.
    :param dataset_container: List of datasets.
    :param harmonic_num: Number of harmonics. One integer number.
    :param trials: List of trial indices. A list of integer numbers.
    :param ch_used: List of channel indices. A list of integer numbers.
    :param t_latency: Latency time (in second). A float number. If ``None``, the suggested latency time of the dataset will be used.
    :param shuffle: If ``True``, trials will be shuffled. Default is ``False``.

.. py:function:: SSVEPAnalysisToolbox.evaluator.performance.cal_performance_onedataset_individual_diffsiglen

    Calculate evaluation performance of ``BaseEvaluator`` whose ``trial_container`` is generated by ``gen_trials_onedataset_individual_diffsiglen``.

    :param evaluator: The instance of the class ``BaseEvaluator``.
    :param dataset_idx: Dataset index.
    :param tw_seq: List of signal lengths (in second)
    :param train_or_test: If ``"train"``, evaluate performance of training trials. If ``"test"``, evaluate performance of testing trials.

    :return:

        + ``acc``: Classification accuracy. The shape is (methods :raw-html:`&#215;` subjects :raw-html:`&#215;` signal length).
        + ``itr``: ITR. The shape is (methods :raw-html:`&#215;` subjects :raw-html:`&#215;` signal length).

.. py:function:: SSVEPAnalysisToolbox.evaluator.performance.cal_confusionmatrix_onedataset_individual_diffsiglen

    Calculate confusion matrices of ``BaseEvaluator`` whose ``trial_container`` is generated by ``gen_trials_onedataset_individual_diffsiglen``.

    :param evaluator: The instance of the class ``BaseEvaluator``.
    :param dataset_idx: Dataset index.
    :param tw_seq: List of signal lengths (in second)
    :param train_or_test: If ``"train"``, evaluate confusion matrices of training trials. If ``"test"``, evaluate confusion matrices of testing trials.

    :return:

        + ``confusion_matrix``: Confusion matrices. The shape is (methods :raw-html:`&#215;` subjects :raw-html:`&#215;` signal lengths :raw-html:`&#215;` true classes :raw-html:`&#215;` predicted classes).

Plot Functions
^^^^^^^^^^^^^^^^

.. py:function:: SSVEPAnalysisToolbox.evaluator.plot.shadowline_plot

    Plot shadow lines. Each group plots one shadow line. 

    :param X: List of variable values.
    :param Y: Plot data. The shape is (groups :raw-html:`&#215;` observations :raw-html:`&#215;` variables). The line is the mean across observations. The shadow is the variation across observations.
    :param fmt: Format of lines. Default is ``'-'``.
    :param x_label: Label of x axis. Default is ``None``.
    :param y_label: Label of y axis. Default is ``None``.
    :param x_ticks: X tick labels. Default is ``None``.
    :param legend: List of line names. Default is ``None``. 
    :param errorbar_type: If ``'std'``, calculate the variation using the standard derivation. If ``'95ci'``, calculate the variation using the 95% confidence interval.
    :param grid: Whether grid. Default is ``True``.
    :param xlim: ``[min_x, max_x]``. Default is ``None``.
    :param ylim: ``[min_y, max_y]``. Default is ``None``.
    :param figsize: Figure size. Default is ``[6.4, 4.8]``.

.. py:function:: SSVEPAnalysisToolbox.evaluator.plot.bar_plot_with_errorbar

    Plot bars with error bars. Each group plots one color bars.

    :param Y: Plot data. The shape is (groups :raw-html:`&#215;` observations :raw-html:`&#215;` variables). The bar height is the mean across observations. The error bar is the variation across observations.
    :param bar_sep: Separate distence of adjacent bars. 
    :param x_label: Label of x axis. Default is ``None``.
    :param y_label: Label of y axis. Default is ``None``.
    :param x_ticks: X tick labels. Default is ``None``.
    :param legend: List of bar names. Default is ``None``. 
    :param errorbar_type: If ``'std'``, calculate the variation using the standard derivation. If ``'95ci'``, calculate the variation using the 95% confidence interval.
    :param grid: Whether grid. Default is ``True``.
    :param xlim: ``[min_x, max_x]``. Default is ``None``.
    :param ylim: ``[min_y, max_y]``. Default is ``None``.
    :param figsize: Figure size. Default is ``[6.4, 4.8]``.

.. py:function:: SSVEPAnalysisToolbox.evaluator.plot.bar_plot

    This function is similar as ``bar_plot_with_errorbar``. But this function only plots one group data and does not plot error bars. 

    :param Y: Plot data. The shape is (observations :raw-html:`&#215;` variables). The bar height is the mean across observations. The error bar is the variation across observations.
    :param bar_sep: Separate distence of adjacent bars. 
    :param x_label: Label of x axis. Default is ``None``.
    :param y_label: Label of y axis. Default is ``None``.
    :param x_ticks: X tick labels. Default is ``None``.
    :param grid: Whether grid. Default is ``True``.
    :param xlim: ``[min_x, max_x]``. Default is ``None``.
    :param ylim: ``[min_y, max_y]``. Default is ``None``.
    :param figsize: Figure size. Default is ``[6.4, 4.8]``.

Recognition algorithms
-------------------------

Common functions for all models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All following recognition models have these functions. The inputs and outputs are same so they will not be repeatedly introduced in following sections.

.. py:function:: __copy__

    Copy the recognition model.

    :return:

        + ``model``: The returned new model is same as the original one.

.. py:function:: fit

    Train the recognition model. The trained model parameters will be stored in the class parameter `model`.

    :param freqs: List of stimulus frequencies. 

    :param X: List of training EEG signals. Each element is one 3D single trial EEG signal (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

    :param Y: List of training labels. Each element is one single trial label that is an integer number.

    :param ref_sig: List of reference signals. Each element is the reference signal of one stimulus. 

.. py:function:: predict

    Recognize the testing signals.

    :param X: List of testing EEG signals. Each element is one 3D single trial EEG signal (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

    :return:

        + ``Y_pred``: List of predicted labels for testing signals. Each element is one single trial label that is an integer number.

Standard CCA and filterbank CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related papers: 

+ Standard CCA: Z. Lin et al., “Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs,” IEEE Trans. Biomed. Eng., vol. 53, no. 12, pp. 2610-2614, 2006. DOI: `10.1109/TBME.2006.886577 <https://doi.org/10.1109/TBME.2006.886577>`_.
+ Filterbank CCA: X. Chen et al., “Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface,” J. Neural Eng., vol. 12, no. 4, p. 046008, 2015. DOI: `10.1088/1741-2560/12/4/046008 <https://doi.org/10.1088/1741-2560/12/4/046008>`_.

In this toolbox, the standard CCA (sCCA) are regarded as a special case of the filterbank CCA (FBCCA) that only have one filterbank. Spatial filters are found to maximize the similarity between the EEG signals and the sine-cosine-based reference signals, which can be presented as

.. math::

    \mathbf{U}_i, \mathbf{V}_i = \arg\max_{\mathbf{u},\mathbf{v}}\frac{\mathbf{u}^T\mathbf{X}\mathbf{Y}_i^T\mathbf{v}}{\sqrt{\mathbf{u}^T\mathbf{X}\mathbf{X}^T\mathbf{u}\mathbf{v}^T\mathbf{Y}_i\mathbf{Y}_i^T\mathbf{v}}}

where :math:`\mathbf{X}` denotes the testing multi-channel EEG signal, :math:`\mathbf{Y}_i` denotes the sine-cosine-based reference signal of the :math:`i\text{-th}` stimulus, :math:`\mathbf{U}_i` is the spatial filter of the :math:`i\text{-th}` stimulus, and :math:`\mathbf{V}_i` is the harmonic weights of the reference signal for the :math:`i\text{-th}` stimulus.

The stimulus with the highest similarity is regarded as the target:

.. math::

    \arg\max_{i\in\left\{1,2,\cdots,I\right\}}\left\{ \frac{\mathbf{U}_i^T\mathbf{X}\mathbf{Y}_i^T\mathbf{V}_i}{\sqrt{\mathbf{U}_i^T\mathbf{X}\mathbf{X}^T\mathbf{U}_i\mathbf{V}_i^T\mathbf{Y}_i\mathbf{Y}_i^T\mathbf{V}_i}} \right\}

where :math:`I` denotes the total number of stimuli.

.. py:function:: SSVEPAnalysisToolbox.algorithms.cca.SCCA_canoncorr

    FBCCA implemented directly following above equations.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param force_output_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be stored. Otherwise, they will not be stored. Default is ``False``.

    :param update_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be re-computed in following testing trials. Otherwise, they will not be re-computed if they are already existed. Default is ``True``.

.. py:function:: SSVEPAnalysisToolbox.algorithms.cca.SCCA_qr

    FBCCA implemented by the QR decomposition. This implementation is almost same as the `"SCCA_canoncorr" model <#SSVEPAnalysisToolbox.algorithms.cca.SCCA_canoncorr>`_. The only difference is that this implementation does not repeatedly compute the QR decomposition of reference signals, which can improve the computational efficiency.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param force_output_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be stored. Otherwise, they will not be stored. Default is ``False``.

    :param update_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be re-computed in following testing trials. Otherwise, they will not be re-computed if they are already existed. Default is ``True``.

.. note::

    Although the FBCCA is a training-free method, these models still need run `"fit" function <#fit>`_ to store reference signals in the model.

Extended CCA
^^^^^^^^^^^^^

Related paper:

    + X. Chen, Y. Wang, M. Nakanishi, X. Gao, T.-P. Jung, and S. Gao, "High-speed spelling with a noninvasive brain-computer interface," *Proc. Natl. Acad. Sci.*, vol. 112, no. 44, pp. E6058-E6067, 2015. DOI: `10.1073/pnas.1508080112 <https://doi.org/10.1073/pnas.1508080112>`_.

The extended CCA (eCCA) not only applies the sine-cosine-based reference signals but also uses the averaged template signals. Three types of spatial filters are computed:

.. math::

    \mathbf{U}_{1,i}, \mathbf{V}_{1,i} = \arg\max_{\mathbf{u},\mathbf{v}}\frac{\mathbf{u}^T\mathbf{X}\mathbf{Y}_i^T\mathbf{v}}{\sqrt{\mathbf{u}^T\mathbf{X}\mathbf{X}^T\mathbf{u}\mathbf{v}^T\mathbf{Y}_i\mathbf{Y}_i^T\mathbf{v}}}

.. math::

    \mathbf{U}_{2,i}, \mathbf{V}_{2,i} = \arg\max_{\mathbf{u},\mathbf{v}}\frac{\mathbf{u}^T\mathbf{X}\overline{\mathbf{X}}_i^T\mathbf{v}}{\sqrt{\mathbf{u}^T\mathbf{X}\mathbf{X}^T\mathbf{u}\mathbf{v}^T\overline{\mathbf{X}}_i\mathbf{Y}_i^T\mathbf{v}}}

.. math::

    \mathbf{U}_{3,i}, \mathbf{V}_{3,i} = \arg\max_{\mathbf{u},\mathbf{v}}\frac{\mathbf{u}^T\overline{\mathbf{X}}_i\mathbf{Y}_i^T\mathbf{v}}{\sqrt{\mathbf{u}^T\overline{\mathbf{X}}_i\overline{\mathbf{X}}_i^T\mathbf{u}\mathbf{v}^T\mathbf{Y}_i\mathbf{Y}_i^T\mathbf{v}}}

where :math:`\overline{\mathbf{X}}_i` denotes the averaged template signal of the :math:`i\text{-th}` stimulus. 

Four types of corresponding correlation coefficients can be computed:

.. math::

    r_{1,i} = \frac{\mathbf{U}_{1,i}^T\mathbf{X}\mathbf{Y}_i^T\mathbf{V}_{1,i}}{\sqrt{\mathbf{U}_{1,i}^T\mathbf{X}\mathbf{X}^T\mathbf{U}_{1,i}\mathbf{V}_{1,i}^T\mathbf{Y}_i\mathbf{Y}_i^T\mathbf{V}_{1,i}}}

.. math::

    r_{2,i} = \frac{\mathbf{U}_{2,i}^T\mathbf{X}\overline{\mathbf{X}}_i^T\mathbf{U}_{2,i}}{\sqrt{\mathbf{U}_{2,i}^T\mathbf{X}\mathbf{X}^T\mathbf{U}_{2,i}\mathbf{U}_{2,i}^T\overline{\mathbf{X}}_i\overline{\mathbf{X}}_i^T\mathbf{U}_{2,i}}}

.. math::

    r_{3,i} = \frac{\mathbf{U}_{1,i}^T\mathbf{X}\overline{\mathbf{X}}_i^T\mathbf{U}_{1,i}}{\sqrt{\mathbf{U}_{1,i}^T\mathbf{X}\mathbf{X}^T\mathbf{U}_{1,i}\mathbf{U}_{1,i}^T\overline{\mathbf{X}}_i\overline{\mathbf{X}}_i^T\mathbf{U}_{1,i}}}

.. math::

    r_{4,i} = \frac{\mathbf{U}_{3,i}^T\mathbf{X}\overline{\mathbf{X}}_i^T\mathbf{U}_{3,i}}{\sqrt{\mathbf{U}_{3,i}^T\mathbf{X}\mathbf{X}^T\mathbf{U}_{3,i}\mathbf{U}_{3,i}^T\overline{\mathbf{X}}_i\overline{\mathbf{X}}_i^T\mathbf{U}_{3,i}}}

The target stimulus is predicted by combining four correlation coefficients together:

.. math::

    \arg\max_{i\in\left\{1,2,\cdots,I\right\}}\left\{ \sum_{k=1}^4 \text{sign}\left\{r_{k,i}\right\}\cdot r_{k,i}^2 \right\}

where :math:`\text{sign}\left\{\cdot\right\}` is the `signum function <https://en.wikipedia.org/wiki/Sign_function>`_.

.. py:function:: SSVEPAnalysisToolbox.algorithms.cca.ECCA

    The eCCA. The implementation is similar as the `"SCCA_qr" model <#SSVEPAnalysisToolbox.algorithms.cca.SCCA_qr>`_.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param update_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be re-computed in following training and testing trials. Otherwise, they will not be re-computed if they are already existed. Default is ``True``.

Multi-stimulus CCA
^^^^^^^^^^^^^^^^^^^

Related paper:

+ C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: `10.1088/1741-2552/ab2373 <https://doi.org/10.1088/1741-2552/ab2373>`_.

The multi-stimulus CCA (ms-CCA) considers reference signals and template signals of target stimulus and stimuli with stimulus frequencies are close to that of target stimulus, which includes the phase information and thus improve the recognition accuracy. The spatial filters are computed by

.. math::

    \mathbf{U}_i, \mathbf{V}_i = \arg\max_{\mathbf{u},\mathbf{v}}\frac{\mathbf{u}^T\mathbf{A}_i\mathbf{B}_i^T\mathbf{v}}{\sqrt{\mathbf{u}^T\mathbf{A}_i\mathbf{A}_i^T\mathbf{u}\mathbf{v}^T\mathbf{B}_i\mathbf{B}_i^T\mathbf{v}}}

where :math:`\mathbf{A}_i` is the concatenated template signal defined as :math:`\mathbf{A}_i = \left[\overline{\mathbf{X}}_{i-m},\cdots,\overline{\mathbf{X}}_{i},\cdots,\overline{\mathbf{X}}_{i+n}\right]`, and :math:`\mathbf{B}_i` is the concatenated reference signal defined as :math:`\mathbf{A}_i = \left[\mathbf{Y}_{i-m},\cdots,\mathbf{Y}_{i},\cdots,\mathbf{Y}_{i+n}\right]`.

Two types of correlation coefficients are computed:

.. math::

    r_{1,i} = \frac{\mathbf{U}_{i}^T\mathbf{X}\mathbf{Y}_i^T\mathbf{V}_{i}}{\sqrt{\mathbf{U}_{i}^T\mathbf{X}\mathbf{X}^T\mathbf{U}_{i}\mathbf{V}_{i}^T\mathbf{Y}_i\mathbf{Y}_i^T\mathbf{V}_{i}}}

.. math::

    r_{2,i} = \frac{\mathbf{U}_{i}^T\mathbf{X}\overline{\mathbf{X}}_i^T\mathbf{U}_{i}}{\sqrt{\mathbf{U}_{i}^T\mathbf{X}\mathbf{X}^T\mathbf{U}_{i}\mathbf{U}_{i}^T\overline{\mathbf{X}}_i\overline{\mathbf{X}}_i^T\mathbf{U}_{i}}}

The target stimulus is predicted by combining two correlation coefficients:

.. math::

    \arg\max_{i\in\left\{1,2,\cdots,I\right\}}\left\{ \sum_{k=1}^2 \text{sign}\left\{r_{k,i}\right\}\cdot r_{k,i}^2 \right\}

.. py:function:: SSVEPAnalysisToolbox.algorithms.cca.MSCCA

    ms-CCA. The implementation directly follows above equations.

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``12``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Multi-set CCA (MsetCCA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related paper:

+ Y. Zhang, G. Zhou, J. Jin, X. Wang, A. Cichocki, "Frequency recognition in SSVEP-based BCI using multiset canonical correlation analysis," *Int J Neural Syst.*, vol. 24, 2014, p. 1450013. DOI: `10.1142/ S0129065714500130 <https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130>`_.

For each stimulus, the MsetCCA maximize the inter-trial covariance to compute the spatial filters. The spatial filters of different trials are different.

.. math::

    \widetilde{\mathbf{w}}_{i,1}, \cdots, \widetilde{\mathbf{w}}_{i,N_t}=\arg\max_{\mathbf{w}_1,\cdots,\mathbf{w}_{N_t}}\sum_{h_1=1}^{N_t}\sum_{h_2=1,h_1\neq h_2}^{N_t}\mathbf{w}_{h_1}^T\cal{X}_{i,h_1}\cal{X}_{i,h_2}^T\mathbf{w}_{h_2}

.. math:: 

    \text{subject to }\frac{1}{N_t} \sum_{h_1=1}^{N_t}\sum_{h_2=1,h_1\neq h_2}^{N_t}\mathbf{w}_{h_1}^T\cal{X}_{i,h_1}\cal{X}_{i,h_2}^T\mathbf{w}_{h_2} = 1

where :math:`\widetilde{\mathbf{w}}_{i,n}` denotes the spatial filter of the :math:`i\text{-th}` stimulus and the :math:`n\text{-th}` training trial, :math:`N_t` denotes the number of training trials, and :math:`\cal{X}_{i,n}` denotes the calibration data of the :math:`i\text{-th}` stimulus and the :math:`n\text{-th}` training trial.

Solving this optimization problem is equivalent to solving the following eigenvalue problem:

.. math:: 

    \left( \mathbf{R}_i-\mathbf{S}_i \right)\mathbf{w} = \rho\mathbf{S}_i\mathbf{w}

where 

.. math:: 

    \mathbf{R}_i = \left[  
                    \begin{array}{ccc}
                        \cal{X}_{i,1}\cal{X}_{i,1}^T & \cdots & \cal{X}_{i,1}\cal{X}_{i,N_t}^T\\
                        \vdots & \ddots & \vdots\\
                        \cal{X}_{i,N_t}\cal{X}_{i,1}^T & \cdots & \cal{X}_{i,N_t}\cal{X}_{i,N_t}^T
                    \end{array}
                   \right]

.. math:: 

    \mathbf{S}_i = \left[  
                    \begin{array}{ccc}
                        \cal{X}_{i,1}\cal{X}_{i,1}^T & \cdots & 0\\
                        \vdots & \ddots & \vdots\\
                        0 & \cdots & \cal{X}_{i,N_t}\cal{X}_{i,N_t}^T
                    \end{array}
                   \right]

.. math:: 

    \mathbf{w} = \left[ \mathbf{w}_1, \cdots, \mathbf{w}_{N_t} \right]^T.

.. py:function:: SSVEPAnalysisToolbox.algorithms.cca.MsetCCA

    Multi-set CCA. The implementation directly follows above equations.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Task-related component analysis (TRCA) and ensemble TRCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related paper:

+ M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component Analysis," *IEEE Trans. Biomed. Eng.*, vol. 65, no. 1, pp. 104-112, 2018. DOI: `10.1109/TBME.2017.2694818 <https://doi.org/10.1109/TBME.2017.2694818>`_.

For each stimulus, the TRCA and the ensemble TRCA (eTRCA) maximize the inter-trial covariance to compute the common spatial filter across trials, which can be achieved by solving

.. math::

    \left( \sum_{j,k=1,\; j\neq k}^{N_t} \mathbf{X}_i^{(j)}\left(\mathbf{X}_i^{(k)}\right)^T \right)\mathbf{U}_i = \left( \sum_{j=1}^{N_t} \mathbf{X}_i^{(j)}\left(\mathbf{X}_i^{(j)}\right)^T \right) \mathbf{U}_i\mathbf{\Lambda}_i

where :math:`\mathbf{X}_i^{(j)}` denotes the :math:`j\text{-th}` trial training EEG signals of :math:`i\text{-th}` stimulus.

The target stimulus can be predicted by 

.. math::

    \arg\max_{i\in\left\{1,2,\cdots,I\right\}}\left\{ \frac{\mathbf{U}_i^T\mathbf{X}\overline{\mathbf{X}}_i^T\mathbf{U}_i}{\sqrt{\mathbf{U}_i^T\mathbf{X}\mathbf{X}^T\mathbf{U}_i\mathbf{U}_i^T\overline{\mathbf{X}}_i\overline{\mathbf{X}}_i^T\mathbf{U}_i}} \right\}

.. py:function:: SSVEPAnalysisToolbox.algorithms.trca.TRCA

    TRCA. The implementation directly follows above equations.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

.. py:function:: SSVEPAnalysisToolbox.algorithms.trca.ETRCA

    eTRCA. The spatial computation is same as the TRCA. The only difference is that the recognition uses the same set of spatial filters for all stimuli. This set of saptial filters contain all eigen vectors with the highest eigen value of all stimuli.

    :param n_component: This parameter will not be considered in the eTRCA. 

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Multi-stimulus TRCA
^^^^^^^^^^^^^^^^^^^^^^

Related paper:

+ C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: `10.1088/1741-2552/ab2373 <https://doi.org/10.1088/1741-2552/ab2373>`_.

The multi-stimulus TRCA (ms-TRCA) is similar as the `ms-CCA <#multi-stimulus-cca>`_. It also considers training EEG signals of stimuli whose stimulus frequencies are close to the target stimulus to compute spatial filters:

.. math::

    \sum_{d=i-m}^{i+n}\left\{ \sum_{j,k=1,\; j\neq k}^{N_t} \mathbf{X}_d^{(j)}\left(\mathbf{X}_d^{(k)}\right)^T \right\}\mathbf{U}_i = \sum_{d=i-m}^{i+n}\left\{ \sum_{j=1}^{N_t} \mathbf{X}_d^{(j)}\left(\mathbf{X}_d^{(j)}\right)^T \right\} \mathbf{U}_i\mathbf{\Lambda}_i

.. py:function:: SSVEPAnalysisToolbox.algorithms.trca.MSETRCA

    ms-TRCA. In this toolbox, the ms-TRCA follows the `eTRCA <#SSVEPAnalysisToolbox.algorithms.trca.ETRCA>`_ scheme to emsemble spatial filters of all stimuli for the recognition. 

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: This parameter will not be considered in this function. 

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

.. py:function:: SSVEPAnalysisToolbox.algorithms.trca.MSCCA_and_MSETRCA

    This method ensembles correlation coefficients of the `ms-CCA <#SSVEPAnalysisToolbox.algorithms.cca.MSCCA>`_ and the `ms-TRCA <#SSVEPAnalysisToolbox.algorithms.trca.MSETRCA>`_ to recognize the target stimulus. Suppose that :math:`r_{1,i}` and :math:`r_{2,i}` are correlation coefficients obtained from the ms-CCA and the ms-TRCA respectively, then the ensembled correlation coefficient is 

    .. math::

        r_\text{ms-CCA + ms-TRCA} = \sum_{k=1}^2 \text{sign}\left\{r_{k,i}\right\}\cdot r_{k,i}^2 

    :param n_neighbor_mscca: Number of neighbers considered for computing the spatial filter of one stimulus in the ms-CCA. Default is ``12``.

    :param n_neighber_msetrca: Number of neighbers considered for computing the spatial filter of one stimulus in the ms-TRCA. Default is ``2``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters in the ms-CCA. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.


Task-discriminant component analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related paper:

+ B. Liu, X. Chen, N. Shi, Y. Wang, S. Gao, X. Gao, "Improving the performance of individually calibrated SSVEP-BCI by task-discriminant component analysis." *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 1998-2007, 2021. DOI: `10.1109/TNSRE.2021.3114340 <https://doi.org/10.1109/TNSRE.2021.3114340>`_.

Compared with other methods, the task-discriminant component analysis (TDCA) have following three key differences:

1. The dimensionality of EEG signals is elevated. For one trial EEG signal :math:`\mathbf{X}`, the augmented EEG trial :math:`\widetilde{\mathbf{X}}` is

   .. math::
      
      \widetilde{\mathbf{X}} = \left[ \begin{array}{cc}
                                        \mathbf{X}, & \mathbf{O}_0\\
                                        \mathbf{X}_1, & \mathbf{O}_1\\
                                        \vdots & \\
                                        \mathbf{X}_L, & \mathbf{O}_L
                                      \end{array} \right]
   
   where :math:`\mathbf{X}_l` denotes the EEG trial delayed by :math:`l` points, :math:`\mathbf{O}_l\in\mathbb{R}^{N_\text{ch}\times l}` denotes the zero matrix, and :math:`L` is the total number of delays. 

2. After elevating the dimension, EEG trials are then further extended for each stimulus as

   .. math::

      \mathbf{X}_a = \left[ \widetilde{\mathbf{X}},\;\; \widetilde{\mathbf{X}}\mathbf{Q}_i\mathbf{Q}_i^T \right]

   where :math:`\mathbf{Q}_i` is the orthogonal factor of the reference signal of the :math:`i\text{-th}` stimulus and can be obtained by the QR decomposition :math:`\mathbf{Q}_i\mathbf{R}_i=\mathbf{Y}_i^T`.

3. The two-dimensional linear discriminant analysis is applied to compute spatial filters by solving

   .. math::

      \mathbf{S}_b\mathbf{U} = \mathbf{S}_w\mathbf{U}\mathbf{\Lambda}

   The :math:`\mathbf{S}_b` is the covariance of between-class differences:

   .. math::

      \mathbf{S}_b = \frac{1}{I} \sum_{i=1}^I \left( \overline{\mathbf{X}}_a^{(i)} - \frac{1}{I}\sum_{i=1}^I\overline{\mathbf{X}}_a^{(i)} \right)\left( \overline{\mathbf{X}}_a^{(i)} - \frac{1}{I}\sum_{i=1}^I\overline{\mathbf{X}}_a^{(i)} \right)^T

   where :math:`\overline{\mathbf{X}}_a^{(i)}` is the averaged :math:`\mathbf{X}_a` over all trials of the :math:`i\text{-th}` stimulus, and :math:`I` is the total number of stimuli.

   The :math:`\mathbf{S}_w` is the covariance of within-class differences:

   .. math::

      \mathbf{S}_w = \frac{1}{N_t} \sum_{i=1}^I \sum_{j=1}^{N_t} \left( \mathbf{X}_a^{(i,j)} - \overline{\mathbf{X}}_a^{(i)} \right) \left( \mathbf{X}_a^{(i,j)} - \overline{\mathbf{X}}_a^{(i)} \right)^T

   where :math:`N_t` denotes the total number of trials, and :math:`\mathbf{X}_a^{(i,j)}` denotes :math:`\mathbf{X}_a` of the :math:`j\text{-th}` trial for the :math:`i\text{-th}` stimulus.

Finally, the target stimulus can be predicted by 

.. math::

    \arg\max_{i\in\left\{1,2,\cdots,I\right\}}\left\{ \frac{\mathbf{U}^T\mathbf{X}_a\left(\overline{\mathbf{X}}_a^{(i)}\right)^T\mathbf{U}}{\sqrt{\mathbf{U}^T\mathbf{X}_a\mathbf{X}_a^T\mathbf{U}\mathbf{U}_i^T\left(\overline{\mathbf{X}}_a^{(i)}\right)\left(\overline{\mathbf{X}}_a^{(i)}\right)^T\mathbf{U}_i}} \right\}

.. py:function:: SSVEPAnalysisToolbox.algorithms.tdca.TDCA

    TDCA. The implementation directly follows above equations.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.

Oneline adaptive CCA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related paper:

+ C. M. Wong et al., “Online adaptation boosts SSVEP-based BCI performance,” *IEEE Trans. Biomed. Eng.*, vol. 69, no. 6, pp. 2018-2028, 2022. DOI: `10.1109/TBME.2021.3133594 <https://doi.org/10.1109/TBME.2021.3133594>`_.

Compared to sCCA, the spatial filters are fine-turned online. Therefore, as more trials are processed, the performance will be improved until achieving the upper bound. 

The OACCA ensembles three parts:

1. Spatial filters based on the sCCA. These spatial filters can be considered as the baseline reference of the recognition result.

2. Prototype spatial filters. These spatial filters are fine-turned online. They are calculated by

   .. math::

      \mathbf{u}_0^{[t+1]}=\arg\max_{\mathbf{u}}\frac{\mathbf{u}^T\left[ \sum_{m=1}^t\widetilde{\mathbf{u}}^{[m]}\left(\widetilde{\mathbf{u}}^{[m]}\right)^T \right]\mathbf{u}}{\mathbf{u}^T\mathbf{u}}

   where :math:`\widetilde{\mathbf{u}}^{[t]} = \frac{\mathbf{u}^{[t]}}{\left\|\mathbf{u}^{[t]}\right\|}` and :math:`\mathbf{u}^{[t]}` is the spatial filtered obtained from the sCCA at the :math:`t\text{-th}` trial.

3. Spatial filters based on the online ms-CCA. These spatial filters are also fine-turned online. They are calculated by 

   .. math::

      \mathbf{u}^{[t+1]},\;\mathbf{v}^{[t+1]}=\max_{\mathbf{u},\;\mathbf{v}}\frac{\mathbf{u}^T\mathbf{C}_{XY}^{[t]}\mathbf{v}}{\sqrt{\mathbf{u}^T\mathbf{C}_{XY}^{[t]}\mathbf{u} \cdot \mathbf{v}^T\mathbf{v}}}

   where :math:`\mathbf{C}_{XY}^{[t]}=\sum_{m=1}^t\left(\mathbf{X}^{[t]}\right)^T\mathbf{Y}`. 

The basic idea of the OACCA is shown below:

.. image:: ./_static/oacca.png

.. py:function:: SSVEPAnalysisToolbox.algorithms.cca.OACCA

    OACCA. The implementation directly follows above equations.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

