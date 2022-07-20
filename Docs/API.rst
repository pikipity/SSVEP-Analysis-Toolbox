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

Utility Function
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



