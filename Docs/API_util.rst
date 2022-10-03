.. role::  raw-html(raw)
    :format: html

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