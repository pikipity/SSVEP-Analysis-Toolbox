.. role::  raw-html(raw)
    :format: html

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

    It contains following attributes:

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

    It contains following attributes:

    + ``true_label_train``: After training, to evaluate the training performance, the list of true labels of training trials is stored in this attribute. The format is 

      .. code-block:: python

        [[true_label_1, true_label_2, ...],
         [true_label_1, true_label_2, ...],
         ...,
         [true_label_1, true_label_2, ...]]

      where ``true_label_1``, ``true_label_2``, ... are true labels of different evaluation trials.
    
    + ``pred_label_train``: After training, to evaluate the training performance, the list of predicted labels of training trials is stored in this attribute. The format is same as ``true_label_train``.
    + ``true_label_test``: The list of true labels of testing trials is stored in this attribute. The format is same as ``true_label_train``.
    + ``pred_label_test``: The list of predicted labels of testing trials is stored in this attribute. The format is same as ``true_label_train``.
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