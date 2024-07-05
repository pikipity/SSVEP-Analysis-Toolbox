.. role::  raw-html(raw)
    :format: html

Example 2: How to run and compare various methods 
-----------------------------------------------------

This example shows 

1. How to training and testing different recognition methods under the same simulation settings for the performance comparisons;
2. How to use the build-in functions to conduct the individual recognition with the leave-one-block-out cross validation.

The Benchmark Dataset is applied as an example. You can find the related code in :file:`demo/benchmarkdataset.py` or :file:`demo/benchmarkdataset.ipynb`.

Same as the 1st example, we firstly need to add the toolbox into the
search path, prepare the dataset, and hook the preprocessing and
filter-bank methods. Because we already download the dataset in the 1st
example, the dataset will not be downloaded again.

.. code:: ipython3

    from SSVEPAnalysisToolbox.datasets import BenchmarkDataset
    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import (
        preprocess, filterbank
    )
    dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(filterbank)

Now, we can prepare the simulation. In this example,

1. We only use 9 occiple channels;
2. All 40 classes in this dataset are included;
3. 5 harmonic components are considered in the SSVEP reference signals;
4. The performance with signal lengths from 0.25s to 1.00s will be
   verified and compared in this example;
5. Based on above simulation settings, we will use the build-in function
   to automatically generate the training and testing trials of the
   individual recognition with the leave-one-block-out cross validation.
   Users also can build their own evaluation trials by referring this
   build-in function.

.. code:: ipython3

    dataset_container = [ dataset ]
    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import suggested_ch
    ch_used = suggested_ch()
    all_trials = [i for i in range(dataset.trial_num)]
    harmonic_num = 5
    tw_seq = [i/100 for i in range(25,100+5,5)]
    from SSVEPAnalysisToolbox.evaluator import gen_trials_onedataset_individual_diffsiglen
    trial_container = gen_trials_onedataset_individual_diffsiglen(dataset_idx = 0,
                                                                 tw_seq = tw_seq,
                                                                 dataset_container = dataset_container,
                                                                 harmonic_num = harmonic_num,
                                                                 trials = all_trials,
                                                                 ch_used = ch_used,
                                                                 t_latency = None,
                                                                 shuffle = False)

Then, we need to initialize the recognition methods for the performance
comparisions. In this example, we compare the eTRCA implemented based on
the QR decomposition and the least-square framework. For other methods,
we only provide the suggested parameters for the Benchmark Dataset for
your reference.

.. code:: ipython3

    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import suggested_weights_filterbank
    weights_filterbank = suggested_weights_filterbank()
    from SSVEPAnalysisToolbox.algorithms import (
        SCCA_qr, SCCA_canoncorr, ECCA, MSCCA, MsetCCA, MsetCCAwithR,
        TRCA, ETRCA, MSETRCA, MSCCA_and_MSETRCA, TRCAwithR, ETRCAwithR, SSCOR, ESSCOR,
        TDCA,
        SCCA_ls, SCCA_ls_qr,
        ECCA_ls, ITCCA_ls,
        MSCCA_ls,
        TRCA_ls, ETRCA_ls,
        MsetCCA_ls,
        MsetCCAwithR_ls,
        TRCAwithR_ls, ETRCAwithR_ls,
        MSETRCA_ls,
        TDCA_ls
    )
    model_container = [
                       # SCCA_qr(weights_filterbank = weights_filterbank),
                       # SCCA_canoncorr(weights_filterbank = weights_filterbank),
                       # MsetCCA(weights_filterbank = weights_filterbank),
                       # MsetCCAwithR(weights_filterbank = weights_filterbank),
                       # ECCA(weights_filterbank = weights_filterbank),
                       # MSCCA(n_neighbor = 12, weights_filterbank = weights_filterbank),
                       # SSCOR(weights_filterbank = weights_filterbank),
                       # ESSCOR(weights_filterbank = weights_filterbank),
                       # TRCA(weights_filterbank = weights_filterbank),
                       # TRCAwithR(weights_filterbank = weights_filterbank),
                       ETRCA(weights_filterbank = weights_filterbank),
                       # ETRCAwithR(weights_filterbank = weights_filterbank),
                       # MSETRCA(n_neighbor = 2, weights_filterbank = weights_filterbank),
                       # MSCCA_and_MSETRCA(n_neighbor_mscca = 12, n_neighber_msetrca = 2, weights_filterbank = weights_filterbank),
                       # TDCA(n_component = 8, weights_filterbank = weights_filterbank, n_delay = 6)
                       ETRCA_ls(weights_filterbank = weights_filterbank),
                      ]

After preparing the dataset, the recognition methods and the simulation
settings, we can use the build-in function to run the evaulation. The
parameter ``n_jobs`` is the number of threading. Higher number requires
the computer with higher performance. You can adjust this parameter
based on your own situation, or set it as ``-1`` to automatically
generate the threading number based on your core number in your CPU.

.. code:: ipython3

    from SSVEPAnalysisToolbox.evaluator import BaseEvaluator
    evaluator = BaseEvaluator(dataset_container = dataset_container,
                              model_container = model_container,
                              trial_container = trial_container,
                              save_model = False,
                              disp_processbar = True)
    
    evaluator.run(n_jobs = 5,
                  eval_train = False)


.. parsed-literal::

    
    ========================
       Start
    ========================

    100.000%|████████████████████████████████████████████████████████████| 3360/3360 [Time: 4:28:51<00:00]

    ========================
       End
    ========================
    


All simulation results has been stored in ``evaluator``. We can save it
for further analysis.

.. code:: ipython3

    evaluator_file = 'res/benchmarkdataset_evaluator.pkl'
    evaluator.save(evaluator_file)

Then, we can use the build-in function to calculate the recognition the
accuracy, the ITR, and the confusion matrix. It should be noticed that
the following build-in functions are only designed to evaluate the
individual recognition performance with the leave-one-block-out cross
evaluation. In other words, the training and testing trails must be
generated by the function
``gen_trials_onedataset_individual_diffsiglen``. Otherwise, you may need
to use other build-in functions or write your own calculation functions
by referring these build-in functions.

.. code:: ipython3

    from SSVEPAnalysisToolbox.evaluator import (
        cal_performance_onedataset_individual_diffsiglen, 
        cal_confusionmatrix_onedataset_individual_diffsiglen
    )
    acc_store, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                             dataset_idx = 0,
                                                                             tw_seq = tw_seq,
                                                                             train_or_test = 'test')
    confusion_matrix = cal_confusionmatrix_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                            dataset_idx = 0,
                                                                            tw_seq = tw_seq,
                                                                            train_or_test = 'test')  

We also can separate the training and testing time from ``evaluator``.
This part also demonstrates how to get evaluation results from
``evaluator``. You can follow the idea to compute the recognition
accuracy or ITR.

.. code:: ipython3

    import numpy as np
    train_time = np.zeros((len(model_container), len(evaluator.performance_container)))
    test_time = np.zeros((len(model_container), len(evaluator.performance_container)))
    for trial_idx, performance_trial in enumerate(evaluator.performance_container):
        for method_idx, method_performance in enumerate(performance_trial):
            train_time[method_idx, trial_idx] = sum(method_performance.train_time)
            test_time[method_idx, trial_idx] = sum(method_performance.test_time_test)
    train_time = train_time.T
    test_time = test_time.T

Finally, we can store all results for further analysis. This example
will show you how to store all results in ``mat`` file (MATLAB format).
You also can use this function to store results as ``np`` file (numpy
data file).

.. code:: ipython3

    from SSVEPAnalysisToolbox.utils.io import savedata
    data = {"acc_store": acc_store,
            "itr_store": itr_store,
            "train_time": train_time,
            "test_time": test_time,
            "confusion_matrix": confusion_matrix,
            "tw_seq":tw_seq,
            "method_ID": [model.ID for model in model_container]}
    data_file = 'res/benchmarkdataset_res.mat'
    savedata(data_file, data, 'mat')
