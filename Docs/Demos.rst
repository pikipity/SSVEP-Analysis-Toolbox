Demo Codes
==========================

Recognition Performance in Benchmark Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The individual performance on the Benchmark Dataset with various signal lengths is verified in this demo. The classification accuracies, ITRs, training time, testing time, and confusion matrices are verified and stored in :file:`res/benchmarkdataset_res.mat`. 

This demo shows the following points: 

+ How to use the Benchmark Dataset. When you first try this demo, the benchmark dataset will be downloaded in the folder  :file:`2016_Tsinghua_SSVEP_database`. 
+ How to create recognition models. 
+ How to use the provided evaluator ``BaseEvaluator`` to verify recognition performance.

.. tip::

  + This demo uses ``gen_trials_onedataset_individual_diffsiglen`` to generate evaluation trials used for ``BaseEvaluator``. These trials are used to evaluate indivudal performance on various signal lengths. If your target is not to evaluate these performance, you can follow this function to prepare your own evaluation trials.
  + This demo uses ``cal_performance_onedataset_individual_diffsiglen`` and ``cal_confusionmatrix_onedataset_individual_diffsiglen`` to calculate recognition performance (accuracies and ITRs) and confusion matrices. These two functions are also used to calculate individual performance on various signal lengths. For your own evaluation trials, you can follow these two functions to evaluate your own performance.
  + You can adjust the threading number by changing ``n_jobs`` in ``evaluator.run``. Higher number requires the computer with higher performance. The current demo may occupy mora than 10 hours. You may reduce the number of models or the number of signal lengths to condense the running time.
  + ITRs are related to computational time. Different implementations may lead to different computational time. You may check the recorded testing time to get know the time used for ITR computation. We are also try to optimize implementations to reduce the computational time. For example, the sCCA implemented based on the QR decomposition is faster than the sCCA implemented based on the conventional canonical correlation with the same performance as shown in `"Plot Recognition Performance" demo <#plot-recognition-performance>`_. 

Demo file: :file:`demo/benchmarkdataset.py`

.. literalinclude:: ../demo/benchmarkdataset.py
    :language: python

Recognition Performance in BETA Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This demo is almost same as the above demo. The only difference is that this demo uses the BETA Dataset. Results are stored in :file:`res/betadataset_res.mat`.

Demo file: :file:`demo/betadataset.py`

.. literalinclude:: ../demo/betadataset.py
    :language: python

Recognition Performance of Online Adaptive Method (OACCA) in Benchmark Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This demo is also similar as the first demo. The key differences are how to evaluation trials and how to calculate performance. If you want to define your own functions of generating evaluation trials and calculating related performance, you may refer this demo.

Demo file: :file:`demo/benchmarkdataset_online.py`

.. literalinclude:: ../demo/benchmarkdataset_online.py
    :language: python

Plot Recognition Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This demo uses bar graph with error bars and shadow lines to plot classification accuracies, ITRs, training time, and testing time. Before running this demo, please run above two demos to obtain :file:`res/benchmarkdataset_res.mat` and :file:`res/betadataset_res.mat` files. 

This demo shows the following points:

+ How to use provided ``bar_plot_with_errorbar`` to plot the bar grapth with error bars.
+ How to use provided ``shadowline_plot`` to plot shadow lines.

Demo file: :file:`demo/plot_performance.py`

.. literalinclude:: ../demo/plot_performance.py
    :language: python

Generated graphs are stored in :file:`demo/res`. Parts of graphs are shown below.

+ Classification accuracies of the Benchmark Dataset:

  .. image:: ../demo/res/benchmark_acc_bar.jpg

  .. image:: ../demo/res/benchmark_acc_shadowline.jpg

+ Classification accuracies of the BETA Dataset:

  .. image:: ../demo/res/beta_acc_bar.jpg

  .. image:: ../demo/res/beta_acc_shadowline.jpg

+ Testing time of the Benchmark Dataset:

  .. image:: ../demo/res/benchmark_testtime_bar.jpg

+ Testing time of the BETA Dataset:

  .. image:: ../demo/res/beta_testtime_bar.jpg

Plot Recognition Performance of Online Adaptive Method (OACCA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This demo is similar as the above demo. The key difference is that this demo shows the performance changes along trials.

Demo file: :file:`demo/plot_performance_online.py`

.. literalinclude:: ../demo/plot_performance_online.py
    :language: python

+ Classification accuracies of the Benchmark Dataset:

  .. image:: ../demo/res/benchmark_OACCA_acc_shadowline.jpg

Plot Confusion Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This demo provides an example of plotting confusion matrices. This demo directly uses ``imshow`` in ``matplotlib`` to plot confusion matrices. You also can use ``heatmap`` [#heatmap]_ in ``seaborn`` or ``plot_confusion_matrix`` [#plot_confusion_matrix]_ in ``sklearn`` to plot these confusion matrices. This demo only shows confusion matrices at 0.5-s signal length, which is controlled by ``target_time`` in the demo file. Moreover, all subjects' confusion matrices are summed together. 

.. [#heatmap] `Plot confusion matrices using seaborn <https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/>`_
.. [#plot_confusion_matrix] `Plot confusion matrices using sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html>`_

Demo file: :file:`demo/plot_confusion_matrix.py`

.. literalinclude:: ../demo/plot_confusion_matrix.py
    :language: python

Generated graphs are stored in :file:`demo/res/benchmarkdataset_confusionmatrix` and :file:`demo/res/beta_confusionmatrix`. Parts of graphs are shown below.

+ eCCA (0.5s) in Benchmark Dataset

  .. image:: ../demo/res/benchmarkdataset_confusionmatrix/eCCA_T0.5.jpg

+ eCCA (0.5s) in BETA Dataset

  .. image:: ../demo/res/beta_confusionmatrix/eCCA_T0.5.jpg

+ eTRCA (0.5s) in Benchmark Dataset

  .. image:: ../demo/res/benchmarkdataset_confusionmatrix/eTRCA_T0.5.jpg

+ eTRCA (0.5s) in BETA Dataset

  .. image:: ../demo/res/beta_confusionmatrix/eTRCA_T0.5.jpg
