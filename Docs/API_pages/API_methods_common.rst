.. role::  raw-html(raw)
    :format: html

.. _common-functions-in-methods:

Common methods for all models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All following recognition models have these methods. The inputs and outputs are same so they will not be repeatedly introduced in following sections. 

When you define your own algorithm class, You may use the ``BaseModel`` as the father class and re-define the ``__init__`` method and the following methods. 

.. py:function:: __copy__

    Copy the recognition model.

    :return:

        + ``model``: The returned new model is same as the original one.

.. py:function:: fit

    Train the recognition model. The trained model parameters will be stored in the class parameter ``model``. Different methods may require different input parameters. You may follow the below parameter names to define your own fit function. 

    :param X: List of training EEG signals. Each element is one 3D single trial EEG signal (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

    :param Y: List of training labels. Each element is one single trial label that is an integer number.

    :param ref_sig: List of reference signals. Each element is the reference signal of one stimulus. 

    :param freqs: List of stimulus frequencies. 

.. py:function:: predict

    Recognize the testing signals.

    :param X: List of testing EEG signals. Each element is one 3D single trial EEG signal (filterbank :raw-html:`&#215;` channels :raw-html:`&#215;` samples).

    :return:

        + ``Y_pred``: List of predicted labels for testing signals. Each element is one single trial label that is an integer number.