.. role::  raw-html(raw)
    :format: html

Implementations based on least-square unified framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We summarized various correlation analysis (CA) based spatial filtering algorithms as a least-square unified framework:

.. math:: 

    \arg\min_{\mathbf{W},\mathbf{M}} \left\|\mathbf{E}\mathbf{W}\mathbf{M}^T-\mathbf{T}\right\|^2_F,

.. math::

    \text{subject to} \mathbf{M}^T\mathbf{M}=\mathbf{I},

where :math:`\mathbf{W}` is the spatial filter. The matrix :math:`\mathbf{E}` presents the combined EEG features of inter-classes, i.e.,

.. math::

    \mathbf{E}=\mathbf{L}_\mathbf{E}\mathbf{P}_\mathbf{E}\mathbf{Z},

where :math:`\mathbf{L}_\mathbf{E}` denotes the combination matrix of inter-class EEG features, :math:`\mathbf{P}_\mathbf{E}` denotes the orthogonal matrix applied for generating inter-class EEG features, and :math:`\mathbf{Z}` denotes the EEG data. In addition, the matrix :math:`\mathbf{T}` is

.. math::

    \mathbf{T}=\mathbf{E}\,\mathcal{S}\!\left(\mathbf{K}\right).

Similarly as :math:`\mathbf{E}`, :math:`\mathbf{K}` presents the combined EEG features of intra-classes, i.e., 

.. math::

    \mathbf{K}=\mathbf{L}_\mathbf{K}\mathbf{P}_\mathbf{K}\mathbf{Z},

where :math:`\mathbf{L}_\mathbf{K}` denotes the combination matrix of intra-class EEG features, and :math:`\mathbf{P}_\mathbf{K}` denotes the orthogonal matrix applied for generating intra-class EEG features. 

The values of :math:`\mathbf{Z}`, :math:`\mathbf{L}_\mathbf{E}`, :math:`\mathbf{P}_\mathbf{E}`, :math:`\mathbf{L}_\mathbf{K}`, and :math:`\mathbf{P}_\mathbf{K}` in the CA-based spatial filtering methods are shown in the following table.

.. image:: ../_static/lsframework_table.png

This reduced rank regression problem can be solved by  the alternated least squares:

1. Initilize :math:`\mathbf{M}`: 

    .. math::

        \mathbf{M}=\mathbf{V}^{\left( \mathbf{T} \right)}\text{ where }\mathbf{V}^{\left( \mathbf{T} \right)}\text{ is obtained from the SVD of }\mathbf{T}.

2. Update :math:`\mathbf{W}`:

    .. math::

        \mathbf{W}=\left(\mathbf{E}^T\mathbf{E}\right)^{-1}\mathbf{E}^T\mathbf{T}\mathbf{M}.

3. Update :math:`\mathbf{M}`:

    .. math::

        \mathbf{M}=\mathbf{U}^{\left(\mathbf{P}\right)}\left(\mathbf{V}^{\left(\mathbf{P}\right)}\right)^T.\text{ where }\mathbf{U}^{\left(\mathbf{P}\right)}\text{ and }\mathbf{V}^{\left(\mathbf{P}\right)}\text{ are obtained from the SVD of }\mathbf{P}

4. Repeat the steps 2 and 3 until :math:`\mathbf{W}` and :math:`\mathbf{M}` are converged. 

Standard CCA and filterbank CCA
""""""""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.SCCA_ls

    FBCCA implemented based on the least-square unified framework

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param force_output_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be stored. Otherwise, they will not be stored. Default is ``False``.

    :param update_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be re-computed in following testing trials. Otherwise, they will not be re-computed if they are already existed. Default is ``True``.

.. note::

    Although the FBCCA is a training-free method, these models still need run `"fit" function <#fit>`_ to store reference signals in the model.

Individual template CCA (itCCA) and extended CCA (eCCA)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.ITCCA_ls

    itCCA implemented based on the least-square unified framework

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param force_output_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be stored. Otherwise, they will not be stored. Default is ``False``.

    :param update_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be re-computed in following testing trials. Otherwise, they will not be re-computed if they are already existed. Default is ``True``.

.. py:function:: SSVEPAnalysisToolbox.algorithms.ECCA_ls

    eCCA implemented based on the least-square unified framework

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param update_UV: If ``True``, :math:`\left\{\mathbf{U}_i,\mathbf{V}_i\right\}_{i=1,2,\cdots,I}` will be re-computed in following training and testing trials. Otherwise, they will not be re-computed if they are already existed. Default is ``True``.

Multi-stimulus CCA
"""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.MSCCA_ls

    ms-CCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``12``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Multi-set CCA (MsetCCA)
"""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.MsetCCA_ls

    Multi-set CCA implemented based on the least-square unified framework

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Multi-set CCA with reference signals (MsetCCA-R)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.MsetCCAwithR_ls

    Multi-set CCA with reference signals implemented based on the least-square unified framework

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Task-related component analysis (TRCA) and ensemble TRCA (eTRCA)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.TRCA_ls

    TRCA implemented based on the least-square unified framework

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

.. py:function:: SSVEPAnalysisToolbox.algorithms.ETRCA_ls

    eTRCA implemented based on the least-square unified framework

    :param n_component: This parameter will not be considered in the eTRCA. 

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

TRCA with reference signals (TRCA-R) and eTRCA with reference signals (eTRCA-R)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.TRCAwithR_ls

    TRCA-R implemented based on the least-square unified framework

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

.. py:function:: SSVEPAnalysisToolbox.algorithms.ETRCAwithR_ls

    eTRCA-R implemented based on the least-square unified framework

    :param n_component: This parameter will not be considered in the eTRCA-R. 

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Multi-stimulus TRCA
"""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.MSETRCA_ls

    ms-TRCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: This parameter will not be considered in this function. 

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

Task-discriminant component analysis
"""""""""""""""""""""""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.TDCA_ls

    TDCA implemented based on the least-square unified framework

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.

ms-eTRCA-R-1
""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.ms_trca_r_1

    TDCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.

ms-eTRCA-R-2
""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.ms_trca_r_2

    TDCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.

ms-MsetCCA-R-1
""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.ms_msetcca_r_1

    TDCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.

ms-MsetCCA-R-2
""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.ms_msetcca_r_2

    TDCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.

ms-MsetCCA-R-3
""""""""""""""""""""

.. py:function:: SSVEPAnalysisToolbox.algorithms.ms_msetcca_r_3

    TDCA implemented based on the least-square unified framework

    :param n_neighbor: Number of neighbers considered for computing the spatial filter of one stimulus. Default is ``2``.

    :param n_component: Number of components of eigen vectors that will be applied as the spatial filters. The default number is ``1``, which means the eigen vector with the highest eigen value is regarded as the spatial filter.

    :param n_jobs: Number of threadings. If the given value is larger than 1, the parallel computation will be applied to improve the computational speed. Default is ``None``, which means the parallel computation will not be applied. 

    :param weights_filterbank: Weights of filterbanks. It is a list of float numbers. Default is ``None``, which means all weights of filterbanks are 1.

    :param n_delay: Total number of delays. Default is ``0``, which means no delay.