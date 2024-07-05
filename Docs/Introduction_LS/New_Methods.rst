.. role::  raw-html(raw)
    :format: html

New Spatial Filtering Methods
------------------------------------------------------------

To fill the research gaps and demonstrate that the proposed LS framework could
facilitate the development of new methods, 5 new spatial filtering methods are designed by
integrating the strategies. 

New TRCA-related methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ms-eTRCA-R-2:

    + Only include the projection matrix in :math:`\mathbf{P}_\mathbf{E}` (NOT :math:`\mathbf{P}_\mathbf{K}`);

    + :math:`\mathbf{L}_\mathbf{E}`, :math:`\mathbf{L}_\mathbf{K}` and :math:`\mathbf{Z}` are designed following the TDCA;

    + The utilization strategies of spatial filters follow the eTRCA (Combine spatial filters of multiple stimuli).

2. ms-eTRCA-R-1:

    + Include the projection matrix in both :math:`\mathbf{P}_\mathbf{E}` and :math:`\mathbf{P}_\mathbf{K}`;

    + :math:`\mathbf{L}_\mathbf{E}`, :math:`\mathbf{L}_\mathbf{K}` and :math:`\mathbf{Z}` are designed following the TDCA;

    + The utilization strategies of spatial filters follow the eTRCA (Combine spatial filters of multiple stimuli).


New MsetCCA-related methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ms-MsetCCA-2:

    + Combine multi-stimulus data in :math:`\mathbf{Z}` of the MsetCCA;

    + Only include the projection matrix in :math:`\mathbf{P}_\mathbf{E}` (NOT :math:`\mathbf{P}_\mathbf{K}`);

    + :math:`\mathbf{L}_\mathbf{E}`, :math:`\mathbf{L}_\mathbf{K}` and :math:`\mathbf{Z}` are designed following the TDCA;

    + The utilization strategies of spatial filters follow the ms-eTRCA (Combine spatial filters of multiple trials of multiple stimuli).

2. ms-MsetCCA-1:

    + Combine multi-stimulus data in :math:`\mathbf{Z}` of the MsetCCA;

    + Include the projection matrix in both :math:`\mathbf{P}_\mathbf{E}` and :math:`\mathbf{P}_\mathbf{K}`;

    + :math:`\mathbf{L}_\mathbf{E}`, :math:`\mathbf{L}_\mathbf{K}` and :math:`\mathbf{Z}` are designed following the TDCA;

    + The utilization strategies of spatial filters follow the ms-eTRCA (Combine spatial filters of multiple trials of multiple stimuli).

3. ms-MsetCCA-3:

    + Combine multi-stimulus data in :math:`\mathbf{Z}` of the MsetCCA;

    + Include the projection matrix in both :math:`\mathbf{P}_\mathbf{E}` and :math:`\mathbf{P}_\mathbf{K}`;

    + :math:`\mathbf{L}_\mathbf{E}`, :math:`\mathbf{L}_\mathbf{K}`, `\mathbf{P}_\mathbf{E}`, :math:`\mathbf{P}_\mathbf{K}` and :math:`\mathbf{Z}` are designed following the TDCA;

    + The utilization strategies of spatial filters follow the ms-eTRCA (Combine spatial filters of multiple trials of multiple stimuli).


Performance evaluation of new TRCA- and MsetCCA-related methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+ Compare TRCA-related methods in Benchmark Dataset:

    .. image:: ../_static/TRCA_performance.png

+ Compare MsetCCA-related methods in Benchmark Dataset:

    .. image:: ../_static/MsetCCA_performance.png

+ Compare multi-stimulus methods in Benchmark Dataset:

    .. image:: ../_static/MultiFreq_performance.png