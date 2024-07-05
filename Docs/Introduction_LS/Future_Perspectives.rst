.. role::  raw-html(raw)
    :format: html

Future Perspectives
------------------------------------------------------------

We outline 3 potential avenues for future investigations by using the unified LS framework:

1. The proposed LS framework also offers insights into the development of novel computation strategies:

    + Consider trial weights in the combination matrices :math:`\mathbf{L}_\mathbf{E}` and :math:`\mathbf{L}_\mathbf{K}`;
    + Use the SSVEP template signals to construct the projection matrices :math:`\mathbf{P}_\mathbf{E}` and :math:`\mathbf{P}_\mathbf{K}` instead of the SSVEP reference signals.

2. The non-linearity and the regularization can be easily integrated into the computations of spatial filters. According to the generalized reduced rank regression (RRR) model, the proposed LS framework can be modified into their non-linear form with regularization, i.e., 

    .. math::

        \arg\min_{\mathbf{W},\mathbf{M}}\left\{\left\| \Phi\left(\mathbf{E}\right)\mathbf{W}\mathbf{M}^T-\Theta\left(\mathbf{T}\right)  \right\|^2_F + \rho\mathcal{R}\left(\mathbf{W}\right)\right\}

    where :math:`\Phi\left(\cdot\right)` and :math:`\Theta\left(\cdot\right)` denote the non-linear kernel methods of EEG features, :math:`\rho` is the regularization parameter, and :math:`\mathcal{R}\left(\cdot\right)` presents the regularization function. This toolbox already integrates common regularization functions in recognition methods implemented based on the LS framework. Users may check the source code to know how to integrate the regularization items.

3. The proposed LS framework demonstrates great potential for incorporating strategies in spatial filtering methods into ANN modelsmodels, and further envisions a promising future for developing the knowledge-embedded ANN-based models in the SSVEP recognition. For example, the loss function can be modified by using the proposed LS framework. 