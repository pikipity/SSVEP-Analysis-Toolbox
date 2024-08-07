.. SSVEP-Analysis_Toolbox documentation master file, created by
   sphinx-quickstart on Sun Jul 17 17:58:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SSVEP Analysis Toolbox
==================================================

This repository provides a python package for SSVEP datasets and recognition algorithms. The goal of this toolbox is to make researchers be familier with SSVEP signals and related recognition algorithms quickly, and focus on improving algorithms with shortest preparation time.

Most conventional recognition algorithms are implemented using both **eigen decomposition** and **least-square unified framework**. The least-square unified framework demonstrates various design strategies applied in the correlatiion analysis (CA)-based SSVEP spatial filtering algorithms and their relationships. 

The lease-square unified framework has been published in **IEEE TNSRE**: `<https://ieeexplore.ieee.org/document/10587150/>`_. If you use this toolbox, the lease-square unified framework, or the new spatial filtering methods (ms-MsetCCA-R-1, ms-MsetCCA-R-2, ms-MsetCCA-R-3, ms-(e)TRCA-R-1, ms-(e)TRCA-R-2), please cite this paper. 

.. parsed-literal::
  @article{LSFrameWork,
    title = {A least-square unified framework for spatial filtering in {SSVEP}-based {BCIs}},
    volume = {32},
    url = {https://ieeexplore.ieee.org/document/10587150/},
    doi = {10.1109/TNSRE.2024.3424410},
    journal = {IEEE Trans. Neural Syst. Rehabil. Eng.},
    author = {Wang, Ze and Shen, Lu and Yang, Yi and Ma, Yueqi and Wong, Chi Man and Liu, Zige and Lin, Cuiyun and Hon, Chi Tin and Qian, Tao and Wan, Feng},
    year = {2024},
    pages = {2470--2481},
  }

Contents
-------------

.. toctree::
   :maxdepth: 2
   
   Introduction_LS/Introduction_LS
   Installation_pages/Installation
   API_pages/API
   Demos_pages/Examples
   AskforHelp
   JoinUs
   Contributors
   License

Features
--------------

+ Mutiple implementations of various algorithms:
  + Eigen decomposition
  + Least-square unified framework
+ Unify formats of SSVEP datasets
+ Provide a standard processing procedure for faire performance comparisons
+ Python implementations of SSVEP recognition algorithms

Datasets and Algorithms
--------------------------

+ Datasets
  
  + **Benchmark** dataset
  + **BETA** dataset
  + **Nakanishi2015** dataset
  + **eldBETA** dataset
  + **openBMI** dataset
  + **Wearable** dataset

  .. todo::
    
    + `160 Targets SSVEP BCI Dataset <https://iopscience.iop.org/article/10.1088/1741-2552/ac0bfa>`_
    + `Dual Frequency and Phase Modulation SSVEP-BCI Dataset <https://iopscience.iop.org/article/10.1088/1741-2552/abaa9b/meta>`_
  
+ Algorithms
    + Implementations based on eigen decomposition
        + Standard canonical correlation analysis (**sCCA**) and filterbank CCA (**FBCCA**)
        + Individual template CCA (**itCCA**) and Extended CCA (**eCCA**)
        + Multi-stimulus CCA (**ms-CCA**)
        + Multi-set CCA (**MsetCCA**)
        + Multi-set CCA with reference signals (**MsetCCA-R**)
        + Task-related component analysis (**TRCA**) and Ensemble TRCA (**eTRCA**)
        + Task-related component analysis with reference signals (**TRCA-R**) and Ensemble TRCA with reference signals (**eTRCA-R**)
        + Sum of squared correlations (**SSCOR**) and Ensemble sum of squared correlations (**eSSCOR**)
        + Multi-stimulus TRCA (**ms-TRCA**)
        + Task-discriminant component analysis (**TDCA**)
        + Online adaptive CCA (**OACCA**)

    + Implementations based on least-square unified framework
        + sCCA, itCCA, eCCA, (e)TRCA, (e)TRCA-R, MsetCCA, MsetCCA-R, ms-CCA, ms-(e)TRCA, TDCA
        + **ms-MsetCCA-R-1**
        + **ms-MsetCCA-R-2**
        + **ms-MsetCCA-R-3**
        + **ms-(e)TRCA-R-1**
        + **ms-(e)TRCA-R-2**
  


  .. todo::
    
    + `ConvCA <https://github.com/yaoli90/Conv-CA>`_
    + `GuneyNet <https://github.com/pikipity/brainda/blob/master/brainda/algorithms/deep_learning/guney_net.py>`_
    + `Subject transfer CCA <https://github.com/edwin465/SSVEP-stCCA>`_
    + `Inter- and intra-subject maximal correlation <https://ieeexplore.ieee.org/document/9350285/>`_
    + `Least-squares transformation-based transfer learning <https://doi.org/10.1088/1741-2552/abcb6e>`_
    + `Leveraging cross-device shared latent responses <https://doi.org/10.1109/TBME.2019.2929745>`_
    + `Align and pool for EEG headset domain adaptation <https://ieeexplore.ieee.org/document/9516951/>`_
    + `Transfer learning CCA <https://github.com/edwin465/SSVEP-tlCCA>`_
    + `Stimulus-stimulus transafer based on time-frequency-joint representation <https://ieeexplore.ieee.org/document/9857586/>`_




