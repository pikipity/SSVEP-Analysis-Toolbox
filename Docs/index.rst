.. SSVEP-Analysis_Toolbox documentation master file, created by
   sphinx-quickstart on Sun Jul 17 17:58:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SSVEP Analysis Toolbox
==================================================

This repository provides a python package for SSVEP datasets and recognition algorithms. The goal of this toolbox is to make researchers be familier with SSVEP signals and related recognition algorithms quickly, and focus on improving algorithms with shortest preparation time.

Contents
-------------

.. toctree::
   :maxdepth: 2
   
   Installation
   API
   Demos
   AskforHelp
   JoinUs
   Contributors
   License

Features
--------------

+ Unify formats of SSVEP datasets
+ Provide a standard processing procedure for faire performance comparisons
+ Python implementations of SSVEP recognition algorithms

Datasets and Algorithms
--------------------------

+ Datasets
  
  + **Benchmark** dataset: 

      Y. Wang, X. Chen, X. Gao, and S. Gao, "A benchmark dataset for SSVEP-based braincomputer interfaces," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, no. 10, pp. 1746-1752, 2017. DOI: `10.1109/TNSRE.2016.2627556 <https://doi.org/10.1109/TNSRE.2016.2627556>`_.

  + **BETA** dataset: 
  
      B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "BETA: A large benchmark database toward SSVEP-BCI application," *Front. Neurosci.*, vol. 14, p. 627, 2020. DOI: `10.3389/fnins.2020.00627 <https://doi.org/10.3389/fnins.2020.00627>`_.

  + **Nakanishi2015** dataset: 
  
      M. Nakanishi, Y. Wang, Y.-T. Wang, T.-P. Jung, "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," *PLoS ONE*, vol. 10, p. e0140703, 2015. DOI: `10.1371/journal.pone.0140703 <https://doi.org/10.1371/journal.pone.0140703>`_.

  + **eldBETA** dataset: 
  
      B. Liu, Y. Wang, X. Gao, and X. Chen, "eldBETA: A Large eldercare-oriented benchmark database of SSVEP-BCI for the aging population," *Scientific Data*, vol. 9, no. 1, pp.1-12, 2022. DOI: `10.1038/s41597-022-01372-9 <https://www.nature.com/articles/s41597-022-01372-9>`_. 

  .. todo::
    
    + `BMI dataset <https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giz002/5304369>`_
    + `Wearable SSVEP BCI Dataset <http://bci.med.tsinghua.edu.cn/download.html>`_
    + `160 Targets SSVEP BCI Dataset <https://iopscience.iop.org/article/10.1088/1741-2552/ac0bfa>`_
    + `Dual Frequency and Phase Modulation SSVEP-BCI Dataset <https://iopscience.iop.org/article/10.1088/1741-2552/abaa9b/meta>`_
  
+ Algorithms
  
  + Standard canonical correlation analysis (**sCCA**) and filterbank CCA (**FBCCA**): 
  
      X. Chen, Y. Wang, S. Gao, T.-P. Jung, and X. Gao, , "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface," *J. Neural Eng.*, vol. 12, no. 4, p. 046008, 2015. DOI: `10.1088/1741-2560/12/4/046008 <https://doi.org/10.1088/1741-2560/12/4/046008>`_.

  + Individual template CCA (**itCCA**) and Extended CCA (**eCCA**): 
  
      X. Chen, Y. Wang, M. Nakanishi, X. Gao, T.-P. Jung, and S. Gao, "High-speed spelling with a noninvasive brain-computer interface," *Proc. Natl. Acad. Sci.*, vol. 112, no. 44, pp. E6058-E6067, 2015. DOI: `10.1073/pnas.1508080112 <https://doi.org/10.1073/pnas.1508080112>`_.

  + Multi-stimulus CCA (**ms-CCA**): 
  
      C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: `10.1088/1741-2552/ab2373 <https://doi.org/10.1088/1741-2552/ab2373>`_.

  + Multi-set CCA (**MsetCCA**):
  
      Y. Zhang, G. Zhou, J. Jin, X. Wang, A. Cichocki, "Frequency recognition in SSVEP-based BCI using multiset canonical correlation analysis," *Int J Neural Syst.*, vol. 24, 2014, p. 1450013. DOI: `10.1142/ S0129065714500130 <https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130>`_.

  + Multi-set CCA with reference signals (**MsetCCA-R**): 
  
      C. M. Wong, B. Wang, Z. Wang, K. F. Lao, A. Rosa, and F. Wan, "Spatial filtering in SSVEP-based BCIs: Unified framework and new improvements.," *IEEE Transactions on Biomedical Engineering*, vol. 67, no. 11, pp. 3057-3072, 2020. DOI: `10.1109/TBME.2020.2975552 <https://ieeexplore.ieee.org/document/9006809/>`_.

  + Task-related component analysis (**TRCA**) and Ensemble TRCA (**eTRCA**): 

      M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component Analysis," *IEEE Trans. Biomed. Eng.*, vol. 65, no. 1, pp. 104-112, 2018. DOI: `10.1109/TBME.2017.2694818 <https://doi.org/10.1109/TBME.2017.2694818>`_.

  + Task-related component analysis with reference signals (**TRCA-R**) and Ensemble TRCA with reference signals (**eTRCA-R**): 

      C. M. Wong, B. Wang, Z. Wang, K. F. Lao, A. Rosa, and F. Wan, "Spatial filtering in SSVEP-based BCIs: Unified framework and new improvements.," *IEEE Transactions on Biomedical Engineering*, vol. 67, no. 11, pp. 3057-3072, 2020. DOI: `10.1109/TBME.2020.2975552 <https://ieeexplore.ieee.org/document/9006809/>`_.

  + Sum of squared correlations (**SSCOR**) and Ensemble sum of squared correlations (**eSSCOR**): 

      G. K. Kumar, and M. R. Reddy, "Designing a sum of squared correlations framework for enhancing SSVEP-based BCIs," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 27, no. 10, pp. 2044-2050, 2019. DOI: `10.1109/TNSRE.2019.2941349 <https://doi.org/10.1109/TNSRE.2019.2941349>`_.

  + Multi-stimulus TRCA (**ms-TRCA**): 

      C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: `10.1088/1741-2552/ab2373 <https://doi.org/10.1088/1741-2552/ab2373>`_.

  + Task-discriminant component analysis (**TDCA**)

      B. Liu, X. Chen, N. Shi, Y. Wang, S. Gao, X. Gao, "Improving the performance of individually calibrated SSVEP-BCI by task-discriminant component analysis." *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 1998-2007, 2021. DOI: `10.1109/TNSRE.2021.3114340 <https://doi.org/10.1109/TNSRE.2021.3114340>`_.

  + Online adaptive CCA (**OACCA**)

      C. M. Wong et al., “Online adaptation boosts SSVEP-based BCI performance,” *IEEE Trans. Biomed. Eng.*, vol. 69, no. 6, pp. 2018-2028, 2022. DOI: `10.1109/TBME.2021.3133594 <https://doi.org/10.1109/TBME.2021.3133594>`_.

  .. todo::
    
    + `ConvCA <https://github.com/yaoli90/Conv-CA>`_
    + `GuneyNet <https://github.com/pikipity/brainda/blob/master/brainda/algorithms/deep_learning/guney_net.py>`_




