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
  
+ Algorithms
  
  + Standard canonical correlation analysis (**sCCA**) and filterbank CCA (**FBCCA**): 
  
      X. Chen, Y. Wang, S. Gao, T.-P. Jung, and X. Gao, , "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface," *J. Neural Eng.*, vol. 12, no. 4, p. 046008, 2015. DOI: `10.1088/1741-2560/12/4/046008 <https://doi.org/10.1088/1741-2560/12/4/046008>`_.

  + Extended CCA (**eCCA**): 
  
      X. Chen, Y. Wang, M. Nakanishi, X. Gao, T.-P. Jung, and S. Gao, "High-speed spelling with a noninvasive brain-computer interface," *Proc. Natl. Acad. Sci.*, vol. 112, no. 44, pp. E6058-E6067, 2015. DOI: `10.1073/pnas.1508080112 <https://doi.org/10.1073/pnas.1508080112>`_.

  + Multi-stimulus CCA (**ms-CCA**): 
  
      C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: `10.1088/1741-2552/ab2373 <https://doi.org/10.1088/1741-2552/ab2373>`_.

  + Task-related component analysis (**TRCA**) and Ensemble TRCA (**eTRCA**): 

      M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component Analysis," *IEEE Trans. Biomed. Eng.*, vol. 65, no. 1, pp. 104-112, 2018. DOI: `10.1109/TBME.2017.2694818 <https://doi.org/10.1109/TBME.2017.2694818>`_.

  + Multi-stimulus TRCA (**ms-TRCA**): 

      C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: `10.1088/1741-2552/ab2373 <https://doi.org/10.1088/1741-2552/ab2373>`_.

  + ...




