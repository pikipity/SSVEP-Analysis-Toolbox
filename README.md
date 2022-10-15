# SSVEP Analysis Toolbox

This repository provides a python package for SSVEP datasets and recognition algorithms. The goal of this toolbox is to make researchers be familier with SSVEP signals and related recognition algorithms quickly, and focus on improving algorithms with shortest preparation time.

More detailed information can be found in [document](http://ssvep-analysis-toolbox.readthedocs.io/) ([Chinese version](https://ssvep-analysis-toolbox.readthedocs.io/zh_CN/latest/)).

## Features

+ Unify formats of SSVEP datasets
+ Provide a standard processing procedure for faire performance comparisons
+ Python implementations of SSVEP recognition algorithms

## Datasets and Algorithms

+ Datasets
  + **Benchmark** dataset: Y. Wang, X. Chen, X. Gao, and S. Gao, "A benchmark dataset for SSVEP-based braincomputer interfaces," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, no. 10, pp. 1746–1752, 2017. DOI: [10.1109/TNSRE.2016.2627556](https://doi.org/10.1109/TNSRE.2016.2627556).
  + **BETA** dataset: B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "BETA: A large benchmark database toward SSVEP-BCI application," *Front. Neurosci.*, vol. 14, p. 627, 2020. DOI: [10.3389/fnins.2020.00627](https://doi.org/10.3389/fnins.2020.00627).
  + **Nakanishi2015** dataset: M. Nakanishi, Y. Wang, Y.-T. Wang, T.-P. Jung, "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," *PLoS ONE*, vol. 10, p. e0140703, 2015. DOI: [10.1371/journal.pone.0140703](https://doi.org/10.1371/journal.pone.0140703).
  + **eldBETA** dataset: B. Liu, Y. Wang, X. Gao, and X. Chen, "eldBETA: A Large eldercare-oriented benchmark database of SSVEP-BCI for the aging population," *Scientific Data*, vol. 9, no. 1, pp.1-12, 2022. DOI: [10.1038/s41597-022-01372-9](https://www.nature.com/articles/s41597-022-01372-9).
  + **openBMI** dataset: M.-H. Lee, O.-Y. Kwon, Y.-J. Kim, H.-K. Kim, Y.-E. Lee, J. Williamson, S. Fazli, and S.-W. Lee, "EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy," GigaScience, vol. 8, no. 5, p. giz002, 2019. DOI: [10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002).
  + **Wearable** dataset: F. Zhu, L. Jiang, G. Dong, X. Gao, and Y. Wang, “An Open Dataset for Wearable SSVEP-Based Brain-Computer Interfaces,” Sensors, vol. 21, no. 4, p. 1256, 2021. DOI: [10.3390/s21041256](https://www.mdpi.com/1424-8220/21/4/1256).
  + ...

+ Algorithms
  + Standard canonical correlation analysis (**sCCA**) and filterbank CCA (**FBCCA**): Chen, X., Wang, Y., Gao, S., Jung, T.P. and Gao, X., "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface," *J. Neural Eng.*, vol. 12, no. 4, p. 046008, 2015. DOI: [10.1088/1741-2560/12/4/046008](https://doi.org/10.1088/1741-2560/12/4/046008).
  + Individual template CCA (**itCCA**) and Extended CCA (**eCCA**): X. Chen, Y. Wang, M. Nakanishi, X. Gao, T.-P. Jung, and S. Gao, "High-speed spelling with a noninvasive brain–computer interface," *Proc. Natl. Acad. Sci.*, vol. 112, no. 44, pp. E6058–E6067, 2015. DOI: [10.1073/pnas.1508080112](https://doi.org/10.1073/pnas.1508080112).
  + Multi-stimulus CCA (**ms-CCA**): C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: [10.1088/1741-2552/ab2373](https://doi.org/10.1088/1741-2552/ab2373).
  + Multi-set CCA (**MsetCCA**): Y. Zhang, G. Zhou, J. Jin, X. Wang, A. Cichocki, "Frequency recognition in SSVEP-based BCI using multiset canonical correlation analysis," *Int J Neural Syst.*, vol. 24, 2014, p. 1450013. DOI: [10.1142/ S0129065714500130](https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130).
  + Multi-set CCA with reference signals (**MsetCCA-R**): C. M. Wong, B. Wang, Z. Wang, K. F. Lao, A. Rosa, and F. Wan, "Spatial filtering in SSVEP-based BCIs: Unified framework and new improvements.," *IEEE Transactions on Biomedical Engineering*, vol. 67, no. 11, pp. 3057-3072, 2020. DOI: [10.1109/TBME.2020.2975552](https://ieeexplore.ieee.org/document/9006809/).
  + Task-related component analysis (**TRCA**) and Ensemble TRCA (**eTRCA**): M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component Analysis," *IEEE Trans. Biomed. Eng.*, vol. 65, no. 1, pp. 104–112, 2018. DOI: [10.1109/TBME.2017.2694818](https://doi.org/10.1109/TBME.2017.2694818).
  + Task-related component analysis with reference signals (**TRCA-R**) and Ensemble TRCA with reference signals (**eTRCA-R**): C. M. Wong, B. Wang, Z. Wang, K. F. Lao, A. Rosa, and F. Wan, "Spatial filtering in SSVEP-based BCIs: Unified framework and new improvements.," *IEEE Transactions on Biomedical Engineering*, vol. 67, no. 11, pp. 3057-3072, 2020. DOI: [10.1109/TBME.2020.2975552](https://ieeexplore.ieee.org/document/9006809/).
  + Sum of squared correlations (**SSCOR**) and Ensemble sum of squared correlations (**eSSCOR**): G. K. Kumar, and M. R. Reddy, "Designing a sum of squared correlations framework for enhancing SSVEP-based BCIs," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 27, no. 10, pp. 2044-2050, 2019. DOI: [10.1109/TNSRE.2019.2941349](https://doi.org/10.1109/TNSRE.2019.2941349).
  + Multi-stimulus TRCA (**ms-TRCA**): C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: [10.1088/1741-2552/ab2373](https://doi.org/10.1088/1741-2552/ab2373).
  + Task-discriminant component analysis (**TDCA**): B. Liu, X. Chen, N. Shi, Y. Wang, S. Gao, X. Gao, "Improving the performance of individually calibrated SSVEP-BCI by task-discriminant component analysis." *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 1998-2007, 2021. DOI: [10.1109/TNSRE.2021.3114340](https://doi.org/10.1109/TNSRE.2021.3114340).
  + Online adaptive CCA (OACCA) (**OACCA**): C. M. Wong et al., “Online adaptation boosts SSVEP-based BCI performance,” IEEE Trans. Biomed. Eng., vol. 69, no. 6, pp. 2018-2028, 2022. DOI: [10.1109/TBME.2021.3133594](https://doi.org/10.1109/TBME.2021.3133594).
  + ...

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgements

+ [edwin465/SSVEP-MSCCA-MSTRCA](https://github.com/edwin465/SSVEP-MSCCA-MSTRCA)
+ [edwin465/SSVEP-OACCA](https://github.com/edwin465/SSVEP-OACCA)
+ [mnakanishi/TRCA-SSVEP](https://github.com/mnakanishi/TRCA-SSVEP)
+ [TBC-TJU/brainda](https://github.com/TBC-TJU/brainda)


