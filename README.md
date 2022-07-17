# SSVEP Analysis Toolbox

This repository provides a python package for SSVEP datasets and recognition algorithms. The goal of this toolbox is to make researchers be familier with SSVEP signals and related recognition algorithms quickly, and focus on improving algorithms with shortest preparation time.

## Features

+ Unify formats of SSVEP datasets
+ Provide a standard processing procedure for faire performance comparisons
+ Python implementations of SSVEP recognition algorithms

## Datasets and Algorithms

+ Datasets
  + **Benchmark** dataset: Y. Wang, X. Chen, X. Gao, and S. Gao, "A benchmark dataset for SSVEP-based braincomputer interfaces," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 25, no. 10, pp. 1746–1752, 2017. DOI: [10.1109/TNSRE.2016.2627556](https://doi.org/10.1109/TNSRE.2016.2627556).
  + **BETA** dataset: B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "BETA: A large benchmark database toward SSVEP-BCI application," *Front. Neurosci.*, vol. 14, p. 627, 2020. DOI: [10.3389/fnins.2020.00627](https://doi.org/10.3389/fnins.2020.00627).

+ Algorithms
  + Standard canonical correlation analysis (**sCCA**) and filterbank CCA (**FBCCA**): Chen, X., Wang, Y., Gao, S., Jung, T.P. and Gao, X., "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface," *J. Neural Eng.*, vol. 12, no. 4, p. 046008, 2015. DOI: [10.1088/1741-2560/12/4/046008](https://doi.org/10.1088/1741-2560/12/4/046008).
  + Extended CCA (**eCCA**): X. Chen, Y. Wang, M. Nakanishi, X. Gao, T.-P. Jung, and S. Gao, "High-speed spelling with a noninvasive brain–computer interface," *Proc. Natl. Acad. Sci.*, vol. 112, no. 44, pp. E6058–E6067, 2015. DOI: [10.1073/pnas.1508080112](https://doi.org/10.1073/pnas.1508080112).
  + Multi-stimulus CCA (**ms-CCA**): C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: [10.1088/1741-2552/ab2373](https://doi.org/10.1088/1741-2552/ab2373).
  + Task-related component analysis (**TRCA**) and Ensemble TRCA (**eTRCA**): M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, "Enhancing detection of SSVEPs for a high-speed brain speller using task-related component Analysis," *IEEE Trans. Biomed. Eng.*, vol. 65, no. 1, pp. 104–112, 2018. DOI: [10.1109/TBME.2017.2694818](https://doi.org/10.1109/TBME.2017.2694818).
  + Multi-stimulus TRCA (**ms-TRCA**): C. M. Wong, F. Wan, B. Wang, Z. Wang, W. Nan, K. F. Lao, P. U. Mak, M. I. Vai, and A. Rosa, "Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs," *J. Neural Eng.*, vol. 17, no. 1, p. 016026, 2020. DOI: [10.1088/1741-2552/ab2373](https://doi.org/10.1088/1741-2552/ab2373).
  + ...

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgements

+ [edwin465/SSVEP-MSCCA-MSTRCA](https://github.com/edwin465/SSVEP-MSCCA-MSTRCA)
+ [mnakanishi/TRCA-SSVEP](https://github.com/mnakanishi/TRCA-SSVEP)
+ [TBC-TJU/brainda](https://github.com/TBC-TJU/brainda)


