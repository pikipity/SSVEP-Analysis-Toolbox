# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from SSVEPAnalysisToolbox.datasets import (
    BenchmarkDataset, BETADataset, ELDBETADataset
)
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import (
    preprocess, filterbank, suggested_ch
)

from SSVEPAnalysisToolbox.evaluator.plot import hist, close_fig
from SSVEPAnalysisToolbox.utils.io import savedata
from SSVEPAnalysisToolbox.utils.algsupport import nextpow2

snr_list = []
legend = []
harmonic_num = 5

# Benchmark dataset
dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                      NFFT = 2 ** nextpow2((dataset.trial_len+2)*dataset.srate)) # filterbank index is 0
snr_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)

# BETA datset
dataset = BETADataset(path = '2020_BETA_SSVEP_database_update')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                      NFFT = 2 ** nextpow2((dataset.trial_len+2)*dataset.srate)) # filterbank index is 0
snr_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)

# eldBETA dataset
dataset = ELDBETADataset(path = 'eldBETA_database')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                      NFFT = 2 ** nextpow2((dataset.trial_len+2)*dataset.srate)) # filterbank index is 0
snr_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)

# Store results
data = {"snr_list": snr_list,
        "legend": legend}
data_file = 'res/snr.mat'
savedata(data_file, data, 'mat')

# plot histogram of SNR
fig, ax = hist(snr_list, bins = list(range(-30,0+1)), range = (-30, 0), density = True,
               color = ['blue', 'green', 'orange'], alpha = 0.3, fit_line = True, line_points = 1000,
               x_label = 'SNR (dB)',
               y_label = 'Probability',
               grid = True,
               legend = legend)
fig.savefig('res/SNR.jpg', bbox_inches='tight', dpi=300)
close_fig(fig)