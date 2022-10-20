# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from SSVEPAnalysisToolbox.datasets import (
    BenchmarkDataset, BETADataset, ELDBETADataset, NakanishiDataset, openBMIDataset,
    WearableDataset_wet, WearableDataset_dry
)
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import (
    preprocess, filterbank, suggested_ch
)

from SSVEPAnalysisToolbox.evaluator import (
    polar_phase_shadow, close_fig, gen_colors
)
from SSVEPAnalysisToolbox.utils.io import savedata
from SSVEPAnalysisToolbox.utils.algsupport import nextpow2

import numpy as np

phase_list = []
legend = []
dataset_no = 0
harmonic_num = 5
sig_len = 1

# Benchmark dataset
dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# BETA datset
dataset = BETADataset(path = '2020_BETA_SSVEP_database_update')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# eldBETA dataset
dataset = ELDBETADataset(path = 'eldBETA_database')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# Nakanishi dataset
from SSVEPAnalysisToolbox.utils.nakanishipreprocess import (
    preprocess, filterbank, suggested_ch
)
dataset = NakanishiDataset(path = 'Nakanishi_2015')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# openBMI dataset
from SSVEPAnalysisToolbox.utils.openbmipreprocess import (
    preprocess, filterbank, suggested_ch
)
dataset = openBMIDataset(path = 'openBMI')
downsample_srate = 100
dataset.regist_preprocess(lambda dataself, X: preprocess(dataself, X, downsample_srate))
dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, downsample_srate))
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# Wearable dataset
from SSVEPAnalysisToolbox.utils.wearablepreprocess import (
    preprocess, filterbank, suggested_ch
)
dataset = WearableDataset_wet(path = 'Wearable')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, 5))
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# dataset = WearableDataset_dry(path = 'Wearable')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, 5))
print("{:s}: ".format(dataset.ID))
snr = dataset.get_phase(display_progress = True, 
                      sig_len = sig_len,
                      remove_break = False, remove_pre_and_latency = False, remove_target_phase = True,
                      NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
phase_list.append(snr[:,:,:,suggested_ch()])
legend.append(dataset.ID)
dataset_no += 1

# Store results
data = {"phase_list": phase_list,
        "legend": legend}
data_file = 'res/phase.mat'
savedata(data_file, data, 'mat')

# plot histogram of SNR
# phase_list_plot = []
# for phase in phase_list:
#     phase_mean = np.mean(phase, axis = 1)
#     phase_mean = np.mean(phase_mean, axis = 1)
#     phase_list_plot.append(phase_mean)
color = gen_colors(dataset_no)
fig, ax = polar_phase_shadow(phase_list,
                            color = color,
                            grid = True,
                            legend = legend,
                            errorbar_type = 'std')
fig.savefig('res/phase.jpg', bbox_inches='tight', dpi=300)
close_fig(fig)