# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from SSVEPAnalysisToolbox.datasets import (
    BenchmarkDataset, BETADataset, ELDBETADataset, NakanishiDataset, openBMIDataset,
    WearableDataset_wet, WearableDataset_dry
)

from SSVEPAnalysisToolbox.evaluator import (
    hist, close_fig, gen_colors
)
from SSVEPAnalysisToolbox.utils.io import savedata
from SSVEPAnalysisToolbox.utils.algsupport import nextpow2

for snr_type in ['fft','sine']:

    snr_list = []
    legend = []
    dataset_no = 0
    harmonic_num = 5
    sig_len = 1
    # filterbank index is 0

    # Benchmark dataset
    from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import (
        preprocess, filterbank, suggested_ch
    )
    dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(filterbank)
    print("{:s}: ".format(dataset.ID))
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*dataset.srate)) 
        snr_list.append(snr[:,:,:,suggested_ch()])
    legend.append(dataset.ID)
    dataset_no += 1

    # BETA datset
    dataset = BETADataset(path = '2020_BETA_SSVEP_database_update')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(filterbank)
    print("{:s}: ".format(dataset.ID))
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
        snr_list.append(snr[:,:,:,suggested_ch()])
    legend.append(dataset.ID)
    dataset_no += 1

    # eldBETA dataset
    dataset = ELDBETADataset(path = 'eldBETA_database')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(filterbank)
    print("{:s}: ".format(dataset.ID))
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
        snr_list.append(snr[:,:,:,suggested_ch()])
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
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
        snr_list.append(snr[:,:,:,suggested_ch()])
    legend.append(dataset.ID)
    dataset_no += 1

    # openBMI dataset
    from SSVEPAnalysisToolbox.utils.openbmipreprocess import (
        preprocess, filterbank, suggested_ch, ref_sig_fun
    )
    dataset = openBMIDataset(path = 'openBMI')
    downsample_srate = 100
    dataset.regist_preprocess(lambda dataself, X: preprocess(dataself, X, downsample_srate))
    dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, downsample_srate))
    dataset.regist_ref_sig_fun(lambda dataself, sig_len, N, phases: ref_sig_fun(dataself, sig_len, N, phases, downsample_srate))
    print("{:s}: ".format(dataset.ID))
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            srate = downsample_srate,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*downsample_srate)) # filterbank index is 0
        snr_list.append(snr[:,:,:,suggested_ch()])
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
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
        snr_list.append(snr[:,:,:,suggested_ch()])
    legend.append(dataset.ID)
    dataset_no += 1

    dataset = WearableDataset_dry(path = 'Wearable')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, 5))
    print("{:s}: ".format(dataset.ID))
    if snr_type == 'sine':
        snr = dataset.get_snr(type = 'sine', ch_used_recog=suggested_ch(), display_progress = True)
        snr_list.append(snr)
    else:
        snr = dataset.get_snr(Nh = harmonic_num, display_progress = True, 
                            sig_len = sig_len,
                            remove_break = False, remove_pre_and_latency = False,
                            NFFT = 2 ** nextpow2(10*dataset.srate)) # filterbank index is 0
        snr_list.append(snr[:,:,:,suggested_ch()])
    legend.append(dataset.ID)
    dataset_no += 1

    # Store results
    data = {"snr_list": snr_list,
            "legend": legend}
    data_file = 'res/snr_' + snr_type + '.mat'
    savedata(data_file, data, 'mat')

    # plot histogram of SNR
    if snr_type == 'sine':
        hist_bins = list(range(-100,0+1))
        hist_range = (-100, 0)
    else:
        hist_bins = list(range(-30,0+1))
        hist_range = (-30, 0)
    color = gen_colors(dataset_no)
    fig, ax = hist(snr_list, bins = hist_bins, range = hist_range, density = True,
                color = color, alpha = 0.3, fit_line = True, line_points = 1000,
                x_label = 'SNR (dB)',
                y_label = 'Probability',
                grid = True,
                legend = legend)
    fig.savefig('res/SNR_' + snr_type + '.jpg', bbox_inches='tight', dpi=300)
    close_fig(fig)