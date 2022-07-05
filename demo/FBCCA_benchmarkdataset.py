# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import time

from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr

weights_filterbank = suggested_weights_filterbank()
ch_used = suggested_ch()
harmonic_num = 5

dataset = BenchmarkDataset(path = '/data/2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(lambda X: preprocess(X,dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))

tw = 0.2 
sub_idx = 0
block_idx = 0

test_block, train_block = dataset.leave_one_block_out(block_idx)

ref_sig = dataset.get_ref_sig(tw,harmonic_num)
X, Y = dataset.get_data_all_stim(sub_idx,test_block,ch_used,tw,
                                 shuffle=True)

t_star1 = time.time()
Model1 = SCCA_qr(weights_filterbank = weights_filterbank)
Model1.fit(ref_sig = ref_sig)
t1_train = time.time() - t_star1
t_star1 = time.time()
Y_pred1 = Model1.predict(X)
t1_test = time.time() - t_star1

t_star2 = time.time()
Model2 = SCCA_canoncorr(weights_filterbank = weights_filterbank)
Model2.fit(ref_sig = ref_sig)
t2_train = time.time() - t_star2
t_star2 = time.time()
Y_pred2 = Model2.predict(X)
t2_test = time.time() - t_star2




