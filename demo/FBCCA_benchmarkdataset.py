# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch

ch_used = suggested_ch()
harmonic_num = 5

dataset = BenchmarkDataset(path = '/data/2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(lambda X: preprocess(X,dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))

tw = 1 
sub_idx = 0
block_idx = 0

test_block, train_block = dataset.generate_test_train_blocks_for_specific_block(block_idx)

ref_sig = dataset.get_ref_sig(tw,harmonic_num)
X, Y = dataset.get_data_all_stim(sub_idx,train_block,ch_used,tw)
