# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA

weights_filterbank = suggested_weights_filterbank()
ch_used = suggested_ch()
harmonic_num = 5

dataset = BenchmarkDataset(path = '/data/2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(lambda X: preprocess(X,dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))

tw = 4 
sub_idx = 0
block_idx = 0

test_block, train_block = dataset.leave_one_block_out(block_idx)

ref_sig = dataset.get_ref_sig(tw,harmonic_num)
X, Y = dataset.get_data_all_stim(sub_idx,test_block,ch_used,tw,
                                 shuffle=True)

sCCAModel = SCCA(weights_filterbank = weights_filterbank)
sCCAModel.fit(ref_sig = ref_sig)
Y_pred = sCCAModel.predict(X)




