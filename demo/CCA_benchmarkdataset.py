# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr, ECCA
from SSVEPAnalysisToolbox.evaluator.baseevaluator import BaseEvaluator, gen_trials_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.evaluator.performance import cal_performance_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.utils.io import savedata, loaddata

import numpy as np

# Prepare dataset
dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(lambda X: preprocess(X, dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))
ch_used = suggested_ch()
all_trials = [i for i in range(dataset.stim_info['stim_num'])]
harmonic_num = 5
dataset_container = [
                        dataset
                    ]


# Prepare train and test trials
tw_seq = [i*0.25 for i in range(1,4+1,1)]
trial_container = gen_trials_onedataset_individual_diffsiglen(dataset_idx = 0,
                                                             tw_seq = tw_seq,
                                                             dataset_container = dataset_container,
                                                             harmonic_num = harmonic_num,
                                                             trials = all_trials,
                                                             ch_used = ch_used,
                                                             t_latency = None,
                                                             shuffle = False)


# Prepare models
weights_filterbank = suggested_weights_filterbank()
model_container = [
                   SCCA_qr(weights_filterbank = weights_filterbank),
                   SCCA_canoncorr(weights_filterbank = weights_filterbank),
                   ECCA(weights_filterbank = weights_filterbank)
                  ]

# Evaluate models
cca_evaluator = BaseEvaluator(dataset_container = dataset_container,
                              model_container = model_container,
                              trial_container = trial_container,
                              save_model = False,
                              disp_processbar = True)

cca_evaluator.run(n_jobs = 7,
                  eval_train = False)

# Calculate performance
acc_store, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = cca_evaluator,
                                                                         dataset_idx = 0,
                                                                         tw_seq = tw_seq,
                                                                         train_or_test = 'test')
            
# Save results
data = {"acc_store": acc_store,
        "itr_store": itr_store}
data_file = 'res/cca_benchmarkdataset_res.mat'
savedata(data_file, data, 'mat')

# Load data
# data_file = 'cca_benchmarkdataset_res.mat'
# data = loaddata(data_file, 'mat')
# acc_store = data["acc_store"]
# itr_store = data["itr_store"]


# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar([1,2,3,4],np.mean(acc_store,1))
# plt.show()





