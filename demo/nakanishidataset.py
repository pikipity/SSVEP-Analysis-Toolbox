# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets.nakanishidataset import NakanishiDataset
from SSVEPAnalysisToolbox.utils.nakanishipreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr, ECCA, MSCCA, MsetCCA, MsetCCAwithR
from SSVEPAnalysisToolbox.algorithms.trca import TRCA, ETRCA, MSETRCA, MSCCA_and_MSETRCA, TRCAwithR, ETRCAwithR
from SSVEPAnalysisToolbox.algorithms.tdca import TDCA
from SSVEPAnalysisToolbox.evaluator.baseevaluator import BaseEvaluator, gen_trials_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.evaluator.performance import cal_performance_onedataset_individual_diffsiglen, cal_confusionmatrix_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.utils.io import savedata

import numpy as np

# Prepare dataset
dataset = NakanishiDataset(path = 'Nakanishi_2015')
dataset.regist_preprocess(lambda X: preprocess(X, dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))
ch_used = suggested_ch()
all_trials = [i for i in range(dataset.stim_info['stim_num'])]
harmonic_num = 3
dataset_container = [
                        dataset
                    ]


# Prepare train and test trials
tw_seq = [i/100 for i in range(50,100+10,10)]
# tw_seq = [tw-dataset.default_t_latency for tw in tw_seq]
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
                   MsetCCA(weights_filterbank = weights_filterbank),
                   MsetCCAwithR(weights_filterbank = weights_filterbank),
                   ECCA(weights_filterbank = weights_filterbank),
                #    MSCCA(n_neighbor = 6, weights_filterbank = weights_filterbank),
                   TRCA(weights_filterbank = weights_filterbank),
                   TRCAwithR(weights_filterbank = weights_filterbank),
                   ETRCA(weights_filterbank = weights_filterbank),
                   ETRCAwithR(weights_filterbank = weights_filterbank),
                #    MSETRCA(n_neighbor = 6, weights_filterbank = weights_filterbank),
                #    MSCCA_and_MSETRCA(n_neighbor_mscca = 6, n_neighber_msetrca = 6, weights_filterbank = weights_filterbank),
                #    TDCA(n_component = 8, weights_filterbank = weights_filterbank, n_delay = 6)
                  ]

# Evaluate models
evaluator = BaseEvaluator(dataset_container = dataset_container,
                          model_container = model_container,
                          trial_container = trial_container,
                          save_model = False,
                          disp_processbar = True)

evaluator.run(n_jobs = 10,
              eval_train = False)

# Calculate performance
acc_store, itr_store = cal_performance_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                         dataset_idx = 0,
                                                                         tw_seq = tw_seq,
                                                                         train_or_test = 'test')
confusion_matrix = cal_confusionmatrix_onedataset_individual_diffsiglen(evaluator = evaluator,
                                                                        dataset_idx = 0,
                                                                        tw_seq = tw_seq,
                                                                        train_or_test = 'test')                                                                       

# Calculate training time and testing time
train_time = np.zeros((len(model_container), len(evaluator.performance_container)))
test_time = np.zeros((len(model_container), len(evaluator.performance_container)))
for trial_idx, performance_trial in enumerate(evaluator.performance_container):
    for method_idx, method_performance in enumerate(performance_trial):
        train_time[method_idx, trial_idx] = sum(method_performance.train_time)
        test_time[method_idx, trial_idx] = sum(method_performance.test_time_test)
train_time = train_time.T
test_time = test_time.T
            
# Save results
data = {"acc_store": acc_store,
        "itr_store": itr_store,
        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix,
        "tw_seq":tw_seq,
        "method_ID": [model.ID for model in model_container]}
data_file = 'res/nakanishidataset_res.mat'
savedata(data_file, data, 'mat')
