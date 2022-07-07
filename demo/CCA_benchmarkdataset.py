# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import time

from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr, ECCA
from SSVEPAnalysisToolbox.evaluator.baseevaluator import TrialInfo, BaseEvaluator

# Prepare dataset
dataset = BenchmarkDataset(path = '/data/2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(lambda X: preprocess(X,dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))
ch_used = suggested_ch()
harmonic_num = 5
all_trials = [i for i in range(dataset.stim_info['stim_num'])]
dataset_container = [
                        dataset
                    ]


# Prepare train and test trials
tw_seq = [i*0.25 for i in range(1,4+1,1)]
sub_num = len(dataset.subjects)
dataset_idx_tmp = 0
trial_container = []
for tw in tw_seq:
    for sub_idx in range(sub_num):
        for block_idx in range(dataset.block_num):
            test_block, train_block = dataset_container[dataset_idx_tmp].leave_one_block_out(block_idx)
            train_trial = TrialInfo().add_dataset(dataset_idx = dataset_idx_tmp,
                                                  sub_idx = sub_idx,
                                                  block_idx = train_block,
                                                  trial_idx = all_trials,
                                                  ch_idx = ch_used,
                                                  harmonic_num = 5,
                                                  tw = tw,
                                                  t_latency = None,
                                                  shuffle = False)
            test_trial = TrialInfo().add_dataset(dataset_idx = dataset_idx_tmp,
                                                  sub_idx = sub_idx,
                                                  block_idx = test_block,
                                                  trial_idx = all_trials,
                                                  ch_idx = ch_used,
                                                  harmonic_num = 5,
                                                  tw = tw,
                                                  t_latency = None,
                                                  shuffle = False)
            trial_container.append([train_trial, test_trial])


# Prepare models
weights_filterbank = suggested_weights_filterbank()
model_container = [
                   SCCA_qr(weights_filterbank = weights_filterbank),
                   SCCA_canoncorr(weights_filterbank = weights_filterbank),
                   ECCA(weights_filterbank = weights_filterbank)
                  ]

cca_evaluator = BaseEvaluator(dataset_container = dataset_container,
                              model_container = model_container,
                              trial_container = trial_container,
                              save_model = False,
                              disp_processbar = True)

cca_evaluator.run(n_jobs = 7)




# from SSVEPAnalysisToolbox.pipline.piplineindividual import PiplineIndividual, gen_train_test_blocks_leave_one_block_out, gen_train_test_trials_all_trials

# # Prepare dataset
# dataset = BenchmarkDataset(path = '/data/2016_Tsinghua_SSVEP_database')
# dataset.regist_preprocess(lambda X: preprocess(X,dataset.srate))
# dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))
# dataset_container = [
#                         dataset
#                     ]
# ch_used = [
#               suggested_ch()
#           ]
# harmonic_num = [
#                    5
#                ]

# # Prepare models
# weights_filterbank = suggested_weights_filterbank()
# model_container = [
#                    SCCA_qr(weights_filterbank = weights_filterbank),
#                    SCCA_canoncorr(weights_filterbank = weights_filterbank),
#                    ECCA(weights_filterbank = weights_filterbank)
#                   ]

# # Prepare simulation parameters
# tw_seq = [i*0.25 for i in range(1,2+1,1)]
# tw_seq.append(4)

# testing_blocks, training_blocks = gen_train_test_blocks_leave_one_block_out(dataset_container)

# testing_trials, training_trials = gen_train_test_trials_all_trials(dataset_container)


# pipline = PiplineIndividual(ch_used = ch_used,
#                             harmonic_num = harmonic_num,
#                             model_container = model_container,
#                             dataset_container = dataset_container,
#                             save_model = False,
#                             tw_seq = tw_seq,
#                             training_blocks = training_blocks,
#                             testing_blocks = testing_blocks,
#                             training_trials = training_trials,
#                             testing_trials = testing_trials,
#                             disp_processbar = True,
#                             shuffle_trials = False)

# pipline.run(n_jobs = 5)

# dataset_idx = 0
# method_idx = 2
# acc_store = np.zeros((35,3))

# for tw_idx in range(3):
#     for sub_idx in range(35):
#         tmp=[]
#         for b_idx in range(6):
#             y_prd = pipline.performance_container[dataset_idx][tw_idx][sub_idx][b_idx][method_idx].container['predict-labels'][0]
#             y_true = pipline.performance_container[dataset_idx][tw_idx][sub_idx][b_idx][method_idx].container['true-labels'][0]
#             acc = cal_acc(y_true, y_prd)
#             tmp.append(acc)
#         acc_store[sub_idx,tw_idx] = sum(tmp)/len(tmp)


