# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import time

from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr, ECCA
from SSVEPAnalysisToolbox.pipline.piplineindividual import PiplineIndividual, gen_train_test_blocks_leave_one_block_out, gen_train_test_trials_all_trials

# Prepare dataset
dataset = BenchmarkDataset(path = '/data/2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(lambda X: preprocess(X,dataset.srate))
dataset.regist_filterbank(lambda X: filterbank(X, dataset.srate))
dataset_container = [
                        dataset
                    ]
ch_used = [
              suggested_ch()
          ]
harmonic_num = [
                   5
               ]

# Prepare models
weights_filterbank = suggested_weights_filterbank()
model_container = [
                   SCCA_qr(weights_filterbank = weights_filterbank),
                   SCCA_canoncorr(weights_filterbank = weights_filterbank),
                   ECCA(weights_filterbank = weights_filterbank)
                  ]

# Prepare simulation parameters
tw_seq = [i*0.25 for i in range(1,2+1,1)]
tw_seq.append(4)

testing_blocks, training_blocks = gen_train_test_blocks_leave_one_block_out(dataset_container)

testing_trials, training_trials = gen_train_test_trials_all_trials(dataset_container)


pipline = PiplineIndividual(ch_used = ch_used,
                            harmonic_num = harmonic_num,
                            model_container = model_container,
                            dataset_container = dataset_container,
                            save_model = False,
                            tw_seq = tw_seq,
                            training_blocks = training_blocks,
                            testing_blocks = testing_blocks,
                            training_trials = training_trials,
                            testing_trials = testing_trials,
                            disp_processbar = True,
                            shuffle_trials = False)

pipline.run(n_jobs = 5)

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


