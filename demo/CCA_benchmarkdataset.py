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
# tw_seq = [i*0.25 for i in range(1,4+1,1)]
tw_seq = [4]

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


