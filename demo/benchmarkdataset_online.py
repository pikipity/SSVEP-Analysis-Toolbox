# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import (
        preprocess, filterbank, suggested_ch, suggested_weights_filterbank
)
from SSVEPAnalysisToolbox.algorithms import (
        SCCA_qr, OACCA
)
from SSVEPAnalysisToolbox.evaluator import (
        BaseEvaluator,
        gen_trials_onedataset_individual_online,
        cal_performance_onedataset_individual_online,
        cal_confusionmatrix_onedataset_individual_online
)
from SSVEPAnalysisToolbox.utils.io import savedata

import numpy as np

# Prepare dataset
dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)
ch_used = suggested_ch()
all_trials = [i for i in range(dataset.trial_num)]
harmonic_num = 5
dataset_container = [
                        dataset
                    ]


# Prepare train and test trials
tw_seq = [i/100 for i in range(50,150+10,10)]
trial_container = gen_trials_onedataset_individual_online(dataset_idx = 0,
                                                             tw_seq = tw_seq,
                                                             dataset_container = dataset_container,
                                                             harmonic_num = harmonic_num,
                                                             repeat_num = 5,
                                                             trials = all_trials,
                                                             ch_used = ch_used,
                                                             t_latency = None,
                                                             shuffle = True)


# Prepare models
weights_filterbank = suggested_weights_filterbank()
model_container = [
                   SCCA_qr(weights_filterbank = weights_filterbank),
                   OACCA(weights_filterbank = weights_filterbank)
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
acc_store, itr_store = cal_performance_onedataset_individual_online(evaluator = evaluator,
                                                                    dataset_idx = 0,
                                                                    tw_seq = tw_seq,
                                                                    train_or_test = 'test')
confusion_matrix = cal_confusionmatrix_onedataset_individual_online(evaluator = evaluator,
                                                                    dataset_idx = 0,
                                                                    tw_seq = tw_seq,
                                                                    train_or_test = 'test')        
            
# Save results
data = {"acc_store": acc_store,
        "itr_store": itr_store,
        # "confusion_matrix": confusion_matrix,
        "tw_seq":tw_seq,
        "method_ID": [model.ID for model in model_container]}
data_file = 'res/benchmarkdataset_res_online.npy'
savedata(data_file, data, 'np')
