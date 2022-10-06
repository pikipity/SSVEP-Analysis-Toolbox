# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets.wearabledataset import WearableDataset_wet, WearableDataset_dry
from SSVEPAnalysisToolbox.utils.wearablepreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr, ECCA, MSCCA, MsetCCA, MsetCCAwithR
from SSVEPAnalysisToolbox.algorithms.trca import TRCA, ETRCA, MSETRCA, MSCCA_and_MSETRCA, TRCAwithR, ETRCAwithR, SSCOR, ESSCOR
from SSVEPAnalysisToolbox.algorithms.tdca import TDCA
from SSVEPAnalysisToolbox.evaluator.baseevaluator import BaseEvaluator, gen_trials_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.evaluator.performance import cal_performance_onedataset_individual_diffsiglen, cal_confusionmatrix_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.utils.io import savedata

import numpy as np

num_subbands = 5
data_type_list = ['wet','dry']

for data_type in data_type_list:
    print("Data type: {:s}".format(data_type))
    
    # Prepare dataset
    if data_type.lower() == 'wet':
        dataset = WearableDataset_wet(path = 'Wearable')
    else:
        dataset = WearableDataset_dry(path = 'Wearable')
    dataset.regist_preprocess(preprocess)
    dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, num_subbands))
    ch_used = suggested_ch()
    all_trials = [i for i in range(dataset.trial_num)]
    harmonic_num = 5
    dataset_container = [
                            dataset
                        ]


    # Prepare train and test trials
    tw_seq = [i/100 for i in range(50,200+50,50)]
    trial_container = gen_trials_onedataset_individual_diffsiglen(dataset_idx = 0,
                                                                tw_seq = tw_seq,
                                                                dataset_container = dataset_container,
                                                                harmonic_num = harmonic_num,
                                                                trials = all_trials,
                                                                ch_used = ch_used,
                                                                t_latency = None,
                                                                shuffle = False)


    # Prepare models
    model_container = [
                    SCCA_qr(weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'cca')),
                    #    SCCA_canoncorr(weights_filterbank = weights_filterbank),
                    #    MsetCCA(weights_filterbank = weights_filterbank),
                    #    MsetCCAwithR(weights_filterbank = weights_filterbank),
                    #    ECCA(weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'cca')),
                    #    MSCCA(n_neighbor = 12, weights_filterbank = weights_filterbank),
                    #    SSCOR(weights_filterbank = weights_filterbank),
                    #    ESSCOR(weights_filterbank = weights_filterbank),
                    TRCA(weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'trca')),
                    #    TRCAwithR(weights_filterbank = weights_filterbank),
                    ETRCA(weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'trca')),
                    #    ETRCAwithR(weights_filterbank = weights_filterbank),
                    #    MSETRCA(n_neighbor = 2, weights_filterbank = weights_filterbank),
                    #    MSCCA_and_MSETRCA(n_neighbor_mscca = 12, n_neighber_msetrca = 2, weights_filterbank = weights_filterbank),
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
    if data_type.lower() == 'wet':
        data_file = 'res/wearable_wet_res.mat'
    else:
        data_file = 'res/wearable_dry_res.mat'
    savedata(data_file, data, 'mat')






