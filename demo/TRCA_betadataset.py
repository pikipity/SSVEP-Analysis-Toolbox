# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets.betadataset import BETADataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import ECCA
from SSVEPAnalysisToolbox.algorithms.trca import TRCA, ETRCA
from SSVEPAnalysisToolbox.evaluator.baseevaluator import BaseEvaluator, gen_trials_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.evaluator.performance import cal_performance_onedataset_individual_diffsiglen
from SSVEPAnalysisToolbox.utils.io import savedata, loaddata
from SSVEPAnalysisToolbox.evaluator.plot import bar_plot_with_errorbar, shadowline_plot, bar_plot

import numpy as np

# Prepare dataset
dataset = BETADataset(path = '2020_BETA_SSVEP_database_update')
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
                   ECCA(weights_filterbank = weights_filterbank),
                   TRCA(weights_filterbank = weights_filterbank),
                   ETRCA(weights_filterbank = weights_filterbank)
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

# Calculate training time and testing time
train_time = np.zeros((1,len(model_container)))
test_time = np.zeros((1,len(model_container)))
for performance_trial in cca_evaluator.performance_container:
    for method_idx, method_performance in enumerate(performance_trial):
        train_time[0,method_idx] = train_time[0,method_idx] + sum(method_performance.train_time)
        test_time[0,method_idx] = test_time[0,method_idx] + sum(method_performance.test_time_test)
            
# Save results
data = {"acc_store": acc_store,
        "itr_store": itr_store,
        "train_time": train_time,
        "test_time": test_time}
data_file = 'res/trca_betadataset_res.mat'
savedata(data_file, data, 'mat')

# Load data
# data_file = 'res/trca_betadataset_res.mat'
# data = loaddata(data_file, 'mat')
# acc_store = data["acc_store"]
# itr_store = data["itr_store"]

# Plot training time and testing time
fig, _ = bar_plot(train_time,
                  x_label = 'Methods',
                  y_label = 'Total training time',
                  x_ticks = [model.ID for model in model_container],
                  grid = True)
fig.savefig('res/trca_betadataset_traintime_bar.jpg', bbox_inches='tight', dpi=300)

fig, _ = bar_plot(test_time,
                  x_label = 'Methods',
                  y_label = 'Total testing time',
                  x_ticks = [model.ID for model in model_container],
                  grid = True)
fig.savefig('res/trca_betadataset_testtime_bar.jpg', bbox_inches='tight', dpi=300)


# Plot Performance of bar plots
fig, _ = bar_plot_with_errorbar(acc_store,
                                x_label = 'Signal Length (s)',
                                y_label = 'Acc',
                                x_ticks = tw_seq,
                                legend = [model.ID for model in model_container],
                                errorbar_type = '95ci',
                                grid = True,
                                ylim = [0, 1])
fig.savefig('res/trca_betadataset_acc_bar.jpg', bbox_inches='tight', dpi=300)

fig, _ = bar_plot_with_errorbar(itr_store,
                                x_label = 'Signal Length (s)',
                                y_label = 'ITR',
                                x_ticks = tw_seq,
                                legend = [model.ID for model in model_container],
                                errorbar_type = '95ci',
                                grid = True,
                                ylim = [0, 190])
fig.savefig('res/trca_betadataset_itr_bar.jpg', bbox_inches='tight', dpi=300)

# Plot Performance of shadow lines
fig, _ = shadowline_plot(tw_seq,
                        acc_store,
                        'x-',
                        x_label = 'Signal Length (s)',
                        y_label = 'Acc',
                        legend = [model.ID for model in model_container],
                        errorbar_type = '95ci',
                        grid = True,
                        ylim = [0, 1])
fig.savefig('res/trca_betadataset_acc_shadowline.jpg', bbox_inches='tight', dpi=300)

fig, _ = shadowline_plot(tw_seq,
                        itr_store,
                        'x-',
                        x_label = 'Signal Length (s)',
                        y_label = 'ITR',
                        legend = [model.ID for model in model_container],
                        errorbar_type = '95ci',
                        grid = True,
                        ylim = [0, 190])
fig.savefig('res/trca_betadataset_itr_shadowline.jpg', bbox_inches='tight', dpi=300)




