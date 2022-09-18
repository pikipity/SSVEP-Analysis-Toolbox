# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.utils.io import loaddata
from SSVEPAnalysisToolbox.evaluator.plot import bar_plot_with_errorbar, shadowline_plot

import numpy as np

data_file_list = ['res/benchmarkdataset_res_online.mat']
sub_title = ['benchmark']

for dataset_idx, data_file in enumerate(data_file_list):
    data = loaddata(data_file, 'mat')
    acc_store = data["acc_store"]
    # print(acc_store.shape)
    itr_store = data["itr_store"]
    tw_seq = data["tw_seq"]
    method_ID = [data["method_ID"]]
    method_ID = [name.strip() for name in method_ID]

    for method_idx, method_name in enumerate(method_ID):
        # Plot Performance of bar plots
        # acc_store_online = acc_store[:,:,method_idx,:,:] # (subject_num, signal_len_num, method_num, trial_num, sub_trial_num)
        # acc_store_online = np.mean(acc_store_online, 2)
        acc_store_online = np.transpose(acc_store, (1,0,2)) 
        # x_ticks_label = [str(i+1) for i in range(acc_store_online.shape[2])]
        x_value = [i+1 for i in range(acc_store_online.shape[2])]
        legend_label = [str(tw)+'s' for tw in tw_seq]
        # fig, _ = bar_plot_with_errorbar(acc_store_online,
        #                                 x_label = 'Trial',
        #                                 y_label = 'Acc',
        #                                 x_ticks = x_value,
        #                                 legend = legend_label,
        #                                 errorbar_type = '95ci',
        #                                 grid = True,
        #                                 ylim = [0, 1],
        #                                 figsize=[6.4*3, 4.8])
        # fig.savefig('res/{:s}_{:s}_acc_bar.jpg'.format(sub_title[dataset_idx],method_name), bbox_inches='tight', dpi=300)

        # itr_store_online = itr_store[:,:,method_idx,:,:] # (subject_num, signal_len_num, method_num, trial_num, sub_trial_num)
        # itr_store_online = np.mean(itr_store_online, 2)
        itr_store_online = np.transpose(itr_store, (1,0,2)) 
        # fig, _ = bar_plot_with_errorbar(itr_store_online,
        #                                 x_label = 'Trial',
        #                                 y_label = 'ITR (bits/min)',
        #                                 x_ticks = x_value,
        #                                 legend = legend_label,
        #                                 errorbar_type = '95ci',
        #                                 grid = True,
        #                                 figsize=[6.4*3, 4.8])
        # fig.savefig('res/{:s}_{:s}_itr_bar.jpg'.format(sub_title[dataset_idx],method_name), bbox_inches='tight', dpi=300)

        # Plot Performance of shadow lines
        fig, _ = shadowline_plot(x_value,
                                acc_store_online,
                                'x-',
                                x_label = 'Trial',
                                y_label = 'Acc',
                                legend = legend_label,
                                errorbar_type = '95ci',
                                grid = True,
                                ylim = [0, 1],
                                figsize=[6.4*3, 4.8])
        fig.savefig('res/{:s}_{:s}_acc_shadowline.jpg'.format(sub_title[dataset_idx],method_name), bbox_inches='tight', dpi=300)

        fig, _ = shadowline_plot(x_value,
                                itr_store_online,
                                'x-',
                                x_label = 'Trial',
                                y_label = 'ITR (bits/min)',
                                legend = legend_label,
                                errorbar_type = '95ci',
                                grid = True,
                                figsize=[6.4*3, 4.8])
        fig.savefig('res/{:s}_{:s}_itr_shadowline.jpg'.format(sub_title[dataset_idx],method_name), bbox_inches='tight', dpi=300)