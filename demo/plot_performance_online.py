# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.utils.io import loaddata
from SSVEPAnalysisToolbox.evaluator.plot import bar_plot_with_errorbar, shadowline_plot, close_fig

import numpy as np

data_file_list = ['res/benchmarkdataset_res_online.npy']
sub_title = ['benchmark']
repeat_num = 5

for dataset_idx, data_file in enumerate(data_file_list):
    data = loaddata(data_file, 'np')
    acc_store = data["acc_store"]
    # print(acc_store.shape)
    itr_store = data["itr_store"]
    tw_seq = data["tw_seq"]
    if type(data["method_ID"]) is not list:
        method_ID = [data["method_ID"]]
    else:
        method_ID = data["method_ID"]
    method_ID = [name.strip() for name in method_ID]
    if len(method_ID)==1:
        acc_store = np.expand_dims(acc_store, 2)
        itr_store = np.expand_dims(itr_store, 2)

    for method_idx, method_name in enumerate(method_ID):
        # Plot Performance of bar plots
        subject_num, signal_len_num, method_num, trial_num = acc_store.shape
        trial_each_repeat_num = int(np.floor(trial_num/repeat_num))
        acc_store_online_tmp = None
        for repeat_idx in range(repeat_num):
            if acc_store_online_tmp is None:
                acc_store_online_tmp = acc_store[:,:,method_idx,(repeat_idx*trial_each_repeat_num):((repeat_idx+1)*trial_each_repeat_num)]
            else:
                acc_store_online_tmp = acc_store_online_tmp + acc_store[:,:,method_idx,(repeat_idx*trial_each_repeat_num):((repeat_idx+1)*trial_each_repeat_num)]
        acc_store_online = acc_store_online_tmp / repeat_num
        acc_store_online = np.transpose(acc_store_online, (1,0,2)) 
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
        # close_fig(fig)

        subject_num, signal_len_num, method_num, trial_num = itr_store.shape
        trial_each_repeat_num = int(np.floor(trial_num/repeat_num))
        itr_store_online_tmp = None
        for repeat_idx in range(repeat_num):
            if itr_store_online_tmp is None:
                itr_store_online_tmp = itr_store[:,:,method_idx,(repeat_idx*trial_each_repeat_num):((repeat_idx+1)*trial_each_repeat_num)]
            else:
                itr_store_online_tmp = itr_store_online_tmp + itr_store[:,:,method_idx,(repeat_idx*trial_each_repeat_num):((repeat_idx+1)*trial_each_repeat_num)]
        itr_store_online = itr_store_online_tmp / repeat_num
        itr_store_online = np.transpose(itr_store_online, (1,0,2)) 
        # fig, _ = bar_plot_with_errorbar(itr_store_online,
        #                                 x_label = 'Trial',
        #                                 y_label = 'ITR (bits/min)',
        #                                 x_ticks = x_value,
        #                                 legend = legend_label,
        #                                 errorbar_type = '95ci',
        #                                 grid = True,
        #                                 figsize=[6.4*3, 4.8])
        # fig.savefig('res/{:s}_{:s}_itr_bar.jpg'.format(sub_title[dataset_idx],method_name), bbox_inches='tight', dpi=300)
        # close_fig(fig)

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
        close_fig(fig)

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
        close_fig(fig)