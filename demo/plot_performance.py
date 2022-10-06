# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('..')
from SSVEPAnalysisToolbox.utils.io import loaddata
from SSVEPAnalysisToolbox.evaluator.plot import bar_plot_with_errorbar, shadowline_plot, close_fig

data_file_list = ['res/benchmarkdataset_res.mat',
                  'res/betadataset_res.mat',
                  'res/nakanishidataset_res.mat',
                  'res/eldbetadataset_res.mat',
                  'res/openbmidataset_res.mat']
sub_title = ['benchmark',
             'beta',
             'nakanishi',
             'eldbeta',
             'openbmi']


for dataset_idx, data_file in enumerate(data_file_list):
    if not os.path.isfile(data_file):
        print("'{:s}' does not exist.".format(data_file))
        continue
    data = loaddata(data_file, 'mat')
    acc_store = data["acc_store"]
    itr_store = data["itr_store"]
    train_time = data["train_time"]
    test_time = data["test_time"]
    tw_seq = data["tw_seq"]
    method_ID = data["method_ID"]
    method_ID = [name.strip() for name in method_ID]

    # Plot training time and testing time
    fig, _ = bar_plot_with_errorbar(train_time,
                                    x_label = 'Methods',
                                    y_label = 'Training time (s)',
                                    x_ticks = method_ID,
                                    grid = True,
                                    figsize=[6.4*2, 4.8])
    fig.savefig('res/{:s}_traintime_bar.jpg'.format(sub_title[dataset_idx]), bbox_inches='tight', dpi=300)
    close_fig(fig)
    

    fig, _ = bar_plot_with_errorbar(test_time,
                                    x_label = 'Methods',
                                    y_label = 'Testing time (s)',
                                    x_ticks = method_ID,
                                    grid = True,
                                    figsize=[6.4*2, 4.8])
    fig.savefig('res/{:s}_testtime_bar.jpg'.format(sub_title[dataset_idx]), bbox_inches='tight', dpi=300)
    close_fig(fig)


    # Plot Performance of bar plots
    fig, _ = bar_plot_with_errorbar(acc_store,
                                    x_label = 'Signal Length (s)',
                                    y_label = 'Acc',
                                    x_ticks = tw_seq,
                                    legend = method_ID,
                                    errorbar_type = '95ci',
                                    grid = True,
                                    ylim = [0, 1],
                                    figsize=[6.4*3, 4.8])
    fig.savefig('res/{:s}_acc_bar.jpg'.format(sub_title[dataset_idx]), bbox_inches='tight', dpi=300)
    close_fig(fig)

    fig, _ = bar_plot_with_errorbar(itr_store,
                                    x_label = 'Signal Length (s)',
                                    y_label = 'ITR (bits/min)',
                                    x_ticks = tw_seq,
                                    legend = method_ID,
                                    errorbar_type = '95ci',
                                    grid = True,
                                    figsize=[6.4*3, 4.8])
    fig.savefig('res/{:s}_itr_bar.jpg'.format(sub_title[dataset_idx]), bbox_inches='tight', dpi=300)
    close_fig(fig)

    # Plot Performance of shadow lines
    fig, _ = shadowline_plot(tw_seq,
                            acc_store,
                            'x-',
                            x_label = 'Signal Length (s)',
                            y_label = 'Acc',
                            legend = method_ID,
                            errorbar_type = '95ci',
                            grid = True,
                            ylim = [0, 1],
                            figsize=[6.4*3, 4.8])
    fig.savefig('res/{:s}_acc_shadowline.jpg'.format(sub_title[dataset_idx]), bbox_inches='tight', dpi=300)
    close_fig(fig)

    fig, _ = shadowline_plot(tw_seq,
                            itr_store,
                            'x-',
                            x_label = 'Signal Length (s)',
                            y_label = 'ITR (bits/min)',
                            legend = method_ID,
                            errorbar_type = '95ci',
                            grid = True,
                            figsize=[6.4*3, 4.8])
    fig.savefig('res/{:s}_itr_shadowline.jpg'.format(sub_title[dataset_idx]), bbox_inches='tight', dpi=300)
    close_fig(fig)