# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets.benchmarkdataset import BenchmarkDataset
from SSVEPAnalysisToolbox.datasets.betadataset import BETADataset
from SSVEPAnalysisToolbox.utils.io import loaddata
import matplotlib.pyplot as plt
import matplotlib.patches as pach
import os

import numpy as np

data_file_list = ['res/benchmarkdataset_res.mat',
                  'res/betadataset_res.mat']
save_folder = ['benchmarkdataset_confusionmatrix',
               'beta_confusionmatrix']
dataset_container = [
                        BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database'),
                        BETADataset(path = '2020_BETA_SSVEP_database_update')
                    ]
target_time = 0.5

for dataset_idx, data_file in enumerate(data_file_list):
    data = loaddata(data_file, 'mat')
    confusion_matrix = data["confusion_matrix"]
    method_ID = data["method_ID"]
    tw_seq = data["tw_seq"]
    freqs = dataset_container[dataset_idx].stim_info['freqs']
    sort_idx = list(np.argsort(freqs))

    signal_len_idx = int(np.where(np.array(tw_seq)==target_time)[0])

    # for signal_len_idx in range(len(tw_seq)):
    for method_idx, method in enumerate(method_ID):
        confusion_matrix_plot = confusion_matrix[method_idx, :, signal_len_idx, :, :]
        confusion_matrix_plot = np.sum(confusion_matrix_plot, axis = 0)
        confusion_matrix_plot = confusion_matrix_plot[sort_idx,:]
        confusion_matrix_plot = confusion_matrix_plot[:,sort_idx]
        N, _ = confusion_matrix_plot.shape
        min_v = 0
        max_v = np.amax(np.reshape(confusion_matrix_plot - np.diag(np.diag(confusion_matrix_plot)),(-1)))

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])

        im = ax.imshow(confusion_matrix_plot,
                        interpolation = 'none',
                        origin = 'upper',
                        vmin = min_v,
                        vmax = max_v,
                        cmap='winter')

        for n in range(N):
            ax.add_patch(
                pach.Rectangle(xy=(n-0.5, n-0.5), width=1, height=1, facecolor='white')
            )
        for i in range(N):
            for j in range(N):
                if i==j:
                    text_color = 'black'
                else:
                    text_color = 'white'
                ax.text(i,j,"{:n}".format(int(confusion_matrix_plot[j,i])),
                    fontsize=5,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color=text_color)
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(list(range(N)))
        ax.set_yticks(list(range(N)))
        ax.spines[:].set_visible(False)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=10)
        ax.tick_params(top=True, bottom=False,
                        labeltop=True, labelbottom=False)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis='x',labelsize=5)
        ax.tick_params(axis='y',labelsize=5)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')

        save_path = 'res/{:s}/{:s}_T{:n}.jpg'.format(save_folder[dataset_idx],
                                                        method_ID[method_idx].strip(),
                                                        tw_seq[signal_len_idx])
        desertation_dir = os.path.dirname(save_path)
        if not os.path.exists(desertation_dir):
            os.makedirs(desertation_dir)
        fig.savefig(save_path, 
                    bbox_inches='tight', dpi=300)