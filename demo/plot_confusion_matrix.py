# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.utils.io import loaddata
import matplotlib.pyplot as plt

import numpy as np

data_file_list = ['res/benchmarkdataset_res.mat',
                  'res/betadataset_res.mat']
save_folder = ['benchmarkdataset_confusionmatrix',
               'beta_confusionmatrix']

for dataset_idx, data_file in enumerate(data_file_list):
    data = loaddata(data_file, 'mat')
    confusion_matrix = data["confusion_matrix"]
    method_ID = data["method_ID"]
    tw_seq = data["tw_seq"]

    for signal_len_idx in range(len(tw_seq)):
        for method_idx, method in enumerate(method_ID):
            confusion_matrix_plot = confusion_matrix[method_idx, :, signal_len_idx, :, :]
            confusion_matrix_plot = np.sum(confusion_matrix_plot, axis = 0)
            min_v = 0
            max_v = np.amax(np.reshape(confusion_matrix_plot - np.expand_dims(np.diag(confusion_matrix_plot),axis=0) @ np.eye(N),(-1)))
            N, _ = confusion_matrix_plot.shape

            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])

            im = ax.imshow(confusion_matrix_plot,
                            interpolation = 'none',
                            origin = 'upper',
                            vmin = min_v,
                            vmax = max_v)
            for n in range(N):
                ax.add_patch(
                    plt.patches.Rectangle(xy=(n-0.5, n-0.5), width=1, height=1, facecolor='white')
                )
            for i in range(N):
                for j in range(N):
                    ax.text(i,j,"{:n}".format(int(confusion_matrix_plot[j,i])),
                        fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='black')
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks(list(range(N)))
            ax.set_yticks(list(range(N)))
            ax.spines[:].set_visible(False)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=1)
            ax.tick_params(top=True, bottom=False,
                           labeltop=True, labelbottom=False)
            ax.tick_params(which="minor", bottom=False, left=False)

            fig.savefig('res/{:s}/{:s}_T{:n}.jpg'.format(save_folder[dataset_idx],
                                                         method_ID[method_idx],
                                                         signal_len_idx), 
                        bbox_inches='tight', dpi=300)