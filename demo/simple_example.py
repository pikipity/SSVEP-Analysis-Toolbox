# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets import WearableDataset_wet, WearableDataset_dry
from SSVEPAnalysisToolbox.utils.wearablepreprocess import preprocess, filterbank, suggested_ch, suggested_weights_filterbank
from SSVEPAnalysisToolbox.algorithms.cca import SCCA_qr, SCCA_canoncorr, ECCA, MSCCA, MsetCCA, MsetCCAwithR, ITCCA
from SSVEPAnalysisToolbox.algorithms.trca import TRCA, ETRCA, MSETRCA, MSCCA_and_MSETRCA, TRCAwithR, ETRCAwithR, SSCOR, ESSCOR
from SSVEPAnalysisToolbox.algorithms.tdca import TDCA
from SSVEPAnalysisToolbox.evaluator.performance import cal_acc,cal_itr

import time

num_subbands = 5
data_type = 'dry'

# Prepare dataset
if data_type.lower() == 'wet':
    dataset = WearableDataset_wet(path = 'Wearable')
else:
    dataset = WearableDataset_dry(path = 'Wearable')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(lambda dataself, X: filterbank(dataself, X, num_subbands))

# Prepare recognition model
weights_filterbank = suggested_weights_filterbank(num_subbands, data_type, 'trca')
recog_model = ETRCAwithR(weights_filterbank = weights_filterbank)

# Set simulation parameters
ch_used = suggested_ch()
all_trials = [i for i in range(dataset.trial_num)]
harmonic_num = 5
tw = 2
sub_idx = 98
test_block_idx = 8
test_block_list, train_block_list = dataset.leave_one_block_out(block_idx = test_block_idx)

# Get training data and train the recognition model
ref_sig = dataset.get_ref_sig(tw, harmonic_num)
freqs = dataset.stim_info['freqs']
X_train, Y_train = dataset.get_data(sub_idx = sub_idx,
                                    blocks = train_block_list,
                                    trials = all_trials,
                                    channels = ch_used,
                                    sig_len = tw)
tic = time.time()
recog_model.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs) 
toc_train = time.time()-tic

# Get testing data and test the recognition model
X_test, Y_test = dataset.get_data(sub_idx = sub_idx,
                                    blocks = test_block_list,
                                    trials = all_trials,
                                    channels = ch_used,
                                    sig_len = tw)
tic = time.time()
pred_label = recog_model.predict(X_test)
toc_test = time.time()-tic
toc_test_onetrial = toc_test/len(Y_test)

# Calculate performance
acc = cal_acc(Y_true = Y_test, Y_pred = pred_label)
itr = cal_itr(tw = tw, t_break = dataset.t_break, t_latency = dataset.default_t_latency, t_comp = toc_test_onetrial,
              N = len(freqs), acc = acc)
print("""
Simulation Information:
    Method Name: {:s}
    Dataset: {:s}
    Signal length: {:.3f} s
    Channel: {:s}
    Subject index: {:n}
    Testing block: {:s}
    Training block: {:s}
    Training time: {:.5f} s
    Total Testing time: {:.5f} s
    Testing time of single trial: {:.5f} s

Performance:
    Acc: {:.3f} %
    ITR: {:.3f} bits/min
""".format(recog_model.ID,
           dataset.ID,
           tw,
           str(ch_used),
           sub_idx,
           str(test_block_list),
           str(train_block_list),
           toc_train,
           toc_test,
           toc_test_onetrial,
           acc*100,
           itr))
