# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from SSVEPAnalysisToolbox.datasets import BenchmarkDataset
from SSVEPAnalysisToolbox.utils.benchmarkpreprocess import (
    preprocess, filterbank, suggested_ch, suggested_weights_filterbank
)
from SSVEPAnalysisToolbox.algorithms import (
    SCCA_qr, SCCA_canoncorr, 
    ITCCA, ECCA, 
    MSCCA, 
    MsetCCA, MsetCCAwithR,
    TRCA, ETRCA, 
    MSETRCA, MSCCA_and_MSETRCA, 
    TRCAwithR, ETRCAwithR, 
    SSCOR, ESSCOR,
    TDCA,
    SCCA_ls, SCCA_ls_qr,
    ECCA_ls, ITCCA_ls,
    MSCCA_ls,
    TRCA_ls, ETRCA_ls,
    MsetCCA_ls,
    MsetCCAwithR_ls,
    TRCAwithR_ls, ETRCAwithR_ls,
    MSETRCA_ls,
    TDCA_ls
)
from SSVEPAnalysisToolbox.evaluator import cal_acc,cal_itr

import time

num_subbands = 5

# Prepare dataset
dataset = BenchmarkDataset(path = '2016_Tsinghua_SSVEP_database')
dataset.regist_preprocess(preprocess)
dataset.regist_filterbank(filterbank)

# Prepare recognition model
weights_filterbank = suggested_weights_filterbank()
# recog_model = SCCA_qr(weights_filterbank = weights_filterbank, force_output_UV=True)
# # recog_model_ls = SCCA_ls_qr(weights_filterbank = weights_filterbank, force_output_UV=True,
# #                                     LSMethod = 'lasso',
# #                                     l1_alpha = 0.0001,
# #                                     reg_iter = 1000,
# #                                     reg_tol = 1e-4,
# #                                     M_error_threshold = 1e-4,
# #                                     W_error_threshold = 1e-4,
# #                                     max_n = 100)
# recog_model_ls = SCCA_ls_qr(weights_filterbank = weights_filterbank, force_output_UV=True,
#                             LSMethod = 'elastic_net',
#                             l1_alpha = 0.0001,
#                             alpha = 0.1,
#                             reg_iter = 1000,
#                             reg_tol = 1e-4,
#                             M_error_threshold = 1e-4,
#                             W_error_threshold = 1e-4,
#                             max_n = 100)
# recog_model = TRCAwithR_ls(weights_filterbank = weights_filterbank)
# recog_model_ls = TRCAwithR_ls(weights_filterbank = weights_filterbank,
#                                 LSMethod = 'lasso',
#                                 l1_alpha = 0.01,
#                                 reg_iter = 2000,
#                                 reg_tol = 1e-6,
#                                 M_error_threshold = 1e-4,
#                                 W_error_threshold = 1e-4,
#                                 max_n = 100)
recog_model = MsetCCAwithR_ls(weights_filterbank = weights_filterbank)
recog_model_ls = MsetCCAwithR_ls(weights_filterbank = weights_filterbank,
                                LSMethod = 'elastic_net',
                                l1_alpha = 0.01,
                                alpha = 0.3,
                                reg_iter = 1000,
                                reg_tol = 1e-4,
                                M_error_threshold = 1e-4,
                                W_error_threshold = 1e-4,
                                max_n = 100)

# Set simulation parameters
# 19ch: 
# ch1 = list(range(43,59)); ch2 = list(range(60,63)); ch_used = [y for x in [ch1,ch2] for y in x]
# 32ch: 
# ch1 = list(range(32,64)); ch_used = [y for x in [ch1] for y in x]
# 9ch:
ch_used = suggested_ch()
all_trials = [i for i in range(dataset.trial_num)]
harmonic_num = 5
tw = 0.25
sub_idx = 0
test_block_idx = 1
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

tic_ls = time.time()
recog_model_ls.fit(X=X_train, Y=Y_train, ref_sig=ref_sig, freqs=freqs) 
toc_train_ls = time.time()-tic

# Get testing data and test the recognition model
X_test, Y_test = dataset.get_data(sub_idx = sub_idx,
                                    blocks = test_block_list,
                                    trials = all_trials,
                                    channels = ch_used,
                                    sig_len = tw)
tic = time.time()
pred_label, _ = recog_model.predict(X_test)
toc_test = time.time()-tic
toc_test_onetrial = toc_test/len(Y_test)

tic_ls = time.time()
pred_label_ls, _ = recog_model_ls.predict(X_test)
toc_test_ls = time.time()-tic
toc_test_ls_onetrial = toc_test_ls/len(Y_test)

# Calculate performance
acc = cal_acc(Y_true = Y_test, Y_pred = pred_label)
itr = cal_itr(tw = tw, t_break = dataset.t_break, t_latency = dataset.default_t_latency, t_comp = toc_test_onetrial,
              N = len(freqs), acc = acc)
acc_ls = cal_acc(Y_true = Y_test, Y_pred = pred_label_ls)
itr_ls = cal_itr(tw = tw, t_break = dataset.t_break, t_latency = dataset.default_t_latency, t_comp = toc_test_ls_onetrial,
              N = len(freqs), acc = acc_ls)
print("""
Simulation Information:
    Method Name: {:s} vs. {:s}
    Dataset: {:s}
    Signal length: {:.3f} s
    Channel: {:s}
    Subject index: {:n}
    Testing block: {:s}
    Training block: {:s}

Performance_ori:
    Training time: {:.5f} s
    Total Testing time: {:.5f} s
    Testing time of single trial: {:.5f} s
    Acc: {:.3f} %
    ITR: {:.3f} bits/min

Performance_ls:
    Training time: {:.5f} s
    Total Testing time: {:.5f} s
    Testing time of single trial: {:.5f} s
    Acc: {:.3f} %
    ITR: {:.3f} bits/min
""".format(recog_model.ID,
           recog_model_ls.ID,
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
           itr,
           toc_train_ls,
           toc_test_ls,
           toc_test_ls_onetrial,
           acc_ls*100,
           itr_ls))

print('finish')