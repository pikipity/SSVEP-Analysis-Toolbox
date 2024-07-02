# -*- coding: utf-8 -*-

from .baseevaluator import (
    BaseEvaluator, 
    gen_trials_onedataset_individual_diffsignlen_specfic_trainblcokNum,
    gen_trials_onedataset_individual_diffsiglen,
    gen_trials_onedataset_individual_online,
    gen_trials_onedataset_cross_subj_notarget_diffsignlen_specfic_trainblcokNum
)

from .performance import (
    cal_confusionmatrix_onedataset_individual_online,
    cal_performance_onedataset_individual_online,
    cal_confusionmatrix_onedataset_individual_diffsiglen,
    cal_performance_onedataset_individual_diffsiglen,
    cal_itr_trials_onebyone, cal_itr_trials, 
    cal_acc_trials_onebyone, cal_acc_trials,
    cal_acc, cal_itr,
    cal_performance_onedataset_crossSubj_diffsiglen
)

from .plot import (
    close_fig,
    shadowline_plot, bar_plot, bar_plot_with_errorbar, cal_CI95,
    hist, gen_colors,
    polar_phase, polar_phase_shadow
)