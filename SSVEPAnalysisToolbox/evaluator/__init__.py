# -*- coding: utf-8 -*-

from .baseevaluator import (
    BaseEvaluator, 
    gen_trials_onedataset_individual_diffsiglen,
    gen_trials_onedataset_individual_online
)

from .performance import (
    cal_confusionmatrix_onedataset_individual_online,
    cal_performance_onedataset_individual_online,
    cal_confusionmatrix_onedataset_individual_diffsiglen,
    cal_performance_onedataset_individual_diffsiglen,
    cal_itr_trials_onebyone, cal_itr_trials, 
    cal_acc_trials_onebyone, cal_acc_trials,
    cal_acc, cal_itr
)

from .plot import (
    close_fig,
    shadowline_plot, bar_plot, bar_plot_with_errorbar, cal_CI95,
    hist, gen_colors,
    polar_phase, polar_phase_shadow
)