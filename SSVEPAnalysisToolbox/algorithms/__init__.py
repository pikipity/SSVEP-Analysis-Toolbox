# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple, Callable
from numpy import ndarray

from .basemodel import BaseModel
from .cca import (
    MsetCCA, MsetCCAwithR, 
    OACCA, 
    SCCA_canoncorr, SCCA_qr, 
    ITCCA, ECCA, 
    MSCCA
)
from .tdca import TDCA
from .trca import (
    TRCA, TRCAwithR, 
    ETRCA, ETRCAwithR, 
    MSETRCA, MSCCA_and_MSETRCA, 
    SSCOR, ESSCOR
)
from .lsframework import (
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

from .ms_msetcca_r_1 import eMsetCCA_multi_f as ms_msetcca_r_1
from .ms_msetcca_r_2 import eMsetCCA_multi_f as ms_msetcca_r_2
from .ms_msetcca_r_3 import eMsetCCA_multi_f as ms_msetcca_r_3

from .ms_trca_r_1 import eTRCAwithR_multi_f as ms_trca_r_1
from .ms_trca_r_2 import eTRCAwithR_multi_f as ms_trca_r_2



def SCCA(n_component: int = 1,
         n_jobs: Optional[int] = None,
         weights_filterbank: Optional[List[float]] = None,
         force_output_UV: bool = False,
         update_UV: bool = True,
         cca_type: str = 'qr'):
    """
    Generate sCCA model

    Parameters
    ----------
    n_component : Optional[int], optional
        Number of eigvectors for spatial filters. The default is 1.
    n_jobs : Optional[int], optional
        Number of CPU for computing different trials. The default is None.
    weights_filterbank : Optional[List[float]], optional
        Weights of spatial filters. The default is None.
    force_output_UV : Optional[bool] 
        Whether store U and V. Default is False
    update_UV: Optional[bool]
        Whether update U and V in next time of applying "predict" 
        If false, and U and V have not been stored, they will be stored
        Default is True
    cca_type : Optional[str], optional
        Methods for computing corr.
        'qr' - QR decomposition
        'canoncorr' - Canoncorr
        The default is 'qr'.

    Returns
    -------
    sCCA model: Union[SCCA_qr, SCCA_canoncorr]
        if cca_type is 'qr' -> SCCA_qr
        if cca_type is 'canoncorr' -> SCCA_canoncorr
    """
    if cca_type.lower() == 'qr':
        return SCCA_qr(n_component,
                       n_jobs,
                       weights_filterbank,
                       force_output_UV,
                       update_UV)
    elif cca_type.lower() == 'canoncorr':
        return SCCA_canoncorr(n_component,
                              n_jobs,
                              weights_filterbank,
                              force_output_UV,
                              update_UV)
    else:
        raise ValueError('Unknown cca_type')