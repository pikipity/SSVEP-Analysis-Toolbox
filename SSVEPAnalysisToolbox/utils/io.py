# -*- coding: utf-8 -*-

from numpy import ndarray
from typing import Union, Optional, Dict, List, Tuple

def loadmat(file_path: str) -> Dict[ndarray]:
    """
    Load mat file

    Parameters
    -------------------
    file_path: str
        Full file path

    Returns
    -------
    mat_data: Dict[ndarray]
        Data in mat file
    """

    

