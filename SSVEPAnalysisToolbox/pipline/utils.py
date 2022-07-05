# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple
from numpy import ndarray

from tqdm import tqdm

def create_pbar(loop_list_num: List[int],
                desc: str):
    """
    Create process bar

    Parameters
    ----------
    loop_list_num : List[int]
        List of following loop numbers
    desc : str
        Prefix for the progressbar

    Returns
    -------
    pbar : tqdm
        processbar
    """
    total_num = 1
    for loop_num in loop_list_num:
        total_num = total_num * loop_num
    pbar = tqdm(total = total_num,
                desc = desc,
                bar_format = '{desc}{percentage:3.3f}%|{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}]')
    return pbar
 