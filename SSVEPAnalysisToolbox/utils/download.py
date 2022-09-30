# -*- coding: utf-8 -*-

import os
from typing import Union, Optional, Dict, List, Tuple

from pooch import retrieve

def download_single_file(source_url: str, 
                         desertation: str,
                         known_hash: Optional[str] = None,
                         progressbar: bool = True):
    """
    Download one file

    Parameters
    ----------
    source_url : str
        Source url
    desertation : str
        Local file path
    """
    
    file_name = os.path.basename(desertation)
    desertation_dir = os.path.dirname(desertation)
    
    if not os.path.exists(desertation_dir):
        os.makedirs(desertation_dir)
    if os.path.isfile(desertation):
        os.remove(desertation)
        
    retrieve(source_url, known_hash, 
             fname = file_name,
             path = desertation_dir,
             progressbar=progressbar)

    if not os.path.exists(desertation):
        raise ValueError("The following file has been downloaded but cannot find it: {:s}".format(desertation))
    