# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, List, Tuple

class SubInfo:
    """
    Subject Information
    """
    def __init__(self,
                 ID: str,
                 path: Optional[str] = None,
                 name: Optional[str] = None,
                 age: Optional[int] = None,
                 gender: Optional[str] = None):
        """
        Parameters required for subjects
        
        Parameters
        ------------------------
        ID: str
            Unique identifier for subject
        path: Optional[str]
            data path
        name: Optional[str]
            Subject name
        age: Optional[int]
            Subject age
        gender: Optional[str]
            Subject gender, including
            - Male: 'M'
            - Female: 'F'
        """
        self.ID = ID
        self.path = path
        self.name = name
        self.age = age
        self.gender = gender