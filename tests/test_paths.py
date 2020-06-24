#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unitary tests of the functions in ../kabl/paths.py
Must be executed inside the `tests/` directory
"""
from kabl.paths import *
import os

def test_localdirectory():
    assert os.path.split(os.getcwd())[-1]=='tests'
    
def test_paths_tofiles():
    locvar = globals()
    fileOK = []
    
    for it in locvar.items():
        key,val = it
        if key[:4]=='file':
            fileOK.append(os.path.isfile(val()))
    
    assert all(fileOK) and len(fileOK)>3

def test_paths_todirs():
    locvar = globals()
    dirOK = []
    
    for it in locvar.items():
        key,val = it
        if key[:4]=='dir_':
            dirOK.append(os.path.isdir(val()))
    
    assert all(dirOK) and len(dirOK)>1
