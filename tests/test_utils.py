#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unitary tests of the functions in ../kabl/utils.py
Must be executed inside the `tests/` directory
"""

from kabl.utils import *
from kabl import paths

import os
import datetime as dt

# get_default_params
#--------------------

def test_get_default_params_isdict():
    params = get_default_params()
    assert isinstance(params,dict)

def test_get_default_params_dictkeys():
    params = get_default_params()
    required_keys = ["algo",
            "n_clusters",
            "classif_score",
            "n_inits",
            "n_profiles",
            "max_k",
            "init",
            "cov_type",
            "predictors",
            "max_height",
            "sunrise_shift",
            "sunset_shift"
        ]
    
    assert set(params.keys())==set(required_keys)


# where_and_when
#----------------

def test_where_and_when():
    testFile = paths.file_defaultlidardata()
    
    trueLocation = 'Trappes'
    trueDay = dt.datetime(2018,8,2)
    trueLat = 48.7743
    trueLon = 2.0098
    
    location, day, lat, lon = where_and_when(testFile)
    
    assert location==trueLocation and day==trueDay and lat==trueLat and lon==trueLon


# get_lat_lon
#----------------

def test_get_lat_lon():
    
    trueLat = 48.7743
    trueLon = 2.0098
    
    lat, lon = get_lat_lon('Trappes')
    
    assert lat==trueLat and lon==trueLon


# create_file_from_source
#-------------------------

def test_create_file_from_source():
    
    testFile = paths.file_defaultlidardata()
    outputFile = os.path.join(
        paths.resultrootdir,
        os.path.split(testFile)[-1][:-3] + ".out.nc"
    )
    create_file_from_source(testFile, outputFile)
    
    assert os.path.isfile(outputFile)


# extract_data
#--------------

def test_extract_data_typenkeys():
    
    testFile = paths.file_defaultlidardata()
    needed_data = ["rcs_0","rcs_1","rcs_2"]
    t_values, z_values, rcss = extract_data(testFile,to_extract=needed_data)
    
    assert isinstance(rcss,dict) and set(rcss.keys())==set(needed_data)

def test_extract_data_shapes():
    
    testFile = paths.file_defaultlidardata()
    t_values, z_values, rcss = extract_data(testFile,to_extract=["rcs_1"])
    rcs_1 = rcss["rcs_1"]
    
    assert t_values.shape==(288,) and z_values.shape==(146,) and rcs_1.shape==(288,146)


# extract_testprofile
#---------------------

def test_extract_testprofile():
    
    testFile = paths.file_defaultlidardata()
    z_values, rcs_1, rcs_2, coords = extract_testprofile(
        testFile, profile_id=2, return_coords=True
    )
    
    assert z_values.shape==(146,) and rcs_1.shape==(146,)

# blh_from_labels
#-----------------

def test_blh_from_labels():
    
    labl = np.zeros(100)
    labl[30:50] = 1
    labl[50:] = 2
    z = np.linspace(0,2970,100)
    
    zi = blh_from_labels(labl,z)
    
    assert zi==885.0



# grid/scatter conversion
#-------------------------

def test_grid_to_scatter():
    
    x=np.linspace(12.1,23.4,32)
    y=np.linspace(0.6,1.4,22)
    z=np.arange(len(x)*len(y)).reshape((len(x),len(y)))
    
    X,Y=grid_to_scatter(x,y,z)
    
    assert X.shape==(len(x)*len(y),2) and Y.shape==(len(x)*len(y),)

def test_scatter_to_grid():
    
    x=np.linspace(12.1,23.4,32)
    y=np.linspace(0.6,1.4,22)
    z=np.arange(len(x)*len(y)).reshape((len(x),len(y)))
    
    X,Y=grid_to_scatter(x,y,z)
    x1,y1,z1 = scatter_to_grid(X,Y)
    assert (x==x1).all() and (y==y1).all() and (z==z1).all()


# colocate_instruments
#----------------------

def test_colocate_instruments():
    
    Nl = 100
    Nr = 6
    
    t_lidar = np.linspace(20,420,Nl)
    t_rs = np.linspace(20,420,Nr)
    
    val1_lidar = np.random.rand(Nl)*0.2
    val2_lidar = np.random.rand(Nl)*24
    
    val_rs = np.random.rand(Nr)*24
    
    t_coloc, val_rs_coloc, val_lidar_coloc = colocate_instruments(
                            t_lidar,
                            t_rs,
                            [val1_lidar,val2_lidar],
                            val_rs,
                            tol=20
                        )
    
    assert all([arr.shape==(Nr,) for arr in val_lidar_coloc+[t_coloc,val_rs_coloc]])
