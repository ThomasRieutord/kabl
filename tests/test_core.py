#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unitary tests of the functions in ../kabl/core.py
Must be executed inside the `tests/` directory
"""

from kabl.core import *

from kabl import paths
from kabl import utils
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    silhouette_score,
    calinski_harabaz_score,
    davies_bouldin_score,
)

# prepare_data
#--------------

def test_prepare_data_singleprof():
    
    testFile = paths.file_defaultlidardata()
    z_values, rcs_1, rcs_2, coords = utils.extract_testprofile(
        testFile, profile_id=2, return_coords=True
    )
    
    params = utils.get_default_params()
    params["predictors"]={"day": ["rcs_1","rcs_2"], "night": ["rcs_1", "rcs_2"]}
    
    X, Z = prepare_data(
            coords,
            z_values,
            rcss ={
                "rcs_1":rcs_1,
                "rcs_2":rcs_2
            },
            params=params
        )
    
    assert X.shape==(146, 2) and Z.shape==(146,)


def test_prepare_data_multiprof():
    
    n_profiles = 3
    testFile = paths.file_defaultlidardata()
    t_values, z_values, rcss = utils.extract_data(
        testFile, to_extract=["rcs_1", "rcs_2"]
    )
    rcs_1 = rcss["rcs_1"]
    rcs_2 = rcss["rcs_2"]
    
    params = utils.get_default_params()
    params["predictors"]={"day": ["rcs_1","rcs_2"], "night": ["rcs_1", "rcs_2"]}
    
    loc, dateofday, lat, lon = utils.where_and_when(testFile)
    t = 55
    coords = {"time": dt.datetime.utcfromtimestamp(t_values[t]), "lat": lat, "lon": lon}
    t_back = max(t - n_profiles + 1, 0)
    rcss ={"rcs_1":rcs_1[t_back : t + 1, :], "rcs_2":rcs_2[t_back : t + 1, :]}
    
    X, Z = prepare_data(coords, z_values, rcss=rcss, params=params)
    
    assert X.shape==(438, 2) and Z.shape==(438,)


def test_prepare_data_cl31():
    
    n_profiles = 3
    testFile = paths.file_defaultcl31data()
    t_values, z_values, rcss = utils.extract_data(
        testFile, to_extract=["rcs_0"]
    )
    rcs_0 = rcss["rcs_0"]
    
    params = utils.get_default_params()
    params["predictors"]={"day": ["rcs_0"], "night": ["rcs_0"]}
    
    loc, dateofday, lat, lon = utils.where_and_when(testFile)
    t = 55
    coords = {"time": dt.datetime.utcfromtimestamp(t_values[t]), "lat": lat, "lon": lon}
    t_back = max(t - n_profiles + 1, 0)
    rcs_0 = rcs_0[t_back : t + 1, :]
    
    X, Z = prepare_data(coords, z_values, rcss={"rcs_0":rcs_0}, params=params)
    
    assert X.shape==(1347, 1) and Z.shape==(1347,)

# apply_algo
#------------

def test_apply_algo():
    
    X, y = make_blobs(n_samples=100,centers=3,random_state=418)
    
    params = utils.get_default_params()
    params["init"]="advanced"
    
    labels = apply_algo(X, n_clusters=3, params=params)
    
    # cluster identification numbers are random. Only borders matter
    assert np.array_equal(np.diff(labels)==0, np.diff(y)==0)


# apply_algo_k_auto
#-----------------

def test_apply_algo_k_auto():
    
    X, y = make_blobs(n_samples=100,centers=3,random_state=418)
    
    params = utils.get_default_params()
    params["init"]="advanced"
    
    labels,K,sc = apply_algo_k_auto(X, params=params)
    
    # cluster identification numbers are random. Only borders matter
    assert np.array_equal(np.diff(labels)==0, np.diff(y)==0) and K==3


# blh_estimation
#----------------

def test_blh_estimation():
    
    testFile = paths.file_defaultlidardata()
    blh = blh_estimation(testFile)
    assert blh.shape==(288,) and np.isnan(blh).sum()==0



# apply_algo_k_3scores
#----------------------

def test_apply_algo_k_3scores():
    
    X, y = make_blobs(n_samples=100,centers=3,random_state=418)
    
    params = utils.get_default_params()
    params["init"]="advanced"
    
    labels,K,sil,db,ch = apply_algo_k_3scores(X, params=params)
    
    # cluster identification numbers are random. Only borders matter
    assert K==3 and sil==silhouette_score(X, y) and db==davies_bouldin_score(X, y) and ch==calinski_harabaz_score(X, y)


# kabl_qualitymetrics
#----------------

def test_kabl_qualitymetrics():
    
    testFile = paths.file_defaultlidardata()
    scores = kabl_qualitymetrics(testFile)
    
    everythingOK =all([
        scores[0]<1000.0,   # errl2
        scores[1]<1000.0,   # errl1
        scores[2]<3000.0,   # errl0
        scores[3]>0.4,      # corr
        scores[4]>1200,     # ch
        scores[5]<0.4,      # db
        scores[6]>0.7,      # sil
        scores[7]<3,      # chrono
        scores[8]==0,      # n_invalid
    ])
    
    assert everythingOK


# blh_estimation_returnlabels
#----------------

def test_blh_estimation_returnlabels():
    
    testFile = paths.file_defaultlidardata()
    blh,labels = blh_estimation_returnlabels(testFile)
    assert labels.shape==(288,146) and blh.shape==(288,) and np.isnan(blh).sum()==0
