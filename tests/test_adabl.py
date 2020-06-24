#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unitary tests of the functions in ../kabl/adabl.py
Must be executed inside the `tests/` directory
"""

from kabl.adabl import *
from kabl import paths


# prepare_supervised_dataset
#----------------------------

def test_prepare_supervised_dataset():
    lidarDir = paths.dir_lidardata()
    dataFiles = [
        os.path.join(lidarDir, "DAILY_MPL_5025_20180802.nc"),
        os.path.join(lidarDir, "DAILY_MPL_5029_20180224.nc"),
    ]
    refDir = paths.dir_handmadedata()
    refFiles = [
        os.path.join(refDir, "blhref_Trappes_20180802.csv"),
        os.path.join(refDir, "blhref_Brest_20180224.csv"),
    ]
    
    df = prepare_supervised_dataset(dataFiles, refFiles, saveInCSV=False)
    
    assert df.shape==(86400, 6) and set(df.columns) == set(['sec0', 'alti', 'rcs0', 'rcs1', 'rcs2', 'isBL'])


# generate_algokey
#----------------------

def test_generate_algokey():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.semi_supervised import LabelSpreading
    
    n_weaks = 50
    tree_depth = 3
    K = 6
    
    rfc = RandomForestClassifier(n_estimators=n_weaks, max_depth=tree_depth)
    knn = KNeighborsClassifier(n_neighbors=K)
    dtc = DecisionTreeClassifier(max_depth=tree_depth)
    abc = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=tree_depth),
        n_estimators=n_weaks,
    )
    lsc = LabelSpreading(kernel="knn")
    
    
    funcRes = [generate_algokey(clf) for clf in [rfc, knn, dtc, abc, lsc]]
    trueRes = [
        'RandomForest_M50_D3',
        '6NearestNeighbors',
        'DecisionTree_D3',
        'AdaBoost_M50_D3',
        'LabelSpreading_knn',
    ]
    
    assert funcRes==trueRes

# generate_predkey
#----------------------

def test_generate_predkey():
    
    
    funcRes = [generate_predkey(predictors) for predictors in [
            ['sec0', 'alti', 'rcs0'],
            ['rcs0', 'alti', 'rcs2'],
            ['rcs0', 'rcs1', 'rcs2'],
            ['alti', 'rcs0', 'rcs1', 'sec0', 'rcs2']
        ]
    ]
    trueRes = [
        'tzRCS0',
        'zRCS02',
        'RCS012',
        'tzRCS012',
    ]
    
    assert funcRes==trueRes

# train_adabl
#-------------

def test_train_adabl():
    datasetfile = paths.file_labelleddataset()
    model, scaler = train_adabl(datasetfile)
    
    assert set(model.classes_) == set([0,1]) and scaler.n_samples_seen_==86400

def test_train_adabl_save():
    datasetfile = paths.file_labelleddataset()
    train_adabl(datasetfile,saveResults = True)
    
    modelFile = os.path.join(paths.dir_trainedadabl(),"Classifier_AdaBoost_M200_D5_tzRCS0.pkl")
    scalerFile = os.path.join(paths.dir_trainedadabl(),"Scaler_tzRCS0.pkl")
    assert os.path.isfile(modelFile) and os.path.isfile(scalerFile)


# random_crossval
#-----------------
def test_random_crossval():
    datasetfile = paths.file_labelleddataset()
    
    accuracies, chronos, classifiers_keys = random_crossval(datasetfile, n_random_splits=3, plot_on=False)
    
    assert accuracies.shape==(5,3) and chronos.shape==(5,3) and set(classifiers_keys)==set(['RandomForestClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier', 'LabelSpreading'])


# block_crossval
#-----------------
def test_block_crossval():
    datasetfile = paths.file_labelleddataset()
    
    accuracies, chronos, classifiers_keys = block_crossval(datasetfile, n_folds=3, plot_on=False)
    
    assert accuracies.shape==(5,3) and chronos.shape==(5,3) and set(classifiers_keys)==set(['RandomForestClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier', 'LabelSpreading'])


# adabl_blh_estimation
# ---------------------

def test_adabl_blh_estimation():
    dataFile = paths.file_defaultlidardata()
    modelFile = paths.file_trainedmodel()
    scalerFile = paths.file_trainedscaler()

    blh_kabl = adabl_blh_estimation(
        dataFile, modelFile, scalerFile, storeInNetcdf=False
    )
    assert blh_kabl.shape==(288,) and np.isnan(blh_kabl).sum()==0

# adabl_qualitymetrics
#----------------------
def test_adabl_qualitymetrics():
    dataFile = paths.file_defaultlidardata()
    modelFile = paths.file_trainedmodel()
    scalerFile = paths.file_trainedscaler()
    
    scores = adabl_qualitymetrics(dataFile, modelFile, scalerFile)
    
    everythingOK =all([
        scores[0]<1000.0,   # errl2
        scores[1]<1000.0,   # errl1
        scores[2]<3000.0,   # errl0
        scores[3]>0.4,      # corr
        scores[4]<12,      # chrono
        scores[5]==0,      # n_invalid
    ])
    
    assert everythingOK
