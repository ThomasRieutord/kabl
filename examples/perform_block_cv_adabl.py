#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from kabl import paths
from kabl import adabl
from kabl import graphics

datasetfile = paths.file_labelleddataset()
df = pd.read_csv(datasetfile)
predictors = ["sec0", "alti", "rcs1","rcs2"]
X = df.loc[:, predictors].values
y = df.loc[:, "isBL"].values
kf = KFold(n_splits=5)
gkf = GroupKFold(n_splits=3)

group=np.zeros_like(y)
for itx in df.groupby(np.floor(df.sec0/(8*3600))).indices.items():
    grval,grdex = itx
    group[grdex]=int(grval)

for ii, (idx_train, idx_test) in enumerate(gkf.split(X=X, y=y, groups=group)):
    print("\n--- Fold #",ii)
    print("Training set starts",df.sec0[idx_train[0]]/3600,"ends",df.sec0[idx_train[-1]]/3600)
    print("Testing set starts",df.sec0[idx_test[0]]/3600,"ends",df.sec0[idx_test[-1]]/3600)
    X_train = X[idx_train,:]
    X_test = X[idx_test,:]
    y_train = y[idx_train]
    y_test = y[idx_test]

accuracies, chronos, classifiers_keys = adabl.block_cv_adabl(df,n_folds=6, predictors=predictors)

graphics.estimator_quality(accuracies,chronos,classifiers_keys)
