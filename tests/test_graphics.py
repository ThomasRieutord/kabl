#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unitary tests of the functions in ../kabl/graphics.py
Must be executed inside the `tests/` directory
"""

# Local packages
from kabl.graphics import *
from kabl import paths
from kabl import utils
from kabl import core


# Test of quicklook_data
# ------------------------
print("\n --------------- Test of quicklook_data")
testFile = paths.file_defaultlidardata()
quicklook_data(testFile)

# Test of quicklook_testprofiles
# ------------------------
print("\n --------------- Test of quicklook_testprofiles")
testFile = paths.file_defaultlidardata()
quicklook_testprofiles(testFile)

# Test of blhs_over_profile
# ------------------------
print("\n --------------- Test of blhs_over_profile")
testFile = paths.file_defaultlidardata()
z_values, rcs_1, rcs_2, coords = utils.extract_testprofile(
    testFile, profile_id=3, return_coords=True
)
X, Z = core.prepare_data(coords, z_values, {"rcs_1":rcs_1, "rcs_2":rcs_2})
labels = core.apply_algo(X, 3)
blh = utils.blh_from_labels(labels, Z)

blhs_over_profile(z_values, rcs_1, blh, labels=labels)

plt.figure()
plt.hist(rcs_1, 35)
plt.title("Histogram of a single profile of RCS")
plt.show(block=False)

# Test of blhs_over_data
# ------------------------
print("\n --------------- Test of blhs_over_data")
testFile = paths.file_defaultlidardata()
blh = core.blh_estimation(testFile)
t_values, z_values, rcss = utils.extract_data(testFile)
rcs_1 = rcss["rcs_1"]
rcs_2 = rcss["rcs_2"]

blhs_over_data(t_values, z_values, rcs_1, blh)

# Test of scatterplot_blhs
# ------------------------
print("\n --------------- Test of scatterplot_blhs")
outputFile = paths.file_defaultoutput()
t_values, z_values, dat = utils.extract_data(
    outputFile, to_extract=["blh_kabl", "pbl"]
)
blh_new = dat["blh_kabl"]
blh_mnf  = dat["pbl"]
scatterplot_blhs(t_values, blh_mnf[:, 0], blh_new)


# Test of quicklook_output
# ------------------------
print("\n --------------- Test of quicklook_output")
outputFile = paths.file_defaultoutput()
quicklook_output(outputFile)


# Test of clusterZTview
# ---------------------

testFile = paths.file_defaultlidardata()
blh,labels = core.blh_estimation_returnlabels(testFile)

clusterZTview(t_values,z_values,labels)


# Test of bar_scores
# ---------------------

scores = [0.22,0.34,0.76]
bar_scores(scores,"corr")

# Test of plot_cv_indices
# ---------------------

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

datasetfile = paths.file_labelleddataset()
df = pd.read_csv(datasetfile)
predictors = ["sec0", "alti", "rcs0"]
X = df.loc[:, predictors].values
y = df.loc[:, "isBL"].values
kf = KFold(n_splits=5)
gkf = GroupKFold(n_splits=3)

group=np.zeros_like(y)
for itx in df.groupby(np.floor(df.sec0/(8*3600))).indices.items():
    grval,grdex = itx
    group[grdex]=int(grval)

subsize=100
plot_cv_indices(
    gkf,
    np.random.choice(X[:,0],subsize),
    np.random.choice(y,subsize),
    group=np.random.choice(group,subsize)
)



input("\n Press Enter to exit (close down all figures)\n")
