#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display the labels in a time-altitude grid, as suggested by Referee #2 (RC2.Q21)
"""

from kabl import paths
from kabl import utils
from kabl import core
from kabl import graphics

params = utils.get_default_params()
params["init"]="advanced"
params["n_inits"]=15

testFile = paths.file_defaultlidardata()
t_values, z_values, rcss = utils.extract_data(
    testFile, to_extract=["rcs_1", "rcs_2"]
)
rcs_1 = rcss["rcs_1"]
rcs_2 = rcss["rcs_2"]

blh,labels = core.blh_estimation_returnlabels(testFile, params=params)

graphics.blhs_over_data(t_values, z_values, rcs_1, blh)
graphics.clusterZTview(t_values,z_values,labels)
