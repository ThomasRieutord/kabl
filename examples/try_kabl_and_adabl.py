#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAUNCHING SCRIPT FOR KABL PROGRAM.
"""

# Local packages
from kabl import core
from kabl import utils
from kabl import graphics
from kabl import adabl
from kabl import paths

# Usual Python packages
import os
import pickle
import numpy as np
import datetime as dt
import netCDF4 as nc

# To load a specific day, change the loaded lidar file from default
lidarFile = paths.file_defaultlidardata()
rsFile = paths.file_blhfromrs()

t_values, z_values, rcss = utils.extract_data(
    lidarFile, max_height=4620, to_extract=["rcs_1","pbl"]
)
rcs_1 = rcss["rcs_1"]
blh_mnf = rcss["pbl"]

# Estimation with KABL
# ----------------------
params = utils.get_default_params()
params["n_clusters"] = 3
params["predictors"] = {"day": ["rcs_1"], "night": ["rcs_1"]}
params["n_profiles"] = 1
params["init"] = "given"

blh_kabl = core.blh_estimation(lidarFile, storeInNetcdf=False, params=params)

# Estimation with ADABL
#-----------------------
modelFile = paths.file_trainedmodel()
scalerFile = paths.file_trainedscaler()

blh_adabl=adabl.adabl_blh_estimation(lidarFile,modelFile,scalerFile,storeInNetcdf=False)


# Estimation with RS
#--------------------
rsdata=nc.Dataset(rsFile)
t_rs=rsdata.variables['time']
blh_rs=rsdata.variables['BEST_BLH']

rsdex=np.where(np.logical_and(t_rs>t_values[0],t_rs<t_values[-1]))[0]

# Plot
# ------
graphics.storeImages = False

graphics.blhs_over_data(
    t_values,
    z_values,
    rcs_1,
    [blh_kabl,blh_mnf[:,0],blh_adabl],
    ['KABL','INDUS','ADABL'],
    blh_rs=[t_rs[rsdex],blh_rs[rsdex]],
)

input("\n Press Enter to exit (close down all figures)\n")
