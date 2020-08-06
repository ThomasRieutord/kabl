#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAUNCHING SCRIPT FOR KABL PROGRAM.
Test on CL31 data
"""

# Local packages
from kabl import core
from kabl import utils
from kabl import graphics
from kabl import adabl
from kabl import paths

# Usual Python packages
import os.path
import numpy as np
import datetime as dt
import time
import netCDF4 as nc


lidarFile = paths.file_defaultcl31data()

t_values, z_values, rcss = utils.extract_data(
    lidarFile, max_height=4620, to_extract=["rcs_0"]
)
rcs_0 = rcss["rcs_0"]

# Estimation with KABL
# ----------------------
params = utils.get_default_params()
params["n_clusters"] = 3
params["predictors"] = {"day": ["rcs_0"], "night": ["rcs_0"]}
params["n_profiles"] = 1
params["init"] = "advanced"

blh_kabl = core.blh_estimation(lidarFile, storeInNetcdf=False, params=params)

# Plot
# ------
graphics.storeImages = False

graphics.blhs_over_data(t_values, z_values, rcs_0, [blh_kabl], ['KABL'])

input("\n Press Enter to exit (close down all figures)\n")
