#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
LAUNCHING SCRIPT FOR KABL PROGRAM.

 +-----------------------------------------+
 |  Date of creation: 4 Oct. 2019          |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  DSO/DOA/IED and CNRM/GMEI/LISA         |
 +-----------------------------------------+
 
Copyright Meteo-France, 2019, [CeCILL-C](https://cecill.info/licences.en.html) license (open source)

This module is a computer program that is part of the KABL (K-means for 
Atmospheric Boundary Layer) program. This program performs boundary layer
height estimation for concentration profiles using K-means algorithm.

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C 
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.
'''

# Local packages
from kabl import core
from kabl import utils
from kabl import graphics
from kabl import adabl
# Usual Python packages
import pickle
import numpy as np
import datetime as dt
import pytz
import sys
import time
import netCDF4 as nc



lidarFile='../data_samples/lidar/DAILY_MPL_5025_20180802.nc'
rsFile='../data_samples/radiosoundings/BLH_RS_liss3_BRNliss10_BREST.nc'

t_values,z_values,rcs_1,rcs_2,blh_mnf=utils.extract_data(lidarFile,max_height=4620,to_extract=['rcs_1','rcs_2','pbl'])

# Estimation with KABL
#----------------------
params=dict()
params['algo']='kmeans'
params['n_clusters']=3
params['predictors']={'day':['rcs_1'],'night':['rcs_1']}
params['classif_score']='db'
params['n_inits']=1
params['n_profiles']=1
params['max_k']=6
params['init']='given'
params['cov_type']='full'
params['max_height']=4500
params['sunrise_shift']=1
params['sunset_shift']=-1

blh_kabl=core.blh_estimation(lidarFile,storeInNetcdf=False,params=params)

# Estimation with ADABL
#-----------------------
modelFile='../pre-trained-adabl/adabl_classifier_tzRCS12_M200_D5.pkl'
scalerFile='../pre-trained-adabl/adabl_scaler_tzRCS12_M200_D5.pkl'
blh_adabl=adabl.adabl_blh_estimation(lidarFile,modelFile,scalerFile,storeInNetcdf=False)


# Estimation with RS
#--------------------
rsdata=nc.Dataset(rsFile)
t_rs=rsdata.variables['time']
blh_rs=rsdata.variables['BEST_BLH']

rsdex=np.where(np.logical_and(t_rs>t_values[0],t_rs<t_values[-1]))[0]

# Plot
#------
graphics.blhs_over_data(t_values,z_values,rcs_1,
            [blh_kabl,blh_mnf[:,0],blh_adabl],['KABL','INDUS','ADABL'],
            blh_rs=[t_rs[rsdex],blh_rs[rsdex]],
            storeImages=False)

input("\n Press Enter to exit (close down all figures)\n")
