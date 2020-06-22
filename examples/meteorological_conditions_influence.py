#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Study the influence of meteorological conditions on the BLH (RC1.Q8, RC1.Q9)

 +-----------------------------------------+
 |  Date of creation: 11 June 2020         |
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
"""

import numpy as np
import pandas as pd
import datetime as dt
import netCDF4 as nc
import matplotlib.pyplot as plt

from kabl.ephemerid import Sun
from kabl import paths
from kabl import utils
from kabl import graphics


# LOAD AND PREPARE DATA
# ========================

# Load data
# -----------

NAMEXP = "Trappes_kmeans_K3_Igiven_1prof_RCS1"
loc = NAMEXP.split("_")[0]
paths.site = loc.upper()
paths.namexp = NAMEXP

rsFile = paths.file_blhfromrs()
lidarFile = paths.file_blhfromlidar()

# Check if RS and lidar are in the same location
loc_rs = rsFile.split("_")[-1].split(".")[0]
if loc_rs.upper() != loc.upper():
    raise Exception("Location of RS is not the same as for the lidar")


rcf = nc.Dataset(rsFile)
lcf = nc.Dataset(lidarFile)

# Set up graphics outputs
graphics.storeImages = False
graphics.figureDir = paths.resultrootdir
graphics.fmtImages = "_" + loc + "_test1.png"

# Colocate LIDAR and RS
# -----------------------
timeL = np.array(lcf.variables["time"])
timeRS = np.array(rcf.variables["time"])

BLH_KABL = np.array(lcf.variables["BLH_KABL"])
BLH_ADABL = np.array(lcf.variables["BLH_ADABL"])
BLH_INDUS = np.array(lcf.variables["BLH_INDUS"])
CBH = np.array(lcf.variables["CLOUD_BASE_HEIGHT"])
CH = np.array(lcf.variables["CH"])
DB = np.array(lcf.variables["DB"])
SILH = np.array(lcf.variables["SILH"])
MASK_RAIN = np.array(lcf.variables["MASK_RAIN"]).astype(bool)
MASK_FOG = np.array(lcf.variables["MASK_FOG"]).astype(bool)
N_CLUSTERS = np.array(lcf.variables["N_CLUSTERS"])
Chronos = np.array(lcf.variables["computation_time"])

BLH_RS=np.array(rcf.variables['BEST_BLH'])
# ~ BLH_RS = np.array(rcf.variables["parcel"])

t_coloc, blh_rs, var_lidar = utils.colocate_instruments(
    timeL,
    timeRS,
    [BLH_KABL, BLH_ADABL, BLH_INDUS, MASK_RAIN, MASK_FOG, CBH],
    BLH_RS,
    tol=10 * 60,
)
blh_kabl, blh_adabl, blh_indus, mask_rain, mask_fog, cbh = var_lidar

# Mask and filters
# ------------------

### Cloud
cbh_threshold = 3000
MASK_CLOUD = np.logical_and(CBH[:] < cbh_threshold, CBH[:] > 0)
mask_cloud = np.logical_and(cbh[:] < cbh_threshold, cbh[:] > 0)

### Low RS
mask_lowrs = blh_rs < 120

### Day/Night
sec_intheday = np.mod(t_coloc, 24 * 3600)
tmin_day = 8 * 3600
tmax_day = 16 * 3600
mask_day = np.logical_and(sec_intheday > tmin_day, sec_intheday < tmax_day)
day_or_night = np.full(len(t_coloc), "night")
day_or_night[mask_day] = "day"

print("Conversion to datetime...")
dtimeL = np.array([dt.datetime.utcfromtimestamp(t) for t in timeL])
dtimeRS = np.array([dt.datetime.utcfromtimestamp(t) for t in timeRS])
dtime_coloc = np.array([dt.datetime.utcfromtimestamp(t) for t in t_coloc])

lat, lon = utils.get_lat_lon(loc)
s = Sun(lat=lat, long=lon)
DAY_OR_NIGHT = np.full(len(dtimeL), "night")
for i in range(len(dtimeL)):
    t = dtimeL[i]
    sunrise = s.sunrise(t)
    sunset = s.sunset(t)
    sunrise = dt.datetime(
        t.year, t.month, t.day, sunrise.hour, sunrise.minute, sunrise.second
    )
    sunset = dt.datetime(
        t.year, t.month, t.day, sunset.hour, sunset.minute, sunset.second
    )
    if t > sunrise and t < sunset:
        DAY_OR_NIGHT[i] = "day"

MASK_DAY = DAY_OR_NIGHT == "day"


# ANALYSIS
# ========

# Count and trend
# ---------------
list_masks = [mask_fog, mask_rain, mask_cloud, mask_day, mask_lowrs]
list_masknames = ["fog", "rain", "cloud", "day", "lowrs"]

mean = lambda x,m:np.nanmean(x[m])
rmse = lambda x,m:np.sqrt(np.nanmean((x[m]-blh_rs[m])**2))
corr = lambda x,m:np.corrcoef(x[m],blh_rs[m])[0,1]
list_func = [mean,rmse,corr]
list_funcnames = ["mean","rmse","corr"]
list_colormaps = ["Blues","Reds","bwr"]

f=0
calc=list_func[f]

matmasks = np.diag(np.array(list_masks).sum(axis=1))
mat_blh_rs = np.diag([calc(blh_rs,m) for m in list_masks])
mat_blh_kabl = np.diag([calc(blh_kabl,m) for m in list_masks])
mat_blh_adabl = np.diag([calc(blh_adabl,m) for m in list_masks])
mat_blh_indus = np.diag([calc(blh_indus,m) for m in list_masks])

for m in range(len(list_masks)):
    print("Mask",m,list_masknames[m])
    for mm in range(m):
        comask=np.logical_and(list_masks[m],list_masks[mm])
        matmasks[mm,m]=comask.sum()
        mat_blh_rs[mm,m]=calc(blh_rs,comask)
        mat_blh_kabl[mm,m]=calc(blh_kabl,comask)
        mat_blh_adabl[mm,m]=calc(blh_adabl,comask)
        mat_blh_indus[mm,m]=calc(blh_indus,comask)
        

plt.matshow(matmasks,cmap=list_colormaps[f])
plt.xticks(np.arange(len(list_masks)), list_masknames)
plt.yticks(np.arange(len(list_masks)), list_masknames)
plt.colorbar()
plt.title('counts '+list_funcnames[f])
plt.show(block=False)


plt.matshow(mat_blh_rs,cmap=list_colormaps[f])
plt.xticks(np.arange(len(list_masks)), list_masknames)
plt.yticks(np.arange(len(list_masks)), list_masknames)
plt.colorbar()
plt.title('blh_rs '+list_funcnames[f])
plt.show(block=False)

plt.matshow(mat_blh_kabl,cmap=list_colormaps[f])
plt.xticks(np.arange(len(list_masks)), list_masknames)
plt.yticks(np.arange(len(list_masks)), list_masknames)
plt.colorbar()
plt.title('blh_kabl '+list_funcnames[f])
plt.show(block=False)

plt.matshow(mat_blh_adabl,cmap=list_colormaps[f])
plt.xticks(np.arange(len(list_masks)), list_masknames)
plt.yticks(np.arange(len(list_masks)), list_masknames)
plt.colorbar()
plt.title('blh_adabl '+list_funcnames[f])
plt.show(block=False)

plt.matshow(mat_blh_indus,cmap=list_colormaps[f])
plt.xticks(np.arange(len(list_masks)), list_masknames)
plt.yticks(np.arange(len(list_masks)), list_masknames)
plt.colorbar()
plt.title('blh_indus '+list_funcnames[f])
plt.show(block=False)

# Timeline
# --------
avg,ct = graphics.mean_by_month(list_masks,
    [dtime_coloc for m in list_masknames],
    list_masknames
)

LIST_MASKS = [MASK_FOG, MASK_RAIN, MASK_CLOUD, MASK_DAY]
LIST_MASKNAMES = ["FOG", "RAIN", "CLOUD", "DAY"]

avg,ct = graphics.mean_by_6min(LIST_MASKS,
    [dtimeL for m in LIST_MASKNAMES],
    LIST_MASKNAMES
)

# Scatterplot
# -----------

graphics.scatterplot_blhs(
    t_coloc,
    blh_x=blh_kabl,
    blh_y=blh_rs,
    blh_xlabel="BLH estimated by KABL",
    blh_ylabel="BLH estimated by RS",
    titre="BLH estimated by KABL and RS ("+str(blh_kabl.size)+" values)"
)

notnan = ~np.isnan(blh_adabl)
graphics.scatterplot_blhs(
    t_coloc[notnan],
    blh_x=blh_adabl[notnan],
    blh_y=blh_rs[notnan],
    blh_xlabel="BLH estimated by ADABL",
    blh_ylabel="BLH estimated by RS",
    titre="BLH estimated by ADABL and RS ("+str(blh_adabl[notnan].size)+" values)"
)

# Correlation vs height
# ---------------------

Nz = 15
z_tops = np.linspace(150,4000,Nz)
zcorr_kabl = np.zeros(Nz)
zcorr_indus = np.zeros(Nz)
zcorr_adabl = np.zeros(Nz)
zcount = np.zeros(Nz)

for iz in range(Nz):
    zmask = blh_rs < z_tops[iz]
    zcount[iz] = zmask.sum()/zmask.size
    zcorr_kabl[iz] = np.corrcoef(blh_rs[zmask],blh_kabl[zmask])[0,1]
    zcorr_indus[iz] = np.corrcoef(blh_rs[zmask],blh_indus[zmask])[0,1]
    
    zmask = np.logical_and(blh_rs < z_tops[iz],~np.isnan(blh_adabl))
    zcorr_adabl[iz] = np.corrcoef(blh_rs[zmask],blh_adabl[zmask])[0,1]

plt.figure()
plt.plot(z_tops,zcorr_kabl,label="KABL")
plt.plot(z_tops,zcorr_indus,label="INDUS")
plt.plot(z_tops,zcorr_adabl,label="ADABL")
plt.plot(z_tops,zcount,':',label="Nb of points")
plt.grid()
plt.legend()
plt.show(block=False)

input("\n Press Enter to exit (close down all figures)\n")
