#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT GENERATING THE FIGURES OF THE STUDY BASED ON KABL PROGRAM

 +-----------------------------------------+
 |  Date of creation: 22 Oct. 2019         |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  DSO/DOA/IED and CNRM/GMEI/LISA         |
 +-----------------------------------------+
"""

# Usual Python packages
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import datetime as dt
import netCDF4 as nc
import seaborn as sns
import matplotlib.pyplot as plt

from kabl.ephemerid import Sun
from kabl import paths
from kabl import utils
from kabl import graphics


# LOAD AND PREPARE DATA
# ========================

# Load data
# -----------

NAMEXP = "Brest_kmeans_K3_Igiven_1prof_RCS1"
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
graphics.figureDir = paths.file_defaultlidardata()
graphics.fmtImages = ".svg"

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


### Combination
badvalues = np.logical_or(
    mask_lowrs, np.logical_or(
        ~mask_day, np.logical_or(
            mask_rain, np.logical_or(
                mask_fog,
                mask_cloud
            )
        )
    )
)
print(100 * np.sum(mask_rain) / mask_rain.size, "% of RS with rain")
print(100 * np.sum(mask_cloud) / mask_cloud.size, "% of RS with cloud under 3000m")
print(100 * (1 - np.sum(badvalues) / badvalues.size), "% of RS used")

BADVALUES = np.logical_or(
    ~MASK_DAY, np.logical_or(
        MASK_RAIN, np.logical_or(
            MASK_FOG,
            MASK_CLOUD
        )
    )
)

print(
    "BADVALUES : ",
    100 * np.sum(BADVALUES) / BADVALUES.size,
    " percent of data discarded",
)


# GRAPHICS
# ==========

cmap = plt.get_cmap("tab20")
color_kabl = cmap(0.01)
color_indus = cmap(0.11)
color_adabl = cmap(0.51)
color_rs = cmap(0.31)
color_kabl_no = cmap(0.09)
color_indus_no = cmap(0.19)
color_adabl_no = cmap(0.59)
color_rs_no = cmap(0.39)

# Seasonal cycle
# ----------------
avg, ct = graphics.mean_by_month(
    [
        BLH_KABL[~BADVALUES],
        BLH_ADABL[~BADVALUES],
        BLH_INDUS[~BADVALUES],
        blh_rs[~badvalues],
    ],
    [
        dtimeL[~BADVALUES],
        dtimeL[~BADVALUES],
        dtimeL[~BADVALUES],
        dtime_coloc[~badvalues],
    ],
    ["KABL", "ADABL", "INDUS", "RS"],
    colorList=[color_kabl, color_adabl, color_indus, color_rs],
)

graphics.plot_samplesize(ct,"month")


# Diurnal cycle
# ---------------

# Remove ~mask_day and mask_lowrs from bad values
rsbadvalues = np.logical_or(mask_rain, np.logical_or(mask_fog, mask_cloud))
rsbadvalues = np.logical_or(
    mask_lowrs, np.logical_or(
        mask_rain, np.logical_or(
            mask_fog,
            mask_cloud
        )
    )
)

onlyday = np.logical_and(mask_day, ~rsbadvalues)
mean11utc = np.nanmean(blh_rs[onlyday])
q25_11utc = np.nanquantile(blh_rs[onlyday], 0.25)
q75_11utc = np.nanquantile(blh_rs[onlyday], 0.75)
onlynight = np.logical_and(~mask_day, ~rsbadvalues)
mean23utc = np.nanmean(blh_rs[onlynight])
q25_23utc = np.nanquantile(blh_rs[onlynight], 0.25)
q75_23utc = np.nanquantile(blh_rs[onlynight], 0.75)

# Remove ~MASK_DAY from bad values
BADVALUES = np.logical_or(
    MASK_RAIN, np.logical_or(
        MASK_FOG,
        MASK_CLOUD
    )
)

avg6m, ct6m = graphics.mean_by_6min(
    [BLH_KABL[~BADVALUES], BLH_ADABL[~BADVALUES], BLH_INDUS[~BADVALUES]],
    [dtimeL[~BADVALUES], dtimeL[~BADVALUES], dtimeL[~BADVALUES]],
    ["KABL", "ADABL", "INDUS"],
    dataRS=[mean11utc, q25_11utc, q75_11utc, mean23utc, q25_23utc, q75_23utc],
    colorList=[color_kabl, color_adabl, color_indus, color_rs],
)

graphics.plot_samplesize(ct6m,"6min")

# Overall metrics
# -----------------

valid_blh_rs = blh_rs[~badvalues]
rmse = lambda x,m:np.sqrt(np.nanmean((x[m]-valid_blh_rs[m])**2))
corr = lambda x,m:np.corrcoef(x[m],valid_blh_rs[m])[0,1]

# Average gap
errl1_kabl = np.nanmean(np.abs(blh_kabl[~badvalues] - blh_rs[~badvalues]))
errl1_adabl = np.nanmean(np.abs(blh_adabl[~badvalues] - blh_rs[~badvalues]))
errl1_indus = np.nanmean(np.abs(blh_indus[~badvalues] - blh_rs[~badvalues]))

# RMSE
errl2_kabl = np.sqrt(np.nanmean((blh_kabl[~badvalues] - blh_rs[~badvalues]) ** 2))
errl2_adabl = np.sqrt(np.nanmean((blh_adabl[~badvalues] - blh_rs[~badvalues]) ** 2))
errl2_indus = np.sqrt(np.nanmean((blh_indus[~badvalues] - blh_rs[~badvalues]) ** 2))
ci_errl2_kabl = utils.bootstrap_confidence_interval(
    rmse,
    blh_kabl[~badvalues],
    1000
)
ci_errl2_adabl = utils.bootstrap_confidence_interval(
    rmse,
    blh_adabl[~badvalues],
    1000
)
ci_errl2_indus = utils.bootstrap_confidence_interval(
    rmse,
    blh_indus[~badvalues],
    1000
)
ci_rmse = np.array(
    [
        [ci_errl2_kabl[0],ci_errl2_adabl[0],ci_errl2_indus[0]],
        [ci_errl2_kabl[1],ci_errl2_adabl[1],ci_errl2_indus[1]],
    ]
)

# Correlation
badvalues = np.logical_or(badvalues,np.isnan(blh_adabl))
valid_blh_rs = blh_rs[~badvalues]

corr_kabl = corr(blh_kabl[~badvalues],np.arange(blh_kabl[~badvalues].size))
corr_indus = np.corrcoef(blh_indus[~badvalues], blh_rs[~badvalues])[0, 1]
corr_adabl = np.corrcoef(
    blh_adabl[np.logical_and(~np.isnan(blh_adabl), ~badvalues)],
    blh_rs[np.logical_and(~np.isnan(blh_adabl), ~badvalues)],
)[0, 1]

ci_corr_kabl = utils.bootstrap_confidence_interval(
    corr,
    blh_kabl[~badvalues],
    1000
)
ci_corr_indus = utils.bootstrap_confidence_interval(
    corr,
    blh_indus[~badvalues],
    1000
)
ci_corr_adabl = utils.bootstrap_confidence_interval(
    corr,
    blh_adabl[np.logical_and(~np.isnan(blh_adabl), ~badvalues)],
    1000
)
ci_corr = np.array(
    [
        [ci_corr_kabl[0],ci_corr_adabl[0],ci_corr_indus[0]],
        [ci_corr_kabl[1],ci_corr_adabl[1],ci_corr_indus[1]],
    ]
)

# Bar charts
graphics.bar_scores(
    [errl2_kabl, errl2_adabl, errl2_indus],
    scorename = "errl2",
    algos = ["KABL", "ADABL", "INDUS"],
    lowupbounds = ci_rmse,
    colors=[color_kabl, color_adabl, color_indus]
)
graphics.bar_scores(
    [corr_kabl, corr_adabl, corr_indus],
    scorename = "corr",
    algos = ["KABL", "ADABL", "INDUS"],
    lowupbounds = ci_corr,
    colors=[color_kabl, color_adabl, color_indus]
)


input("\n Press Enter to exit (close down all figures)\n")
