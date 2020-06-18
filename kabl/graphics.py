#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE OF GRAPHICAL TOOLS FOR THE KABL PROGRAM.
Provides functions to make usual plots in the problem of estimating the
boundary layer height with K-means.

Features:
    - quicklook_data
    - quicklook_testprofiles
    - blhs_over_profile
    - blhs_over_data
    - scatterplot_blhs
    - quicklook_output

Test of the functions: `python graphics.py`
Requires the test file at '../data_samples/lidar/DAILY_MPL_5025_20180802.nc'

 +-----------------------------------------+
 |  Date of creation: 6 Aug. 2019          |
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

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc

# Local packages
from kabl import utils
from kabl import paths

fmtImages=".png"
# Images will be saved under this format (suffix of plt.savefig)

figureDir=paths.resultrootdir
# Images will be saved in this directory (prefix of plt.savefig)

storeImages=False
# If True, figures are saved in files but not shown
# If False, figures are not saved in files but always shown

def quicklook_data(nc_file, max_height=4500, with_pbl=False, with_cbh=False):
    """Give a quick look of the data, only the data.
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
    
    [OUT]
         (matplotlib.pyplot figure): same as blhs_over_data"""

    location, day, lat, lon = utils.where_and_when(nc_file)

    to_be_extracted = ["rcs_0"]
    if with_pbl:
        to_be_extracted.append("pbl")
    if with_cbh:
        to_be_extracted.append("cloud_base_height")

    t, z, dat = utils.extract_data(
        nc_file, max_height=max_height, to_extract=to_be_extracted
    )
    
    rcs = dat["rcs_0"]
    if "pbl" in to_be_extracted:
        pbl = dat["pbl"]
    if "cloud_base_height" in to_be_extracted:
        cbh = dat["cloud_base_height"]

    plt.figure(figsize=(14, 7))
    plt.pcolormesh(t, z, rcs.T, alpha=0.8, cmap="rainbow", vmin=-0.1, vmax=0.8)
    if with_pbl:
        pbl[pbl == -999] = np.nan
        for layer in range(pbl.shape[1]):
            plt.plot(t, pbl[:, layer], "k*")
    if with_cbh:
        cbh[cbh == -999] = np.nan
        for layer in range(cbh.shape[1]):
            plt.plot(t, cbh[:, layer], "r.")
    axes = plt.gca()
    plt.title("Lidar backscatter | " + location + " " + day.strftime("%Y/%m/%d"))
    axes.set_xlabel("Hour")
    axes.set_ylabel("Height (m agl)")
    plt.tight_layout()
    plt.grid(color="white", ls="solid")
    plt.colorbar(label="Range corrected signal", alpha=0.8)

    locs, labels = plt.xticks()
    labels = [dt.datetime.utcfromtimestamp(loc).strftime("%H:%M") for loc in locs]

    axes.set_xticks(locs)
    axes.set_xticklabels(labels)
    plt.gcf().autofmt_xdate()
    plt.show(block=False)


def quicklook_testprofiles(nc_file):
    """Give a quick look of the preselected profiles, only the data.
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
    
    [OUT]
        - (matplotlib.pyplot figure): same as blhs_over_profile"""

    location, day, lat, lon = utils.where_and_when(nc_file)

    plt.figure(figsize=(14, 7))

    n_profiles = 4
    hours = ["0:17", "9:22", "16:17", "21:17"]

    for prof_id in range(n_profiles):
        z, rcs = utils.extract_testprofile(
            nc_file, to_extract=["rcs_0"], profile_id=prof_id
        )
        plt.subplot(1, n_profiles, prof_id + 1)
        plt.plot(rcs, z, linewidth=2, label=hours[prof_id])
        plt.legend()
    plt.tight_layout()
    plt.suptitle("Lidar backscatter | " + location + " " + day.strftime("%Y/%m/%d"))
    plt.show(block=False)


def blhs_over_profile(
    z_values,
    data_values,
    blhs,
    blhs_names=None,
    labels=None,
    titre=None,
):
    """Plot the profile of data and the BLH (more than one can be superimposed)
    
    [IN]
        - z_values (np.array([Nz])): array of height values
        - data_values (np.array([Nz])): array of data values (profile)
        - blhs (list of np.array([Nt])): list of BLHs time series
        - blhs_names (list of str): corresponding names of BLHs. Numbered by default.
        - labels (np.array([Nz])): array of clusters labels. Different for each cluster, but not meaningful on its own.
        - titre (str): Title of plot. Default is "Lidar backscatter | "+day
        - storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
        - fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .png
        - figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
    
    [OUT]
        - (matplotlib.pyplot figure): profile plot with horizontal bars for BLHs
            In the X-axis is the data from which we draw the profile (usually the range-corrected signal)
            In the Y-axis are the altitude values
    """

    if blhs_names is None:
        if isinstance(blhs, list):
            blhs_names = ["BLH {}".format(k + 1) for k in range(len(blhs))]
        else:
            blhs_names = "BLH"

    plt.figure(figsize=(14, 7))
    plt.plot(data_values, z_values, linewidth=2, label="RCS profile")

    vmin = np.nanmin(data_values)
    vmax = np.nanmax(data_values)
    print("vmin=", vmin, "vmax=", vmax)
    if isinstance(blhs, list):
        for ib in range(len(blhs)):
            plt.plot(
                [vmin, vmax],
                [blhs[ib], blhs[ib]],
                "--",
                label=blhs_names[ib],
                alpha=0.3,
                markersize=12,
            )
    else:
        plt.plot(
            [vmin, vmax], [blhs, blhs], "--", label=blhs_names, alpha=0.3, markersize=12
        )

    if labels is not None:
        for k in np.unique(labels):
            kdex = np.where(labels == k)[0]
            plt.plot(
                data_values[kdex],
                z_values[kdex],
                linewidth=0,
                marker=np.mod(k + 4, 11),
                label="Cluster {}".format(k + 1),
            )

    plt.legend()
    plt.tight_layout()
    if storeImages:
        plt.savefig(os.path.join(figureDir,"blhs_over_profile"+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+"blhs_over_profile"+fmtImages)
    else:
        plt.show(block=False)
    plt.show(block=False)


def blhs_over_data(
    t_values,
    z_values,
    bckgrd_data,
    blhs,
    blhs_names=None,
    blh_rs=None,
    titre=None,
):
    """
    Plot the BLH time series over some background data (usually the 
    range-corrected signal). More than one BLH time series can be
    superimposed.
    
    [IN]
        - t_values (np.array([Nt])): array of time values (POSIX timestamps)
        - z_values (np.array([Nz])): array of height values
        - bckgrd_data (np.array([Nt,Nz])): array of backscatter signal Nt is len of time, Nz is len of height
        - blhs (list of np.array([Nt])): list of BLHs time series
        - blhs_names (list of str): corresponding names of BLHs. Numbered by default.
        - blh_rs (tuple[2] or list[2]): if any, BLH measured by radiosondes. First element is the list of time, second is the list of BLH values.
        - titre (str): Title of plot. Default is "Lidar backscatter | "+day
        - storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
        - fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .png
        - figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
    
    [OUT]
        - (matplotlib.pyplot figure): display profil plot:
            It has color distribution of backscatter signal (RCS) with the manufacturer blh in black and estimated blh in yellow
            In the X-axis are the Nt times inputs of backscatter signal
            In the Y-axis are the altitude values"""
    
        
    if blhs_names is None:
        if isinstance(blhs, list):
            blhs_names = ["BLH {}".format(k + 1) for k in range(len(blhs))]
        else:
            blhs_names = "BLH"

    day = dt.datetime.utcfromtimestamp(t_values[1]).strftime("%Y/%m/%d")
    date = dt.datetime.utcfromtimestamp(t_values[1]).strftime("%Y%m%d")
    if titre is None:
        titre = "Lidar backscatter | " + day

    fig = plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        t_values,
        z_values,
        bckgrd_data.T,
        alpha=0.8,
        cmap="rainbow",
        vmin=-0.1,
        vmax=0.8,
    )
    plt.colorbar(alpha=0.5)
    plt.grid(color="white", ls="solid")

    plt.title(titre)

    if isinstance(blhs, list):
        for ib in range(len(blhs)):
            plt.plot(t_values, blhs[ib], ".:", label=blhs_names[ib])
    else:
        plt.plot(t_values, blhs, ".:", label=blhs_names)

    if blh_rs is not None:
        rsTimes = blh_rs[0]
        rsValues = blh_rs[1]
        plt.plot(rsTimes, rsValues, "kP", markersize=12, linewidth=3, label="RS")

    axes = plt.gca()
    axes.set_xlabel("Hour")
    axes.set_ylabel("Height (m agl)")
    locs, labs = plt.xticks()
    labs = [dt.datetime.utcfromtimestamp(loc).strftime("%H:%M") for loc in locs]
    axes.set_xticks(locs)
    axes.set_xticklabels(labs)
    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.tight_layout()

    if storeImages:
        fileName = "_".join(["blhsOverData", date])
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)

    return fig


def quicklook_output(nc_file):
    """Same as blhs_over_data, but directly from the output netcf file
    (and with less flexibility).
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
    
    [OUT]
        - (matplotlib.pyplot figure): same as blhs_over_data"""

    location, day, lat, lon = utils.where_and_when(nc_file)
    t, z, dat = utils.extract_data(
        nc_file, to_extract=["rcs_0", "blh_kabl", "pbl"]
    )
    rcs =  dat["rcs_0"]
    blh_new = dat["blh_kabl"]
    blh_mnf = dat["pbl"]
    
    fig = blhs_over_data(
        t,
        z,
        rcs,
        [blh_new, blh_mnf[:, 0]],
        blhs_names=["BLH KABL", "BLH manufacturer"],
        titre="Lidar backscatter | " + location + " " + day.strftime("%Y/%m/%d"),
    )

    return fig


def quicklook_benchmark(
    data_file,
    blh_file,
    rs_file=None,
):
    """Same as blhs_over_data, but directly from the output netcf file
    (and with less flexibility).
    
    [IN]
        - data_file (str): path to the netcdf file containing the data
        - data_file (str): path to the netcdf file containing the BLH estimation
    
    [OUT]
        - (matplotlib.pyplot figure): same as blhs_over_data"""

    location, day, lat, lon = utils.where_and_when(data_file)
    t, z, rcss = utils.extract_data(data_file, to_extract=["rcs_0"])
    rcs = rcss["rcs_0"]

    BLHS = []
    BLH_NAMES = []
    ncf = nc.Dataset(blh_file)
    for key in ncf.variables.keys():
        if "BLH" in key:
            BLHS.append(np.array(ncf.variables[key]))
            BLH_NAMES.append(key[4:])

    if rs_file is not None:
        blh_rs = utils.extract_rs(rs_file, t[0], t[-1])
    else:
        blh_rs = None

    fig = blhs_over_data(
        t,
        z,
        rcs,
        BLHS,
        blhs_names=BLH_NAMES,
        blh_rs=blh_rs,
        titre="Lidar backscatter | " + location + " " + day.strftime("%Y/%m/%d"),
    )

    return fig


def scatterplot_blhs(
    time,
    blh_x,
    blh_y,
    blh_xlabel=None,
    blh_ylabel=None,
    titre=None,
):
    """
    Daily plot of differences between manufacturer BLH and estimated BLH
    
    [IN]
        - time (np.array([Nt])): array of time values
        - blh_ref (np.array([Nt])): array of BLH estimated by constructor
        - blh_new (np.array([Nt])): array of BLH estimation by Kmeans algorithm
        - titre (str): Title of plot. Default is 'Manufacturer vs estimated BLH \n corr: {}'
        - storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
        - fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .png
        - figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
    
    [OUT]
        - (matplotlib.pyplot figure): display profil plot:
            It has points distribution of backscatter signal (RCS) with the manufactuer BLH and estimated BLH.
            In the X-axis are the calculated values of BLH by constructor
            In the Y-axis are the calculated values of BLH by Kmeans algorithm
    """

    date = dt.datetime.utcfromtimestamp(time[1]).strftime("%Y%m%d")
    corr = np.corrcoef(blh_x, blh_y)[0, 1]
    if titre is None:
        titre = "Manufacturer vs estimated BLH \n" + date + " corr={0:.2f}".format(corr)
    
    fig = plt.figure(figsize=(14, 7))
    plt.plot([200, 2500], [200, 2500], "k-", linewidth=2)
    print("date=", date, type(date))
    Nt = len(time)
    heure = [
        float(dt.datetime.utcfromtimestamp(time[i]).strftime("%H"))
        for i in np.arange(0, Nt, 1)
    ]
    print("heure[0]=", heure[0], type(heure[0]), "heure[-1]=", heure[-1])
    axes = plt.gca()
    if not blh_xlabel is None:
        axes.set_xlabel(blh_xlabel)
    if not blh_ylabel is None:
        axes.set_ylabel(blh_ylabel)
    label = ["00:00", "06:00", "12:00", "18:00", "22:00", "00:00"]
    plt.scatter(blh_x, blh_y, c=heure, cmap="hsv", s=6 * 2)
    # ~ plt.scatter(blh_x, blh_y, s=6 * 2)
    cbar = plt.colorbar()
    cbar.set_label("Hour")
    plt.annotate("Corr = {0:.2f}".format(corr), (1000, 2000))

    plt.title(titre)
    plt.tight_layout()
    
    if storeImages:
        fileName = "_".join(["scatterplotBLHs", date])
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)
    
    return fig


def estimator_quality(
    accuracies,
    chronos,
    estimator_names,
    titl=None,
):
    """Display score versus computation time for a series of estimators.
    Best estimators are in the bottom-right corner.
    
    Abcissa is the R2-score (1-mse/variance: the higher, the better)
    Ordinate is the computing time.
    Both are recorded for Ne estimators and Nr random split of testing
    and training sets.
    
    [IN]
        - accuracies (np.array[Ne,Nr]): R2-score for all estimators and random split
        - chronos (np.array[Ne,Nr]): computing time for all estimators and random split
        - estimator_names (list[Ne] of str): names of the estimators
        - titl (str): if provided, change the title of the figure
    
    [OUT] figure
    """

    if titl is None:
        titl = "Performance/speed comparison of estimators"

    plt.figure(figsize=(8, 8))
    plt.title(titl)
    for icl in range(len(estimator_names)):
        xtext = np.mean(accuracies[icl, :])
        ytext = np.mean(chronos[icl, :])
        plt.scatter(
            accuracies[icl, :], chronos[icl, :], alpha=0.8, label=estimator_names[icl]
        )
        plt.text(xtext, ytext, estimator_names[icl], fontweight="bold")
    plt.grid()
    plt.xlabel("R2-score")
    plt.ylabel("Comp. time")
    plt.legend(loc="best")
    if storeImages:
        fileName = "estimator_quality"
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)


def clusterZTview(t_values,z_values,zoneID,titl=None):
    '''Plots cluster labels in the same time and altitude grid where
    measurements have been done (boundary layer classification).
    
    [IN]
        - t_values (np.array[nt]): vector of time
        - z_values (np.array[nalt]): vector of altitude
        - zoneID (np.array[N]): cluster labels of each obs
        - delete_mask (np.array[nt*nalt]): mask at True when observation has been removed by the deletelines function (to avoid NaNs)
        - fileName (str): customised file name for saving the figure
        - clustersIDs (dict): dictionary associating cluster labels to boundary layer types
                Example: {0:"CL",1:"SBL",2:"FA",3:"ML"}. Default is {0:0,1:1,...}.
        - displayClustersIDs (bool): if True, displays the clusterIDs over the graph, at the center of the cluster.
        - titl (str): customised title for the figure
        
    [OUT] clusters labels on a time-altitude grid
        In X-axis is the time
        In Y-axis is the height (m agl)
        Clusters are shown with differents colors.'''
    
    
    clusterMarks={0:'bo',1:'gx',2:'r^',3:'cv',4:'ys',5:'m*',6:'kp',7:'gd',8:'bx',
        9:'ro',10:'c*',11:'y+',12:'m<',13:'k,'}
    
    K=np.max(zoneID)+1
    clustersIDs=np.arange(K)
    
    if titl is None:
        titl="Cluster in time-altitude grid"
    
    clist = []
    cticks = []
    cticklabels = []
    for k in range(K):
        cticks.append(k+0.5)
        cticklabels.append(clustersIDs[k])
        clist.append(clusterMarks[clustersIDs[k]][0])
    colormap=ListedColormap(clist)
    
    dt_values = [dt.datetime.utcfromtimestamp(t) for t in t_values]
    
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig=plt.figure()
    plt.title(titl)
    plt.pcolormesh(dt_values,z_values,zoneID.T,vmin=0,vmax=K,cmap=colormap)
    cbar=plt.colorbar(label="Cluster label")
    cbar.set_ticks(cticks)
    cbar.set_ticklabels(cticklabels)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time (UTC)")
    plt.ylabel("Alt (m agl)")
    if storeImages:
        fileName="clusterZTview"
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



def mean_by_month(
    datavaluesList, datatimesList, datanamesList=None, colorList=None
):
    """Calculate the average of all values in the same month.
    """

    if not isinstance(datavaluesList, list):
        raise TypeError("Argument datavaluesList must be a list")
    if not isinstance(datatimesList, list):
        raise TypeError("Argument datatimesList must be a list")

    if datanamesList is None:
        datanamesList = ["Data " + str(k) for k in range(len(datavaluesList))]
    if colorList is None:
        cmap = plt.get_cmap("jet")
        colorList = [
            cmap(np.mod(0.1 + k * np.pi / 5, 1)) for k in range(len(datavaluesList))
        ]
    
    
    averages = []
    effectifs = []
    monthLabels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    
    plt.figure()
    plt.title("Monthly average and quartiles")
    
    for i in range(len(datavaluesList)):
        dataValues = datavaluesList[i]
        dataTime = datatimesList[i]
        
        df = pd.DataFrame(dataValues, index=dataTime)
        gbm = df.groupby(df.index.month)
        
        plt.plot(
            gbm.mean().index,
            gbm.mean().values,
            "-",
            linewidth=2,
            label=datanamesList[i],
            c=colorList[i],
        )
        plt.fill_between(
            gbm.mean().index,
            gbm.quantile(0.25).values[:, 0],
            gbm.quantile(0.75).values[:, 0],
            color=colorList[i],
            alpha=0.1,
        )
        
        averages.append(gbm.mean().values)
        effectifs.append(gbm.count())
    
    plt.xticks(gbm.mean().index, monthLabels)
    plt.grid()
    plt.legend(loc="best")
    if storeImages:
        fileName="seasonalcycle_BLHs"
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)

    return averages, effectifs


def mean_by_6min(
    datavaluesList,
    datatimesList,
    datanamesList=None,
    dataRS=None,
    colorList=None,
):
    """Calculate the average of all values in the same 6 minutes slot.
    
    [IN]
        - datavaluesList (float,list): list of values
        - datatimesList (datetime,list): list of times values
        - dataRS (opt, list): list of info about radiosoundings, if any.
        - datanameList (opt,str,list): list of description values to draw
        - colorList (opt,list): list of colors to use
        - plot (opt,bool): If True show plot
    
    [OUT]
        - (matplotlib.pyplot figure): diurnal cycle mean over datatimes values
        Each plot represents the diurnal cycle of given in input
        In X-axis is the local hour
        In Y-axis is fonction of datavaluesList entered
    """

    if not isinstance(datavaluesList, list):
        raise TypeError("Argument datavaluesList must be a list")
    if not isinstance(datatimesList, list):
        raise TypeError("Argument datatimesList must be a list")

    if datanamesList is None:
        datanamesList = ["Data " + str(k) for k in range(len(datavaluesList))]

    if colorList is None:
        cmap = plt.get_cmap("jet")
        colorList = [
            cmap(np.mod(0.1 + k * np.pi / 5, 1)) for k in range(len(datavaluesList) + 1)
        ]
    
    plt.figure()
    plt.title("6-minute average and quartiles")

    averages = []
    effectifs = []

    if dataRS is not None:
        mean11utc, q25_11utc, q75_11utc, mean23utc, q25_23utc, q75_23utc = dataRS
        plt.plot(
            [11.5, 23.5], [mean11utc, mean23utc], "o", color=colorList[-1], label="RS"
        )
        plt.plot([11.5, 11.5], [q25_11utc, q75_11utc], "-+", color=colorList[-1])
        plt.plot([23.5, 23.5], [q25_23utc, q75_23utc], "-+", color=colorList[-1])

    for i in range(len(datavaluesList)):
        dataValues = datavaluesList[i]
        dataTime = datatimesList[i]

        df = pd.DataFrame(dataValues, index=dataTime)
        gbm = df.groupby(df.index.hour + np.round(df.index.minute / 60, 1))

        plt.plot(
            gbm.mean().index,
            gbm.mean().values,
            "-",
            linewidth=2,
            label=datanamesList[i],
            c=colorList[i],
        )
        plt.fill_between(
            gbm.mean().index,
            gbm.quantile(0.25).values[:, 0],
            gbm.quantile(0.75).values[:, 0],
            color=colorList[i],
            alpha=0.1,
        )

        averages.append(gbm.mean().values)
        effectifs.append(gbm.count())

    plt.grid()
    plt.legend(loc="best")
    if storeImages:
        fileName="diurnalcycle_BLHs"
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)

    return averages, effectifs


def plot_samplesize(
    effectifs,
    groupby,
):
    """
    Display the sample size for each average previously computed
    """
    
    algos = ["KABL", "ADABL", "INDUS", "RS"]
    
    if groupby=="6min":
        titl = "Number of values for 6-minute average"
        fileName="diurnalcycle_samplesize"
        x = np.linspace(0,24,241)
    elif groupby=="month":
        titl = "Number of values for monthly average"
        fileName="seasonalcycle_samplesize"
        x = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    else:
        raise ValueError("Unsupported groupby type:",groupby)
        
    plt.figure()
    plt.title(titl)
    for k in range(len(effectifs)):
        plt.plot(x, effectifs[k], label=algos[k])
    plt.grid()
    plt.legend()
    if storeImages:
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)
        

def bar_scores(scores, scorename, algos = None, colors = None):
    
    
    
    if algos is None:
        algos = ["Algo "+str(k+1) for k in range(len(scores))]
    
    if colors is None:
        colors = [plt.get_cmap("tab20")(k/len(scores)) for k in range(len(scores))]
    
    if scorename == "errl1":
        titl = "Overall average gap with RS"
    elif scorename == "errl2":
        titl = "Overall RMSE w.r.t RS"
    elif scorename == "corr":
        titl = "Overall correlation with RS"
    else:
        raise ValueError("Unknown score:",scorename)
    
    
    sns.set()
    
    plt.figure()
    plt.title(titl)
    plt.bar(
        algos,
        scores,
        color=colors,
    )
    if storeImages:
        fileName = scorename
        plt.savefig(os.path.join(figureDir,fileName+fmtImages))
        plt.close()
        print("Figure saved:",figureDir+fileName+fmtImages)
    else:
        plt.show(block=False)


def plot_cv_indices(cv, X, y, group=None, lw=10):
    """Create a sample plot for indices of a cross-validation object.
    
    16/06/2020
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
    """
    
    fig, ax = plt.subplots()
    
    if group is None:
        group = np.zeros_like(y)
    
    n_splits = cv.get_n_splits()
    
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm,
                   vmin=-.2, vmax=1.2, alpha=0.1)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Paired)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=plt.cm.Paired)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    
    plt.show(block=False)
    
    return ax
