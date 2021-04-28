#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE OF GRAPHICAL TOOLS FOR THE KABL PROGRAM.

 +-----------------------------------------+
 |  Date of creation: 6 Aug. 2019          |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  DSO/DOA/IED and CNRM/GMEI/LISA         |
 +-----------------------------------------+
"""

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc
import os.path

# Local packages
from kabl import utils
from kabl import paths

fmtImages = ".png"
# Images will be saved under this format (suffix of plt.savefig)

figureDir = paths.resultrootdir
# Images will be saved in this directory (prefix of plt.savefig)

storeImages = False
# If True, figures are saved in files but not shown
# If False, figures are not saved in files but always shown


def quicklook_data(nc_file, max_height=4500, with_pbl=False, with_cbh=False):
    """Give a quick look of the data, only the data.
    
    Parameters
    ----------
    nc_file : str
        Path to the netcdf file containing the data
    
    max_height : {float, int}, default=4500
        Top height on the graphic
    
    with_pbl : bool, default=False
        If True, add onto the data the boundary layer height calculated
        by the manufacturer
    
    with_cbh : bool, default=False
        If True, add onto the data the first cloud base height
        calculated by the manufacturer
    
    Returns
    -------
    None
    """

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
    plt.pcolormesh(
        t,
        z,
        rcs.T,
        alpha=0.8,
        cmap="rainbow",
        vmin=-0.1,
        vmax=0.8,
        shading="auto"
    )
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
    plt.colorbar(label="Range corrected signal")

    locs, labels = plt.xticks()
    labels = [dt.datetime.utcfromtimestamp(loc).strftime("%H:%M") for loc in locs]

    axes.set_xticks(locs)
    axes.set_xticklabels(labels)
    plt.gcf().autofmt_xdate()
    plt.show(block=False)


def quicklook_testprofiles(nc_file):
    """Give a quick look of the preselected profiles, only the data.
    
    Parameters
    ----------
    nc_file : str
        Path to the netcdf file containing the data
    
    Returns
    -------
    None
    """

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
    z_values, data_values, blhs, blhs_names=None, labels=None, titre=None,
):
    """Plot a single profile of data and the BLH (more than one can be
    superimposed)
    
    Parameters
    ----------
    z_values : ndarray of shape (Nz,)
        Height values
        
    data_values : ndarray of shape (Nz,)
        Data values along height
        
    blhs : list of array-like of shape (Nt,)
        Boundary layer height time series
        
    blhs_names : list of str, default=None
        Corresponding names of BLHs. Numbered by default.
    
    labels : ndarray of shape (Nz,), default=None
        Clusters labels. BEWARE: the cluster identification number are
        random. Only borders matter.
    
    titre : str, default=None
        Title of plot. Default is "Lidar backscatter | "+day
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Profile plot with horizontal bars for BLHs. In the X-axis is the
        data from which we draw the profile (usually the
        range-corrected signal). In the Y-axis are the altitude values
    """

    if blhs_names is None:
        if isinstance(blhs, list):
            blhs_names = ["BLH {}".format(k + 1) for k in range(len(blhs))]
        else:
            blhs_names = "BLH"

    fig = plt.figure(figsize=(14, 7))
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
        plt.savefig(os.path.join(figureDir, "blhs_over_profile" + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + "blhs_over_profile" + fmtImages)
    else:
        plt.show(block=False)
    
    return fig


def blhs_over_data(
    t_values, z_values, bckgrd_data, blhs, blhs_names=None, blh_rs=None, titre=None,
):
    """
    Plot the BLH time series over some background data (usually the 
    range-corrected signal). More than one BLH time series can be
    superimposed.
    
    
    Parameters
    ----------
    t_values : ndarray of shape (Nt,)
        Time values as POSIX timestamps:
        number of seconds since 1970/01/01 00:00 UTC
    
    z_values : ndarray of shape (Nz,)
        Height values
    
    bckgrd_data : ndarray of shape (Nt,Nz)
        Data values along height
    
    blhs : list of array-like of shape (Nt,)
        Boundary layer height time series
    
    blhs_names : list of str, default=None
        Corresponding names of BLHs. Numbered by default.
    
    blh_rs : {tuple, list} of length 2, default=None
        BLH measured by radiosondes, if any. First element is the list
        of time, second is the list of BLH values.
        
    titre : str, default=None
        Title of plot. Default is "Lidar backscatter | "+day
    
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Time-altitude graph with backscatter signal (RCS) in shades of
        colors, with BLHs estimation superimposed. In the X-axis are the
        Nt times. In the Y-axis are the altitude values
    """

    if blhs_names is None:
        if isinstance(blhs, list):
            blhs_names = ["BLH {}".format(k + 1) for k in range(len(blhs))]
        else:
            blhs_names = "BLH"

    day = dt.datetime.utcfromtimestamp(t_values[1]).strftime("%Y/%m/%d")
    date = dt.datetime.utcfromtimestamp(t_values[1]).strftime("%Y%m%d")
    if titre is None:
        titre = "Lidar backscatter (log10) | " + day

    fig = plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        t_values,
        z_values,
        bckgrd_data.T,
        alpha=0.8,
        cmap="rainbow",
        vmin=-0.1,
        vmax=0.8,
        shading="auto"
    )
    plt.colorbar()
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
        fileName = "_".join(["blhsOverData", date, paths.site])
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)

    return fig


def quicklook_output(nc_file):
    """Same as blhs_over_data, but directly from the output netcf file
    (and with less flexibility).
    
    Parameters
    ----------
    nc_file : str
        Path to the netcdf file containing the data
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Same as kabl.graphics.blhs_over_data
    """

    location, day, lat, lon = utils.where_and_when(nc_file)
    t, z, dat = utils.extract_data(nc_file, to_extract=["rcs_0", "blh_kabl", "pbl"])
    rcs = dat["rcs_0"]
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
    data_file, blh_file, rs_file=None,
):
    """Same as blhs_over_data, but directly from the output netcf file
    (and with less flexibility).
    
    Parameters
    ----------
    data_file : str
        Path to the netcdf file containing the data
    
    blh_file : str
        Path to the netcdf file containing the BLH estimation
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Same as kabl.graphics.blhs_over_data
    """

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
    time, blh_x, blh_y, blh_xlabel=None, blh_ylabel=None, titre=None,
):
    """Scatter-plot with hour coloration to compare BLHs estimations
    
    Parameters
    ----------
    time : array-like of shape (Nt,)
        Time values as POSIX timestamps:
        number of seconds since 1970/01/01 00:00 UTC
    
    blh_x : array-like of shape (Nt,)
        BLH time series to put on X-axis
    
    blh_y : array-like of shape (Nt,)
        BLH time series to put on Y-axis
    
    blh_xlabel : str, default=None
        Name of the BLH estimation on X-axis
    
    blh_ylabel : str, default=None
        Name of the BLH estimation on Y-axis
        
    titre : str, default=None
        Title of plot. Default is "Manufacturer vs estimated BLH \n corr: {}"
    
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Joint distribution of two estimation of boundary layer height.
        Values of a BLH estimation is on X-axis, the other on Y-axis.
        If they agree, they will align with the y=x line drawn in black.
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
        fileName = "_".join(["scatterplotBLHs", date, paths.site])
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)

    return fig


def estimator_quality(
    accuracies, chronos, estimator_names, titl=None,
):
    """Display score versus computation time for a series of estimators.
    Best estimators are in the bottom-right corner.
    
    Abcissa is the accuracy. Ordinate is the computing time. Both are 
    recorded for Ne estimators and Nr random split of testing
    and training sets.
    
    Parameters
    ----------
    accuracies : ndarray of shape (Ne,Nr)
        Accuracy score for all estimators and random split
        
    chronos : ndarray of shape (Ne,Nr)
        Computing time for all estimators and random split
    
    estimator_names : list of str
        Names of the estimators
        
    titl : str, default=None
        Title of plot. Default is "Performance/speed comparison of estimators"
    
    [Returns
    -------
    `matplotlib.pyplot figure`
        Abcissa is the accuracy. Ordinate is the computing time.
    """

    if titl is None:
        titl = "Performance/speed comparison of estimators"

    fig = plt.figure(figsize=(8, 8))
    plt.title(titl)
    for icl in range(len(estimator_names)):
        xtext = np.mean(accuracies[icl, :])
        ytext = np.mean(chronos[icl, :])
        plt.scatter(
            accuracies[icl, :], chronos[icl, :], alpha=0.8, label=estimator_names[icl]
        )
        plt.text(xtext, ytext, estimator_names[icl], fontweight="bold")
    plt.grid()
    plt.xlabel("Accuracy")
    plt.ylabel("Comp. time")
    plt.legend(loc="best")
    if storeImages:
        fileName = "estimator_quality"
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    
    return fig


def clusterZTview(t_values, z_values, zoneID, lineplot=None, titl=None):
    """Plots cluster labels in the same time and altitude grid where
    measurements have been done
    
    
    Parameters
    ----------
    t_values : ndarray of shape (Nt,)
        Time values as POSIX timestamps:
        number of seconds since 1970/01/01 00:00 UTC
    
    z_values : ndarray of shape (Nz,)
        Height values
    
    zoneID : ndarray of shape (N,)
        Cluster labels of each obs
    
    titl : str, default=None
        Title of plot. Default is "Cluster in time-altitude grid"
    
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Time-altitude graph with cluster labels in shades of colors. In
        the X-axis are the Nt times. In the Y-axis are the altitude values
    """

    clusterMarks = {
        0: "bo",
        1: "gx",
        2: "r^",
        3: "cv",
        4: "ys",
        5: "m*",
        6: "kp",
        7: "gd",
        8: "bx",
        9: "ro",
        10: "c*",
        11: "y+",
        12: "m<",
        13: "k,",
    }

    K = np.max(zoneID) + 1
    clustersIDs = np.arange(K)

    if titl is None:
        titl = "Cluster in time-altitude grid"

    clist = []
    cticks = []
    cticklabels = []
    for k in range(K):
        cticks.append(k + 0.5)
        cticklabels.append(clustersIDs[k])
        clist.append(clusterMarks[clustersIDs[k]][0])
    colormap = ListedColormap(clist)

    dt_values = [dt.datetime.utcfromtimestamp(t) for t in t_values]

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig = plt.figure()
    plt.title(titl)
    plt.pcolormesh(
        dt_values,
        z_values,
        zoneID.T,
        vmin=0,
        vmax=K,
        cmap=colormap,
        shading="auto"
    )
    if lineplot is not None:
        plt.plot(dt_values, lineplot, 'k-')
    cbar = plt.colorbar(label="Cluster label")
    cbar.set_ticks(cticks)
    cbar.set_ticklabels(cticklabels)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time (UTC)")
    plt.ylabel("Alt (m agl)")
    if storeImages:
        fileName = "clusterZTview"
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    return fig


def mean_by_month(datavaluesList, datatimesList, datanamesList=None, colorList=None):
    """Calculate and plot the average of all values in the same month.
    
    Parameters
    ----------
    datavaluesList : list of array-like of shape (Nt,)
        Time series of data to average
    
    datavaluesList : list of array-like of shape (Nt,)
        Time values as datetime.datetime objects, for each data provided
    
    datanamesList : list of str, default=None
        Names of the data provided. Default is numbered
    
    colorList : str, default=None
        Colors to attribute to each data
    
    
    Returns
    -------
    averages : list of array-like
        Monthly average for each data provided
    
    effectifs : list of array-like
        Number of points used to perform the monthly average
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
    plt.title("Monthly average and quartiles | " + paths.site)

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
        effectifs.append(gbm.count().values)

    plt.xticks(gbm.mean().index, monthLabels)
    plt.grid()
    plt.legend(loc="best")
    if storeImages:
        fileName = "_".join(["seasonalcycle","BLHs",paths.site])
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)

    return averages, effectifs


def mean_by_6min(
    datavaluesList, datatimesList, datanamesList=None, dataRS=None, colorList=None,
):
    """Calculate the average of all values in the same 6 minutes slot.
    
    Parameters
    ----------
    datavaluesList : list of array-like of shape (Nt,)
        Time series of data to average
    
    datavaluesList : list of array-like of shape (Nt,)
        Time values as datetime.datetime objects, for each data provided
    
    datanamesList : list of str, default=None
        Names of the data provided. Default is numbered
    
    dataRS : {tuple, list} of float, default=None
        Info about radiosoundings. Contains mean, 0.25 and 0.75 quantiles
        for the launches of 11:15 and 23:15 UTC.
        ```dataRS = (
            mean11utc, q25_11utc, q75_11utc,
            mean23utc, q25_23utc, q75_23utc
        )```
    
    colorList : str, default=None
        Colors to attribute to each data
    
    
    Returns
    -------
    averages : list of array-like
        Monthly average for each data provided
    
    effectifs : list of array-like
        Number of points used to perform the monthly average
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
    plt.title("Six-minute average and quartiles | " + paths.site)

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
        effectifs.append(gbm.count().values)

    plt.grid()
    plt.legend(loc="best")
    if storeImages:
        fileName = "_".join(["diurnalcycle","BLHs",paths.site])
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)

    return averages, effectifs


def plot_samplesize(effectifs, groupby):
    """
    Display the sample size for each average previously computed
    
    
    Parameters
    ----------
    effectifs : list of array-like
        Number of points used to perform the monthly average
    
    groupby : {'6min', 'month'}
        Specify how were grouped the data
    
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Line plot with all sample size and the correct time axis
    
    """

    algos = ["KABL", "ADABL", "INDUS", "RS"]

    if groupby == "6min":
        titl = "Number of values for 6-minute average"
        cycletype = "diurnalcycle"
        x = np.linspace(0, 24, 241)
    elif groupby == "month":
        titl = "Number of values for monthly average"
        cycletype = "seasonalcycle"
        x = [
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
    else:
        raise ValueError("Unsupported groupby type:", groupby)

    titl = titl + " | " + paths.site
    
    fig = plt.figure()
    plt.title(titl)
    for k in range(len(effectifs)):
        plt.plot(x, effectifs[k], label=algos[k])
    plt.grid()
    plt.legend()
    if storeImages:
        fileName = "_".join([cycletype,"samplesize",paths.site])
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    
    return fig


def bar_scores(scores, scorename, algos=None, lowupbounds=None, colors=None):
    """Bar plot with the quality score estimated to evaluate KABL and
    ADABL estimations.
    
    
    Parameters
    ----------
    scores : array-like
        Score values of each algorithm
    
    scorename : {'errl1', 'errl2', 'corr'}
        Name of the score
    
    algos : array-like of str, default=None
        Names of all algorithms. Default is numbered
    
    lowupbounds : ndarray of shape (2,len(scores)), default=None
        Confidence interval for the score values. First line gives lower
        bound, second line gives upper bound.
    
    colors : array-like of matplotlib.pyplot colors
        Colors to attribute to each algorithm
    
    
    Returns
    -------
    `matplotlib.pyplot figure`
        Bar plot with the quality score
    """
    if algos is None:
        algos = ["Algo " + str(k + 1) for k in range(len(scores))]

    if colors is None:
        colors = [plt.get_cmap("tab20")(k / len(scores)) for k in range(len(scores))]

    if scorename == "errl1":
        titl = "Overall average gap with RS"
    elif scorename == "errl2":
        titl = "Overall RMSE w.r.t RS"
    elif scorename == "corr":
        titl = "Overall correlation with RS"
    else:
        raise ValueError("Unknown score:", scorename)
    
    titl = titl + " | " + paths.site
    
    if not lowupbounds is None:
        lowupbounds[0,:]=scores-lowupbounds[0,:]
        lowupbounds[1,:]=lowupbounds[1,:]-scores
        
    sns.set()

    fig = plt.figure()
    plt.title(titl)
    plt.bar(
        algos, scores, yerr=lowupbounds, color=colors,
    )
    if storeImages:
        fileName = "_".join([scorename, paths.site])
        plt.savefig(os.path.join(figureDir, fileName + fmtImages))
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    
    return fig


def plot_cv_indices(cv, X, y, group=None, lw=10):
    """Create a sample plot for indices of a cross-validation object.
    
    
    Parameters
    ----------
    cv : `sklearn.model_selection` object with `split` method
        Cross-validation splitter
    
    X : array-like
        Design matrix, input data
    
    y : array-like
        Response vector, output data
    
    group : array-like of int
        Group labels for the cross-validation splits
    
    
    Returns
    -------
    `matplotlib.pyplot Axes`
        Line plots representing the indices of each cross-validation split
    
    
    Notes
    -----
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
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
            alpha=0.1,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=plt.cm.Paired
    )

    ax.scatter(
        range(len(X)),
        [ii + 2.5] * len(X),
        c=group,
        marker="_",
        lw=lw,
        cmap=plt.cm.Paired,
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, 100],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)

    plt.show(block=False)

    return ax
