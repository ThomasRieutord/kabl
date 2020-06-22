#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE OF SECONDARY FUNCTIONS FOR THE KABL PROGRAM.

Features:
    - get_default_params
    - where_and_when
    - create_file_from_source
    - extract_data
    - extract_testprofile
    - add_blh_to_netcdf

Test of the functions: `python utils.py`
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

import netCDF4 as nc
import datetime as dt
import numpy as np


def get_default_params():
    """Returns a dict with the default settings.
    
    Here is a description of all the settings you can change :
    
        * 'algo': set the machine learning algorithm that will be applied.
            If it is 'gmm', the EM algorithm (Gaussian mixture) is used.
            If it is 'kmeans', the K-means algorithm is used.
            
        * 'n_clusters': set the number of clusters to be formed.
            If it is an integer, such integer is the number of clusters.
            If not (as when it is 'auto'), the number of clusters is
            automatically estimated with the score in params['classif_score'].
        
        * 'classif_score': set the classification score that will be
            used to automatically estimate the number of clusters.
            Three scores are available:
                'silh' or 'silhouette' for silhouette coefficient
                'db' or 'davies_bouldin' for Davies-Bouldin score
                All else use Calinski-Harabasz score (the fastest)
        
        * 'n_inits': set how many times the algorithm is repeated with a
            different initialisation.
        
        * 'n_profiles': set how many profiles are concatenated before the
            application of the algorithm.
            If n_profiles=1, only the current profile is used.
            If n_profiles=3, the current profile and the two previous are
            concatenated and put in input of the algorithm.
            The higher n_profiles, the smoother is the time evolution of
            BLH estimation.
        
        * 'max_k': highest number of clusters tested when automatically
            chosen. The higher the longer is the code.
        
        * 'init': initialisation strategy for both algorithms. Three are
            available:
                'random': pick randomly an individual as starting  point (both Kmeans and GMM)
                'advanced': more sophisticated way to initialize
                'given': start at explicitly passed point coordinates.
        
        * 'cov_type': ONLY FOR GMM. Set the assumption made on covariance
            matrices.
                'full' each component has its own general covariance matrix
                'tied' all components' covariance matrices are proportional
                'diag' each component has its own diagonal covariance matrix
                'spherical' each component has its own single variance
        
        * 'predictors': list of variables used in the classification.
            They can be different at night and at day. For both, it can
            be chosen among:
                'rcs_1': parallel polarised range-corrected backscatter signal
                'rcs_2': orthogonal polarised range-corrected backscatter signal
            The change of predictors between day and night is made at the
            hours of sunrise and sunset as given by the epheremid module
            plus a shift set later in theses parameters.
        
        * 'max_height': height where the measurements are cut (m agl)
        
        * 'sunrise_shift': the limit from night to day is shifted from 
            sunrise to have a more realistic transition of predictors.
            It is given in algebraic hour after sunrise.
        
        * 'sunset_shift': the limit from day to night is shifted from 
            sunset to have a more realistic transition of predictors.
            It is given in algebraic hour after sunset.

    +----------------+---------+---------------------------------------+
    |    Key         |  Type   |        Possible values                |
    +----------------+---------+---------------------------------------+
    | 'algo'         | str     | 'gmm','kmeans'                        |
    | 'n_clusters'   | int/str | 2 to 10 / 'auto'                      |
    | 'classif_score'| str     | 'silh','db','ch'                      |
    | 'n_inits'      | int     | 1 to 1000                             |
    | 'max_k'        | int     | 3 to 15                               |
    | 'init'         | str     | 'random','advanced','given'           |
    | 'cov_type'     | str     | 'full','diag','tied','spherical'      |
    | 'predictors'   | dict    | 'day':['rcs_1','rcs_2'],'night':['rcs_1','rcs_2']
    | 'max_height'   | int     | 1500 to 10000                         |
    | 'sunrise_shift'| float   | -3 to 3                               |
    | 'sunset_shift' | float   | -3 to 3                               |
    +----------------+---------+---------------------------------------+
    """

    params = dict()
    params["algo"] = "kmeans"
    params["n_clusters"] = 3
    params["classif_score"] = "ch"
    params["n_inits"] = 1
    params["n_profiles"] = 1
    params["max_k"] = 6
    params["init"] = "given"
    params["cov_type"] = "full"
    params["predictors"] = {"day": ["rcs_1"], "night": ["rcs_1", "rcs_2"]}
    params["max_height"] = 4500
    params["sunrise_shift"] = 1
    params["sunset_shift"] = -1
    return params


def where_and_when(datafile):
    """Returns the location of the lidar and the day when the measures
    have been done from the name of the file. Files must be named using
    the convention:
            DAILY_MPL_SITEID_YYYYMMDD.nc
    
    [IN]
        - datafile (str): path to file
        
    [OUT]
        - location (str): name of the place where the lidar was
        - day (datetime): day of the measurements
        - lat (float): latitude of the place where the lidar was
        - lon (float): longitude of the place where the lidar was
    """

    # Dictionnary to link site IDs to real location
    sites_id2loc = {
        "5025": "Trappes",
        "5031": "Toulouse",
        "5029": "Brest",
        "5030": "Lille",
        "Trappes": "Trappes",  # For the test file only
    }

    sites_id2lat = {
        "5025": 48.7743,
        "5031": 43.621,
        "5029": 48.45,
        "5030": 50.57,
        "Trappes": 48.7743,  # For the test file only
    }

    sites_id2lon = {
        "5025": 2.0098,
        "5031": 1.3788,
        "5029": -4.3833,
        "5030": 3.0975,
        "Trappes": 2.0098,  # For the test file only
    }

    Filename = datafile.split("/")[-1]
    daily, mpl, siteID, yyyymmdd = Filename.split("_")

    location = sites_id2loc[siteID]
    lat = sites_id2lat[siteID]
    lon = sites_id2lon[siteID]
    day = dt.datetime(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return location, day, lat, lon


def get_lat_lon(location):
    """Return latitude and longitude of the given location"""

    sites_id2lat = {
        "5025": 48.7743,
        "5031": 43.621,
        "5029": 48.45,
        "5030": 50.57,
        "Trappes": 48.7743,
        "Brest": 48.45,
    }

    sites_id2lon = {
        "5025": 2.0098,
        "5031": 1.3788,
        "5029": -4.3833,
        "5030": 3.0975,
        "Trappes": 2.0098,
        "Brest": -4.3833,
    }

    return sites_id2lat[location], sites_id2lon[location]



def create_file_from_source(src_file, trg_file):
    """Copy a netCDF file into another, using only netCDF4 package.
    It is used to create an output file with the same information as the
    input file but without spoiling it.
    
    From Stack Overflow (2019): https://stackoverflow.com/questions/13936563/copy-netcdf-file-using-python
    
    [IN]
        - src_file (str): path to the input file
        - trg_file (str): path to the output file
    
    [OUT] None"""

    src = nc.Dataset(src_file)
    trg = nc.Dataset(trg_file, mode="w")

    # Create the dimensions of the file
    for name, dim in src.dimensions.items():
        trg.createDimension(name, len(dim) if not dim.isunlimited() else None)

    # Copy the global attributes
    trg.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

    # Create the variables in the file
    for name, var in src.variables.items():
        trg.createVariable(name, var.dtype, var.dimensions)

        # Copy the variable attributes
        trg.variables[name].setncatts({a: var.getncattr(a) for a in var.ncattrs()})

        # Copy the variables values (as 'f4' eventually)
        trg.variables[name][:] = src.variables[name][:]

    # Save the file
    trg.close()


def extract_rs(rs_file, t_start, t_end, method="BEST_BLH"):
    """Extract estimation from the radiosounding within a given time period.
    
    [IN]
        - rs_file (str): path to the radiosounding data (netCDF)
        - t_start (timestamp): beginning of the time period
        - t_end (timestamp): ending of the time period
        - method (str): code for the BLH calculation method (see inside rs_file)
    
    [OUT]
        - t_rs (list of timestamp): vector of timestamps
        - blh_rs (list of float): vector of BLH estimation from RS
    """

    rsdata = nc.Dataset(rs_file)
    t_rs = rsdata.variables["time"]
    blh_rs = rsdata.variables[method]
    rsdex = np.where(np.logical_and(t_rs > t_start, t_rs < t_end))[0]

    return t_rs[rsdex], blh_rs[rsdex]


def extract_data(nc_file, to_extract=["rcs_1", "rcs_2"], max_height=4500, params=None):
    """Extract useful variables from the netcdf file into Numpy arrays
    with one line per time and one column per height.
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
        - to_extract (list of str): list of variables to extract. They must be the same as in the netcdf variables.
        - max_height (int): maximum altitude of measurement extracted.
        - params (dict): dict with all settings. If provided, the value params['max_height'] overrides the value given in 'max_height'.
    
    [OUT]
        - t (np.array[Nt]): vector of timestamps
        - z (np.array[Nz]): vector of altitudes
        - result (dict of np.array[Nt,Nz]): measured atmospheric variable at all times and altitude
    """

    if params is not None:
        max_height = params["max_height"]
        required_var = []
        for pr in params["predictors"].values():
            required_var.extend(pr)
    else:
        required_var = []

    if not all([rv in to_extract for rv in required_var]):
        raise ValueError(
            "Some variables required for computation are NOT in the list of extraction."
            +"\nto_extract="+str(to_extract)
            +"\nrequired_var="+str(required_var)
        )

    ncf = nc.Dataset(nc_file)
    z = np.array(ncf.variables["range"])
    zmax = np.where(z <= max_height)[0][-1]

    t = np.ma.array(ncf.variables["time"][:])
    t = t * 24 * 3600
    if t.count() == t.size:
        mask = np.full(t.shape, True)
    else:
        mask = ~t.mask
    
    result ={}
    for var in to_extract:
        val = ncf.variables[var]
        if val.ndim == 2:
            result[var] = np.array(val[mask, 0:zmax])
        if val.ndim == 1:
            result[var] = np.array(val[mask])
    
    return t[mask], z[0:zmax], result


def extract_testprofile(
    nc_file,
    to_extract=["rcs_1", "rcs_2"],
    max_height=4500,
    profile_id=2,
    return_coords=False,
):
    """Extract preselected profiles of atmospheric variables from the
    netcdf file into Numpy arrays.
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
        - to_extract (list of str): list of variables to extract. They must be the same as in the netcdf variables.
        - max_height (int): maximum altitude of measurement extracted.
        - profile_id (int): identification of the test profile to be extracted:
                0=profile of 0:17 (t=3): nocturnal boundary layer with aerosol plume aloft
                1=profile of 9:22 (t=112): morning transition
                2=profile of 16:32 (t=198): well-mixed boundary layer
                3=profile of 21:17 (t=255): evening transition
        - return_coords (bool): if True, the dict 'coords' is returned as last element of the tuple.
                
    [OUT]
        - z (np.array[Nz]): vector of altitudes
        - *args (np.array[Nz]): measured atmospheric variable at all altitudes
        - coords (dict): coordinate of the profile (time, latitude, longitude). ONLY RETURNED WHEN return_coords=True.
    """

    ncf = nc.Dataset(nc_file)
    z = np.array(ncf.variables["range"])
    zmax = np.where(z <= max_height)[0][-1]

    t_index = [3, 112, 198, 255]
    if profile_id in [0, 1, 2, 3]:
        t = t_index[profile_id]
    else:
        raise ValueError("Unknown profile ID. Must be among [0,1,2,3]")
    result = [z[0:zmax]]
    for var in to_extract:
        val = ncf.variables[var]
        if val.ndim == 2:
            result.append(np.array(val[t, 0:zmax]))
        if val.ndim == 1:
            result.append(np.array(val[t]))

    if return_coords:
        loc, dateofday, lat, lon = where_and_when(nc_file)
        hourofday = dt.datetime.utcfromtimestamp(ncf.variables["time"][t] * 24 * 3600)
        coords = {
            "time": dt.datetime(
                dateofday.year,
                dateofday.month,
                dateofday.day,
                hourofday.hour,
                hourofday.minute,
                hourofday.second,
            ),
            "lat": lat,
            "lon": lon,
        }
        result.append(coords)

    return result


def blh_from_labels(labels, z_values):
    """Derive the boundary layer height from clusters labels.
    Boundary layer is by definition the cluster in contact with the ground.
    Its limit is set where the this cluster ends for the first time.
    
    [IN]
        - labels (np.array[N]): vector of cluster number attribution
        - z_values (np.array[N]): vector of altitude
    
    [OUT]
        - blh (float): height of first cluster transition
    """

    if labels is None or len(np.unique(labels)) == 1:
        # Case where no proper BLH can be found.
        blh = np.nan
    else:
        # Order labels and altitude by altitude
        ordered_by_z = np.argsort(z_values, kind="stable")
        z_values = z_values[ordered_by_z]
        labels = labels[ordered_by_z]

        # Find the first change in labels
        dif = np.diff(labels)
        ind = np.where(np.abs(dif) >= 0.9)[0]
        blh = (z_values[ind[0]] + z_values[ind[0] + 1]) / 2

    return blh

def add_blh_to_netcdf(inputFile, outputFile, blh, origin="kabl", quiet=False):
    """Add the BLH estimated with KABL into a copy of the original netcdf file.
    
    [IN]
        - src_file (str): path to the input file
        - trg_file (str): path to the output file
        - blh (np.array[Nt]): time series of BLH as estimated by the KABL algorithm.
        - quiet (bool): if True, all prints are disabled.
    
    [OUT] None"""

    create_file_from_source(inputFile, outputFile)
    ncf = nc.Dataset(outputFile, "r+")
    BLH_NEW = ncf.createVariable("blh_" + origin.lower(), np.float32, ("time",))
    BLH_NEW[:] = blh[:]
    BLH_NEW.units = "BLH from " + origin.upper() + ". Meters above ground level (m agl)"
    ncf.close()
    if not quiet:
        print(
            "New variable 'blh_"
            + origin.lower()
            + "' has been added in the netcdf file ",
            outputFile,
        )


def save_qualitymetrics(
    dropfilename,
    t_values,
    blhs,
    blhs_names,
    scores,
    scores_names,
    masks,
    masks_names,
    n_clusters,
    chrono,
    params,
):
    """Save results of KABL quality metrics on one day into a netcdf file.
    
    This netcdf file contains the time evolution of BLHs (from KABL and 
    other sources to be compared with), of classification scores and of
    the number of clusters. Other useful information, such as the compu-
    ting time and the full dict of KABL's parameters, are also stored.
    
    [IN]
        - dropfilename (str): path to the netcdf file in which results will be stored
        - t_values (np.array([Nt])): array of time values
        - blh_ref (np.array([Nt])): array of BLH estimated by constructor
        - blh_new (np.array([Nt])): array of BLH estimation by Kmeans algorithm
        - score (np.array([Nt])): array of score assessing the classification
        - n_clusters (np.array([Nt])): array of number of clusters choosen by silhouette method
        - mask_rain (np.array([Nt])): array of booleans. True if there is some rain.
        - mask_cloud (np.array([Nt])): array of booleans. True if there is a cloud under 3000 m (agl).
        - chrono (float): computing time for the full day (seconds)
        - params (dict): dict with all KABL' settings.
    
    [OUT]
        - (str): message saying everything is OK
        + netcdf file created at the given location
    """
    import time

    resultnc = nc.Dataset(dropfilename, "w")

    # General information
    resultnc.description = "Boundary layer heights calculated by different methods and quality score for some of them."
    resultnc.source = "Meteo-France CNRM/GMEI/LISA + DSO/DOA/IED"
    resultnc.history = "Created " + time.ctime(time.time())
    resultnc.contactperson = "Thomas Rieutord (thomas.rieutord@meteo.fr) ; Sylvain Aubert (sylvain.aubert@meteo.fr)"
    resultnc.settings = str(params)
    resultnc.computation_time = chrono

    # Coordinate declaration
    resultnc.createDimension("time", len(t_values))

    # Fill in time vector
    Time = resultnc.createVariable("time", np.float64, ("time",))
    Time[:] = t_values
    Time.units = "Seconds since 1970/01/01 00:00 UTC (s)"

    # Fill in BLHs vectors
    for ib in range(len(blhs)):
        BLH = resultnc.createVariable(blhs_names[ib], np.float64, ("time",))
        BLH[:] = blhs[ib][:]
        BLH.units = "Meters above ground level (m agl)"

    # Fill in scores vectors
    for ik in range(len(scores)):
        SCORE = resultnc.createVariable(scores_names[ik], np.float64, ("time",))
        SCORE[:] = scores[ik][:]
        SCORE.units = scores_names[ik] + " (unitless)"

    # Fill in masks vectors
    for ik in range(len(masks)):
        MASK = resultnc.createVariable(masks_names[ik], "i1", ("time",))
        MASK[:] = masks[ik][:]
        MASK.units = masks_names[ik] + " (boolean)"

    # Fill in number of clusters vectors
    K = resultnc.createVariable("N_CLUSTERS", np.int, ("time",))
    n_clusters = np.array(n_clusters)
    n_clusters[np.isnan(n_clusters)] = 0
    K[:] = n_clusters[:]

    # Closing the netcdf file
    resultnc.close()

    return "Results successfully written in the file " + dropfilename


def grid_to_scatter(x,y,z=None):
    '''Convert grid point data into scattered points data.
    
    Grid point data : (x,y,z) with z[i,j]=f(x[i],y[j])
    Scatter point data : (X,Y) with Y[i]=f(X[i,0],X[i,1])
    
    Inverse function -> scatter_to_grid
    
    [IN]
        - x (np.array[nx]): coordinates on X-axis
        - y (np.array[ny]): coordinates on Y-axis
        - z (np.array[nx,ny]): data for each point (x,y)
    
    [OUT]
        - X (np.array[nx*ny,2]): coordinate matrix
        - Y (np.array[nx*ny]): data vector'''
    nx=np.size(x)
    ny=np.size(y)
    
    X=np.full((nx*ny,2),np.nan)
    X[:,0]=np.repeat(x,ny)
    X[:,1]=np.tile(y,nx)
    if np.isnan(X).any():
        print(" WARNING: ",np.isnan(X).sum()," NaN in X (grid_to_scatter)")
    
    if z is None:
        result = X
    else:
        if np.size(z)!=nx*ny:
            raise Exception("Problem with inputs dimensions (grid_to_scatter)")
        Y=z.ravel()
        result = X,Y
    
    return result

def scatter_to_grid(X,Y=None):
    '''Convert scattered points data into grid point data.
    
    Grid point data : (x,y,z) with z[i,j]=f(x[i],y[j])
    Scatter point data : (X,Y) with Y[i]=f(X[i,0],X[i,1])
    
    Inverse function -> grid_to_scatter
    
    [IN]
        - X (np.array[nx*ny,2]): coordinate matrix
        - Y (np.array[nx*ny]): data vector
    
    [OUT]
        - x (np.array[nx]): coordinates on X-axis
        - y (np.array[ny]): coordinates on Y-axis
        - z (np.array[nx,ny]): data for each point (x,y)
    '''
    
    N,d=np.shape(X)
    
    if d!=2:
        raise ValueError("More than 2 columns. Not ready so far (scatter_to_grid)")
    
    if np.sum(np.diff(X[:,0])==0)>np.sum(np.diff(X[:,1])==0):
        xcoord=0
    else:
        xcoord=1
    
    ny = (np.diff(X[:,xcoord])==0).tolist().index(False)+1
    
    if np.mod(N/ny,1)!=0:
        raise ValueError("Number of points doesn't match with dimensions")
    
    nx = int(N/ny)
    
    x = X[0:N:ny,xcoord]
    y = X[0:ny,1-xcoord]
    if Y is None:
        result = x,y
    else:
        if np.size(Y)!=N:
            raise Exception("Inconsistent inputs dimensions (scatter_to_grid)")
        z = np.reshape(Y,(nx,ny))
        result = x,y,z
    
    return result


def colocate_instruments(
    time_lidar, time_rs, values_lidar, values_rs, tol=0, verbose=False
):
    """Return the values that are colocated according to a given tolerance
    
    [IN]
        - time_lidar (np.array[Nl]): vector of timestamps (seconds since 1970-01-01 00:00 UTC) for lidar values
        - time_rs (np.array[Nr]): vector of timestamps (seconds since 1970-01-01 00:00 UTC) for RS values (Nr << Nl)
        - values_lidar (list of np.array[Nl]): list of variables to be colocated with RS
        - values_rs (np.array[Nr]): vector of values measured by RS to be colocated with lidar
        - tol (float): width of the interval (seconds) to consider values to be colocated
        - verbose (bool): if False, kills the prints
        
    [OUT]
        - 
    """

    time_coloc = []
    values_lidar_coloc = [[] for k in range(len(values_lidar))]
    values_rs_coloc = []

    for it in range(len(time_rs)):
        t = time_rs[it]
        ind_t_coloc = np.where(
            np.logical_and(
                time_lidar >= t,
                time_lidar <= t + tol
            )
        )[0]
        
        if verbose:
            print(
                len(ind_t_coloc),
                "shots colocated with RS at ",
                dt.datetime.utcfromtimestamp(t),
            )
        if len(ind_t_coloc)>0 and not np.isnan(values_rs[it]):
            
            time_coloc.append(time_lidar[ind_t_coloc[0]])
            
            if verbose:
                print(
                    "First one at ",
                    dt.datetime.utcfromtimestamp(time_coloc[-1]),
                    "RS at ",
                    dt.datetime.utcfromtimestamp(t),
                )

            values_rs_coloc.append(values_rs[it])
            
            for k in range(len(values_lidar)):
                if values_lidar[k].dtype == bool:
                    mean10_lidar = any(values_lidar[k][ind_t_coloc])
                else:
                    mean10_lidar = np.nanmean(values_lidar[k][ind_t_coloc])
                values_lidar_coloc[k].append(mean10_lidar)

    values_lidar_coloc = [np.array(var) for var in values_lidar_coloc]

    return np.array(time_coloc), np.array(values_rs_coloc), values_lidar_coloc
