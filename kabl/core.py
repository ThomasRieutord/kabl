#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE OF CORE FUNCTIONS FOR THE KABL PROGRAM.

 +-----------------------------------------+
 |  Date of creation: 6 Aug. 2019          |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  DSO/DOA/IED and CNRM/GMEI/LISA         |
 +-----------------------------------------+
"""

# Usual Python packages
import numpy as np
import datetime as dt
import sys
import os.path
import time

# Machine learning packages
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabaz_score,
    davies_bouldin_score,
)

# Local packages
from kabl.ephemerid import Sun
from kabl import paths
from kabl import utils


def prepare_data(coords, z_values, rcss, params=None):
    """Put the data in form to fulfil algorithm requirements.
    
    Five operations are carried out in this function:
      0. Check and reshape inputs
      1. Distinguish night and day for predictors
      2. Concatenate the profiles
      3. Take the logarithm of range-corrected signal
      4. Apply also a standard normalisation (remove mean and divide by
    standard deviation).
    
    
    Parameters
    ----------
    coords : dict
        Time and space coordinate. The dict must have 3 keys:
        'time' (datetime): time of the profile
        'lat' (float): latitude of the measurement site
        'lon' (float): longitude of the measurement site
    
    z_values : array-like of shape (nZ,)
        Vector of altitude values
    
    rcss : dict
        Input data, in the form of a dict of named matrices.
        Example: rcss={"rcs_0":rcs_0, "rcs_1":rcs_1} where rcs_0 and
        rcs_1 are ndarray of shape (nT,nZ)
    
    params : dict
        Dict with all settings. This function depends on 'n_profiles',
        'predictors', 'sunrise_shift', 'sunset_shift'.
      
    Returns
    -------
    X : ndarray of shape (N,p)
        Design matrix to put in input of the algorithm.
        Each line is an observation, each column is a predictor.
    
    Z : ndarray of shape (N,)
        Vector of altitudes for each observation.
    """

    if params is None:
        params = utils.get_default_params()

    # 0. Check and reshape inputs
    # ---------------------------
    needed_data = np.unique(np.concatenate(list(params["predictors"].values())))

    if set(rcss.keys()) != set(needed_data):
        raise Exception("Wrong input data provided.")

    if "rcs_0" in needed_data:
        rcs_0 = rcss["rcs_0"]
        try:
            Nt, Nz = rcs_0.shape
        except ValueError:
            Nz = rcs_0.size
            Nt = 1
    if "rcs_1" in needed_data:
        rcs_1 = rcss["rcs_1"]
        try:
            Nt, Nz = rcs_1.shape
        except ValueError:
            Nz = rcs_1.size
            Nt = 1
    if "rcs_2" in needed_data:
        rcs_2 = rcss["rcs_2"]
        try:
            Nt, Nz = rcs_2.shape
        except ValueError:
            Nz = rcs_2.size
            Nt = 1

    # 1. Distinguish night and day for predictors
    # --------------------------------------------
    t = coords["time"]
    timeofday = t.strftime("%H:%M")
    dateofday = t.strftime("%Y%m%d")

    s = Sun(lat=coords["lat"], long=coords["lon"])
    sunrise = s.sunrise(t)
    sunset = s.sunset(t)

    sunrise = dt.datetime(
        t.year, t.month, t.day, sunrise.hour, sunrise.minute, sunrise.second
    ) + dt.timedelta(hours=params["sunrise_shift"])
    sunset = dt.datetime(
        t.year, t.month, t.day, sunset.hour, sunset.minute, sunset.second
    ) + dt.timedelta(hours=params["sunset_shift"])

    if t >= sunrise and t <= sunset:
        nightorday = "day"
    else:
        nightorday = "night"

    predictors = params["predictors"][nightorday]

    # 2. Concatenate the profiles
    # ----------------------------
    if Nt > 1:
        Z = np.tile(z_values, Nt)
    else:
        Z = z_values

    X = []
    if "rcs_0" in predictors:
        if rcs_0 is None:
            raise ValueError("Missing argument rcs_0 in kabl.core.prepare_data")
        X.append(rcs_0.ravel())
    if "rcs_1" in predictors:
        if rcs_1 is None:
            raise ValueError("Missing argument rcs_1 in kabl.core.prepare_data")
        X.append(rcs_1.ravel())
    if "rcs_2" in predictors:
        if rcs_2 is None:
            raise ValueError("Missing argument rcs_2 in kabl.core.prepare_data")
        X.append(rcs_2.ravel())

    # 3. Take the logarithm of range-corrected signal
    # ------------------------------------------------
    X = np.array(X).T
    X[X <= 0] = 1e-5
    X = np.log10(X)
    if X.ndim == 1:
        X.reshape(-1, 1)

    # 4. Normalisation: remove mean and divide by standard deviation
    # ---------------------------------------------------------------
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    return X, Z


def apply_algo_k_auto(X, init_codification=None, quiet=True, params=None):
    """Apply the machine learning algorithm for various number of
    clusters and choose the best according the specified score.
    
    Parameters
    ----------
    X : ndarray of shape (N,p)
        Design matrix to put in input of the algorithm. Each line is an
        observation, each column is a predictor.
    
    init_codification : dict, default=None
        Link initialisation strategy with actual algorithm inputs. 
        See kabl.core.apply_algo
    
    quiet : bool, default=True
        If True, cut down all prints
        
    params : dict, default=None
        Dict with all settings. This function depends on 'max_k', 'n_clusters'
    
    
    Returns
    -------
    labels : ndarray of shape (N,)
        Vector of cluster number attribution
        BEWARE: the cluster identification number are random. Only
        borders matter.
    
    n_clusters_opt : int
        Optimal number of clusters to be found in the data
    
    classif_scores : float
        Value of classification score (chosen in
        params['n_clusters']) for the returned classification.
    """

    if params is None:
        params = utils.get_default_params()

    # 1. Apply algorithm and compute scores for several number of clusters
    # --------------------------------------------------------------------
    all_labels = []
    classif_scores = []
    for n_clusters in range(2, params["max_k"]):

        labels = apply_algo(
            X, n_clusters, init_codification=init_codification, params=params
        )
        all_labels.append(labels)

        if params["classif_score"] in ["silhouette", "silh"]:
            classif_scores.append(silhouette_score(X, labels))
        elif params["classif_score"] in ["davies_bouldin", "db"]:
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # to avoid itempestive warning ("RuntimeWarning: divide by zero encountered in true_divide...")
                classif_scores.append(davies_bouldin_score(X, labels))
        else:  # Default because fastest
            classif_scores.append(calinski_harabaz_score(X, labels))

    # 2. Choose the best number of clusters
    # -------------------------------------
    if params["classif_score"] in ["silhouette", "silh"]:
        k_best = np.argmax(classif_scores)
        if classif_scores[k_best] < 0.5:
            if not quiet:
                print(
                    "Bad classification according to silhouette score (",
                    classif_scores[k_best],
                    "). BLH is thus NaN",
                )
            k_best = None
    elif params["classif_score"] in ["davies_bouldin", "db"]:
        k_best = np.argmin(classif_scores)
        if classif_scores[k_best] > 0.36:
            if not quiet:
                print(
                    "Bad classification according to Davies-Bouldin score (",
                    classif_scores[k_best],
                    "). BLH is thus NaN",
                )
            k_best = None
    else:
        k_best = np.argmax(classif_scores)
        if classif_scores[k_best] < 200:
            if not quiet:
                print(
                    "Bad classification according to Calinski-Harabasz score (",
                    classif_scores[k_best],
                    "). BLH is thus NaN",
                )
            k_best = None

    # 3. Return the results
    # ---------------------
    if k_best is not None:
        result = all_labels[k_best], k_best + 2, classif_scores[k_best]
    else:
        result = None, None, None

    return result


def apply_algo(X, n_clusters, init_codification=None, params=None):
    """Apply the machine learning algorithm on the prepared data.
    
    Parameters
    ----------
    X : ndarray of shape (N,p)
        Design matrix to put in input of the algorithm. Each line is an
        observation, each column is a predictor.
    
    n_clusters : int
        Number of clusters to be found in the data
    
    init_codification : dict, default=None
        Link initialisation strategy with actual algorithm inputs.
        Keys are the three strategy are available:
            'random': pick randomly an individual as starting  point (both Kmeans and GMM)
            'advanced': more sophisticated way to initialize
            'given': start at explicitly passed point coordinates.
            + special key 'token', where are given the explicit point coordinates to use when the strategy is 'given'
        Values are dictionnaries with, as key, the algorithm name and, as value, the corresponding input in Scikit-learn.
        For 'token', the value is a list of np.arrays (explicit point coordinates)
    
    params : dict, default=None
        Dict with all settings. This function depends  on 'algo',
        'n_inits', 'init', 'cov_type'
        
    Returns
    -------
    labels : ndarray of shape (N,)
        Vector of cluster number attribution
        BEWARE: the cluster identification number are random.
        Only borders matter.
    """

    if params is None:
        params = utils.get_default_params()

    if init_codification is None:
        init_codification = {
            "random": {"kmeans": "random", "gmm": "random"},
            "advanced": {"kmeans": "k-means++", "gmm": "kmeans"},
            "given": {  # When initialization is 'given', the values are given in the 'token' field
                "kmeans": "token",
                "gmm": "kmeans",
            },
            "token": [  # trick to specify centroids in one object
                np.array([-2.7, -0.7]),  # 2 clusters
                np.array([-2.7, -0.7, 1]),  # 3 clusters
                np.array([-3.9, -2.7, -0.7, 1]),  # 4 clusters
                np.array([-3.9, -2.7, -1.9, -0.7, 1]),  # 5 clusters
                np.array([-3.9, -2.7, -1.9, -0.7, 0, 1]),  # 6 clusters
            ],
        }
    initialization = init_codification[params["init"]][params["algo"]]

    # When initialization is 'given', the values are given in the 'token' field
    # The values are accessed afterward to keep the dict init_codification not too hard to read...
    if initialization == "token":
        # Given values are repeated in all predictors
        n_predictors = X.shape[1]
        initialization = np.repeat(
            init_codification["token"][n_clusters - 2], n_predictors
        ).reshape((n_clusters, n_predictors))

    if params["algo"] == "kmeans":
        kmeans = KMeans(
            n_clusters=n_clusters, n_init=params["n_inits"], init=initialization
        )
        kmeans.fit(X)
        labels = kmeans.predict(X)
    elif params["algo"] == "gmm":
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=params["cov_type"],
            n_init=params["n_inits"],
            init_params=initialization,
        )
        gmm.fit(X)
        labels = gmm.predict(X)

    return labels


def blh_estimation(inputFile, outputFile=None, storeInNetcdf=True, params=None):
    """Perform BLH estimation on all profiles of the day and write it into
    a copy of the netcdf file.
    
    
    Parameters
    ----------
    inputFile : str
        Path to the input file, as generated by raw2l1
    
    outputFile : str, default=None
        Path to the output file. Default adds ".out" before ".nc"
    
    storeInNetcdf : bool, default=True
        If True, the field 'blh_kabl', containg BLH estimation, is
        stored in the outputFile
    
    params : dict, default=None
        Dict with all settings. This function depends  on 'n_clusters'
    
    
    Returns
    -------
    blh : ndarray of shape (Nt,)
        Time series of BLH as estimated by the KABL algorithm.
    
    
    Example
    -------
    >>> from kabl import paths
    >>> from kabl import core
    >>> testFile = paths.file_defaultlidardata()
    >>> blh = core.blh_estimation(testFile)
    """

    t0 = time.time()  #::::::::::::::::::::::

    if params is None:
        params = utils.get_default_params()

    # 1. Extract the data
    # ---------------------
    loc, dateofday, lat, lon = utils.where_and_when(inputFile)
    needed_data = np.unique(np.concatenate(list(params["predictors"].values())))
    t_values, z_values, rcss = utils.extract_data(
        inputFile, to_extract=needed_data, params=params
    )

    if "rcs_0" in needed_data:
        rcs_0 = rcss["rcs_0"]
    if "rcs_1" in needed_data:
        rcs_1 = rcss["rcs_1"]
    if "rcs_2" in needed_data:
        rcs_2 = rcss["rcs_2"]

    blh = []

    # setup toolbar
    toolbar_width = int(len(t_values) / 10) + 1
    sys.stdout.write(
        "\nKABL estimation ("
        + loc
        + dateofday.strftime(", %Y/%m/%d")
        + "): [%s]" % ("." * toolbar_width)
    )
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    # Loop on all profile of the day
    for t in range(len(t_values)):
        # toolbar
        if np.mod(t, 10) == 0:
            sys.stdout.write("*")
            sys.stdout.flush()

        # 2. Prepare the data
        # ---------------------
        coords = {
            "time": dt.datetime.utcfromtimestamp(t_values[t]),
            "lat": lat,
            "lon": lon,
        }
        t_back = max(t - params["n_profiles"] + 1, 0)

        rcss = {}
        if "rcs_0" in needed_data:
            rcss["rcs_0"] = rcs_0[t_back : t + 1, :]
        if "rcs_1" in needed_data:
            rcss["rcs_1"] = rcs_1[t_back : t + 1, :]
        if "rcs_2" in needed_data:
            rcss["rcs_2"] = rcs_2[t_back : t + 1, :]

        X, Z = prepare_data(coords, z_values, rcss=rcss, params=params)

        # 3. Apply the machine learning algorithm
        # ---------------------
        if isinstance(params["n_clusters"], int):
            labels = apply_algo(X, params["n_clusters"], params=params)
        else:
            labels, n_clusters, classif_score = apply_algo_k_auto(X, params=params)

        # 4. Derive and store the BLH
        # ---------------------
        blh.append(utils.blh_from_labels(labels, Z))

    if outputFile is None:
        outputFile = paths.file_defaultoutput()

    # end toolbar
    t1 = time.time()  #::::::::::::::::::::::
    chrono = t1 - t0
    sys.stdout.write("] (" + str(np.round(chrono, 4)) + " s)\n")

    # 5. Store the new BLH estimation into a copy of the original netCDF
    # ---------------------
    if storeInNetcdf:
        utils.add_blh_to_netcdf(inputFile, outputFile, blh)

    return np.array(blh)


def apply_algo_k_3scores(X, quiet=True, params=None):
    """Adapation of kabl.core.apply_algo_k_auto in benchmark context.
    
    Parameters
    ----------
    X : ndarray of shape (N,p)
        Design matrix to put in input of the algorithm. Each line is an
        observation, each column is a predictor.
    
    quiet : bool, default=True
        If True, cut down all prints
        
    params : dict, default=None
        Dict with all settings. This function depends on 'max_k', 'n_clusters'
        
    Returns
    -------
    labels : ndarray of shape (N,)
        Vector of cluster number attribution
        BEWARE: the cluster identification number are random. Only
        borders matter.
    
    n_clusters_opt : int
        Optimal number of clusters to be found in the data
    
    classif_scores : float
        Value of classification score (chosen in
        params['n_clusters']) for the returned classification.
    """

    if params is None:
        params = utils.get_default_params()

    # Apply algorithm and compute scores for several number of clusters
    all_labels = []
    s_scores = []
    db_scores = []
    ch_scores = []
    for n_clusters in range(2, params["max_k"] + 1):

        labels = apply_algo(X, n_clusters, params=params)
        all_labels.append(labels)

        if len(np.unique(labels)) > 1:
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # to avoid itempestive warning ("RuntimeWarning: divide by zero encountered in true_divide...")
                db_scores.append(davies_bouldin_score(X, labels))
            s_scores.append(silhouette_score(X, labels))
            ch_scores.append(calinski_harabaz_score(X, labels))
        else:
            db_scores.append(np.nan)
            s_scores.append(np.nan)
            ch_scores.append(np.nan)

    # Choose the best number of clusters
    valid = True
    if params["classif_score"] in ["silhouette", "silh"]:
        k_best = np.nanargmax(s_scores)
        if s_scores[k_best] < 0.6:
            if not quiet:
                print(
                    "Bad classification according to silhouette score (",
                    s_scores[k_best],
                    "). BLH is thus NaN",
                )
            valid = False
    elif params["classif_score"] in ["davies_bouldin", "db"]:
        k_best = np.nanargmin(db_scores)
        if db_scores[k_best] > 0.4:
            if not quiet:
                print(
                    "Bad classification according to Davies-Bouldin score (",
                    db_scores[k_best],
                    "). BLH is thus NaN",
                )
            valid = False
    else:
        k_best = np.nanargmax(ch_scores)
        if ch_scores[k_best] < 200:
            if not quiet:
                print(
                    "Bad classification according to Calinski-Harabasz score (",
                    ch_scores[k_best],
                    "). BLH is thus NaN",
                )
            valid = False

    if all(np.isnan(db_scores)):
        valid = False

    # Return the results
    if valid:
        result = (
            all_labels[k_best],
            k_best + 2,
            s_scores[k_best],
            db_scores[k_best],
            ch_scores[k_best],
        )
    else:
        result = None, np.nan, s_scores[k_best], db_scores[k_best], ch_scores[k_best]

    return result


def kabl_qualitymetrics(
    inputFile,
    outputFile=None,
    reference="None",
    rsFile="None",
    storeResults=True,
    params=None,
):
    """Estimate quality metrics of KABL for one day of measurement.
    
    This function perform the BLH estimation as in
    kabl.core.blh_estimation but its output are the quality metrics, not
    the BLH estimation. As the estimation of quality metrics is greedier
    this function is noticeably longer to execute.
    
    Parameters
    ----------
    inputFile : str
        Path to the input file, as generated by raw2l1
    
    outputFile : str, default=None
        Path to the output file
    
    reference : str, default=None
        Path to handmade BLH estimation, if any, which will serve
        as reference.
    
    rsFile : str
        Path to the radiosounding estimations, if any. Give the
        possibility to store it in the same netcdf
    
    storeResults : bool, default=True
        If True, quality metrics are stored in the `outputFile`
    
    params : dict, default=None
        Dict with all settings. This function depends  on 'n_clusters'
    
    
    Returns
    -------
    errl2_blh : float
        Root mean squared gap between BLH from KABL and the reference
        .. math:: \sqrt{1/N \sum_i^N (Z(i)-Zref(i))^2}
    
    errl1_blh : float
        Mean absolute gap between BLH from KABL and the reference
        .. math:: 1/N \sum_i^N \vert Z(i)-Zref(i) \vert
      
    errl0_blh : float
        Maximum absolute gap between BLH from KABL and the reference
        .. math:: \max_i \vert Z(i)-Zref(i) \vert
    
    ch_score : float
        Average Calinski-Harabasz score (the higher, the better) over
        the full day
        
    db_scores : float
        Average Davies-Bouldin score (the lower, the better) over
        the full day
    
    s_scores : float
        Average silhouette score (the higher, the better) over
        the full day
    
    chrono : float
        Computation time for the full day (seconds)
    
    n_invalid : int
        Number of BLH estimation at NaN or Inf
    """

    t0 = time.time()  #::::::::::::::::::::::

    if params is None:
        params = utils.get_default_params()

    # 1. Extract the data
    # ---------------------
    loc, dateofday, lat, lon = utils.where_and_when(inputFile)
    t_values, z_values, dat = utils.extract_data(
        inputFile, to_extract=["rcs_1", "rcs_2", "pbl", "rr", "vv", "b1"], params=params
    )
    rcs_1 = dat["rcs_1"]
    rcs_2 = dat["rcs_2"]
    blh_mnf = dat["pbl"]
    rr = dat["rr"]
    vv = dat["vv"]
    cbh = dat["b1"]

    blh = []
    K_values = []
    s_scores = []
    db_scores = []
    ch_scores = []

    # setup toolbar
    toolbar_width = int(len(t_values) / 10) + 1
    sys.stdout.write(
        "\nKABL estimation ("
        + loc
        + dateofday.strftime(", %Y/%m/%d")
        + "): [%s]" % ("." * toolbar_width)
    )
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    # Loop on all profile of the day
    for t in range(len(t_values)):
        # toolbar
        if np.mod(t, 10) == 0:
            if any(np.isnan(blh[-11:-1])):
                sys.stdout.write("!")
            else:
                sys.stdout.write("*")
            sys.stdout.flush()

        # 2. Prepare the data
        # ---------------------
        coords = {
            "time": dt.datetime.utcfromtimestamp(t_values[t]),
            "lat": lat,
            "lon": lon,
        }
        t_back = max(t - params["n_profiles"] + 1, 0)
        X, Z = prepare_data(
            coords,
            z_values,
            rcss={"rcs_1": rcs_1[t_back : t + 1, :], "rcs_2": rcs_2[t_back : t + 1, :]},
            params=params,
        )

        # 3. Apply the machine learning algorithm
        # ---------------------

        if isinstance(params["n_clusters"], int):
            n_clusters = params["n_clusters"]
            labels = apply_algo(X, params["n_clusters"], params=params)

            # Compute classification score
            if len(np.unique(labels)) > 1:
                with np.errstate(
                    divide="ignore", invalid="ignore"
                ):  # to avoid itempestive warning ("RuntimeWarning: divide by zero encountered in true_divide...")
                    db_score = davies_bouldin_score(X, labels)
                s_score = silhouette_score(X, labels)
                ch_score = calinski_harabaz_score(X, labels)
            else:
                db_score = np.nan
                s_score = np.nan
                ch_score = np.nan
        else:
            labels, n_clusters, s_score, db_score, ch_score = apply_algo_k_3scores(
                X, params=params
            )

        # 4. Derive and store the BLH
        # ---------------------
        blh.append(utils.blh_from_labels(labels, Z))
        K_values.append(n_clusters)
        s_scores.append(s_score)
        db_scores.append(db_score)
        ch_scores.append(ch_score)

    # end toolbar
    t1 = time.time()  #::::::::::::::::::::::
    chrono = t1 - t0
    sys.stdout.write("] (" + str(np.round(chrono, 4)) + " s)\n")

    if outputFile is None:
        fname = os.path.split(inputFile)[-1]
        outputFile = os.path.join(
            paths.resultrootdir, "DAILY_BENCHMARK_" + fname[10:-3] + ".nc"
        )

    mask_cloud = cbh[:] <= 3000

    if os.path.isfile(reference):
        blh_ref = np.loadtxt(reference)
    else:
        blh_ref = blh_mnf[:, 0]

    if storeResults:
        BLHS = [np.array(blh), np.array(blh_mnf[:, 0])]
        BLH_NAMES = ["BLH_KABL", "BLH_INDUS"]
        if os.path.isfile(reference):
            BLHS.append(blh_ref)
            BLH_NAMES.append("BLH_REF")

        # Cloud base height is added as if it were a BLH though it's not
        BLHS.append(cbh)
        BLH_NAMES.append("CLOUD_BASE_HEIGHT")

        msg = utils.save_qualitymetrics(
            outputFile,
            t_values,
            BLHS,
            BLH_NAMES,
            [s_scores, db_scores, ch_scores],
            ["SILH", "DB", "CH"],
            [rr, vv],
            ["MASK_RAIN", "MASK_FOG"],
            K_values,
            chrono,
            params,
        )

        if os.path.isfile(rsFile):
            blh_rs = utils.extract_rs(rsFile, t_values[0], t_values[-1])
        else:
            blh_rs = None

        print(msg)

    errl2_blh = np.sqrt(np.nanmean((blh - blh_ref) ** 2))
    errl1_blh = np.nanmean(np.abs(blh - blh_ref))
    errl0_blh = np.nanmax(np.abs(blh - blh_ref))
    corr_blh = np.corrcoef(blh, blh_ref)[0, 1]
    n_invalid = np.sum(np.isnan(blh)) + np.sum(np.isinf(blh))

    return (
        errl2_blh,
        errl1_blh,
        errl0_blh,
        corr_blh,
        np.mean(ch_scores),
        np.mean(db_scores),
        np.mean(s_scores),
        chrono,
        n_invalid,
    )


def blh_estimation_returnlabels(
    inputFile, outputFile=None, storeInNetcdf=False, params=None
):
    """Perform BLH estimation on all profiles of the day and return the labels
    of the classification.
    
    
    Parameters
    ----------
    inputFile : str
        Path to the input file, as generated by raw2l1
    
    outputFile : str, default=None
        Path to the output file. Default adds ".out" before ".nc"
    
    storeInNetcdf : bool, default=True
        If True, the field 'blh_kabl', containg BLH estimation, is
        stored in the outputFile
    
    params : dict, default=None
        Dict with all settings. This function depends  on 'n_clusters'
    
    
    Returns
    -------
    blh : ndarray of shape (Nt,)
        Time series of BLH as estimated by the KABL algorithm
    
    zoneID : ndarray of shape (Nt,Nz)
        Cluster labels of every profiles
    """

    t0 = time.time()  #::::::::::::::::::::::

    if params is None:
        params = utils.get_default_params()

    # 1. Extract the data
    # ---------------------
    loc, dateofday, lat, lon = utils.where_and_when(inputFile)
    needed_data = np.unique(np.concatenate(list(params["predictors"].values())))
    t_values, z_values, rcss = utils.extract_data(
        inputFile, to_extract=needed_data, params=params
    )

    if "rcs_0" in needed_data:
        rcs_0 = rcss["rcs_0"]
    if "rcs_1" in needed_data:
        rcs_1 = rcss["rcs_1"]
    if "rcs_2" in needed_data:
        rcs_2 = rcss["rcs_2"]

    blh = []
    zoneID = []

    # setup toolbar
    toolbar_width = int(len(t_values) / 10) + 1
    sys.stdout.write(
        "\nKABL estimation ("
        + loc
        + dateofday.strftime(", %Y/%m/%d")
        + "): [%s]" % ("." * toolbar_width)
    )
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    # Loop on all profile of the day
    for t in range(len(t_values)):
        # toolbar
        if np.mod(t, 10) == 0:
            sys.stdout.write("*")
            sys.stdout.flush()

        # 2. Prepare the data
        # ---------------------
        coords = {
            "time": dt.datetime.utcfromtimestamp(t_values[t]),
            "lat": lat,
            "lon": lon,
        }
        t_back = max(t - params["n_profiles"] + 1, 0)

        rcss = {}
        if "rcs_0" in needed_data:
            rcss["rcs_0"] = rcs_0[t_back : t + 1, :]
        if "rcs_1" in needed_data:
            rcss["rcs_1"] = rcs_1[t_back : t + 1, :]
        if "rcs_2" in needed_data:
            rcss["rcs_2"] = rcs_2[t_back : t + 1, :]

        X, Z = prepare_data(coords, z_values, rcss=rcss, params=params)

        # 3. Apply the machine learning algorithm
        # ---------------------
        if isinstance(params["n_clusters"], int):
            labels = apply_algo(X, params["n_clusters"], params=params)
        else:
            labels, n_clusters, classif_score = apply_algo_k_auto(X, params=params)

        # 4. Derive and store the BLH
        # ---------------------
        blh.append(utils.blh_from_labels(labels, Z))
        zoneID.append(labels)

    if outputFile is None:
        outputFile = paths.file_defaultoutput()

    # end toolbar
    t1 = time.time()  #::::::::::::::::::::::
    chrono = t1 - t0
    sys.stdout.write("] (" + str(np.round(chrono, 4)) + " s)\n")

    # 5. Store the new BLH estimation into a copy of the original netCDF
    # ---------------------
    if storeInNetcdf:
        utils.add_blh_to_netcdf(inputFile, outputFile, blh)

    return np.array(blh), np.array(zoneID)
