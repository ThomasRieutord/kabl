#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
MODULE OF CORE FUNCTIONS FOR THE KABL PROGRAM.

Features:
  - prepare_data
  - apply_algo_k_auto
  - apply_algo
  - blh_from_labels
  - blh_estimation
  - apply_algo_k_3scores
  - kabl_qualitymetrics

Test of the functions: `python core.py`
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
'''

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
from sklearn.metrics import silhouette_score,calinski_harabaz_score,davies_bouldin_score

# Local packages
from kabl.ephemerid import Sun
from kabl import utils
from kabl import graphics


def prepare_data(coords,z_values,rcs_1=None,rcs_2=None,params=None):
    '''Put the data in form to fulfil algorithm requirements.
    
    Four operations are carried out in this function:
     1. Distinguish night and day for predictors
     2. Concatenate the profiles
     3. Take the logarithm of range-corrected signal
     4. Apply also a standard normalisation (remove mean and divide by
    standard deviation).
    
    [IN]
        - coords (dict): dict with time and space coordinate
          'time' (datetime): time of the profile
          'lat' (float): latitude of the measurement site
          'lon' (float): longitude of the measurement site
        - z_values (np.array[nZ]): vector of altitude values
        - rcs_1 (np.array[nT,nZ]): vector of co-polarized backscatter values
        - rcs_2 (np.array[nT,nZ]): vector of cross-polarized backscatter values
        - params (dict): dict with all settings. Depends on 'n_profiles', 'predictors', 'sunrise_shift', 'sunset_shift'.
      
    [OUT]
        - X (np.array[N,p]): design matrix to put in input of the algorithm. Each line is an observation, each column is a predictor.
        - Z (np.array[N]): vector of altitudes for each observation.
    '''
    
    if params is None:
        params=utils.get_default_params()
    
    # 1. Distinguish night and day for predictors
    #--------------------------------------------
    t=coords['time']
    timeofday = t.strftime('%H:%M')
    dateofday = t.strftime('%Y%m%d')
    
    s = Sun(lat=coords['lat'],long=coords['lon'])
    sunrise = s.sunrise(t)
    sunset = s.sunset(t)
    
    sunrise = dt.datetime(t.year,t.month,t.day,sunrise.hour,sunrise.minute,sunrise.second)+ dt.timedelta(hours=params['sunrise_shift'])
    sunset = dt.datetime(t.year,t.month,t.day,sunset.hour,sunset.minute,sunset.second)+ dt.timedelta(hours=params['sunset_shift'])
    
    if t >= sunrise and t <= sunset:
        nightorday='day'
    else:
        nightorday='night'
    
    predictors=params['predictors'][nightorday]
    
    # 2. Concatenate the profiles
    #----------------------------
    try:
        Nt,Nz=rcs_1.shape
        Z=np.tile(z_values,Nt)
    except ValueError:
        Z=z_values
    
    X=[]
    if 'rcs_1' in predictors:
        if rcs_1 is None:
            raise ValueError("Missing argument rcs_1 in kabl.core.prepare_data")
        X.append(rcs_1.ravel())
    if 'rcs_2' in predictors:
        if rcs_2 is None:
            raise ValueError("Missing argument rcs_2 in kabl.core.prepare_data")
        X.append(rcs_2.ravel())
    
    # 3. Take the logarithm of range-corrected signal
    #------------------------------------------------
    X=np.array(X).T
    X[X<=0]=1e-5
    X=np.log10(X)
    
    # 4. Normalisation: remove mean and divide by standard deviation
    #---------------------------------------------------------------
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    
    return X,Z

def apply_algo_k_auto(X,init_codification=None,quiet=True,params=None):
    '''Apply the machine learning algorithm for various number of
    clusters and choose the best according a certain score
    
    [IN]
        - X (np.array[N,p]): design matrix to put in input of the algorithm. Each line is an observation, each column is a predictor.
        - init_codification (dict): dict to link initialisation strategy with actual algorithm inputs. See kabl.core.apply_algo
        - quiet (boolean): if True, cut down all prints
        - params (dict): dict with all settings. Depends on 'max_k', 'n_clusters'
        
    [OUT]
        - labels (np.array[N]): vector of cluster number attribution
            BEWARE: the cluster identification number are random. Only borders matter.
        - n_clusters_opt (int): optimal number of clusters to be found in the data
        - classif_scores (float): value of classification score (chosen in params['n_clusters']) for the returned classification.
    '''
    
    if params is None:
        params=utils.get_default_params()
    
    # Apply algorithm and compute scores for several number of clusters
    all_labels=[]
    classif_scores = []
    for n_clusters in range(2,params['max_k']):
        
        labels = apply_algo(X,n_clusters,init_codification=init_codification,params=params)
        all_labels.append(labels)
        
        if params['classif_score'] in ['silhouette','silh']:
            classif_scores.append(silhouette_score(X,labels))
        elif params['classif_score'] in ['davies_bouldin','db']:
            with np.errstate(divide='ignore',invalid='ignore'):  # to avoid itempestive warning ("RuntimeWarning: divide by zero encountered in true_divide...")
                classif_scores.append(davies_bouldin_score(X,labels))
        else:   # Default because fastest
            classif_scores.append(calinski_harabaz_score(X,labels))
        
    
    # Choose the best number of clusters
    if params['classif_score'] in ['silhouette','silh']:
        k_best=np.argmax(classif_scores)
        if classif_scores[k_best]<0.5:
            if not quiet:
                print("Bad classification according to silhouette score (",classif_scores[k_best],"). BLH is thus NaN")
            k_best=None
    elif params['classif_score'] in ['davies_bouldin','db']:
        k_best=np.argmin(classif_scores)
        if classif_scores[k_best]>0.36:
            if not quiet:
                print("Bad classification according to Davies-Bouldin score (",classif_scores[k_best],"). BLH is thus NaN")
            k_best=None
    else:
        k_best=np.argmax(classif_scores)
        if classif_scores[k_best]<200:
            if not quiet:
                print("Bad classification according to Calinski-Harabasz score (",classif_scores[k_best],"). BLH is thus NaN")
            k_best=None
    
    # Return the results
    if k_best is not None:
        result = all_labels[k_best],k_best+2,classif_scores[k_best]
    else:
        result = None,None,None
    
    return result
    

def apply_algo(X,n_clusters,init_codification=None,params=None):
    '''Apply the machine learning algorithm on the prepared data.
    
    [IN]
        - X (np.array[N,p]): design matrix to put in input of the algorithm. Each line is an observation, each column is a predictor.
        - n_clusters (int): number of clusters to be found in the data
        - init_codification (dict): dict to link initialisation strategy with actual algorithm inputs.
            Keys are the three strategy are available:
                'random': pick randomly an individual as starting  point (both Kmeans and GMM)
                'advanced': more sophisticated way to initialize
                'given': start at explicitly passed point coordinates.
                + special key 'token', where are given the explicit point coordinates to use when the strategy is 'given'
            Values are dictionnaries with, as key, the algorithm name and, as value, the corresponding input in Scikit-learn.
            For 'token', the value is a list of np.arrays (explicit point coordinates)
        - params (dict): dict of parameters. Depends on 'algo', 'n_inits', 'init', 'cov_type'
        
    [OUT]
        - labels (np.array[N]): vector of cluster number attribution
            BEWARE: the cluster identification number are random. Only borders matter.'''
    
    if params is None:
        params=utils.get_default_params()
        
    if init_codification is None:
        init_codification={ 'random':
                                {'kmeans':'random','gmm':'random'},
                            'advanced':
                                {'kmeans':'k-means++','gmm':'kmeans'},
                            'given':# When initialization is 'given', the values are given in the 'token' field
                                {'kmeans':'token','gmm':'kmeans'},
                            'token':    # trick to specify centroids in one object
                                [np.array([-2.7,-0.7]), # 2 clusters
                                    np.array([-2.7,-0.7,1]), # 3 clusters
                                    np.array([-3.9,-2.7,-0.7,1]), # 4 clusters
                                    np.array([-3.9,-2.7,-1.9,-0.7,1]), # 5 clusters
                                    np.array([-3.9,-2.7,-1.9,-0.7,0,1])] # 6 clusters
                            }
    initialization=init_codification[params['init']][params['algo']]
    
    # When initialization is 'given', the values are given in the 'token' field
    # The values are accessed afterward to keep the dict init_codification not too hard to read...
    if initialization=='token':
        # Given values are repeated in all predictors
        n_predictors=X.shape[1]
        initialization=np.repeat(init_codification['token'][n_clusters-2],n_predictors).reshape((n_clusters,n_predictors))
    
    if params['algo']=='kmeans':
        kmeans = KMeans(n_clusters = n_clusters, n_init=params['n_inits'], init=initialization)
        kmeans.fit(X)
        labels = kmeans.predict(X)
    elif params['algo']=='gmm':
        gmm = GaussianMixture(n_components=n_clusters,covariance_type=params['cov_type'],
                n_init=params['n_inits'],init_params=initialization)
        gmm.fit(X)
        labels = gmm.predict(X)
    
    return labels


def blh_from_labels(labels,z_values):
    '''Derive the boundary layer height from clusters labels.
    Boundary layer is by definition the cluster in contact with the ground.
    Its limit is set where the this cluster ends for the first time.
    
    [IN]
        - labels (np.array[N]): vector of cluster number attribution
        - z_values (np.array[N]): vector of altitude
    
    [OUT]
        - blh (float): height of first cluster transition
    '''
    
    if labels is None or len(np.unique(labels))==1:
        # Case where no proper BLH can be found.
        blh=np.nan
    else:
        # Order labels and altitude by altitude
        ordered_by_z=np.argsort(z_values,kind='stable')
        z_values = z_values[ordered_by_z]
        labels= labels[ordered_by_z]
        
        # Find the first change in labels
        dif = np.diff(labels)
        ind = np.where(np.abs(dif) >= 0.9)[0]
        blh = (z_values[ind[0]]+z_values[ind[0]+1])/2
    
    return blh


def blh_estimation(inputFile,outputFile=None,storeInNetcdf=True,params=None):
    '''Perform BLH estimation on all profiles of the day and write it into
    a copy of the netcdf file.
    
    [IN]
      - inputFile (str): path to the input file, as generated by raw2l1
      - outputFile (str): path to the output file. Default adds ".out" before ".nc"
      - storeInNetcdf (bool): if True, the field 'blh_ababl', containg BLH estimation, is stored in the outputFile
      - params (dict): dict of parameters. Depends on 'n_clusters'
    
    [OUT]
      - blh (np.array[Nt]): time series of BLH as estimated by the KABL algorithm.
    '''
    
    t0=time.time()      #::::::::::::::::::::::
    
    if params is None:
        params=utils.get_default_params()
    
    # 1. Extract the data
    #---------------------
    loc,dateofday,lat,lon=utils.where_and_when(inputFile)
    t_values,z_values,rcs_1,rcs_2=utils.extract_data(inputFile,params=params)
    
    blh = []
    
    # setup toolbar
    toolbar_width = int(len(t_values)/10)+1
    sys.stdout.write("KABL estimation ("+loc+dateofday.strftime(', %Y/%m/%d')+"): [%s]" % ("." * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    # Loop on all profile of the day
    for t in range(len(t_values)):
        # toolbar
        if np.mod(t,10)==0:
            sys.stdout.write("*")
            sys.stdout.flush()
        
        # 2. Prepare the data
        #---------------------
        coords={'time':dt.datetime.utcfromtimestamp(t_values[t]),'lat':lat,'lon':lon}
        t_back=max(t-params['n_profiles']+1,0)
        X,Z=prepare_data(coords,z_values,rcs_1[t_back:t+1,:],rcs_2[t_back:t+1,:],params)
        
        # 3. Apply the machine learning algorithm
        #---------------------
        if isinstance(params['n_clusters'],int):
            labels=apply_algo(X,params['n_clusters'],params=params)
            
            # (3.1 OPTIONAL) Compute classification score
            classif_score = silhouette_score(X,labels)
            #ch_score=calinski_harabaz_score(X,labels)
            #db_score=davies_bouldin_score(X,labels)
        else:
            labels,n_clusters,classif_score=apply_algo_k_auto(X,params=params)
        
        # 4. Derive and store the BLH
        #---------------------
        blh.append(blh_from_labels(labels,Z))
    
    if outputFile is None:
        outputFile=inputFile[:-3]+".out.nc"
    
    # end toolbar
    t1=time.time()      #::::::::::::::::::::::
    chrono=t1-t0
    sys.stdout.write("] ("+str(np.round(chrono,4))+" s)\n")
    
    # 5. Store the new BLH estimation into a copy of the original netCDF
    if storeInNetcdf:
        utils.add_blh_to_netcdf(inputFile,outputFile,blh)
    
    return np.array(blh)


def apply_algo_k_3scores(X,params=None,quiet=True):
    '''Adapation of apply_algo_k_auto in benchmark context.
    
    [IN]
        - X (np.array[N,p]): design matrix to put in input of the algorithm. Each line is an observation, each column is a predictor.
        - params (dict): dict with all settings. Depends on 'max_k', 'n_clusters'
        - quiet (bool): if True, all prints are skipped
        
    [OUT]
        - labels (np.array[N]): vector of cluster number attribution
            BEWARE: the cluster identification number are random. Only borders matter.
        - n_clusters_opt (int): optimal number of clusters to be found in the data
        - classif_scores (float): value of classification score (chosen in params['n_clusters']) for the returned classification.
    '''
    
    if params is None:
        params=utils.get_default_params()
    
    # Apply algorithm and compute scores for several number of clusters
    all_labels=[]
    s_scores = []
    db_scores = []
    ch_scores = []
    for n_clusters in range(2,params['max_k']+1):
        
        labels = apply_algo(X,n_clusters,params=params)
        all_labels.append(labels)
        
        if len(np.unique(labels))>1:
            with np.errstate(divide='ignore',invalid='ignore'):  # to avoid itempestive warning ("RuntimeWarning: divide by zero encountered in true_divide...")
                db_scores.append(davies_bouldin_score(X,labels))
            s_scores.append(silhouette_score(X,labels))
            ch_scores.append(calinski_harabaz_score(X,labels))
        else:
            db_scores.append(np.nan)
            s_scores.append(np.nan)
            ch_scores.append(np.nan)
        
    
    # Choose the best number of clusters
    valid=True
    if params['classif_score'] in ['silhouette','silh']:
        k_best=np.nanargmax(s_scores)
        if s_scores[k_best]<0.6:
            if not quiet:
                print("Bad classification according to silhouette score (",s_scores[k_best],"). BLH is thus NaN")
            valid=False
    elif params['classif_score'] in ['davies_bouldin','db']:
        k_best=np.nanargmin(db_scores)
        if db_scores[k_best]>0.4:
            if not quiet:
                print("Bad classification according to Davies-Bouldin score (",db_scores[k_best],"). BLH is thus NaN")
            valid=False
    else:
        k_best=np.nanargmax(ch_scores)
        if ch_scores[k_best]<200:
            if not quiet:
                print("Bad classification according to Calinski-Harabasz score (",ch_scores[k_best],"). BLH is thus NaN")
            valid=False
    
    if all(np.isnan(db_scores)):
        valid=False
    
    # Return the results
    if valid:
        result = all_labels[k_best],k_best+2,s_scores[k_best],db_scores[k_best],ch_scores[k_best]
    else:
        result = None,np.nan,s_scores[k_best],db_scores[k_best],ch_scores[k_best]
    
    return result


def kabl_qualitymetrics(inputFile,outputFile=None,reference='None',rsFile='None',storeResults=True,params=None):
    '''Copy of blh_estimation including calculus and storage of scores
    
    [IN]
      - inputFile (str): path to the input file, as generated by raw2l1
      - outputFile (str): path to the output file. Default adds ".out" before ".nc"
      - reference (str): path to the reference file, if any.
      - rsFile (str): path to the radiosounding estimations, if any (give the possibility to store it in the same netcdf)
      - storeResults (bool): if True, the field 'blh_ababl', containg BLH estimation, is stored in the outputFile
      - params (dict): dict of parameters. Depends on 'n_clusters'
    
    [OUT]
      - errl2_blh (float): root mean squared gap between BLH from KABL and the reference
      - errl1_blh (float): mean absolute gap between BLH from KABL and the reference
      - errl0_blh (float): maximum absolute gap between BLH from KABL and the reference
      - ch_score (float): mean over all day Calinski-Harabasz score (the higher, the better)
      - db_scores (float): mean over all day Davies-Bouldin score (the lower, the better)
      - s_scores (float): mean over all day silhouette score (the higher, the better)
      - chrono (float): computation time for the full day (seconds)
      - n_invalid (int): number of BLH estimation at NaN or Inf
    '''
    
    t0=time.time()      #::::::::::::::::::::::
    
    if params is None:
        params=utils.get_default_params()
    
    
    # 1. Extract the data
    #---------------------
    loc,dateofday,lat,lon=utils.where_and_when(inputFile)
    t_values,z_values,rcs_1,rcs_2,blh_mnf,rr,vv,cbh=utils.extract_data(inputFile,to_extract=['rcs_1','rcs_2','pbl','rr','vv','b1'],params=params)
    
    blh = []
    K_values = []
    s_scores = []
    db_scores = []
    ch_scores = []
    
    # setup toolbar
    toolbar_width = int(len(t_values)/10)+1
    sys.stdout.write("KABL estimation ("+loc+dateofday.strftime(', %Y/%m/%d')+"): [%s]" % ("." * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    # Loop on all profile of the day
    for t in range(len(t_values)):
        # toolbar
        if np.mod(t,10)==0:
            if any(np.isnan(blh[-11:-1])):
                sys.stdout.write("!")
            else:
                sys.stdout.write("*")
            sys.stdout.flush()
        
        # 2. Prepare the data
        #---------------------
        coords={'time':dt.datetime.utcfromtimestamp(t_values[t]),'lat':lat,'lon':lon}
        t_back=max(t-params['n_profiles']+1,0)
        X,Z=prepare_data(coords,z_values,rcs_1[t_back:t+1,:],rcs_2[t_back:t+1,:],params=params)
        
        # 3. Apply the machine learning algorithm
        #---------------------
        
        if isinstance(params['n_clusters'],int):
            n_clusters=params['n_clusters']
            labels=apply_algo(X,params['n_clusters'],params=params)
            
            # Compute classification score
            if len(np.unique(labels))>1:
                with np.errstate(divide='ignore',invalid='ignore'):  # to avoid itempestive warning ("RuntimeWarning: divide by zero encountered in true_divide...")
                    db_score=davies_bouldin_score(X,labels)
                s_score=silhouette_score(X,labels)
                ch_score=calinski_harabaz_score(X,labels)
            else:
                db_score=np.nan
                s_score=np.nan
                ch_score=np.nan
        else:
            labels,n_clusters,s_score,db_score,ch_score=apply_algo_k_3scores(X,params=params)
        
        # 4. Derive and store the BLH
        #---------------------
        blh.append(blh_from_labels(labels,Z))
        K_values.append(n_clusters)
        s_scores.append(s_score)
        db_scores.append(db_score)
        ch_scores.append(ch_score)
            
    
    # end toolbar
    t1=time.time()      #::::::::::::::::::::::
    chrono=t1-t0
    sys.stdout.write("] ("+str(np.round(chrono,4))+" s)\n")
    
    if outputFile is None:
        fname=inputFile.split('/')[-1]
        outputFile="DAILY_BENCHMARK_"+fname[10:-3]+".nc"
    
    mask_cloud = cbh[:]<=3000
    
    if os.path.isfile(reference):
        blh_ref=np.loadtxt(reference)
    else:
        blh_ref=blh_mnf[:,0]
    
    if storeResults:
        BLHS = [np.array(blh),np.array(blh_mnf[:,0])]
        BLH_NAMES = ['BLH_KABL','BLH_INDUS']
        if os.path.isfile(reference):
            BLHS.append(blh_ref)
            BLH_NAMES.append('BLH_REF')
        
        # Cloud base height is added as if it were a BLH though it's not
        BLHS.append(cbh)
        BLH_NAMES.append("CLOUD_BASE_HEIGHT")
        
        msg=utils.save_qualitymetrics(outputFile,t_values,BLHS,BLH_NAMES,
                            [s_scores,db_scores,ch_scores],['SILH','DB','CH'],
                            [rr,vv],['MASK_RAIN','MASK_FOG'],K_values,chrono,params)
        
        if os.path.isfile(rsFile):
            blh_rs=utils.extract_rs(rsFile,t_values[0],t_values[-1])
        else:
            blh_rs=None
        
        # graphics.blhs_over_data(t_values,z_values,rcs_1,BLHS,[s[4:] for s in BLH_NAMES],
                        # blh_rs=blh_rs,storeImages=True,showFigure=False)
        print(msg)
    
    errl2_blh=np.sqrt(np.nanmean((blh-blh_ref)**2))
    errl1_blh=np.nanmean(np.abs(blh-blh_ref))
    errl0_blh=np.nanmax(np.abs(blh-blh_ref))
    corr_blh=np.corrcoef(blh,blh_ref)[0,1]
    n_invalid=np.sum(np.isnan(blh))+np.sum(np.isinf(blh))
    
    return errl2_blh,errl1_blh,errl0_blh,corr_blh,np.mean(ch_scores),np.mean(db_scores),np.mean(s_scores),chrono,n_invalid

########################
#      TEST BENCH      #
########################
# Launch with
# >> python core.py
#
# For interactive mode
# >> python -i core.py
#
if __name__ == '__main__':
    
    # Test of prepare_data
    #----------------------
    print("\n --------------- Test of prepare_data")
    testFile='../data_samples/lidar/DAILY_MPL_5025_20180802.nc'
    print(' ** Single profile **')
    z_values,rcs_1,rcs_2,coords=utils.extract_testprofile(testFile,profile_id=2,return_coords=True)
    print("z_values.shape",z_values.shape,"rcs_1.shape",rcs_1.shape,"rcs_2.shape",rcs_2.shape)
    X,Z=prepare_data(coords,z_values,rcs_1,rcs_2)
    print("X.shape=",X.shape)
    print("Z.shape=",Z.shape)
    
    n_profiles=3
    print(' ** Concatenated profiles ** (',n_profiles,')')
    t_values,z_values,rcs_1,rcs_2=utils.extract_data(testFile,to_extract=['rcs_1','rcs_2'])
    loc,dateofday,lat,lon=utils.where_and_when(testFile)
    t=55
    coords={'time':dt.datetime.utcfromtimestamp(t_values[t]),'lat':lat,'lon':lon}
    t_back=max(t-n_profiles+1,0)
    rcs_1=rcs_1[t_back:t+1,:]
    rcs_2=rcs_2[t_back:t+1,:]
    print("z_values.shape",z_values.shape,"rcs_1.shape",rcs_1.shape,"rcs_2.shape",rcs_2.shape)
    X,Z=prepare_data(coords,z_values,rcs_1,rcs_2)
    print("X.shape=",X.shape)
    print("Z.shape=",Z.shape)
    
    # Test of apply_algo
    #--------------------
    print("\n --------------- Test of apply_algo")
    labels=apply_algo(X,3)
    print("labels=",labels)
    
    # Test of apply_algo_k_auto
    #--------------------
    print("\n --------------- Test of apply_algo_k_auto")
    params=utils.get_default_params()
    params['n_clusters']='silh'
    labels,n_clusters,score=apply_algo_k_auto(X,params=params)
    print("labels=",labels,"n_clusters=",n_clusters,"score=",score)
    
    # Test of blh_from_labels
    #------------------------
    print("\n --------------- Test of blh_from_labels")
    blh=blh_from_labels(labels,z_values)
    print("blh=",blh)
    
    #graphics.blhs_over_profile(z_values,rcs_1,blh,labels=labels)
    
    # Test of blh_estimation
    #------------------------
    print("\n --------------- Test of blh_estimation")
    # Perform BLH estimation on all profiles of the day and write it into a copy of the netcdf file.
    bblh=blh_estimation(testFile)
    print("blh.shape=",bblh.shape)
    
