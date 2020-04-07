#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
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
'''

from matplotlib import pyplot as plt
import datetime as dt
import numpy as np
import netCDF4 as nc

# Local packages
from kabl import utils
from kabl import core

def quicklook_data(nc_file,max_height=4500,with_pbl=False,with_cbh=False):
    '''Give a quick look of the data, only the data.
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
    
    [OUT]
         (matplotlib.pyplot figure): same as blhs_over_data'''
    
    location,day,lat,lon=utils.where_and_when(nc_file)
    
    to_be_extracted=['rcs_0']
    if with_pbl:
        to_be_extracted.append('pbl')
    if with_cbh:
        to_be_extracted.append('cloud_base_height')
        
    data=utils.extract_data(nc_file,max_height=max_height,to_extract=to_be_extracted)
    
    if with_pbl and with_cbh:
        t,z,rcs,pbl,cbh=data
    elif with_pbl:
        t,z,rcs,pbl=data
    elif with_cbh:
        t,z,rcs,cbh=data
    else:
        t,z,rcs=data
    
    plt.figure(figsize=(14,7))
    plt.pcolormesh(t,z,rcs.T,alpha = 0.8,cmap='rainbow',vmin=-0.1, vmax=0.8)
    if with_pbl:
        pbl[pbl==-999]=np.nan
        for layer in range(pbl.shape[1]):
            plt.plot(t,pbl[:,layer],'k*')
    if with_cbh:
        cbh[cbh==-999]=np.nan
        for layer in range(cbh.shape[1]):
            plt.plot(t,cbh[:,layer],'r.')
    axes = plt.gca()
    plt.title("Lidar backscatter | "+location+" "+day.strftime('%Y/%m/%d'))
    axes.set_xlabel('Hour')
    axes.set_ylabel('Height (m agl)')
    plt.tight_layout()
    plt.grid(color='white', ls='solid')
    plt.colorbar(label="Range corrected signal",alpha=0.8)
    
    locs, labels = plt.xticks()
    labels = [dt.datetime.utcfromtimestamp(loc).strftime('%H:%M') 
          for loc in locs]
    
    axes.set_xticks(locs)
    axes.set_xticklabels(labels)
    plt.gcf().autofmt_xdate()
    plt.show(block=False)


def quicklook_testprofiles(nc_file):
    '''Give a quick look of the preselected profiles, only the data.
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
    
    [OUT]
        - (matplotlib.pyplot figure): same as blhs_over_profile'''
    
    location,day,lat,lon=utils.where_and_when(nc_file)
    
    plt.figure(figsize=(14,7))
    
    n_profiles=4
    hours=['0:17','9:22','16:17','21:17']
    
    for prof_id in range(n_profiles):
        z,rcs=utils.extract_testprofile(nc_file,to_extract=['rcs_0'],profile_id=prof_id)
        plt.subplot(1,n_profiles,prof_id+1)
        plt.plot(rcs,z,linewidth=2,label=hours[prof_id])
        plt.legend()
    plt.tight_layout()
    plt.suptitle("Lidar backscatter | "+location+" "+day.strftime('%Y/%m/%d'))
    plt.show(block=False)

def blhs_over_profile(z_values,data_values,blhs,blhs_names=None,labels=None,titre=None,storeImages=False,fmtImages=".png",figureDir=""):
    '''Plot the profile of data and the BLH (more than one can be superimposed)
    
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
    '''
    
    if blhs_names is None:
        if isinstance(blhs,list):
            blhs_names=["BLH {}".format(k+1) for k in range(len(blhs))]
        else:
            blhs_names="BLH"
        
    plt.figure(figsize=(14,7))
    plt.plot(data_values,z_values,linewidth=2,label="RCS profile")
    
    vmin=np.nanmin(data_values)
    vmax=np.nanmax(data_values)
    print("vmin=",vmin,"vmax=",vmax)
    if isinstance(blhs,list):
        for ib in range(len(blhs)):
            plt.plot([vmin,vmax],[blhs[ib],blhs[ib]],'--',  label = blhs_names[ib],alpha = 0.3, markersize=12)
    else:
        plt.plot([vmin,vmax],[blhs,blhs],'--', label = blhs_names,alpha = 0.3, markersize=12)
    
    if labels is not None:
        for k in np.unique(labels):
            kdex=np.where(labels==k)[0]
            plt.plot(data_values[kdex],z_values[kdex],linewidth=0,marker=np.mod(k+4,11),label="Cluster {}".format(k+1))
    
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def blhs_over_data(t_values,z_values,bckgrd_data,blhs,blhs_names=None,blh_rs=None,titre=None,showFigure=True,storeImages=False,fmtImages=".png",figureDir=""):
    '''
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
            In the Y-axis are the altitude values'''
    
    if blhs_names is None:
        if isinstance(blhs,list):
            blhs_names=["BLH {}".format(k+1) for k in range(len(blhs))]
        else:
            blhs_names="BLH"
    
    day = dt.datetime.utcfromtimestamp(t_values[1]).strftime('%Y/%m/%d')
    date = dt.datetime.utcfromtimestamp(t_values[1]).strftime('%Y%m%d')
    if titre is None:
        titre = "Lidar backscatter | "+day
    
    fig=plt.figure(figsize=(10,5))
    plt.pcolormesh(t_values,z_values,bckgrd_data.T,alpha = 0.8,cmap='rainbow',vmin=-0.1, vmax=0.8)
    plt.colorbar(alpha=0.5)
    plt.grid(color='white', ls='solid')
    
    plt.title(titre)
    axes = plt.gca()
    axes.set_xlabel('Hour')
    axes.set_ylabel('Height (m agl)')
    
    locs, labs = plt.xticks()
    labs = [dt.datetime.utcfromtimestamp(loc).strftime('%H:%M') for loc in locs]
    axes.set_xticks(locs)
    axes.set_xticklabels(labs)
    plt.gcf().autofmt_xdate()
    
    if isinstance(blhs,list):
        for ib in range(len(blhs)):
            plt.plot(t_values,blhs[ib],'.:', label = blhs_names[ib])
    else:
        plt.plot(t_values,blhs,'.:', label = blhs_names)
        
    if blh_rs is not None:
        rsTimes=blh_rs[0]
        rsValues=blh_rs[1]
        plt.plot(rsTimes,rsValues,'kP',markersize=12,linewidth=3,label='RS')
        
    plt.legend()
    plt.tight_layout()
    
    if storeImages:
        plt.savefig(figureDir+"_".join(["blhsOverData",date])+fmtImages)
    
    if showFigure:
        plt.show(block=False)
    else:
        plt.close()
    
    return fig

def quicklook_output(nc_file):
    '''Same as blhs_over_data, but directly from the output netcf file
    (and with less flexibility).
    
    [IN]
        - nc_file (str): path to the netcdf file containing the data
    
    [OUT]
        - (matplotlib.pyplot figure): same as blhs_over_data'''
    
    location,day,lat,lon=utils.where_and_when(nc_file)
    t,z,rcs,blh_new,blh_mnf=utils.extract_data(nc_file,to_extract=['rcs_0','blh_kabl','pbl'])
    
    fig=blhs_over_data(t,z,rcs,[blh_new,blh_mnf[:,0]],blhs_names=['BLH KABL','BLH manufacturer'],titre="Lidar backscatter | "+location+" "+day.strftime('%Y/%m/%d'))
        
    
    return fig
    
def quicklook_benchmark(data_file,blh_file,rs_file=None,showFigure=True,storeImages=False,fmtImages=".png"):
    '''Same as blhs_over_data, but directly from the output netcf file
    (and with less flexibility).
    
    [IN]
        - data_file (str): path to the netcdf file containing the data
        - data_file (str): path to the netcdf file containing the BLH estimation
    
    [OUT]
        - (matplotlib.pyplot figure): same as blhs_over_data'''
    
    location,day,lat,lon=utils.where_and_when(data_file)
    t,z,rcs=utils.extract_data(data_file,to_extract=['rcs_0'])
    
    BLHS=[]
    BLH_NAMES=[]
    ncf=nc.Dataset(blh_file)
    for key in ncf.variables.keys():
        if "BLH" in key:
            BLHS.append(np.array(ncf.variables[key]))
            BLH_NAMES.append(key[4:])
    
    if rs_file is not None:
        blh_rs=utils.extract_rs(rs_file,t[0],t[-1])
    else:
        blh_rs=None
    
    fig=blhs_over_data(t,z,rcs,BLHS,blhs_names=BLH_NAMES,blh_rs=blh_rs,
            titre="Lidar backscatter | "+location+" "+day.strftime('%Y/%m/%d'),
            showFigure=showFigure,storeImages=storeImages,fmtImages=fmtImages)
    
    return fig

def scatterplot_blhs(time,blh_ref,blh_new,titre=None,storeImages=False,fmtImages=".png",figureDir=""):
    '''
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
    '''
    
    fig=plt.figure(figsize=(14,7))
    plt.plot([200,2500],[200,2500],'k-',linewidth=5)
    date = dt.datetime.utcfromtimestamp(time[1]).strftime('%Y%m%d')
    print("date=",date,type(date))
    Nt = len(time)
    heure = [float(dt.datetime.utcfromtimestamp(time[i]).strftime('%H')) for i in np.arange(0,Nt,1)]
    print("heure[0]=",heure[0],type(heure[0]),"heure[-1]=",heure[-1])
    axes = plt.gca()
    axes.set_xlabel('Manufacturer BLH (m agl)')
    axes.set_ylabel('Estimated BLH (m agl)')
    label = ['00:00','06:00','12:00','18:00','22:00','00:00']
    plt.scatter(blh_ref,blh_new,c=heure,cmap='hsv',s=60*2)
    cbar = plt.colorbar()
    cbar.set_label("Hour")
    corr = np.corrcoef(blh_ref, blh_new)[0,1]
    plt.annotate('Corr = {0:.2f}'.format(corr),(1000,2000))
    
    if titre is None:
        titre = "Manufacturer vs estimated BLH \n"+date+" corr={0:.2f}".format(corr)
    
    plt.title(titre)
    plt.tight_layout()
    
    if storeImages:
        plt.savefig(figureDir+"_".join(["scatterplotBLHs",date])+fmtImages)
    plt.show(block=False)
    return fig

########################
#      TEST BENCH      #
########################
# Launch with
# >> python graphics.py
#
# For interactive mode
# >> python -i graphics.py
#
if __name__ == '__main__':
    
    # Test of quicklook_data
    #------------------------
    print("\n --------------- Test of quicklook_data")
    testFile='../data_samples/lidar/DAILY_MPL_5025_20180802.nc'
    quicklook_data(testFile)
    
    
    # Test of quicklook_testprofiles
    #------------------------
    print("\n --------------- Test of quicklook_testprofiles")
    testFile='../data_samples/lidar/DAILY_MPL_5025_20180802.nc'
    quicklook_testprofiles(testFile)
    
    
    # Test of blhs_over_profile
    #------------------------
    print("\n --------------- Test of blhs_over_profile")
    testFile='../data_samples/lidar/DAILY_MPL_5025_20180802.nc'
    z_values,rcs_1,rcs_2,coords=utils.extract_testprofile(testFile,profile_id=3,return_coords=True)
    X,Z=core.prepare_data(coords,z_values,rcs_1,rcs_2)
    labels=core.apply_algo(X,3)
    blh=core.blh_from_labels(labels,Z)
    
    blhs_over_profile(z_values,rcs_1,blh,labels=labels)
    
    plt.figure()
    plt.hist(rcs_1,35)
    plt.title("Histogram of a single profile of RCS")
    plt.show(block=False)
    
    # Test of blhs_over_data
    #------------------------
    print("\n --------------- Test of blhs_over_data")
    testFile='../data_samples/lidar/DAILY_MPL_5025_20180802.nc'
    blh=core.blh_estimation(testFile)
    t_values,z_values,rcs_1,rcs_2=utils.extract_data(testFile)
    
    blhs_over_data(t_values,z_values,rcs_1,blh)
    
    # Test of scatterplot_blhs
    #------------------------
    print("\n --------------- Test of scatterplot_blhs")
    outputFile='../data_samples/lidar/DAILY_MPL_5025_20180802.out.nc'
    t_values,z_values,blh_new,blh_mnf=utils.extract_data(outputFile,to_extract=['blh_kabl','pbl'])
    
    scatterplot_blhs(t_values,blh_mnf[:,0],blh_new)
    
    
    # Test of quicklook_output
    #------------------------
    print("\n --------------- Test of quicklook_output")
    outputFile='../data_samples/lidar/DAILY_MPL_5025_20180802.out.nc'
    quicklook_output(outputFile)
    
    input("\n Press Enter to exit (close down all figures)\n")
