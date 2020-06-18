#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pathfile to access the data OS-independently."""
import os

# HARD PATHS
#============

# Data
#------
site='TRAPPES'
datarootdir = '../data-samples/'
defaultlidarfile = 'DAILY_MPL_5025_20180802.nc'

# Artifacts
#-----------
artifactrootdir = '../artifacts/'

# Results
#---------
resultrootdir='../results/'
namexp="Brest_kmeans_K3_Igiven_1prof_RCS1"


# SOFT PATHS
#============

# Radiosoundings
#----------------

def file_blhfromrs():
    """Default file containing 2-year long BLH estimations from radiosoundings"""
    return os.path.join(datarootdir,'blh-from-rs','BLH_RS_liss3_BRNliss10_'+site+'.nc')


# Lidar
#-------

def dir_lidardata():
    """Directory containing samples of lidar data"""
    return os.path.join(datarootdir,'lidar')

def file_defaultlidardata():
    """Default MMPL data file, used for testing most of the methods"""
    return os.path.join(dir_lidardata(),defaultlidarfile)

def file_defaultcl31data():
    """Default CL31 data file, used for testing most of the methods"""
    return os.path.join(dir_lidardata(),"DAILY_CL31_5029_20200404.nc")

# def dir_lidar2years():
    # """Directory containing all lidar data from raw2l1 (2-year long dataset)"""
    # if not os.path.isdir(os.path.join(datarootdir,site+'_LIDAR_2ANS')):
        # print("WARNING: full dataset directory does not exist")
    # return os.path.join("../../KABL-data/",site+'_LIDAR_2ANS')


# Handmade data
#---------------

def dir_handmadedata():
    """Directory containing human expertised BLH estimations"""
    return os.path.join(datarootdir,'blh-from-human')


# Artifacts
#-----------

def file_labelleddataset():
    """File with all required material to train ADABL"""
    return os.path.join(artifactrootdir,"labelled-datasets","adabl_supervised_dataset.csv")

def file_trainedmodel():
    """File to Pickle object containing trained ADABL"""
    return os.path.join(artifactrootdir,"trained-adabl","adabl_classifier_tzRCS12_M200_D5.pkl")

def file_trainedscaler():
    """File to Pickle object containing corresponding scaler for ADABL"""
    return os.path.join(artifactrootdir,"trained-adabl","adabl_scaler_tzRCS12_M200_D5.pkl")


# Results
#---------
def file_defaultoutput():
    """File to an example of KABL output: a raw2l1 file with an extra field 'blh_kabl'"""
    return os.path.join(resultrootdir,defaultlidarfile[:-3] + ".out.nc")

def file_blhfromlidar():
    return os.path.join(datarootdir,'blh-from-lidar','BENCHMARK_'+namexp+'.nc')
