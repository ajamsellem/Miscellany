"""
Usage: python process_zbins.py  Output_Filename Number_of_Jackknifes Number_of_Z_bins Results_Directory

Secondary Usage: python process_zbins.py  Output_Filename Number_of_Jackknifes Number_of_Z_bins Results_Directory Used_Area Correct_Area
Clusters_Data_Filename Minimum_Z_Value Maximum_Z_Value Minimum_Richness_Value Maximum_Richness_Value

Example python process_zbins.py Redmapper_2 100 7 . 1249.123412341234 1793.123412341324 RM_jk.fits 0.2 0.55 20. 100.
"""

import numpy as np
import os
import sys
from astropy.cosmology import FlatLambdaCDM
import astropy.io.fits as pf
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

name = sys.argv[1]
result_dir = sys.argv[2]
redshift_bin = sys.argv[3]
i = str(redshift_bin)

R = np.load(result_dir+'/Sigmag_' + i + '.npz')['R']

infile = np.load(result_dir+'/Sigmag_' + i + '.npz')
xi = infile['xi']
varxi = infile['varxi']
w = infile['w']
ave_dens = infile['ave_dens']
nclust = infile['nclust']

dens     = np.sum(ave_dens*nclust)/np.sum(nclust)
mean     = dens*xi
mean_err = dens*np.sqrt(varxi)

Sg_mean = np.array(mean)
Sg_sig = np.array(mean_err)

np.savez(result_dir+'/splashback_cov_'+str(name)+'.npz', r_data=R, sg_mean=Sg_mean, sg_sig=Sg_sig, redshift_bin = i)
