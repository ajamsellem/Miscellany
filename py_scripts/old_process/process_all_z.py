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

R = np.load(result_dir+'/Sigmag_all_z.npz')['R']

xi = []
varxi = []
w = []
ave_dens = []
nclust = []
ngal = []
infile = np.load(result_dir+'/Sigmag_all_z.npz')
xi.append(infile['xi'])
varxi.append(infile['varxi'])
w.append(infile['w'])
ave_dens.append(infile['ave_dens'])
nclust.append(infile['nclust'])

xi = np.array(xi)[0]
varxi = np.array(varxi)[0]
w  = np.array(w)[0]
ave_dens = np.array(ave_dens)
nclust = np.array(nclust)

dens     = np.sum(ave_dens*nclust)/np.sum(nclust)
#mean     = dens*(xi*w/np.sum(w))
mean     = dens*xi
print(np.sum(w, axis=0)/10**13)
print(w)
#mean_err = dens*np.sqrt(np.sum(varxi*w**2, axis=0))/np.sum(w, axis=0)
mean_err = dens*np.sqrt(varxi)

Sg_mean = np.array(mean)
Sg_sig = np.array(mean_err)


np.savez(result_dir+'/splashback_cov_'+str(name)+'.npz', r_data=R, sg_mean=Sg_mean, sg_sig=Sg_sig)
#np.savez(result_dir+'/splashback_cov_'+str(name)+'.npz', r_data=R, sg_mean=Sg_mean, sg_sig=np.sum(varxi, axis=0))
#np.savez(result_dir+'/splashback_cov_'+str(name)+'.npz', r_data=R, sg_mean=Sg_mean, sg_sig=np.sum(err_xi, axis=0))
