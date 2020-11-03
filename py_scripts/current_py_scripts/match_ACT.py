import sys
import os
sys.path.insert(0, '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/py_scripts/')
sys.path.insert(0, '/opt/anaconda3/lib/python3.7/site-packages')
import numpy as np
import pylab as pyl
import astropy.io.fits as pf
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from sympy import *
import matplotlib as mpl
from matplotlib.patches import Rectangle
import pickle

from SPT_functions import *

# Construct Cluster Classes
class ACT_cluster:
    def __init__(self, ra, dec, z, dz, lam, dlam):
        self.ra            = ra
        self.dec           = dec
        self.z             = z
        self.dz            = dz
        self.lam           = lam
        self.dlam          = dlam
        self.match_id      = None
        self.arcmin_diff   = None
        self.z_diff        = None
        self.lam_diff      = None
        self.sigmaz        = None

class RM_cluster:
    def __init__(self, ra, dec, z, dz, lam, dlam, ids):
        self.ra          = ra
        self.dec         = dec
        self.z           = z
        self.dz          = dz
        self.lam         = lam
        self.dlam        = dlam
        self.ids         = ids
        self.match       = 0.
        self.arcmin_diff = None
        self.z_diff      = None
        self.lam_diff    = None
        self.sigmaz      = None

# ACT Data
ACT_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/ACT_data_files/AdvACT_S18Clusters_v1.0-beta_DESY3Only.fits'
ACT     = pf.open(ACT_dat)[1].data
z_a     = ACT['redshift']
in_RM   = ACT['RMDESY3']
mask_a  = (z_a>0.25)*(z_a<0.7)*(in_RM==True)
ra_a    = ACT['RADeg'][mask_a]
dec_a   = ACT['decDeg'][mask_a]
z_a     = ACT['redshift'][mask_a]
dz_a    = ACT['redshiftErr'][mask_a]
lam_a   = ACT['RMDESY3_LAMBDA_CHISQ'][mask_a]
dlam_a  = ACT['RMDESY3_LAMBDA_CHISQ_E'][mask_a]

# Redmapper DES Y3 Data
Y3_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl02_catalog.fit'
Y3     = pf.open(Y3_dat)[1].data
lam_3  = Y3['lambda_chisq']
z_3    = Y3['Z_LAMBDA']
dec_3  = Y3['dec']
mask_3 = (z_3>0.25)*(z_3<0.7)

ra_3   = Y3['ra'][mask_3]
dec_3  = Y3['dec'][mask_3]
z_3    = Y3['Z_LAMBDA'][mask_3]
dz_3   = Y3['Z_LAMBDA_E'][mask_3]
lam_3  = Y3['lambda_chisq'][mask_3]
ids_3  = Y3['MEM_MATCH_ID'][mask_3]
dlam_3 = Y3['LAMBDA_CHISQ_E'][mask_3]

# Sigmaz column for Redmapper Clusters
sigz_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/sigma_zmeasurement_desy3_lgt15.fits'
XX       = pf.open(sigz_dat)[1].data
ids_z    = XX['MEM_MATCH_ID']
sigmaz   = XX['sigma_z']

clus_a = []
for j, val in enumerate(ra_a):
    x = ACT_cluster(ra_a[j], dec_a[j], z_a[j], dz_a[j], lam_a[j], dlam_a[j])
    clus_a.append(x)
clus_a = np.array(clus_a)
print("Number of Potential Matches:")
print(len(clus_a))
print()
print()
dat_3 = np.array(list(zip(ra_3, dec_3, z_3, dz_3, lam_3, dlam_3, ids_3)))

for i, A_cluster in enumerate(clus_a):
    if i%10. == 0.:
        print(i)
    # Find richness differences between a single ACT cluster and all RM clusters
    lam_diff_list  = np.absolute(A_cluster.lam-dat_3[:,4])
    match_idx_temp = np.where(lam_diff_list == 0.)[0]

    if len(match_idx_temp) >= 1.:
        ra_temp  = ra_3[match_idx_temp]
        dec_temp = dec_3[match_idx_temp]
        z_temp   = z_3[match_idx_temp]
        lam_temp = lam_3[match_idx_temp]
        ids_temp = ids_3[match_idx_temp]

        seps = find_sep(ra_temp, dec_temp, A_cluster.ra, A_cluster.dec)
        match_idx = np.argmin(seps)

        # If the matched clusters are reasonably close...
        if np.min(seps.arcminute <= 3.):
            # Find the difference between matched RM and ACT clusters in arcminutes, redshift, and richness
            z_diff   = np.absolute(z_temp[match_idx]-A_cluster.z)
            lam_diff = np.absolute(lam_temp[match_idx]-A_cluster.lam)

            # Save information of the matched RM cluster on the ACT cluster object
            A_cluster.match_id    = ids_temp[match_idx]
            A_cluster.arcmin_diff = seps[match_idx]
            A_cluster.z_diff      = z_diff
            A_cluster.lam_diff    = lam_diff

file_Name = "/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/ACT_data_files/ACT_matched_info.dat"
fileObject = open(file_Name,'wb')
pickle.dump(clus_a,fileObject)

fileObject.close()

clus_a = pickle.load(open("/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/ACT_data_files/ACT_matched_info.dat", "rb" ))
