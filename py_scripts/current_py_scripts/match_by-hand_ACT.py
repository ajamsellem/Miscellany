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

        m, dm  = convert_rich_to_mass(lam, dlam, z, dz)
        self.m  = m
        self.dm = dm

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
Y3 = pf.open(Y3_dat)[1].data
lam_3 = Y3['lambda_chisq']
z_3   = Y3['Z_LAMBDA']
dec_3 = Y3['dec']
mask_3 = (z_3>0.25)*(z_3<0.7)*(lam_3>=20.)

ra_3  = Y3['ra'][mask_3]
dec_3 = Y3['dec'][mask_3]
z_3   = Y3['Z_LAMBDA'][mask_3]
dz_3 = Y3['Z_LAMBDA_E'][mask_3]
lam_3 = Y3['lambda_chisq'][mask_3]
ids_3 = Y3['MEM_MATCH_ID'][mask_3]
dlam_3 = Y3['LAMBDA_CHISQ_E'][mask_3]
print(ra_3.shape)

clus_a = []
for j, val in enumerate(ra_a):
    x = ACT_cluster(ra_a[j], dec_a[j], z_a[j], dz_a[j], lam_a[j], dlam_a[j])
    clus_a.append(x)
clus_a = np.array(clus_a)

dat_3 = np.array(list(zip(ra_3, dec_3, z_3, dz_3, lam_3, dlam_3, ids_3)))

for i, A_cluster in enumerate(clus_a):
    print(i)
    # Find the angular separation, redshift difference, and lambda difference between the all RM clusters and the ACT cluster
    sep      = find_sep(dat_3[:,0], dat_3[:,1], A_cluster.ra, A_cluster.dec)
    z_diff   = np.absolute(dat_3[:,2] - A_cluster.z  )
    lam_diff = np.absolute(dat_3[:,4] - A_cluster.lam)

    # Requirements to be considered a match
    reqs = (sep.degree <= 1.) & (lam_diff <= 10.)

    # If only one match is found...
    if lam_diff[reqs].shape[0] == 1.:
        match_idx = 0
        print('one match')
    # If multiple matches are found...
    # But one match has exactly equal richnesses...
    elif lam_diff[reqs].shape[0] > 1.:
        if len(np.where(lam_diff[reqs]==0.)[0]) == 1.:
            match_idx = np.where(lam_diff[reqs]==0.)[0][0]
            print('0 match')
    # Otherwise...
        else:
            # Print info on cluster match goodness
            print("Arcmin Differences:   " + str(sep.arcminute[reqs]))
            print("Redshift Differences: " + str(z_diff[reqs]))
            print("Richness Differences: " + str(lam_diff[reqs]))

            # Select which RM cluster mathces
            val = int(input("Matched Index? "))
            match_idx = val-1

    # Save information of the matched RM cluster on the ACT cluster object
    if lam_diff[reqs].shape[0] >= 1.:
        A_cluster.match_id      = dat_3[:,6][reqs][match_idx]
        A_cluster.arcmin_diff   = sep.arcminute[reqs][match_idx]
        A_cluster.z_diff        = z_diff[reqs][match_idx]
        A_cluster.lamm_diff     = lam_diff[reqs][match_idx]

    os.system("clear")

file_Name = "/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/ACT_data_files/ACT_hand-matched_info_2.dat"
fileObject = open(file_Name,'wb')
pickle.dump(clus_a,fileObject)

fileObject.close()
