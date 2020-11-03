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

class SPT_cluster:
    def __init__(self, ra, dec, z, dz, m, dmu, dml):
        self.ra  = ra
        self.dec = dec
        self.z   = z
        self.dz  = dz
        self.m   = m
        self.dmu = dmu
        self.dml = dml
        self.match_id = None
        self.arcmin_diff = None
        self.z_diff = None
        self.m_diff = None
        self.match_precise = None
        self.sigmaz = None

class RM_cluster:
    def __init__(self, ra, dec, z, dz, lam, dlam, ids):
        self.ra   = ra
        self.dec  = dec
        self.z    = z
        self.dz   = dz
        self.lam  = lam
        self.dlam = dlam
        self.ids  = ids
        self.ang_match = 0.
        self.z_match = 0.
        self.mass_match = 0.
        self.SPT_ra = None
        self.SPT_dec = None
        self.SPT_z = None
        self.SPT_mass = None
        self.sigmaz = None

        m, dm  = convert_rich_to_mass(lam, dlam, z, dz)
        self.m  = m
        self.dm = dm

# SPT Data
SPT = pf.open('/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/2500d_cluster_sample_Bocquet19_SPT.fits')[1].data
z_s = SPT['REDSHIFT']
mask_s = (z_s>0.25)*(z_s<0.7)

ra_s  = SPT['ra'][mask_s]
dec_s = SPT['dec'][mask_s]
z_s   = SPT['REDSHIFT'][mask_s]
dz_s = SPT['REDSHIFT_UNC'][mask_s]
min_dz_s = np.min(dz_s[np.where(dz_s != 0.)[0]])

mass_s = SPT['M200_marge'][mask_s]
mass_uerr_s = SPT['M200_marge_uerr'][mask_s]
mass_lerr_s = SPT['M200_marge_lerr'][mask_s]
print(ra_s.shape)

# Redmapper DES Y3 Data
Y3_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl02_catalog.fit'
Y3 = pf.open(Y3_dat)[1].data
lam_3 = Y3['lambda_chisq']
z_3   = Y3['Z_LAMBDA']
dec_3 = Y3['dec']
mask_3 = (z_3>0.25)*(z_3<0.7)*(dec_3<=-39.)#*(lam_3>=58.)

ra_3  = Y3['ra'][mask_3]
dec_3 = Y3['dec'][mask_3]
z_3   = Y3['Z_LAMBDA'][mask_3]
dz_3 = Y3['Z_LAMBDA_E'][mask_3]
lam_3 = Y3['lambda_chisq'][mask_3]
ids_3 = Y3['MEM_MATCH_ID'][mask_3]
dlam_3 = Y3['LAMBDA_CHISQ_E'][mask_3]
print(ra_3.shape)

# Sigmaz column for Redmapper Clusters
sigz_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/sigma_zmeasurement_desy3_lgt15.fits'
XX = pf.open(sigz_dat)[1].data
ids_z = XX['MEM_MATCH_ID']
sigmaz = XX['sigma_z']

SPT_matches = pf.open('/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/SPT_in_DES_foot.fits')[1].data
ra_s        = SPT_matches['RA']
dec_s       = SPT_matches['DEC']
z_s         = SPT_matches['Z_LAMBDA']
dz_s       = SPT_matches['DZ_LAMBDA']
mass_s      = SPT_matches['M200_marge']
mass_uerr_s = SPT_matches['M200_marge_uerr']
mass_lerr_s = SPT_matches['M200_marge_lerr']
print(len(ra_s))

clus_s = []
for j, val in enumerate(ra_s):
    x = SPT_cluster(ra_s[j], dec_s[j], z_s[j], dz_s[j], mass_s[j], mass_uerr_s[j], mass_lerr_s[j])
    clus_s.append(x)
clus_s = np.array(clus_s)

dat_3 = np.array(list(zip(ra_3, dec_3, z_3, dz_3, lam_3, dlam_3, ids_3)))
for i, S_cluster in enumerate(clus_s):
    print(i)
    # Find the 5 clusters with the least separation from a given SPT cluster
    sep = find_sep(dat_3[:,0], dat_3[:,1], S_cluster.ra, S_cluster.dec)
    idx = np.argpartition(sep.arcminute, 5)[:5]

    # print separation/redshift difference/mass difference between the 5 RM clusters and the SPT cluster and the lambda values of all 5 RM clusters
    print("Degree Differences:   " + str(sep.degree[idx]))
    z_diff = np.absolute(S_cluster.z-z_3[idx])
    print("Redshift Differences: " + str(z_diff))
    mass_3 = []
    print("Richness Values:      " + str(lam_3[idx]))
    for k, lam_val in enumerate(lam_3[idx]):
        m_3, dm_3 = convert_rich_to_mass(lam_val, dlam_3[idx][k], z_3[idx][k], dz_3[idx][k])
        mass_3.append(m_3)
    mass_3 = np.array(mass_3)
    m_diff = np.absolute(S_cluster.m-mass_3)
    print("Mass Differences:     " + str(m_diff))

    # Select which RM cluster mathces
    val = int(input("Matched Index? "))
    match_precise = input("Bad Match? ")
    match_idx = idx[val-1]

    # Save information of the matched RM cluster on the SPT cluster object
    S_cluster.match_id = ids_3[match_idx]
    S_cluster.arcmin_diff = sep.arcminute[match_idx]
    S_cluster.z_diff = z_diff[val-1]
    S_cluster.m_diff = m_diff[val-1]
    S_cluster.match_precise = match_precise

    os.system("clear")

file_Name = "/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/SPT_matched_info.dat"
fileObject = open(file_Name,'wb')
pickle.dump(clus_s,fileObject)

fileObject.close()

clus_s = pickle.load(open("/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/SPT_matched_info.dat", "rb" ))
print(clus_s[0].match_id)
