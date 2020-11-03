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
        self.min_num       = 1

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

def check_dups(a_list):
    a_set = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False

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
mask_3 = (z_3>0.25)*(z_3<0.7)*(lam_3>=20.)

ra_3  = Y3['ra'][mask_3]
dec_3 = Y3['dec'][mask_3]
ids_3 = Y3['MEM_MATCH_ID'][mask_3]

# List all ACT clusters
clus_a = []
for j, val in enumerate(ra_a):
    x = ACT_cluster(ra_a[j], dec_a[j], z_a[j], dz_a[j], lam_a[j], dlam_a[j])
    clus_a.append(x)
clus_a = np.array(clus_a)

# List all RM clusters
dat_3 = np.array(list(zip(ra_3, dec_3, ids_3)))

id_already = []
A_cluster_already = []
for i, A_cluster in enumerate(clus_a):
    # Find the nearest RM cluster to the given ACT cluster
    sep       = find_sep(dat_3[:,0], dat_3[:,1], A_cluster.ra, A_cluster.dec)
    match_idx = np.where(sep.arcminute==np.min(sep.arcminute))[0][0]

    # Save the RM cluster id and angular separation between the matched RM cluster and ACT cluster on the ACT cluster object
    A_cluster.match_id    = dat_3[:,2][match_idx]
    A_cluster.arcmin_diff = sep.arcminute[match_idx]

    # Save the match id of the RM cluster and the ACT cluster
    id_already.append(match_idx)
    A_cluster_already.append(A_cluster)

    # When two ACT clusters match to the same RM cluster...
    while len(np.unique(id_already)) != len(id_already):
        # Find the id of the RM cluster that is matched twice
        dup_id = set([x for x in id_already if id_already.count(x) > 1]).pop()

        # Get the separation between the RM cluster and the first of the matched ACT clusters
        loc_1 = np.where(id_already == dup_id)[0][0]
        clus_1 = A_cluster_already[loc_1]
        min_sep_1 = clus_1.arcmin_diff

        # Get the separation between the RM cluster and the second of the matched ACT clusters
        loc_2 = np.where(id_already == dup_id)[0][1]
        clus_2 = A_cluster_already[loc_2]
        min_sep_2 = clus_2.arcmin_diff

        # If the second ACT cluster is farther away from the match...
        if min_sep_1 < min_sep_2:
            # Find the next closest RM cluster to match the first ACT cluster to
            seps_2 = find_sep(dat_3[:,0], dat_3[:,1], clus_2.ra, clus_2.dec)
            sep_next_min_2 = np.partition(seps_2.arcminute, clus_2.min_num+1)[clus_2.min_num+1]
            match_idx_2 = np.where(seps_2.arcminute == sep_next_min_2)[0][0]
            clus_2.match_id    = dat_3[:,2][match_idx_2]
            clus_2.arcmin_diff = seps_2.arcminute[match_idx_2]
            clus_2.min_num    += 1
            # Remove the match info of the first ACT cluster from lists, and add the new info
            del id_already[loc_2]
            del A_cluster_already[loc_2]
            id_already.append(match_idx_2)
            A_cluster_already.append(clus_2)

        # If the first ACT cluster is farther away from the match...
        if min_sep_2 < min_sep_1:
            # Find the next closest RM cluster to match the second ACT cluster to
            seps_1 = find_sep(dat_3[:,0], dat_3[:,1], clus_1.ra, clus_1.dec)
            sep_next_min_1 = np.partition(seps_1.arcminute, clus_1.min_num+1)[clus_1.min_num+1]
            match_idx_1 = np.where(seps_1.arcminute == sep_next_min_1)[0][0]
            clus_1.match_id    = dat_3[:,2][match_idx_1]
            clus_1.arcmin_diff = seps_1.arcminute[match_idx_1]
            clus_1.min_num    += 1
            # Remove the match info of the second ACT cluster from lists, and add the new info
            del id_already[loc_1]
            del A_cluster_already[loc_1]
            id_already.append(match_idx_1)
            A_cluster_already.append(clus_1)

file_Name = "/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/ACT_data_files/ACT_pure-RADEC-matched_info.dat"
fileObject = open(file_Name,'wb')
pickle.dump(clus_a,fileObject)

fileObject.close()
