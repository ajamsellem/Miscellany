# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ajamsellem/.local/bin')
sys.path.insert(0, '/home/ajamsellem/kmeans_radec')
import numpy as np
print(np.__version__)
import astropy.io.fits as pf
import healpy as hp
import kmeans_radec

def make_jk(ra_ran, dec_ran, ra, dec, N=100, dilute_factor=1, rand_out=1, large_mem=True, maxiter=500, tol=1e-05, seed=100):
    """
    Given coordinate of random points, generate JK indecies 
    for another catalog of positions. Include the possibility 
    of diluting the random catalog. Return an array of JK 
    indicies the same length of ra and dec.  
    """

    RADEC_ran = np.zeros((len(ra_ran),2))
    RADEC_ran[:,0] = ra_ran
    RADEC_ran[:,1] = dec_ran

    RADEC = np.zeros((len(ra),2))
    RADEC[:,0] = ra
    RADEC[:,1] = dec

    np.random.seed(seed)
    ids = np.arange(len(ra_ran))
    np.random.shuffle(ids)
    RADEC_ran_dilute = np.zeros((int(len(ra_ran)/dilute_factor),2))
    RADEC_ran_dilute[:,0] = ra_ran[ids[:int(len(ra_ran)/dilute_factor)]]
    RADEC_ran_dilute[:,1] = dec_ran[ids[:int(len(ra_ran)/dilute_factor)]]

    km = kmeans_radec.kmeans_sample(RADEC_ran_dilute, N, maxiter=500, tol=1e-05)
    print(np.unique(km.labels))

    if large_mem == True:
        Ntotal = len(RADEC)
        Ntotal_ran = len(RADEC_ran)

        JK = np.array([])
        JK_ran = np.array([])

        for i in range(99):
            #print i
            JK = np.concatenate((JK, km.find_nearest(RADEC[i*int(Ntotal/100):(i+1)*int(Ntotal/100)])), axis=0)
            print(np.unique(JK))

            if rand_out==1:
                print(i)
                JK_ran = np.concatenate((JK_ran, km.find_nearest(RADEC_ran[i*int(Ntotal_ran/100):(i+1)*int(Ntotal_ran/100)])), axis=0)

        JK = np.concatenate((JK, km.find_nearest(RADEC[99*int(Ntotal/100):])), axis=0)
        if rand_out==1:
            JK_ran = np.concatenate((JK_ran, km.find_nearest(RADEC_ran[99*int(Ntotal_ran/100):])), axis=0)
        print('len of random', len(ra_ran))
        print('len of JK', len(JK_ran))

    else:
        JK = km.find_nearest(RADEC)
        if rand_out==1:
            JK_ran = km.find_nearest(RADEC_ran)
    
    if rand_out==1:    
        return JK_ran, JK
    else:
        return JK

# Read in Redmapper data
RM = pf.open('/home/ajamsellem/Fits_files/redmapper.fit')[1].data
ra_r = RM['RA']
dec_r = RM['DEC']
z_r = RM['Z_Lambda']
lam_r = RM['Lambda_Chisq']

print("Number of RM Clusters after cuts: " + str(len(ra_r)))

# Read in WaZP data
W = pf.open('/home/ajamsellem/Fits_files/WaZP_jk.fits')[1].data
print('Number of WaZP Clusters before cuts: ' + str(len(W)))
ra_w = W['ra']
dec_w = W['dec']
z_w = W['z']
ngal_w = W['lambda']
jk_w = W['jk']

print("Redmapper RA Max: " + str(np.max(ra_r)))
print("Redmapper RA Min: " + str(np.min(ra_r)))
print("Redmapper DEC Max: " + str(np.max(dec_r)))
print("Redmapper DEC Min: " + str(np.min(dec_r)))
print()
print("WaZP RA Max: " + str(np.max(ra_w)))
print("WaZP RA Min: " + str(np.min(ra_w)))
print("WaZP Dec Max: " + str(np.max(dec_w)))
print("WaZP Dec Min: " + str(np.min(dec_w)))

# Masking parts of catalogues that do not overlap
mask = (dec_w >= -60.)
ra_w = ra_w[mask]
dec_w = dec_w[mask]
z_w = z_w[mask]
ngal_w = ngal_w[mask]
jk_w = jk_w[mask]

mask = (dec_r <= -38.)
ra_r = ra_r[mask]
dec_r = dec_r[mask]
z_r = z_r[mask]
lam_r = lam_r[mask]

print("Redmapper RA Max: " + str(np.max(ra_r)))
print("Redmapper RA Min: " + str(np.min(ra_r)))
print("Redmapper DEC Max: " + str(np.max(dec_r)))
print("Redmapper DEC Min: " + str(np.min(dec_r)))
print()
print("WaZP RA Max: " + str(np.max(ra_w)))
print("WaZP RA Min: " + str(np.min(ra_w)))
print("WaZP Dec Max: " + str(np.max(dec_w)))
print("WaZP Dec Min: " + str(np.min(dec_w)))

# Order indices from least to great lambda/ngal
idx_r = np.argsort(lam_r)
idx_w = np.argsort(ngal_w)

ra_r_2 = []
dec_r_2 = []
z_r_2 = []
lam_r_2 = []

for index in idx_r:
    ra_r_2.append(ra_r[index])
    dec_r_2.append(dec_r[index])
    z_r_2.append(z_r[index])
    lam_r_2.append(lam_r[index])
ra_r_2 = np.flip(np.asarray(ra_r_2))
dec_r_2 = np.flip(np.asarray(dec_r_2))
z_r_2 = np.flip(np.asarray(z_r_2))
lam_r_2 = np.flip(np.asarray(lam_r_2))

ra_w_2 = []
dec_w_2 = []
z_w_2 = []
ngal_w_2 = []
jk_w_2 = []

for index in idx_w:
    ra_w_2.append(ra_w[index])
    dec_w_2.append(dec_w[index])
    z_w_2.append(z_w[index])
    ngal_w_2.append(ngal_w[index])
    jk_w_2.append(jk_w[index])
ra_w_2 = np.flip(np.asarray(ra_w_2))
dec_w_2 = np.flip(np.asarray(dec_w_2))
z_w_2 = np.flip(np.asarray(z_w_2))
ngal_w_2 = np.flip(np.asarray(ngal_w_2))
jk_w_2 = np.flip(np.asarray(jk_w_2))

# Make Fiducial RM Cuts
cuts_mask_r = (z_r_2 >= 0.2)*(z_r_2 <= 0.55)*(lam_r_2 >= 20.)*(lam_r_2 <= 100.)
ra_r_f = ra_r_2[cuts_mask_r]
dec_r_f = dec_r_2[cuts_mask_r]
z_r_f = z_r_2[cuts_mask_r]
lam_r_f = lam_r_2[cuts_mask_r]

print("RM Fiducial Population Size: " + str(len(z_r_f)))

# Make JKs for Redmapper
random = pf.open('/home/ajamsellem/Fits_files/Master_cluster_random_overlapped.fits')[1].data
ra_rand = random['ra']
dec_rand = random['dec']

jk_r_f = make_jk(ra_rand, dec_rand, ra_r_f, dec_r_f, 100, 1, 0, True, 500, 1e-05, 100)

c1 = pf.Column(name='RA', format='E', array=ra_r_f)
c2 = pf.Column(name='DEC', format='E', array=dec_r_f)
c3 = pf.Column(name='Z', format='E', array=z_r_f)
c4 = pf.Column(name='LAMBDA', format='E', array=lam_r_f)
c5 = pf.Column(name='JK', format='E', array=jk_r_f)

CC = [c1,c2,c3,c4,c5]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(ra_r_f))
hdu.writeto('/home/ajamsellem/Fits_files/RM_clusters_fiducial_by_mass.fits')

# Make Fiducial WaZP Cuts
matched_idx = np.where(cuts_mask_r == True)[0]
ra_w_f = ra_w_2[matched_idx]
dec_w_f = dec_w_2[matched_idx]
z_w_f = z_w_2[matched_idx]
ngal_w_f = ngal_w_2[matched_idx]
jk_w_f = jk_w_2[matched_idx]

print("WaZP Fiducial Population Size: " + str(len(z_w_f)))

c6 = pf.Column(name='RA', format='E', array=ra_w_f)
c7 = pf.Column(name='DEC', format='E', array=dec_w_f)
c8 = pf.Column(name='Z', format='E', array=z_w_f)
c9 = pf.Column(name='LAMBDA', format='E', array=ngal_w_f)
c10 = pf.Column(name='JK', format='E', array=jk_w_f)

CC2 = [c6, c7, c8, c9, c10]
hdu = pf.BinTableHDU.from_columns(CC2, nrows=len(ra_w_f))
hdu.writeto('/home/ajamsellem/Fits_files/WaZP_clusters_fiducial_by_mass.fits')
