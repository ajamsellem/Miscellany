# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ajamsellem/.local/bin')
sys.path.insert(0, '/home/ajamsellem/kmeans_radec')
import numpy as np
import astropy.io.fits as pf
import healpy as hp
import astropy.io.fits as fits
import scipy 
import kmeans_radec

# A necessary function from Chihway
def make_jk(ra_ran, dec_ran, ra, dec, N=100, dilute_factor=1, rand_out=1, large_mem=True, maxiter=500, tol=1e-05, seed=100, centers=False):
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

        for i in range(N-1):
            #print i
            JK = np.concatenate((JK, km.find_nearest(RADEC[i*int(Ntotal/N):(i+1)*int(Ntotal/N)])), axis=0)
            print(np.unique(JK))

            if rand_out==1:
                print(i)
                JK_ran = np.concatenate((JK_ran, km.find_nearest(RADEC_ran[i*int(Ntotal_ran/N):(i+1)*int(Ntotal_ran/N)])), axis=0)

        JK = np.concatenate((JK, km.find_nearest(RADEC[(N-1)*int(Ntotal/N):])), axis=0)
        if rand_out==1:
            JK_ran = np.concatenate((JK_ran, km.find_nearest(RADEC_ran[(N-1)*int(Ntotal_ran/N):])), axis=0)
        print('len of random', len(ra_ran))
        print('len of JK Random', len(JK_ran))
    else:
        JK = km.find_nearest(RADEC)
        if rand_out==1:
            JK_ran = km.find_nearest(RADEC_ran)

    if centers==True:
        #Saving the kmeans centers
        assert km.converged > 0, 'Kmeans did not converge! Try more iterations.'
        print('Saving Jackknife Centers...')
        np.savetxt('/project2/chihway/sims/buzzard/y1_gal_member/jk_centers',km.centers)
    
    if rand_out==1:    
        return JK_ran, JK
    else:
        return JK


gal_data = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/galaxy.fits')[1].data
ra = gal_data['ra']
dec = gal_data['dec']

ran_data = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/galaxy_rand.fits')[1].data
ra_rand = ran_data['RA']
dec_rand = ran_data['DEC']

JK_ran, JK = make_jk(ra_rand, dec_rand, ra, dec, 30, 50, 1, True, 500, 1e-05, 100, True)
print(len(JK_ran))
print(len(JK))

# save columns to a Master Fits file
ids = gal_data['ID']
mag = gal_data['MAG_AUTO_I']

c1 = pf.Column(name='ID', format='K', array=ids)
c2 = pf.Column(name='RA', format='E', array=ra)
c3 = pf.Column(name='DEC', format='E', array=dec)
c4 = pf.Column(name='MAG_AUTO_I', format='E', array=mag)
c5 = pf.Column(name='JK', format='E', array=JK)
CC = [c1,c2,c3,c4,c5]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(JK))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/galaxy_1.fits', overwrite=True)
x = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/galaxy_1.fits')[1].data
print(x.columns)
del x 

del c1, CC, JK, hdu

c1 = pf.Column(name='RA', format='E', array=ra_rand)
c2 = pf.Column(name='DEC', format='E', array=dec_rand)
c3 = pf.Column(name='JK', format='E', array=JK_ran)
CC = [c1,c2,c3]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(JK_ran))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/galaxy_rand_1.fits', overwrite=True)

del c2, CC, JK_ran, hdu
del ra, dec, ra_rand, dec_rand

clus_data = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/RM_to_use.fits')[1].data
ra = clus_data['RA']
dec = clus_data['DEC']

clus_ran = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/Master_cluster_random_overlapped.fits')[1].data
ra_rand = clus_ran['RA']
dec_rand = clus_ran['DEC']

JK_ran, JK = make_jk(ra_rand, dec_rand, ra, dec, 30, 10, 1, True, 500, 1e-05, 100, False)
print(len(JK_ran))
print(len(JK))

# save columns to a Master Fits file
z   = clus_data['Z']
lam = clus_data['LAMBDA']

c1 = pf.Column(name='RA', format='E', array=ra)
c2 = pf.Column(name='DEC', format='E', array=dec)
c3 = pf.Column(name='Z', format='E', array=z)
c4 = pf.Column(name='LAMBDA', format='E', array=lam)
c5 = pf.Column(name='JK', format='E', array=JK)

CC = [c1,c2,c3,c4,c5]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(JK))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/RM_to_use_1.fits', overwrite=True)

del c1, CC, JK, hdu
z_rand   = clus_ran['Z']
lam_rand = clus_ran['LAMBDA']

c1 = pf.Column(name='RA', format='E', array=ra_rand)
c2 = pf.Column(name='DEC', format='E', array=dec_rand)
c3 = pf.Column(name='Z', format='E', array=z_rand)
c4 = pf.Column(name='LAMBDA', format='E', array=lam_rand)
c5 = pf.Column(name='JK', format='E', array=JK_ran)
CC = [c1,c2,c3,c4,c5]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(JK_ran))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/Master_cluster_random_overlapped_1.fits', overwrite=True)
