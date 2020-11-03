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


gal_data = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_jk_dependencies_1.fits')[1].data
ra = gal_data['ra']
dec = gal_data['dec']

ran_data = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_jk_dependencies_2.fits')[1].data

ra_rand = ran_data['RA_RAND']
dec_rand = ran_data['DEC_RAND']

JK_ran, JK = make_jk(ra_rand, dec_rand, ra, dec, 100, 1000, 1, True, 500, 1e-05, 100)
print(len(JK_ran))
print(len(JK))

del ra, dec, ra_rand, dec_rand

# save columns to a Master Fits file
c1 = pf.Column(name='JK', format='E', array=JK)
CC = [c1]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(JK))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_jk.fits')

del c1, CC, JK, hdu

c2 = pf.Column(name='JK_RAN', format='E', array=JK_ran)
CC = [c2]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(JK_ran))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_jk_random.fits')
