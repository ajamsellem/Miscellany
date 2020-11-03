# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/ajamsellem/.local/bin')
sys.path.insert(0, './kmeans_radec')
import numpy as np
import astropy.io.fits as pf
import healpy as hp
import kmeans_radec

i = int(sys.argv[1])

# Some necessary functions from Chihway
def radec2thetaphi(ra, dec, nside):
    """
    Convert RA DEC in degrees to THETA and PHI in Healpix 
    convention. 
    """

    theta = (90.0 - dec)*np.pi/180.0
    phi = ra*np.pi/180.0
    return theta, phi

def make_mask(ra, dec, nmin=1, nside=4096):
    """
    Take RA, DEC, build a binary mask just by assigning 
    1 to pixels with count>=nmin and 0 otherwise. Mask 
    is in Healpix format with assigned nside. 
    """
    mask = np.zeros(hp.nside2npix(nside))
    theta, phi = radec2thetaphi(ra, dec, nside)
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    for i in range(len(pix)):
        mask[pix[i]] += 1
    mask[mask>=nmin] = 1

    return mask

data = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_galaxies_no_jk.fits')[1].data
ra = data['ra']
dec = data['dec']

mask = make_mask(ra, dec, 1, 4096)

del data
del ra
del dec

data_rand_ra = np.load('/project2/chihway/sims/buzzard/y1_gal_member/ra_random' + str(i) + '.npz', mmap_mode = 'r')
ra_rand = data_rand_ra['arr_0']
del data_rand_ra

data_rand_dec = np.load('/project2/chihway/sims/buzzard/y1_gal_member/dec_random' + str(i) + '.npz',mmap_mode = 'r')
v = data_rand_dec['arr_0']
del data_rand_dec

decmin = -68.
decmax = -38.
nside = 4096

vmin = np.cos((90.0+decmin)/180.*np.pi)
vmax = np.cos((90.0+decmax)/180.*np.pi)
v *= (vmax-vmin)
v += vmin
dec_rand = np.arccos(v)
del v
np.rad2deg(dec_rand,dec_rand)
dec_rand -= 90.0
theta_rand = (90.0 - dec_rand)*np.pi/180.
phi_rand = ra_rand*np.pi/180.
pix_rand = hp.ang2pix(nside, theta_rand, phi_rand, nest=False)
del theta_rand
del phi_rand
goodm, = np.where(mask[pix_rand]==1)
ra_rand = ra_rand[goodm]
dec_rand = dec_rand[goodm]

ids_rand, mag_rand  = np.ones(len(ra_rand)), np.ones(len(ra_rand))
print(len(ids_rand))

# save columns to a galaxy random Fits file
c6 = pf.Column(name='ID', format='E', array=ids_rand)
c7 = pf.Column(name='RA', format='E', array=ra_rand)
c8 = pf.Column(name='DEC', format='E', array=dec_rand)
c9 = pf.Column(name='MAG_I', format='E', array=mag_rand)

CC = [c6,c7,c8,c9]
hdu = pf.BinTableHDU.from_columns(CC, nrows=len(ra_rand))
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_random_galaxy_no_jk_' + str(i)  + '.fits')

new = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_random_galaxy_no_jk_' + str(i)  + '.fits')[1].data
print(len(new['mag_i']))
