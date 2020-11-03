# -*- coding: utf-8 -*-

import numpy as np
import astropy.io.fits as pf
import sys

N = 0
n = []
for i in range(10):
    ra_p = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_random_galaxy_no_jk_' + str(i)  + '.fits')[1].data['ra']
    N += len(ra_p)
    n.append(len(ra_p))
    print(n)

print(N)
n = np.array(n)
NN = 0
RA_p = np.zeros(N)
for i in range(10):
    ra_p = pf.open('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_random_galaxy_no_jk_' + str(i)  + '.fits')[1].data['ra']
    RA_p[NN:NN+n[i]] = ra_p
    print(RA_p.shape)
    NN += len(ra_p)
print(RA_p)
print(RA_p.shape)

length = len(RA_p)
print()
print(RA_p.nbytes)
RA_p = pf.Column(name='RA', format='E', array=RA_p)

CC = [RA_p]
hdu = pf.BinTableHDU.from_columns(CC, nrows=length)
hdu.writeto('/project2/chihway/sims/buzzard/y1_gal_member/buzzard_random_galaxy_no_jk.fits')
