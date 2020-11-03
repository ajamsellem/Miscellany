# -*- coding: utf-8 -*-
 #!/usr/local/bin/python3

import numpy as np
import sys
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

#area = 4680.2835147815285
area  = 4735
Y3_gal_bin   = [4588446, 7727861, 11812582, 16857851, 22854867, 29754111, 37521078]
buzz_gal_bin = [4592113, 8174313, 12995769, 19058615, 26322296, 34712923, 44158210]

nclust  = [1181, 1332, 1477, 2015, 2268, 2243, 2413]

ave_dens_3 = []
ave_dens_b = []

for i, zmid in enumerate(np.linspace(0.225, 0.525, 7)):
    area_Mpch = area*(np.pi/180.)**2*(cosmo.comoving_distance(zmid).value)**2
    ave_density_3 = Y3_gal_bin[i]*1.0/area_Mpch
    ave_density_b = buzz_gal_bin[i]*1.0/area_Mpch

    ave_dens_3.append(ave_density_3)
    ave_dens_b.append(ave_density_b)

ave_dens_3 = np.array(ave_dens_3)
ave_dens_b = np.array(ave_dens_b)
nclust     = np.array(nclust)

dens_3     = np.sum(ave_dens_3*nclust)/np.sum(nclust)
dens_b     = np.sum(ave_dens_b*nclust)/np.sum(nclust)


print(dens_3)
print(dens_b)
