# -*- coding: utf-8 -*-

import sys
import numpy as np

i = int(sys.argv[1])

x = np.linspace(0,10,11).astype(int)*488366997
idx_min = x[i]
idx_max = x[i+1]

ra_rand_data = np.load('/project2/chihway/sims/buzzard/y1_gal_member/ra_rand.npz', mmap_mode = 'r')
ra_rand = ra_rand_data['arr_0'][idx_min:idx_max]

#dec_rand_data = np.load('/project2/chihway/sims/buzzard/y1_gal_member/dec_rand.npz', mmap_mode = 'r')
#v = dec_rand_data['arr_0'][idx_min:idx_max]

np.savez('/project2/chihway/sims/buzzard/y1_gal_member/ra_random' + str(i), ra_rand)
#np.savez('/project2/chihway/sims/buzzard/y1_gal_member/dec_random' + str(i), v)
