from astropy.coordinates import SkyCoord
## jackknife centers
center_data = np.loadtxt('/data3/taeshin/data/kmeans_centers/kmeans_centers_npix100_desy3.dat',unpack=True)
## jk centers position
pos_jk = SkyCoord(ra=center_data[0]*u.deg,dec=center_data[1]*u.deg)
## calculate distances from THIS jk center to other jk centers
jk_dist = pos_jk[jkid].separation(pos_jk).degree
## masking out jk patches that are too far from the clusters
ind = jk_dist < MAXIMUM_ANGLE_IN_DEGREE + 2*MAXIMUM_DISTANCE_BETWEEN_ADJACENT_JK_PATCHES  ### MAXIMUM_DISTANCE_BETWEEN_ADJACENT_JK_PATCHES is about 5 degrees
jk_neigh = np.arange(NUMBER_OF_JK_PATCHES)[ind]
ind_jk_src = np.in1d(JK_galaxy,jk_neigh)
galaxies = galaxies[ind_jk_src]
