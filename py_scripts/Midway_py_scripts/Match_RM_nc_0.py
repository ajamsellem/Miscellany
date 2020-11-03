# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ajamsellem/.local/bin')
from astropy.io import fits
import numpy as np
import astropy.io.fits as pf
from scipy.constants import c
import math
from decimal import *
c /= 1000
from astropy import units as u
from astropy.coordinates import SkyCoord
getcontext().prec = 6

# READ IN DATA
# Read in Redmapper data
RM = pf.open('/home/ajamsellem/Fits_files/redmapper.fit')[1].data

ra_r = RM['RA']
dec_r = RM['dec']
z_r = RM['z']

print("Number of RM Clusters before cuts: " + str(len(ra_r)))

# Read in WaZP data
W = pf.open('/home/ajamsellem/Fits_files/WaZP_jk.fits')[1].data
print('Number of WaZP Clusters before cuts: ' + str(len(W)))

# The cuts on WaZP clusters in redshift and richness/ngal
idx = np.where((W['Z']>=0.2) & (W['Z']<=0.55) & (W['lambda']>=30.) & (W['lambda']<=110.) )[0]
#idx = np.where((W['Z'] > 0.))[0]
print('Number of WaZP Clusters after cuts: ' + str(len(idx)))

ra_u = []
dec_u = []
z_u = []

for val in idx:
    ra_u.append(W['ra'][val])
    dec_u.append(W['dec'][val])
    z_u.append(W['z'][val])

ra_u = np.asarray(ra_u)
dec_u = np.asarray(dec_u)
z_u = np.asarray(z_u)

# FUNCTIONS
def find_matches(ra_r, dec_r, z_r, ra_u, dec_u, z_u, R, zmin, zmax, match_to):
    # Calculate an average redshift from the aggregate of redshifts in both catalogues
    z_tot = np.append(z_r,z_u)
    z = np.average(z_tot[np.logical_and(z_tot>zmin, z_tot<zmax)])
    
    H_0 = 70 #km/s*Mpc (approximately – not using h)
    
    # Calculate average distance to clusters based on redshift (very rough!)
    D = (((z+1)**2 - 1)/((z+1)**2 + 1)) * c/H_0  # Mpc
    
    # Calculate angular separation (theta) tolerated as error
    tan_theta = R/D     # R in Mpc
    allowed_ang_sep = np.arctan(tan_theta) # radians
    allowed_ang_sep *= 180./np.pi # degrees
    #print("Measured Angular Sep (deg): "+ str(allowed_ang_sep))
    #print('')
    
    # Convert all data to lists and decimals in order to preserve the precision of the data
    ra_r = ra_r.tolist()
    dec_r = dec_r.tolist()
    ra_u = ra_u.tolist()
    dec_u = dec_u.tolist()
    
    ra_r = [Decimal(i) for i in ra_r]
    dec_r = [Decimal(i) for i in dec_r]
    ra_u = [Decimal(i) for i in ra_u]
    dec_u = [Decimal(i) for i in dec_u]

    # Initialize storage variables
    tot_match_idx = np.asarray([])
    tot_separations = np.asarray([])
    loop_val = np.asarray([])
    redo = np.asarray([])
    ra_m = np.asarray([])
    dec_m = np.asarray([])
    # Create a astropy.coordinates.angles.Angle object of separation zero and convert it to a string
    some_coordinate = SkyCoord(5.*u.degree, 60.*u.degree, frame='icrs')
    zero_sep = some_coordinate.separation(some_coordinate)
    zero_sep = str(zero_sep)

    # Decide whether we are matching to WaZP clusters or RM clusters
    if match_to == 'RM':
        for i in range(len(ra_u)):
            c1 = SkyCoord(ra_r*u.degree, dec_r*u.degree, frame='icrs')
            c2 = SkyCoord(ra_u[i]*u.degree, dec_u[i]*u.degree, frame='icrs')
            sep = c1.separation(c2)
            sep = sep.to_string()
                
            # Find indices where the ang separation is zero
            match_index = np.where(sep == str(zero_sep))[0]
            # If the index is not already a match, add it to the list of matched indices
            if len(match_index) == 1. and (np.isin(match_index, tot_match_idx)[0] == False):
                tot_match_idx = np.append(tot_match_idx, match_index)
                    
                        
    elif match_to == 'W':
        for j in range(len(ra_r)):
            c1 = SkyCoord(ra_r[j]*u.degree, dec_r[j]*u.degree, frame='icrs')
            c2 = SkyCoord(ra_u*u.degree, dec_u*u.degree, frame='icrs')
            sep = c1.separation(c2)
            sep = sep.to_string()
                
            # Find indices where the ang separation is zero
            match_index = np.where(sep == str(zero_sep))[0]
            # If the index is not already a match, add it to the list of matched indices
            if len(match_index) == 1.:
                if np.isin(match_index, tot_match_idx)[0] == False:
                    tot_match_idx = np.append(tot_match_idx, match_index) 
                    
    tot_match_idx = tot_match_idx.astype(int)
    return R, allowed_ang_sep, tot_match_idx

R, allowed_ang_sep, matches = find_matches(ra_r, dec_r, z_r, ra_u, dec_u, z_u, 0., 0.2, 0.55, 'RM')
print(R, allowed_ang_sep, len(matches))
Rw, allowed_ang_sepw, matchesw = find_matches(ra_r, dec_r, z_r, ra_u, dec_u, z_u, 0., 0.2, 0.55, 'W')
print(Rw, allowed_ang_sepw, len(matchesw))
print('')
