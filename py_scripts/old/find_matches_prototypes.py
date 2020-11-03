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

'''
def find_matches_brute_force(ra_r, dec_r, z_r, ra_u, dec_u, z_u, R, zmin, zmax):

    # Calculate an average redshift from the aggregate of redshifts in both catalogues
    z_tot = np.append(z_r,z_u)
    z = np.average(z_tot[np.logical_and(z_tot>zmin, z_tot<zmax)])

    # Zip up the ra and dec of each catalogue into one array for each catalogue
    zip_radec_r = list(zip(ra_r, dec_r))
    zip_radec_u = list(zip(ra_u, dec_u))
    zip_radec_r = np.array(zip_radec_r)
    zip_radec_u = np.array(zip_radec_u)


    # Convert dec into phi and all degrees into radians
    dec_r = 90. - dec_r
    dec_u = 90. - dec_u
    dec_r = dec_r*np.pi/180
    dec_u = dec_u*np.pi/180
    ra_r = ra_r*np.pi/180
    ra_u = ra_u*np.pi/180

    H_0 = 70 #km/s*Mpc (approximately – not using h)

    # Calculate average distance to clusters based on redshift (very rough!)
    D = (((z+1)**2 - 1)/((z+1)**2 + 1)) * c/H_0  # Mpc

    # Calculate angular separation (theta) tolerated as error
    tan_theta = R/D
    theta_calc = np.arctan(tan_theta) # radians
    #print("Measured Angular Sep (deg): "+ str(theta_calc))

    # Find number of clusters within the allowed value of theta
    idx_matches = []
    duplicates = 0.

    for i, val_r in enumerate(zip_radec_r):
        # Cosines and sines of declinations
        c_dec_r = np.cos(val_r[1])
        c_dec_u = np.cos(zip_radec_u[:,1])
        s_dec_r = np.sin(val_r[1])
        s_dec_u = np.sin(zip_radec_u[:,1])

        # Calculate cos(theta) using the equation at https://en.wikipedia.org/wiki/Angular_distance (but sin and cos are switched on declinations)
        cos_theta = (c_dec_r*c_dec_u) + (s_dec_r*s_dec_u) * np.cos(val_r[0]-zip_radec_u[:,0])
        if i == 1292:
            print(cos_theta[2620])

        # Because of python floating, some values close to 1 have to made equal to 1
        close_vals = np.where(cos_theta + 0.00000006 == 1.)[0]
        cos_theta[close_vals] = 1.

        # Sometimes python floating causes cos_theta to go to 1.0000001, and I assume that's really a 1.
        # Same thing can happen around -1
        # To avoid a runtime error, we set those values to 1 and -1 respectively
        if len(np.where(cos_theta > 1.)[0]) != 0.:
            cos_theta[np.where(cos_theta > 1.)[0][0]] = 1.
        if len(np.where(cos_theta < -1.)[0]) != 0.:
            cos_theta[np.where(cos_theta < -1.)[0][0]] = -1.

        # Theta based on ra and dec of each point
        theta_meas = np.arccos(cos_theta) # radians


        # Checking why some R-matched clusters are not in the original matching
        #if theta_meas[2819] == 0.:
        #    print(i)


        #print("Measured Angular Sep (deg): "+ str(theta_meas))

        # If statement exists so as not to return an error if there are no matches at all
        if len(np.where(theta_meas <= theta_calc)[0]) != 0:
            # Find index in _u array where a match occurs, and save those indices in idx_matches
            match = np.where(theta_meas <= theta_calc)[0][0]

            if len(np.where(theta_meas <= theta_calc)[0]) > 1.:
                dup = len(np.where(theta_meas <= theta_calc)[0])
            else:
                dup = 0.
            duplicates += dup
            idx_matches.append(match)

        #for index in np.setdiff1d(idx_w, matches_act):
            #print(theta_meas[index])
    print('')
    print(len(idx_matches))
    print("Number of duplicates: " + str(duplicates))
    return idx_matches

N_matches = find_matches(np.array([0., 15.]), np.array([0., 16.]), np.array([0.3, 0.4]), np.array([15., 0.]),  np.array([16.00001, 0.]), np.array([0.3, 0.4]), 1, 0.2, 0.55)

def find_matches_a(ra_r, dec_r, z_r, ra_u, dec_u, z_u, R, zmin, zmax):

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

    tot_match_idx = np.ndarray([0])
    dup_idx = np.ndarray([0])
    duplicates = 0.

    for i in range(len(ra_u)):
        c1 = SkyCoord(ra_r*u.degree, dec_r*u.degree, frame='icrs')
        c2 = SkyCoord(ra_u[i]*u.degree, dec_u[i]*u.degree, frame='icrs')
        sep = c1.separation(c2)
        separation = sep.degree
        match_index = np.where(separation <= allowed_ang_sep)[0]
        if (len(match_index) == 1.):
            tot_match_idx = np.append(tot_match_idx, np.where(separation <= allowed_ang_sep)[0][0])
        elif len(match_index) >= 2.:
            tot_match_idx = np.append(tot_match_idx, np.where(separation <= allowed_ang_sep)[0][0])
            dup_idx = np.append(dup_idx, match_index)
            duplicates += len(np.where(separation <= allowed_ang_sep)[0]) - 1


    return R, allowed_ang_sep, tot_match_idx, duplicates
'''

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

    # Decide whether we are matching to WaZP clusters or RM clusters

    if match_to == 'RM':
        for i in range(len(ra_u)):
            c1 = SkyCoord(ra_r*u.degree, dec_r*u.degree, frame='icrs')
            c2 = SkyCoord(ra_u[i]*u.degree, dec_u[i]*u.degree, frame='icrs')
            sep = c1.separation(c2)
            # Find indices where the ang separation is less than the allowed ang separation
            separation = sep.degree
            match_index = np.where(separation <= allowed_ang_sep)[0]

            # If there exists multiple allowed indices, find the index with the minimum separation
            if len(match_index) >= 2.:
                min_sep = np.amin(separation[match_index])
                match_index = np.where(separation == min_sep)[0]
            if len(match_index) == 1.:
                # If the index is not already a match, add it to the list of matched indices
                if np.isin(match_index, tot_match_idx)[0] == False:
                    tot_match_idx = np.append(tot_match_idx, match_index)
                    tot_separations = np.append(tot_separations, separation[match_index[0]])
                    loop_val = np.append(loop_val, i)
                # If the index is already a match, check which registered match has the minimum separation
                elif np.isin(match_index, tot_match_idx)[0] == True:
                    duplicate_idx = np.where(tot_match_idx == match_index[0])[0][0]
                    other_min_sep = tot_separations[duplicate_idx]
                    if separation[match_index[0]] < other_min_sep:
                        tot_separations[duplicate_idx] = separation[match_index[0]]
                        redo = np.append(redo, loop_val[duplicate_idx])

        redo = redo.astype(int)
        for i in redo:
            c1 = SkyCoord(ra_r*u.degree, dec_r*u.degree, frame='icrs')
            c2 = SkyCoord(ra_u[i]*u.degree, dec_u[i]*u.degree, frame='icrs')
            sep = c1.separation(c2)
            # Find indices where the ang separation is less than the allowed ang separation
            separation = sep.degree
            match_index = np.where(separation <= allowed_ang_sep)[0]

            # If there exists multiple allowed indices, find the index with the minimum separation
            if len(match_index) >= 2.:
                min_sep = np.amin(separation[match_index])
                match_index = np.where(separation == min_sep)[0]
            if len(match_index) == 1.:
                # If the index is not already a match, add it to the list of matched indices
                if np.isin(match_index, tot_match_idx)[0] == False:
                    print('found one')
                    tot_match_idx = np.append(tot_match_idx, match_index)

    elif match_to == 'W':
        for j in range(len(ra_r)):
            c1 = SkyCoord(ra_r[j]*u.degree, dec_r[j]*u.degree, frame='icrs')
            c2 = SkyCoord(ra_u*u.degree, dec_u*u.degree, frame='icrs')
            sep = c1.separation(c2)
            # Find indices where the ang separation is less than the allowed ang separation
            separation = sep.degree
            match_index = np.where(separation <= allowed_ang_sep)[0]

            # If there exists multiple allowed indices, find the index with the minimum separation
            if len(match_index) >= 2.:
                min_sep = np.amin(separation[match_index])
                match_index = np.where(separation == min_sep)[0]
            if len(match_index) == 1.:
                # If the index is not already a match, add it to the list of matched indices
                if np.isin(match_index, tot_match_idx)[0] == False:
                    tot_match_idx = np.append(tot_match_idx, match_index)
                    tot_separations = np.append(tot_separations, separation[match_index[0]])
                    loop_val = np.append(loop_val, j)
                # If the index is already a match, check which registered match has the minimum separation
                elif np.isin(match_index, tot_match_idx)[0] == True:
                    duplicate_idx = np.where(tot_match_idx == match_index[0])[0][0]
                    other_min_sep = tot_separations[duplicate_idx]
                    if separation[match_index[0]] < other_min_sep:
                        tot_separations[duplicate_idx] = separation[match_index[0]]
                        redo = np.append(redo, loop_val[duplicate_idx])

        redo = redo.astype(int)
        for j in redo:
            c1 = SkyCoord(ra_r*u.degree, dec_r*u.degree, frame='icrs')
            c2 = SkyCoord(ra_u[j]*u.degree, dec_u[j]*u.degree, frame='icrs')
            sep = c1.separation(c2)
            # Find indices where the ang separation is less than the allowed ang separation
            separation = sep.degree
            match_index = np.where(separation <= allowed_ang_sep)[0]

            # If there exists multiple allowed indices, find the index with the minimum separation
            if len(match_index) >= 2.:
                min_sep = np.amin(separation[match_index])
                match_index = np.where(separation == min_sep)[0]
            if len(match_index) == 1.:
                # If the index is not already a match, add it to the list of matched indices
                if np.isin(match_index, tot_match_idx)[0] == False:
                    print('found one')
                    tot_match_idx = np.append(tot_match_idx, match_index)

    tot_match_idx = tot_match_idx.astype(int)
    return R, allowed_ang_sep, tot_match_idx
