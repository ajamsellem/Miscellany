"""
Usage: python process_zbins.py  Output_Filename Number_of_Jackknifes Number_of_Z_bins Results_Directory

Secondary Usage: python process_zbins.py  Output_Filename Number_of_Jackknifes Number_of_Z_bins Results_Directory Used_Area Correct_Area
Clusters_Data_Filename Minimum_Z_Value Maximum_Z_Value Minimum_Richness_Value Maximum_Richness_Value

Example python process_zbins.py Redmapper_2 100 7 . 1249.123412341234 1793.123412341324 RM_jk.fits 0.2 0.55 20. 100.
"""


import numpy as np
import os
import sys
from astropy.cosmology import FlatLambdaCDM
import astropy.io.fits as pf
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

name = sys.argv[1]
Njk = int(sys.argv[2])
Nz = int(sys.argv[3])
result_dir = sys.argv[4]

# Apply an area correction. Area had been hard-coded, but it should be based on the area of the mask
if len(sys.argv) != 5:
    # Hard-coded area
    tot_area_old = float(sys.argv[5])
    # Area from the mask
    tot_area_new = float(sys.argv[6])
    # Cluster Data – For purposes of cuts
    clusters = pf.open('/Users/arielamsellem/Desktop/Research/splashback_codes_master/Fits_files/' + sys.argv[7])[1].data
    zmin = float(sys.argv[8])
    zmax = float(sys.argv[9])
    lambmin = float(sys.argv[10])
    lambmax = float(sys.argv[11])
    JK = clusters['JK']
    # Galaxy Random Data – For purposes of JK determination
    galaxy_randoms = pf.open('/Users/arielamsellem/Desktop/Research/splashback_codes_master/Fits_files/galaxy_rand.fits')[1].data
    JK_gal_ran = galaxy_randoms['JK']
    N_allgal = len(JK_gal_ran)

R = np.load(result_dir+'/Sigmag_0_0.npz')['R']
Sg = []
DD_full = []
RR_full = []
DR_full = []
RD_full = []
clusterless_bins = 0.

for i in range(Njk):
    #print("Step: " + str(i))
    xi = []
    w = []
    DD = []
    RR = []
    RD = []
    DR = []
    ave_dens = []
    nclust = []
    ngal = []
    for j in range(Nz):
        if os.path.exists(result_dir+'/Sigmag_'+str(i)+'_'+str(j)+'.npz') == True:
            infile = np.load(result_dir+'/Sigmag_'+str(i)+'_'+str(j)+'.npz')
            #if i == 3.:
            #    print(infile['DD'])
            xi.append(infile['xi'])
            w.append(infile['w'])
            DD.append(infile['DD'])
            RR.append(infile['RR'])
            DR.append(infile['DR'])
            RD.append(infile['RD'])
            if len(sys.argv) != 5:
                # Mask values based on data
                mask = (JK!=i)*(clusters['Z']>=zmin)*(clusters['Z']<zmax)*(clusters['LAMBDA']>=lambmin)*(clusters['LAMBDA']<lambmax)
                Z = clusters['Z'][mask]
                n1 = np.histogram(Z, range=(zmin,zmax), bins=Nz)
                zmid = (n1[1][1:]+n1[1][:-1])/2

                # Mask values based on jackknives
                mask = (JK_gal_ran!=i)
                ra_ran = galaxy_randoms['RA'][mask]
                N_jkgal = len(ra_ran)

                # Calculate old and new areas
                area_new = tot_area_new*(N_jkgal*1.0/N_allgal)
                area_old = tot_area_old*(N_jkgal*1.0/N_allgal)

                # Calculate area factor, and multiply by the average density
                area_Mpch_new = area_new*(np.pi/180.)**2*(cosmo.comoving_distance(zmid[j]).value)**2
                area_Mpch_old = area_old*(np.pi/180.)**2*(cosmo.comoving_distance(zmid[j]).value)**2
                ave_density = infile['ave_dens']*(1.0/area_Mpch_new)*area_Mpch_old

            else:
                ave_density = infile['ave_dens']

            ave_dens.append(ave_density)
            nclust.append(infile['nclust'])
            ngal.append(infile['ngal'])

        else:
            print("File of JK " + str(i) + " Zbin " + str(j) + " had no clusters.")
            clusterless_bins += 1.

    xi = np.array(xi)
    w  = np.array(w)
    DD = np.array(DD)
    #print(DD)
    RR = np.array(RR)
    RD = np.array(RD)
    DR = np.array(DR)
    ave_dens = np.array(ave_dens)
    nclust = np.array(nclust)
    ngal   = np.array(ngal)

    dens = np.sum(ave_dens*nclust)/np.sum(nclust)
    print("xi: " + str(xi))
    print("w: " + str(w))
    mean = np.sum(xi*w, axis=0)/np.sum(w, axis=0)*dens

    Sg.append(mean)
    DD_full.append(DD)
    RR_full.append(RR)
    DR_full.append(DR)
    RD_full.append(RD)

#for i in range(0,100):
#    print(np.sum(np.array(DD_full) , axis = 1)[i])
DD_full = np.sum(np.sum(np.array(DD_full) , axis = 0), axis = 0)
#print(DD_full)
RR_full = np.sum(np.sum(np.array(RR_full) , axis = 0), axis = 0)
DR_full = np.sum(np.sum(np.array(DR_full) , axis = 0), axis = 0)
RD_full = np.sum(np.sum(np.array(RD_full) , axis = 0), axis = 0)

#print(np.sum(np.sum(DD_full, axis = 0), axis = 0))
#print(R)
#print(np.sum(DD_full))
#print(nclust)
#print(np.sum(ngal))
Sg = np.array(Sg)
C = np.cov(Sg.T)*(len(Sg)-1)
Sg_mean = np.mean(Sg, axis=0)
Sg_sig = np.sum((Sg-Sg_mean)**2, axis=0)**0.5/len(Sg)**0.5*(len(Sg)-1)**0.5
if clusterless_bins != 0.:
    print()
    print("Number of bins that had 0 clusters: " + str(int(clusterless_bins)))
    print("Please check the logged error files from the splashback measurement.")

#np.savez(result_dir+'/splashback_cov_'+str(name)+'.npz', cov=C,
#         r_data=R, sg_mean=Sg_mean, sg_sig=Sg_sig)
#         h=meta['h'],
#         R_unit=meta['R_unit'],
#         Sigmag_unit=meta['Sigmag_unit'],
#         DeltaSigma_unit=meta['DeltaSigma_unit'],
#         cat=meta['cat'][id_meta],
#         mean_z=meta['mean_z'][id_meta],
#         mean_lambda=meta['mean_lambda'][id_meta],
#         mean_rlambda=meta['mean_rlambda'][id_meta],
#         mean_rlambda_com=meta['mean_rlambda_com'][id_meta],
#         n_clust=meta['n_clust'][id_meta])

#np.savez(result_dir+'/DD_'+str(name)+'.npz', DD=DD_full, RR=RR_full, DR=DR_full, RD=RD_full, r_data=R,)
print(DD_full)
print(Sg_mean)
