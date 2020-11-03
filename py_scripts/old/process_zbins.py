
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

if len(sys.argv) != 4:
    tot_area_old = 1392.13304984
    tot_area_new = float(sys.argv[5])
    clusters = pf.open('/Users/arielamsellem/Desktop/Research/splashback_codes_master/Fits_files/' + sys.argv[6])[1].data
    zmin = float(sys.argv[7])
    zmax = float(sys.argv[8])
    lambmin = float(sys.argv[9])
    lambmax = float(sys.argv[10])

#meta = np.load('splashback_meta_info.npz')
R = np.load(result_dir+'/Sigmag_0_0.npz')['R']
Sg = []

for i in range(Njk):
    print("Step: " + str(i))
    xi = []
    w = []
    ave_dens = []
    nclust = []
    for j in range(Nz):
        infile = np.load(result_dir+'/Sigmag_'+str(i)+'_'+str(j)+'.npz')
        xi.append(infile['xi'])
        w.append(infile['w'])
        if len(sys.argv) != 4:

            JK = clusters['JK']
            mask = (JK!=i)*(clusters['Z']>=zmin)*(clusters['Z']<zmax)*(clusters['LAMBDA']>=lambmin)*(clusters['LAMBDA']<lambmax)
            Z = clusters['Z'][mask]
            n1 = np.histogram(Z, range=(zmin,zmax), bins=Nz)
            zmid = (n1[1][1:]+n1[1][:-1])/2

            galaxy_randoms = pf.open('/Users/arielamsellem/Desktop/Research/splashback_codes_master/Fits_files/galaxy_rand.fits')[1].data
            JK_gal_ran = galaxy_randoms['JK']
            N_allgal = len(JK_gal_ran)
            mask = (JK_gal_ran!=i)
            ra_ran = galaxy_randoms['RA'][mask]
            N_jkgal = len(ra_ran)

            area_new = tot_area_new*(N_jkgal*1.0/N_allgal)
            area_old = tot_area_old*(N_jkgal*1.0/N_allgal)

            area_Mpch_new = area_new*(np.pi/180.)**2*(cosmo.comoving_distance(zmid[j]).value)**2
            area_Mpch_old = area_old*(np.pi/180.)**2*(cosmo.comoving_distance(zmid[j]).value)**2
            ave_density = infile['ave_dens']*(1.0/area_Mpch_new)*area_Mpch_old

        else:
            ave_density = infile['ave_dens']

        ave_dens.append(ave_density)
        nclust.append(infile['nclust'])
    xi = np.array(xi)
    w = np.array(w)
    ave_dens = np.array(ave_dens)
    nclust = np.array(nclust)

    dens = np.sum(ave_dens*nclust)/np.sum(nclust)
    mean = np.sum(xi*w, axis=0)/np.sum(w, axis=0)*dens

    Sg.append(mean)

Sg = np.array(Sg)
#C = np.cov(Sg.T)*(len(Sg)-1)
Sg_mean = np.mean(Sg, axis=0)
Sg_sig = np.sum((Sg-Sg_mean)**2, axis=0)**0.5/len(Sg)**0.5*(len(Sg)-1)**0.5

np.savez(result_dir+'/splashback_cov_'+str(name)+'.npz', cov=C,
         r_data=R, sg_mean=Sg_mean, sg_sig=Sg_sig)
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
