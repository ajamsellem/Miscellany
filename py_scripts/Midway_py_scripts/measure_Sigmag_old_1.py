# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ajamsellem/.local/bin')
import numpy as np
import astropy.io.fits as pf
from astropy import units as u
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import treecorr
# Uncomment these if using on local machine
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

zmin = float(sys.argv[1])
zmax = float(sys.argv[2])
Nz = int(sys.argv[3])
Maglim1 = float(sys.argv[4])
Maglim2 = float(sys.argv[5])
lambmin = float(sys.argv[6])
lambmax = float(sys.argv[7])

clusters = pf.open(sys.argv[8])[1].data
randoms = pf.open(sys.argv[9])[1].data
galaxies = pf.open(sys.argv[10])[1].data
galaxy_randoms = pf.open(sys.argv[11])[1].data

N_jk = int(sys.argv[12])
area = float(sys.argv[13])
outfile = sys.argv[14]
nR = int(sys.argv[15])
i = int(sys.argv[16])

# Make redshift and ngal cut
mask = (clusters['Z']>=zmin)*(clusters['Z']<zmax)*(clusters['LAMBDA']>=lambmin)*(clusters['LAMBDA']<lambmax) 
RA = clusters['RA'][mask]
DEC = clusters['DEC'][mask]
Z = clusters['Z'][mask]
LAMB = clusters['LAMBDA'][mask]

RA_ran_0 = randoms['RA']
DEC_ran_0 = randoms['DEC']

ra = galaxies['RA']
dec = galaxies['DEC']
mag = galaxies['MAG_AUTO_I']

ra_ran = galaxy_randoms['RA']
dec_ran = galaxy_randoms['DEC']

# make sure we're not occupying memory
clusters=0
randoms=0
galaxies=0
galaxy_randoms=0

Z = Z.astype('float_') # Avoids a weird histogram bug that sometimes occurs because of float32
n1 = np.histogram(Z, range=(zmin,zmax), bins=Nz)
zmid = (n1[1][1:]+n1[1][:-1])/2

# Input parameters
h = 0.7
Rmin = 0.1/h   #Mpc
Rmax = 10.0/h
lnrperp_bins = np.linspace(np.log(Rmin), np.log(Rmax), num = nR+1)
R_edge = np.exp(lnrperp_bins)
R_mid = np.sqrt(R_edge[:-1] * R_edge[1:])
bslop = 0.

# Binning in z
mask = (Z>=n1[1][i])*(Z<n1[1][i+1])
z_cen_bin = Z[mask]
lamb_cen_bin = LAMB[mask]
ra_cen_bin = RA[mask]
dec_cen_bin = DEC[mask]
RA_ran = RA_ran_0[:20*len(ra_cen_bin)]
DEC_ran = DEC_ran_0[:20*len(ra_cen_bin)]  

M = mag - 5*(np.log10(cosmo.luminosity_distance(zmid[i]).value*1e6) - 1)
M = M - 5.0*np.log10(0.7)
mask_gal = (M>Maglim1)*(M<Maglim2)

ra_gal_bin = ra[mask_gal]
dec_gal_bin = dec[mask_gal]

area_Mpch = area*(np.pi/180.)**2*(cosmo.comoving_distance(zmid[i]).value)**2
ave_density = len(ra_gal_bin)*1.0/area_Mpch

# Convert physical to angular distance at this zl
D_l = cosmo.comoving_distance(zmid[i]).value # comiving distance
thmin = np.arctan(Rmin / D_l) * (180./np.pi) * 60.      #arcmin
thmax = np.arctan(Rmax / D_l) * (180./np.pi) * 60.

# CCCCCCCCC
ra_gal_ran_temp = ra_ran[:2*len(ra_gal_bin)]
dec_gal_ran_temp = dec_ran[:2*len(ra_gal_bin)] 

RR = float(len(ra_cen_bin))*float(len(ra_gal_bin))/float(len(ra_gal_ran_temp))

'''
########JUST ONE TIME#############
import coord
ra = np.loadtxt('/project2/chihway/sims/buzzard/y1_gal_member/jk_centers')[:,0]
dec = np.loadtxt('/project2/chihway/sims/buzzard/y1_gal_member/jk_centers')[:,1]
xyz = coord.CelestialCoord.radec_to_xyz(ra*(coord.degrees/coord.radians), dec*(coord.degrees/coord.radians))
centers = np.column_stack(xyz)
gal_cat = treecorr.Catalog(ra=ra_gal_bin, dec=dec_gal_bin, ra_units='degrees', dec_units='degrees', patch_centers=centers)
##################################
'''

print('Galaxies: ' + str(len(ra_gal_bin)))
print('Random Galaxies: ' + str(len(ra_gal_ran_temp)))
print('Clusters: ' + str(len(ra_cen_bin)))
print('Random Clusters: ' + str(len(RA_ran)))

gal_cat = treecorr.Catalog(ra=ra_gal_bin, dec=dec_gal_bin, ra_units='degrees', dec_units='degrees', npatch=N_jk)
gal_ran_cat = treecorr.Catalog(ra=ra_gal_ran_temp, dec=dec_gal_ran_temp, ra_units='degrees', dec_units='degrees', patch_centers=gal_cat.patch_centers)
cen_cat = treecorr.Catalog(ra=ra_cen_bin, dec=dec_cen_bin, ra_units='degrees', dec_units='degrees', patch_centers=gal_cat.patch_centers)
ran_cat = treecorr.Catalog(ra=RA_ran, dec=DEC_ran, ra_units='degrees', dec_units='degrees', patch_centers=gal_cat.patch_centers)

dd = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax, bin_slop = bslop, sep_units = 'arcmin', verbose=0, var_method='jackknife')
rd = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax, bin_slop = bslop, sep_units = 'arcmin', verbose=0, var_method='jackknife')
dr = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax, bin_slop = bslop, sep_units = 'arcmin', verbose=0, var_method='jackknife')
rr = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax, bin_slop = bslop, sep_units = 'arcmin', verbose=0, var_method='jackknife')

dd.process(cen_cat, gal_cat, low_mem=True)
rd.process(ran_cat, gal_cat, low_mem=True)
dr.process(cen_cat, gal_ran_cat, low_mem=True)
rr.process(ran_cat, gal_ran_cat, low_mem=True)
xi,varxi = dd.calculateXi(rr,dr,rd) # xi = (𝐷𝐷−𝐷𝑅−𝑅𝐷+𝑅𝑅)/𝑅𝑅
cov = dd.estimate_cov('jackknife')

outfile_final = outfile + 'Sigmag_'  + str(i) + '.npz'
np.savez(outfile_final, R=R_mid, xi=xi, varxi=varxi, cov=cov, ave_dens=ave_density, w=rr.npairs*RR, nclust=len(ra_cen_bin), ngal=len(ra_gal_bin))

print('Finsihed Bin ' + str(i))
