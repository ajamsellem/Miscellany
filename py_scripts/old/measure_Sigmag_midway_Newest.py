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
jkid = int(sys.argv[12])
tot_area = float(sys.argv[13])
outfile = sys.argv[14]
nR = int(sys.argv[15])
i = int(sys.argv[16])

# read in data, make redshift and ngal cut, and mask JK sub-sample
JK = clusters['JK']
mask = (JK!=jkid)*(clusters['Z']>=zmin)*(clusters['Z']<zmax)*(clusters['LAMBDA']>=lambmin)*(clusters['LAMBDA']<lambmax)
RA = clusters['RA'][mask]
DEC = clusters['DEC'][mask]
Z = clusters['Z'][mask]
LAMB = clusters['LAMBDA'][mask]

JK_ran = randoms['JK']
mask = (JK_ran!=jkid)
RA_ran = randoms['RA'][mask]
DEC_ran = randoms['DEC'][mask]

JK_gal = galaxies['JK']
mask = (JK_gal!=jkid)
ra = galaxies['RA'][mask]
dec = galaxies['DEC'][mask]
mag = galaxies['MAG_AUTO_I'][mask]

JK_gal_ran = galaxy_randoms['JK']
N_allgal = len(JK_gal_ran)
mask = (JK_gal_ran!=jkid)
ra_ran = galaxy_randoms['RA'][mask]
dec_ran = galaxy_randoms['DEC'][mask]
N_jkgal = len(ra_ran)

# make sure we're not occupying memory
clusters=0
randoms=0
galaxies=0
galaxy_randoms=0

Z = Z.astype('float_') # Avoids a weird histogram bug that sometimes occurs because of float32
n1 = np.histogram(Z, range=(zmin,zmax), bins=Nz)
zmid = (n1[1][1:]+n1[1][:-1])/2

# calculate area
area = tot_area*(N_jkgal*1.0/N_allgal)
print("area:", area)

# Measurement parameters
h = 0.7
Rmin = 0.1/h   #Mpc
Rmax = 10.0/h
lnrperp_bins = np.linspace(np.log(Rmin), np.log(Rmax), num = nR+1)
R_edge = np.exp(lnrperp_bins)
R_mid = np.sqrt(R_edge[:-1] * R_edge[1:])
bslop = 0.03


# treecorr
print("bin:", i, "mean z:", zmid[i])
print(zmid[i])
mask = (Z>=n1[1][i])*(Z<n1[1][i+1])
print(Z)
print(mask)
z_cen_bin = Z[mask]
lamb_cen_bin = LAMB[mask]
ra_cen_bin = RA[mask]
dec_cen_bin = DEC[mask]

ra_ran_bin = RA_ran
dec_ran_bin = DEC_ran

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
ra_gal_ran_temp = ra_ran[:7*len(ra_gal_bin)]
dec_gal_ran_temp = dec_ran[:7*len(ra_gal_bin)]

# Define catalogs
cen_cat = treecorr.Catalog(ra=ra_cen_bin, dec=dec_cen_bin,
                                    ra_units='degrees', dec_units='degrees')
ran_cat = treecorr.Catalog(ra=ra_ran_bin, dec=dec_ran_bin,
                                    ra_units='degrees', dec_units='degrees')
gal_cat = treecorr.Catalog(ra=ra_gal_bin, dec=dec_gal_bin,
                                   ra_units='degrees', dec_units='degrees')
gal_ran_cat = treecorr.Catalog(ra=ra_gal_ran_temp, dec=dec_gal_ran_temp,
                                   ra_units='degrees', dec_units='degrees')

# CCCCCCCC
#RR = len(ra_cen_bin)*1.0/len(ra_ran_bin)*len(ra_gal_bin)/len(ra_gal_ran_temp)
RR = len(ra_cen_bin)*len(ra_gal_bin)/len(ra_gal_ran_temp)
ra_gal_ran_temp=0
dec_gal_ran_temp=0

# Numerator
dd = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax,
                                        bin_slop = bslop, sep_units = 'arcmin', verbose=2)

rd = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax,
                                        bin_slop = bslop, sep_units = 'arcmin', verbose=2)

dr = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax,
                                        bin_slop = bslop, sep_units = 'arcmin', verbose=2)

rr = treecorr.NNCorrelation(nbins = nR, min_sep = thmin, max_sep = thmax,
                                        bin_slop = bslop, sep_units = 'arcmin', verbose=2)

dd.process(cen_cat, gal_cat)    # For cross-correlation.
rd.process(ran_cat, gal_cat)
dr.process(cen_cat, gal_ran_cat)
rr.process(ran_cat, gal_ran_cat)

xi,varxi = dd.calculateXi(rr,dr,rd) # xi = (𝐷𝐷−𝐷𝑅−𝑅𝐷+𝑅𝑅)/𝑅𝑅

# CCCCCCCCCCCC
np.savez(outfile, R=R_mid, xi=xi, ave_dens=ave_density, w=rr.weight*RR, nclust=len(ra_cen_bin))
#np.savez(outfile, R=R_mid, xi=xi, ave_dens=ave_density, w=rr.npairs*RR, nclust=len(ra_cen_bin))
