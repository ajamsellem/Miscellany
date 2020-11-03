import numpy as np
import astropy.io.fits as pf
from astropy.coordinates import SkyCoord
from astropy import units as u
%matplotlib inline
import matplotlib.pyplot as plt
from sympy import *

# Functions
def find_sep(ra1, dec1, ra2, dec2):
    # Find angular separation for two locations (ra, dec in degrees) on the sky
    c1 = SkyCoord(ra1*u.degree, dec1*u.degree, frame='icrs')
    c2 = SkyCoord(ra2*u.degree, dec2*u.degree, frame='icrs')
    sep = c1.separation(c2)
    return sep

def range_intersect(val_1, val_2, lerr_1, uerr_1, lerr_2, uerr_2):
    # See if two error regions – each with a central value – intersect
    lim_l_1 = val_1 - lerr_1
    lim_u_1 = val_1 + uerr_1
    lim_l_2 = val_2 - lerr_2
    lim_u_2 = val_2 + uerr_2

    if (lim_u_1 > lim_l_2) & (lim_u_2 > lim_l_1):
        return True
    else:
        return False

def match_sigz(data_loc, sigz_loc):
    data_loc = pf.open(data_loc)[1].data
    sigz_loc = pf.open(sigz_loc)[1].data
    lam = data_loc['lambda_chisq']
    z   = data_loc['Z_LAMBDA']
    mask = (z>0.25)*(z<0.7)*(lam>=58.)

    ra  = data_loc['ra'][mask]
    dec = data_loc['dec'][mask]
    lam = data_loc['lambda_chisq'][mask]
    z   = data_loc['Z_LAMBDA'][mask]
    ids = data_loc['MEM_MATCH_ID'][mask]

    sigmaz = sigz_loc['sigma_z']
    idz    = sigz_loc['MEM_MATCH_ID']

    idxz_list    = []
    sigmaz_list = []

    for i, id_num in enumerate(ids):
        if id_num in idz:
            # Find indices with data in the sigmaz and Data data set
            idxz_list.append(i)
            # Retain the value of sigmaz when there is data in sigmaz and Data
            sigz_index = np.where(idz == id_num)[0][0]
            sigmaz_list.append(sigmaz[sigz_index])

    print("Number of Points Before Sigmaz Matching: " + str(len(ra)))
    print("Number of Points After Sigmaz Matching: " + str(len(ra[idxz_list])))

    return ra[idxz_list], dec[idxz_list], ids[idxz_list], z[idxz_list], lam[idxz_list], np.asarray(sigmaz_list)

def convert_rich_to_mass(richness, drichness, z, dz):
    M0  = 3.081*10**14
    dM0 = np.sqrt((.075*10**14)**2 + (.133*10**14)**2)
    F   = 1.356
    dF  = np.sqrt((.051)**2 + (.008)**2)
    G   = -.3
    dG  = np.sqrt((.3)**2 + (.06)**2)

    m0 = Symbol('m0')
    r  = Symbol('r')
    f  = Symbol('f')
    zz = Symbol('z')
    g  = Symbol('g')

    m_m0 = m0*((richness/40.)**F)*(((1+z)/1.35)**G)
    m_r  = M0*((r/40.)**F)*(((1+z)/1.35)**G)
    m_f  = M0*((richness/40.)**f)*(((1+z)/1.35)**G)
    m_zz = M0*((richness/40.)**F)*(((1+zz)/1.35)**G)
    m_g  = M0*((richness/40.)**F)*(((1+z)/1.35)**g)

    dm_dm0 = m_m0.diff(m0)
    dm_dr  = m_r.diff(r)
    dm_df  = m_f.diff(f)
    dm_dzz = m_zz.diff(zz)
    dm_dg  = m_g.diff(g)

    f_m0 = lambdify(m0, dm_dm0, 'numpy')
    f_r  = lambdify(r , dm_dr , 'numpy')
    f_f  = lambdify(f , dm_df , 'numpy')
    f_zz = lambdify(zz, dm_dzz, 'numpy')
    f_g  = lambdify(g , dm_dg , 'numpy')

    dm_dm0 = f_m0(M0)
    dm_dr  = f_r(richness)
    dm_df  = f_f(F)
    dm_dzz = f_zz(z)
    dm_dg  = f_g(G)

    #log_mass = np.log10(1.05*M0*((richness/40.)**F)*(((1+z)/1.35)**G))
    #log_mass_err = np.sqrt( (dm_dm0*dM0)**2 + (dm_dr*drichness)**2 + (dm_df*dF)**2 + (dm_dzz*dz)**2 + (dm_dg*dG)**2 )
    mass = M0*((richness/40.)**F)*(((1+z)/1.35)**G)
    mass = mass/10**14
    mass_err = np.sqrt( (dm_dm0*dM0)**2 + (dm_dr*drichness)**2 + (dm_df*dF)**2 + (dm_dzz*dz)**2 + (dm_dg*dG)**2 )
    mass_err = mass_err/10**14
    return mass, mass_err

# Main
# SPT Data
SPT = pf.open('/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/2500d_cluster_sample_Bocquet19_SPT.fits')[1].data
z_s = SPT['REDSHIFT']
mask_s = (z_s>0.25)*(z_s<0.7)

ra_s  = SPT['ra'][mask_s]
dec_s = SPT['dec'][mask_s]
z_s   = SPT['REDSHIFT'][mask_s]
z_e_s = SPT['REDSHIFT_UNC'][mask_s]
min_z_e_s = np.min(z_e_s[np.where(z_e_s != 0.)[0]])

mass_s = SPT['M200_marge'][mask_s]
mass_uerr_s = SPT['M200_marge_uerr'][mask_s]
mass_lerr_s = SPT['M200_marge_lerr'][mask_s]

# Redmapper DES Y3 Data
Y3_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl02_catalog.fit'
Y3 = pf.open(Y3_dat)[1].data
lam_3 = Y3['lambda_chisq']
z_3   = Y3['Z_LAMBDA']
mask_3 = (z_3>0.25)*(z_3<0.7)*(lam_3>=58.)

ra_3  = Y3['ra'][mask_3]
dec_3 = Y3['dec'][mask_3]
z_3   = Y3['Z_LAMBDA'][mask_3]
z_e_3 = Y3['Z_LAMBDA_E'][mask_3]
lam_3 = Y3['lambda_chisq'][mask_3]
ids_3 = Y3['MEM_MATCH_ID'][mask_3]
lam_e_3 = Y3['LAMBDA_CHISQ_E'][mask_3]

# Sigmaz column for Redmapper Clusters
sigz_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/sigma_zmeasurement_desy3_lgt15.fits'
XX = pf.open(sigz_dat)[1].data
ids_z = XX['MEM_MATCH_ID']
sigmaz = XX['sigma_z']

# Matching Procedure
dat_s = np.array(list(zip(ra_s, dec_s, z_s, z_e_s, mass_s, mass_lerr_s, mass_uerr_s)))
dat_3 = np.array(list(zip(ra_3, dec_3, z_3, z_e_3, lam_3, lam_e_3, ids_3)))
match_results = np.empty((0,5))
match_results = np.append(match_results, np.array([['RM RA', 'RM DEC', 'RM Z', 'RM Lambda', 'RM IDs']]), axis=0)
max_ang_sep = 3.
dm = 0.
ang_m = 0.
red_m = 0.
mas_m = 0.

for i, val_s in enumerate(dat_s):
    # Angular Separation Matching
    sep = find_sep(dat_3[:,0], dat_3[:,1], val_s[0], val_s[1])
    match_idx = np.where(sep.arcminute <=max_ang_sep)[0]

    # Redshift Matching
    if match_idx.size >= 1.:
        ang_m += 1.
        if val_s[3] == 0.:
            z_SPT_err = min_z_e_s
        else:
            z_SPT_err = val_s[3]

        match_idx_2 = []
        for j, idx in enumerate(match_idx):
            if val_s[2] != 0. and range_intersect(dat_3[:,2][idx], val_s[2], dat_3[:,3][idx], dat_3[:,3][idx], z_SPT_err, z_SPT_err) == True:
                match_idx_2.append(idx)
        match_idx = np.array(match_idx_2)

        # Mass-Richness Matching
        if len(match_idx) >= 1.:
            red_m += 1.
            match_idx_3 = []
            for idx in match_idx:
                mass_3, mass_e_3 = convert_rich_to_mass(dat_3[:,4][idx], dat_3[:,5][idx], dat_3[:,2][idx], dat_3[:,3][idx])
                if range_intersect(mass_3, val_s[4], mass_e_3, mass_e_3,  val_s[5], val_s[6]) == True:
                    match_idx_3.append(idx)
            match_idx = np.array(match_idx_3)

            if len(match_idx) >= 1.:
                mas_m += 1.
            if len(match_idx) > 1.:
                dm += 1.
            if len(match_idx) == 1.:
                idx = match_idx[0]
                match_results = np.append(match_results, np.array([[dat_3[:,0][idx], dat_3[:,1][idx],
                                                                    dat_3[:,2][idx], dat_3[:,4][idx],
                                                                    dat_3[:,6][idx]]]), axis=0)

ra  = match_results[:,0][1:].astype(np.float)
dec = match_results[:,1][1:].astype(np.float)
z   = match_results[:,2][1:].astype(np.float)
lam = match_results[:,3][1:].astype(np.float)
ids = match_results[:,4][1:].astype(np.float)

# Write RM Matches to a file
c6 = pf.Column(name='RA', format='E', array=ra)
c7 = pf.Column(name='DEC', format='E', array=dec)
c8 = pf.Column(name='Z_LAMBDA', format='E', array=z)
c9 = pf.Column(name='LAMBDA_CHISQ', format='E', array=lam)
c10 = pf.Column(name='MEM_MATCH_ID', format='E', array=ids)

CC2 = [c6, c7, c8, c9, c10]
hdu = pf.BinTableHDU.from_columns(CC2, nrows=len(ra))
RM_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/RM_matches_2.fits'
hdu.writeto(RM_dat, overwrite = True)

RM_dat = '/Users/arielamsellem/Desktop/Research/Y3-Buzzard_1.9.8/Plots/SPT_data_files/RM_matches_2.fits'
ra_m, dec_m, ids_m, z_m, lam_m, sigmaz_m = match_sigz(RM_dat, sigz_dat) # Report Sigmaz of Matched clusters
ra_3, dec_3, ids_3, z_3, lam_3, sigmaz_3 = match_sigz(Y3_dat, sigz_dat) # Report Sigmaz of all RM clusters

# Find Sigmaz Percentiles of each cluster (Matched and Total RM)
zbins = np.arange(0,9)

s_0_10   = []
s_10_20  = []
s_20_30  = []
s_30_40  = []
s_40_50  = []
s_50_60  = []
s_60_70  = []
s_70_80  = []
s_80_90  = []
s_90_100 = []

s3_0_10   = []
s3_10_20  = []
s3_20_30  = []
s3_30_40  = []
s3_40_50  = []
s3_50_60  = []
s3_60_70  = []
s3_70_80  = []
s3_80_90  = []
s3_90_100 = []

for zbin in zbins:
    zmin = zbin*.05 + .25
    zmax = zbin*.05 + .3
    mask_m = (z_m>=zmin)*(z_m<=zmax)
    mask_3 = (z_3>=zmin)*(z_3<=zmax)
    sigmaz_m_bin = sigmaz_m[mask_m]
    sigmaz_3_bin = sigmaz_3[mask_3]

    # Find sigmaz percentiles (in intervals of 10) values
    sigmaz_0    = np.percentile(sigmaz_m_bin, 0 )
    sigmaz_10   = np.percentile(sigmaz_m_bin, 10)
    sigmaz_20   = np.percentile(sigmaz_m_bin, 20)
    sigmaz_30   = np.percentile(sigmaz_m_bin, 30)
    sigmaz_40   = np.percentile(sigmaz_m_bin, 40)
    sigmaz_50   = np.percentile(sigmaz_m_bin, 50)
    sigmaz_60   = np.percentile(sigmaz_m_bin, 60)
    sigmaz_70   = np.percentile(sigmaz_m_bin, 70)
    sigmaz_80   = np.percentile(sigmaz_m_bin, 80)
    sigmaz_90   = np.percentile(sigmaz_m_bin, 90)
    sigmaz_100  = np.percentile(sigmaz_m_bin, 100)

    # Store Matched Information
    for sig_val in sigmaz_m_bin:
        if (sig_val > sigmaz_0)  & (sig_val < sigmaz_10):
            s_0_10.append(sig_val)
        if (sig_val > sigmaz_10) & (sig_val < sigmaz_20):
            s_10_20.append(sig_val)
        if (sig_val > sigmaz_20) & (sig_val < sigmaz_30):
            s_20_30.append(sig_val)
        if (sig_val > sigmaz_30) & (sig_val < sigmaz_40):
            s_30_40.append(sig_val)
        if (sig_val > sigmaz_40) & (sig_val < sigmaz_50):
            s_40_50.append(sig_val)
        if (sig_val > sigmaz_50) & (sig_val < sigmaz_60):
            s_50_60.append(sig_val)
        if (sig_val > sigmaz_60) & (sig_val < sigmaz_70):
            s_60_70.append(sig_val)
        if (sig_val > sigmaz_70) & (sig_val < sigmaz_80):
            s_70_80.append(sig_val)
        if (sig_val > sigmaz_80) & (sig_val < sigmaz_90):
            s_80_90.append(sig_val)
        if (sig_val > sigmaz_90) & (sig_val < sigmaz_100):
            s_90_100.append(sig_val)

    # Store RM Information
    for sig_val in sigmaz_3_bin:
        if (sig_val > sigmaz_0 ) & (sig_val < sigmaz_10):
            s3_0_10.append(sig_val)
        if (sig_val > sigmaz_10) & (sig_val < sigmaz_20):
            s3_10_20.append(sig_val)
        if (sig_val > sigmaz_20) & (sig_val < sigmaz_30):
            s3_20_30.append(sig_val)
        if (sig_val > sigmaz_30) & (sig_val < sigmaz_40):
            s3_30_40.append(sig_val)
        if (sig_val > sigmaz_40) & (sig_val < sigmaz_50):
            s3_40_50.append(sig_val)
        if (sig_val > sigmaz_50) & (sig_val < sigmaz_60):
            s3_50_60.append(sig_val)
        if (sig_val > sigmaz_60) & (sig_val < sigmaz_70):
            s3_60_70.append(sig_val)
        if (sig_val > sigmaz_70) & (sig_val < sigmaz_80):
            s3_70_80.append(sig_val)
        if (sig_val > sigmaz_80) & (sig_val < sigmaz_90):
            s3_80_90.append(sig_val)
        if (sig_val > sigmaz_90) & (sig_val < sigmaz_100):
            s3_90_100.append(sig_val)

sigmaz_m_tiles = np.array([len(s_0_10), len(s_10_20), len(s_20_30), len(s_30_40),
                           len(s_40_50), len(s_50_60), len(s_60_70), len(s_70_80),
                           len(s_80_90), len(s_90_100)])
sigmaz_3_tiles = np.array([len(s3_0_10), len(s3_10_20), len(s3_20_30), len(s3_30_40),
                           len(s3_40_50), len(s3_50_60), len(s3_60_70), len(s3_70_80),
                           len(s3_80_90), len(s3_90_100)])

print(sigmaz_m_tiles)
print(sigmaz_3_tiles)
