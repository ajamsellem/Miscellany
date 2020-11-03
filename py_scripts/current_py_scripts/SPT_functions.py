import numpy as np
import healpy as hp
import pylab as pyl
import astropy.io.fits as pf
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from sympy import *
import matplotlib as mpl

def radec2thetaphi(ra, dec, nside):
    """
    Convert RA DEC in degrees to THETA and PHI in Healpix
    convention.
    """

    theta = (90.0 - dec)*np.pi/180.0
    phi = ra*np.pi/180.0
    return theta, phi

def make_mask(ra, dec, nmin=1, nside=4096):
    """
    Take RA, DEC, build a binary mask just by assigning
    1 to pixels with count>=nmin and 0 otherwise. Mask
    is in Healpix format with assigned nside.
    """
    mask = np.zeros(hp.nside2npix(nside))
    theta, phi = radec2thetaphi(ra, dec, nside)
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    for i in range(len(pix)):
        mask[pix[i]] += 1
    mask[mask>=nmin] = 1

    return mask

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

def match_sigz_class(cluster, sigz_loc):
    sigz_loc = pf.open(sigz_loc)[1].data
    id_clus = cluster.ids
    sigmaz = sigz_loc['sigma_z']
    sigz_ids    = sigz_loc['MEM_MATCH_ID']
    match_idx = np.where(sigz_ids == id_clus)[0][0]
    cluster.sigmaz = sigmaz[match_idx]

def convert_rich_to_mass(richness, drichness, z, dz):
    M0  = 1.05*3.081*10**14
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

'''
def convert_rich_to_mass_CH(lam, dlam):
    a  = 14.351
    da = 0.02
    b  = 1.058
    db = 0.074

    h = 0.7

    A = Symbol('A')
    B = Symbol('B')
    R = Symbol('R')

    m_A = (1/h)*(10)**A*(lam/40.)**b
    m_B = (1/h)*(10)**a*(lam/40.)**B
    m_R = (1/h)*(10)**a*(R/40.)**b

    dm_da = m_A.diff(A)
    dm_db = m_B.diff(B)
    dm_dr = m_R.diff(R)

    f_A = lambdify(A, dm_da, 'numpy')
    f_B = lambdify(B, dm_db, 'numpy')
    f_R = lambdify(R, dm_dr, 'numpy')

    dm_da = f_A(a)
    dm_db = f_B(b)
    dm_dr = f_R(lam)

    m200     = (1/h)*(10)**a*(lam/40.)**b
    m200_err = np.sqrt( (dm_da*da)**2 + (dm_db*db)**2 + (dm_dr*dlam)**2 )

    return m200/10**14, m200_err/10**14

'''
def convert_rich_to_mass_CH(lam, dlam):
    a  = 14.351
    da = 0.02
    b  = 1.058
    db = 0.074

    h = 0.7

    m200     = (10)**a*(lam/40.)**b/h
    m200_err = (10**a)*((lam/40)**b)*np.sqrt((np.log(10)*da)**2+(np.log(lam/40)*db)**2+(1.058*np.divide(dlam,lam))**2)/h

    return m200/10**14, m200_err/10**14
