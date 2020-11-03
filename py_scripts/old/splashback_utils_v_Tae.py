import numpy as np
import pdb
from scipy import interpolate
import scipy.integrate as integrate
import scipy.signal as sig
from scipy.interpolate import interp1d


# Initial Parameters for Miscentering Grid
Rmin = 0.05
Rmax = 100
nR = 150
R_sigmag = np.logspace(np.log10(Rmin), np.log10(Rmax), nR)


nphi = 50
phi = np.linspace(0,2*np.pi,nphi)
ndmis = 50
R_grid = np.zeros((ndmis,nphi,nR)) # 50 matrices of 50 matrices of matrices with 150 elements (with each element being 0)
for i in range(nR):
        R_grid[:,:,i] = R_sigmag[i]

xmax = 5.
d_mis = np.linspace(0,xmax,ndmis)
d_mis_grid = np.zeros((ndmis,nphi,nR))
for i in range(ndmis):
        d_mis_grid[i,:,:] = d_mis[i]
p_mis = (d_mis/1.**2)*np.exp(-d_mis**2/(2.*1.**2))
p_mis = np.tile(p_mis,(nR,1))
p_mis = p_mis.T


phi_grid = np.zeros((ndmis,nphi,nR))
for i in range(nphi):
        phi_grid[:,i,:] = phi[i]


def Sigmag(R, z, params, h0, splash):

    minr = 0.01
    maxr = 100.
    numr = 500
    rr = np.exp(np.linspace(np.log(minr), np.log(maxr), num = numr))

    if splash==1:
        ln_alpha, ln_beta, ln_gamma, ln_r_s, ln_r_t, ln_rho_O, ln_rho_s, se, lnmis, f_mis = params
        beta = 10**ln_beta
        gamma = 10**ln_gamma
        r_t = 10**ln_r_t
        f_trans = (1.+(rr/r_t)**beta)**(-1*gamma/beta)

    if splash==0:
        ln_alpha, ln_r_s, ln_rho_O, ln_rho_s, se, lnmis, f_mis = params
        f_trans = 1.0

    alpha = 10**ln_alpha
    mis = np.exp(lnmis+np.log(0.81))
    r_s = 10**ln_r_s
    rho_O = 10**ln_rho_O
    rho_s = 10**ln_rho_s
    r_o = 1.5/h0

    rho_gi = rho_s*np.exp((-2./alpha)*(((rr/r_s)**alpha)-1))
    rho_go = rho_O*(rr/r_o)**(-1*se)
    rho_g = rho_gi * f_trans + rho_go
    rho_g_func = interpolate.interp1d(rr, rho_g)

    sigmag = []
    for i in range(len(R)):
        func_evals = rho_g_func(np.sqrt(R[i]**2.+z**2.))
        sigmag.append(2*integrate.simps(func_evals, z))
        # it appears to make a difference how this integration is done...
    func = interp1d(R,sigmag,fill_value = "extrapolate")

    # Miscentering Corrections
    R_mis = np.sqrt(R_grid**2 + (d_mis_grid*mis)**2 + 2.*R_grid*(d_mis_grid*mis)*np.cos(phi_grid)) # EQ 12
    sigma_tem = func(R_mis)
    sigma_temp = np.mean(sigma_tem,axis=1)
    sigma_mis = np.average(sigma_temp,weights=p_mis,axis=0)
    sigma = func(R_sigmag)
    sigma_tot = (1.-f_mis)*sigma + f_mis*sigma_mis # EQ 11
    func_tot = interp1d(R_sigmag,sigma_tot,kind='linear')

    return func_tot(R)

def lnlike(theta, rdat, z, sig0, covinv0, h0, splash):
    ln_alpha, ln_beta, ln_gamma, ln_r_s, ln_r_t, ln_rho_O, ln_rho_s, se, lnmis, f_mis = theta
    sig_m = Sigmag(rdat, z, theta, h0, splash)
    vec = sig_m - sig0
    like = -0.5*np.matmul(np.matmul(vec,covinv0),vec.T)
    return like

def lnprior(theta):
    ln_alpha, ln_beta, ln_gamma, ln_r_s, ln_r_t, ln_rho_O, ln_rho_s, se, lnmis, f_mis = theta
    if -4. < ln_rho_O < 2. and -4. < ln_rho_s < 4. and np.log10(0.01) < ln_r_s < np.log10(5.0) and np.log10(0.1) < ln_r_t < np.log10(5.0) and 0.1 < se < 10. and 0.01 < f_mis < 0.99 and np.log(0.01) < lnmis < np.log(0.99):
        return -0.5*(-1.13-lnmis)**2/0.22**2 - 0.5*(ln_alpha - np.log10(0.19))**2/0.4**2 - 0.5*(ln_beta - np.log10(6.0))**2/0.4**2 - 0.5*(ln_gamma - np.log10(4.0))**2/0.4**2  -0.5*(f_mis-0.22)**2/0.11**2
    else:
        return -np.inf

def ln_prob(theta, rdat, z, sig0, covinv0, h0, splash):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnl = lnlike(theta, rdat, z, sig0, covinv0, h0, splash) +lp
    lnl = lp
    print(lnl)
    print(lp)
    return lnl


def derivative_savgol(R, data, N=10000, window_length=5, polyorder=3):

    data_sm = sig.savgol_filter(np.log10(data), window_length=window_length, polyorder=polyorder)
    f = interpolate.interp1d(np.log10(R), data_sm, kind='cubic')

    # Evaluate spline across a very fine grid or radii
    lnrad_fine = np.linspace(np.log10(np.min(R)), np.log10(np.max(R)), num=N)
    lnsigma_fine = f(lnrad_fine)

    # Calculate derivative using finite differencing
    dlnsig_dlnr_fine = (lnsigma_fine[1:] - lnsigma_fine[:-1])/(lnrad_fine[1:] - lnrad_fine[:-1])

    return (lnrad_fine[1:]+lnrad_fine[:-1])/2, dlnsig_dlnr_fine

# Alternative to DSigmag, results are basically identical...
def DelSigmag(R, z, params, h0, splash, N=100):

    sigma = Sigmag(R, z, params, h0, splash)
    # interpolate onto a finer grid just so the integration is smooth

    f = interpolate.interp1d(np.log10(R), np.log10(sigma), kind='cubic')
    rad_fine = np.linspace(np.log10(np.min(R)), np.log10(np.max(R)), num=N)

    Dsigma = []
    for i in range(len(rad_fine)-1):
        func_evals = f(rad_fine[:i+1])*2.*np.pi*rad_fine[:i+1]
        sigmag_sum = integrate.simps(func_evals, rad_fine[:i+1])
        sigmag_mean = sigmag_sum/(np.pi*rad_fine[i+1]**2)
        Dsigma.append(sigmag_mean - f(rad_fine[i+1]))

    f = interpolate.interp1d(rad_fine[1:], Dsigma, kind='cubic', bounds_error=False, fill_value=0)
    lnsigma_coarse = f(R)

    return lnsigma_coarse

def DSigmag(R, z, params, h0, splash, N=100):

    sigma = Sigmag(R, z, params, h0, splash)
    # interpolate onto a finer grid just so the integration is smooth
    f = interpolate.interp1d(np.log10(R), np.log10(sigma), kind='cubic')
    lnrad_fine = np.linspace(np.min(np.log10(R)), np.max(np.log10(R)), num=N)
    lnsigma_fine = f(lnrad_fine)

    R_fine = 10**lnrad_fine
    sigma_fine = 10**lnsigma_fine
    R_fine_mid = (R_fine[1:]+R_fine[:-1])/2
    dR_fine = R_fine[1:]-R_fine[:-1]
    sigma_fine_mid = (sigma_fine[1:]+sigma_fine[:-1])/2

    Dsigma = []

    for i in range(len(R_fine_mid)):
        Mean = np.sum(sigma_fine_mid[:i+1]*2*np.pi*R_fine_mid[:i+1]*dR_fine[:i+1])/np.sum(2*np.pi*R_fine_mid[:i+1]*dR_fine[:i+1])
        Dsigma.append(Mean-sigma_fine_mid[i])

    # interpolate back to original R grid
    f = interpolate.interp1d(R_fine_mid, Dsigma, kind='cubic', bounds_error=False, fill_value=0)
    lnsigma_coarse = f(R)

    return lnsigma_coarse


def lnlikelihoodD(params, R, z, data_vec, invcov, h0, splash):

    lnlike_priors = priors(params, h0, splash)
    lnlike_data = 0.0

    if (lnlike_priors > -1.0e5):

        model = DSigmag(R, z, params, h0, splash)
        diff = data_vec - model
        detinvcov = np.linalg.det(invcov)
        detcov = 1./detinvcov
        lnlike_data = -0.5*(len(data_vec)*np.log(2.*np.pi) + np.log(detcov)) -0.5*np.dot(diff, np.dot(invcov, diff))

    lnlike = lnlike_data + lnlike_priors

    return lnlike
