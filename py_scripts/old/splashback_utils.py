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
        ln_alpha, ln_beta, ln_gamma, ln_r_s, ln_r_t, rho_O, rho_s, se, lnmis, f_mis = params
        beta = 10**ln_beta
        gamma = 10**ln_gamma
        r_t = 10**ln_r_t
        f_trans = (1.+(rr/r_t)**beta)**(-1*gamma/beta)

    if splash==0:
        ln_alpha, ln_r_s, rho_O, rho_s, se, lnmis, f_mis = params
        f_trans = 1.0

    alpha = 10**ln_alpha
    mis = np.exp(lnmis+np.log(0.81))
    r_s = 10**ln_r_s
    #rho_O = 10**ln_rho_O
    #rho_s = 10**ln_rho_s
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

def priors(params, h0, splash):
    if splash==1:
        ln_alpha, ln_beta, ln_gamma, ln_r_s, ln_r_t, rho_O, rho_s, se, lnmis, f_mis = params
        beta = 10**ln_beta
        gamma = 10**ln_gamma
        r_t = 10**ln_r_t

    if splash==0:
        ln_alpha, ln_r_s, rho_O, rho_s, se, lnmis, f_mis = params

    alpha = 10**ln_alpha
    mis = np.exp(lnmis+np.log(0.81))
    r_s = 10**ln_r_s
    #rho_O = 10**ln_rho_O
    #rho_s = 10**ln_rho_s

    min_logrs = np.log10(0.1/h0)
    max_logrs = np.log10(5.0/h0)
    min_logrt = np.log10(0.1/h0)
    max_logrt = np.log10(5.0/h0)
    min_se = -10.0
    max_se = 10.0
    f_mis_min = 0.01
    f_mis_max = 0.99
    min_lnmis = np.log(0.01)
    max_lnmis = np.log(0.99)

    if splash==1:
        if (ln_r_s > max_logrs) or (ln_r_s < min_logrs) or (ln_r_t > max_logrt) or (ln_r_t < min_logrt)  or (se < min_se) or (se > max_se) or (f_mis < f_mis_min) or (f_mis > f_mis_max) or (lnmis < min_lnmis) or (lnmis > max_lnmis):
            lnprior = -1.0e10  # equivalent to -np.inf

        else:
            mean_logalpha = np.log10(0.2)
            sigma_logalpha = 0.6
            mean_logbeta = np.log10(4.)
            sigma_logbeta = 0.2
            mean_loggamma = np.log10(6.0)
            sigma_loggamma = 0.2

            lnprior_alpha = -0.5*np.log(2.*np.pi*sigma_logalpha**2.)-0.5*((ln_alpha - mean_logalpha)**2.)/sigma_logalpha**2.
            lnprior_beta =  -0.5*np.log(2.*np.pi*sigma_logbeta**2.)-0.5*((ln_beta - mean_logbeta)**2.)/sigma_logbeta**2.
            lnprior_gamma =  -0.5*np.log(2.*np.pi*sigma_loggamma**2.)-0.5*((ln_gamma- mean_loggamma)**2.)/sigma_loggamma**2.
            lnprior_lnmis = -0.5*(-1.13-lnmis)**2/0.22**2
            lnprior_f_mis = -0.5*(f_mis-0.22)**2/0.11**2
            lnprior = lnprior_alpha + lnprior_beta + lnprior_gamma + lnprior_lnmis + lnprior_f_mis

    if splash==0:

        if ((np.log10(r_s) > max_logrs) or (np.log10(r_s) < min_logrs) or (se < min_se) or (se > max_se)) or (f_mis < f_mis_min) or (f_mis > f_mis_max) or (lnmis < min_lnmis) or (lnmis > max_lnmis):
            lnprior = -1.0e10

        else:
            mean_logalpha = np.log10(0.2)
            sigma_logalpha = 0.6
            lnprior_alpha = -0.5*np.log(2.*np.pi*sigma_logalpha**2.)-0.5*((np.log10(alpha) - mean_logalpha)**2.)/sigma_logalpha**2.
            lnprior_lnmis = -0.5*(-1.13-lnmis)**2/0.22**2
            lnprior_f_mis = -0.5*(f_mis-0.22)**2/0.11**2
            lnprior = lnprior_alpha + lnprior_f_mis + lnprior_lnmis

    if (np.isnan(lnprior)):
        pdb.set_trace()

    return lnprior

def lnlikelihood(params, R, z, data_vec, invcov, h0, splash):

    lnlike_priors = priors(params, h0, splash)
    #lnlike_priors = 0.
    lnlike_data = 0.0

    #if (lnlike_priors > -1.0e5):
    if 0==0:
        model = Sigmag(R, z, params, h0, splash)
        diff = data_vec - model
        detinvcov = np.linalg.det(invcov)
        detcov = 1./detinvcov
        lnlike_data = -0.5*(len(data_vec)*np.log(2.*np.pi) + np.log(detcov)) -0.5*np.dot(diff, np.dot(invcov, diff))
        #lnlike_priors = 0.

    lnlike = lnlike_data + lnlike_priors
    #print(params)
    #print(lnlike_priors)
    #print(lnlike_data)
    #print()

    return lnlike

def profile_range(params, r_sigmag, z, min, max):
    nsamps = params.shape[0]
    numr = 500
    r = np.exp(np.linspace(np.log(0.1), np.log(10.), num = numr))
    prof_stack = np.zeros((nsamps,numr-1))
    #sigmag_stack = np.zeros((nsamps, 150))
    h0 = 0.7
    r_o = 1.5/h0
    for i in range(nsamps):
        lgalpha = params[i][0]
        lgbeta = params[i][1]
        lggamma = params[i][2]
        lgr_s = params[i][3]
        lgr_t = params[i][4]
        rho_0 = params[i][5]
        rho_s = params[i][6]
        s_e = params[i][7]
        ln_mis = params[i][8]
        f_mis = params[i][9]
        alpha = 10.**(lgalpha)
        beta = 10.**(lgbeta)
        gamma = 10.**(lggamma)
        #rho_s = 10.**(lgrho_s)
        #rho_0 = 10.**(lgrho_0)
        r_s = 10.**(lgr_s)
        r_t = 10.**(lgr_t)
        mis = np.exp(ln_mis+np.log(0.81))

        # Compute Rho
        rho_gi = rho_s*np.exp((-2./alpha)*(((r/r_s)**alpha)-1))
        f_trans = (1.+(r/r_t)**beta)**(-1*gamma/beta)
        rho_go = rho_0*(r/r_o)**(-s_e)
        rho = rho_gi*f_trans + rho_go

        # Compute Rho Derivative
        prof_stack[i] = (np.log(rho[1:])-np.log(rho[:-1]))/(np.log(r[1])-np.log(r[0]))
        #np.argmin((np.log(rho[1:])-np.log(rho[:-1]))/(np.log(r[1])-np.log(r[0])))
        '''
        # Compute Sigmag
        rho_func = interpolate.interp1d(r, rho, fill_value = 'extrapolate')
        sigmag = []
        for j in range(len(r_sigmag)):
            func_evals = rho_func(np.sqrt(r_sigmag[j]**2.+z**2.))
            sigmag.append(2*integrate.simps(func_evals, z))
        # it appears to make a difference how this integration is done...
        func = interp1d(r_sigmag,sigmag,fill_value = "extrapolate")

        # Miscentering Corrections
        R_mis = np.sqrt(R_grid**2 + (d_mis_grid*mis)**2 + 2.*R_grid*(d_mis_grid*mis)*np.cos(phi_grid)) # EQ 12
        sigma_tem = func(R_mis)
        sigma_temp = np.mean(sigma_tem,axis=1)
        sigma_mis = np.average(sigma_temp,weights=p_mis,axis=0)
        sigma = func(R_sigmag)
        sigma_tot = (1.-f_mis)*sigma + f_mis*sigma_mis # EQ 11
        sigmag_stack[i] = sigma_tot

    sigmag_low = np.percentile(sigmag_stack,16,axis=0)
    sigmag_high = np.percentile(sigmag_stack,84,axis=0)
    '''
    rhoderiv_low = np.percentile(prof_stack,min,axis=0)
    rhoderiv_high = np.percentile(prof_stack,max,axis=0)

    return rhoderiv_low, rhoderiv_high, prof_stack, r

def find_rho_drho(params):
    minr = 0.1
    maxr = 10.
    numr = 500
    h0 = 0.7

    rr = np.exp(np.linspace(np.log(minr), np.log(maxr), num = numr))

    ln_alpha, ln_beta, ln_gamma, ln_r_s, ln_r_t, rho_0, rho_s, se, ln_mis, f_mis = params
    beta = 10**ln_beta
    gamma = 10**ln_gamma
    r_t = 10**ln_r_t
    f_trans = (1.+(rr/r_t)**beta)**(-1*gamma/beta)

    alpha = 10**ln_alpha
    r_s = 10**ln_r_s
    #rho_0 = 10**ln_rho_O
    #rho_s = 10**ln_rho_s
    r_o = 1.5/h0

    rho_gi = rho_s*np.exp(-2./alpha*((rr/r_s)**alpha-1))
    rho_go = rho_0*(rr/r_o)**(-1*se)
    rho_g = rho_gi * f_trans + rho_go
    rho_g_func = interpolate.interp1d(rr, rho_g)

    LNRHO = np.log(rho_g)
    LNR = np.log(rr)
    Logderiv = (LNRHO[1:]-LNRHO[:-1])/(LNR[1:]-LNR[:-1])

    Rmid = (rr[1:]+rr[:-1])/2

    return rr, Rmid, rho_g, Logderiv, f_trans*rho_gi, rho_go

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
