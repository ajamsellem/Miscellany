#!/usr/bin/python
import numpy as np
import pymultinest as mult
import pylab as plt
import seaborn as sns
sns.set_style("whitegrid")
import scipy.optimize as op
import os
import sys
sys.path.insert(0, '/Users/arielamsellem/Desktop/Research/splashback_codes_master/py_scripts')
from splashback_utils import *
from getdist import plots, MCSamples
from scipy.special import erfcinv

# Prior Functions
def uniform(x_0, x_f, cube_range):
    new_range = (x_f-x_0)*cube_range + x_0
    return new_range

def gaussian(mean, sigma, cube_range):
    new_range = mean + sigma*np.sqrt(2.0)*erfcinv(2.0*(1.0-cube_range))
    return new_range

# Constants and Stuff
h0 = 0.7
z_max = 40./h0
minz_to_integrate = 0.0
maxz_to_integrate = z_max
num_z_tointegrate = 500
z = np.linspace(minz_to_integrate, maxz_to_integrate, num = num_z_tointegrate)
r_o = 1.5/h0

def main():

    if (len(sys.argv) != 5):
        print()
        print("##################################")
        print("Ariel J. Amsellem")
        print("ajamsellem@uchicago.edu")
        print("KICP UChicago")
        print("##################################\n")

        print("run_Multinest.py - Run Multinest on Sigmag Measurements to determine measure of best fit with errors.")
        print("Usage: python run_Multinest.py [output directory/file name] [data filename] [color] [plot label/title]")
        print("Example: python run_Multinest.py Fiducial_RM splashback_cov_Fiducial_RM.npz    r    Fiducial_Redmapper")
        sys.exit(0)

    out_directory = str(sys.argv[1])
    dat_filename  = str(sys.argv[2])
    color         = str(sys.argv[3])
    label         = str(sys.argv[4])
    label         = label.replace("_", " ")

    # Scipy Minimization
    # Load Data
    data = np.load('/Users/arielamsellem/Desktop/Research/splashback_codes_master/npzs/' + dat_filename)
    sigmag = data['sg_mean']
    sigmag_sig = data['sg_sig']
    sigmag_cov = data['cov']
    rperp = data['r_data']

    # Priors
    log_alpha = -0.32085983 -0.1
    log_beta = 0.16309539
    log_gamma = 0.64815634
    log_r_s = 0.85387196 -0.1
    log_r_t = 0.08325509
    log_rho_0 = -0.8865869 -0.5
    log_rho_s = -0.19838697 -0.3
    se = 1.3290722
    ln_mis = -1.146114384
    f_mis = 0.15857366
    # Chihway: alpha, beta, gamma, r_s, r_t, rho_0, rho_s, se, ln_mis, f_mis
    params = np.array([log_alpha, log_beta, log_gamma, log_r_s, log_r_t, log_rho_0, log_rho_s, se, ln_mis, f_mis])

    # Minimized Splashback Model of Data
    print('Running Scipy Minimize...')
    print('')
    nll = lambda *args: -1*lnlikelihood(*args)
    p0 = params.copy()
    bounds = ((None, None), (None,None), (None,None), (np.log10(0.1/h0), np.log10(5.0/h0)), (np.log10(0.1/h0), np.log10(5.0/h0)), (None,None), (None,None), (-10., 10.), (np.log(0.01),np.log(0.99)),(0.01,0.99))
    data_vec = sigmag.copy()
    invcov = np.linalg.inv(sigmag_cov.copy())
    args = (rperp, z, data_vec, invcov, h0, 1)
    result = op.minimize(nll, p0, args=args, options = {'maxiter':200}, bounds = bounds)
    best_params = result.x
    best_lnlike = -result.fun

    # Scipy Stats
    model = Sigmag(rperp, z, best_params, h0, 1)
    diff = data_vec - model
    chisq_min = np.dot(diff, np.dot(invcov, diff))

    # Defining the Multinest Function
    def run_multinest(rperp, sigmag, invcov, splashback, outfile):
        def Prior(cube, ndim, nparams):
            # Sigma Values are from Chang 2018 Table 2. Each sigma is half a prior range
            cube[0] = gaussian(np.log10(0.19), 0.2 , cube[0]) # log(alpha)
            cube[1] = gaussian(np.log10(6.)  , 0.2 , cube[1]) # log(beta)
            cube[2] = gaussian(np.log10(4.)  , 0.2 , cube[2]) # log(gamma)
            cube[3] = uniform(0.1            , 5.  , cube[3]) # r_s
            cube[4] = uniform(0.1            , 5.  , cube[4]) # r_t
            cube[5] = uniform(0.             , 10. , cube[5]) # rho_0
            cube[6] = uniform(0.             , 10. , cube[6]) # rho_s
            cube[7] = uniform(1.             , 10. , cube[7]) # s_e
            cube[8] = gaussian(-1.13         , 0.22, cube[8]) # ln(c_mis)
            cube[9] = gaussian(0.22          , 0.11, cube[9]) # f_mis

        def Loglike(cube, ndim, nparams):
            # Read in parameters
            log_alpha = cube[0]
            log_beta = cube[1]
            log_gamma = cube[2]
            r_s = cube[3]
            r_t = cube[4]
            rho_0 = cube[5]
            rho_s = cube[6]
            se = cube[7]
            ln_mis = cube[8]
            f_mis = cube[9]
            params = [log_alpha, log_beta, log_gamma, r_s, r_t, rho_0, rho_s, se, ln_mis, f_mis]

            # Calculate likelihood
            sig_m = Sigmag(rperp, z, params, h0, splashback)
            vec = sig_m - sigmag
            likelihood = -0.5*np.matmul(np.matmul(vec,invcov),vec.T)

            # Calculate prior
            #prior = -0.5*(-1.13-ln_mis)**2/0.22**2 - 0.5*(log_alpha - np.log10(0.19))**2/0.4**2 - 0.5*(log_beta - np.log10(6.0))**2/0.4**2 - 0.5*(log_gamma - np.log10(4.0))**2/0.4**2  -0.5*(f_mis-0.22)**2/0.11**2
            prior = 0.

            # Total probability
            tot = likelihood + prior

            return tot

        # Run Multinest
        mult.run(Loglike, Prior, 10, outputfiles_basename=outfile, verbose = False)

    # Saving Results
    os.mkdir('/Users/arielamsellem/Desktop/Research/Multinest/' + out_directory)
    out_filename = out_directory
    out_directory = '/Users/arielamsellem/Desktop/Research/Multinest/' + out_directory + '/'
    out_filename = out_directory + out_filename

    # Run Multinest
    run_multinest(rperp, sigmag, invcov, 1, out_filename)

    # Save Output to File "log.txt"
    stdoutOrigin=sys.stdout
    sys.stdout = open(out_directory + "log.txt", "w")
    # Read in Multinest Results
    # Unequal Weights
    #multinest_out = np.genfromtxt(out_filename + '.txt')
    #samples_txt = multinest_out[:,2:]
    #likelihood_txt = -1.*multinest_out[:,1]/2
    # Equal Weights
    multinest_out = np.genfromtxt(out_filename + 'post_equal_weights.dat')
    samples_txt = multinest_out[:,:-1]
    likelihood_txt = multinest_out[:,-1]

    # Multinest Best Parameters
    analyzer = mult.analyse.Analyzer(10, outputfiles_basename=(out_filename), verbose=False)
    bestfit_params_multinest = analyzer.get_best_fit()
    best_params_mult = bestfit_params_multinest['parameters']
    best_loglike_mult = bestfit_params_multinest['log_likelihood']

    # Multinest Stats
    model_mult = Sigmag(rperp, z, best_params_mult, h0, 1)
    diff_mult = data_vec - model_mult
    chisq_mult = np.dot(diff_mult, np.dot(invcov, diff_mult))

    print("Best Parameters From Minimization: " + str(best_params))
    print("Loglike From Minimization: " + str(best_lnlike))
    print("Best Parameters From Multinest: " + str(best_params_mult))
    print("Loglike From Multinest: " + str(best_loglike_mult))
    print("Chi-Squared Scipy Minimize: " + str(chisq_min))
    print("Chi-Squared Multinest: " + str(chisq_mult))
    sys.stdout.close()
    sys.stdout=stdoutOrigin

    # Get Rho Values and Error Range
    low, high = profile_range(samples_txt, rperp, z, 16, 84)
    r_rho, r_rhoderiv, rho, drho, rho_i, rho_o = find_rho_drho(best_params)
    r_rho_mult, r_rhoderiv_mult, rho_mult, drho_mult, rho_i_mult, rho_o_mult  = find_rho_drho(best_params_mult)

    # Plot Results
    print('')
    print('Plotting Results...')
    samples = MCSamples(samples = samples_txt,
                    loglikes = likelihood_txt, names = ['alpha', 'beta', 'gamma', 'rs', 'rt', 'rho0', 'rhos', 'se', 'lnmis', 'fmis'],
                    labels = ['\\alpha', '\\beta', '\\gamma', 'r_s', 'r_t', '\\rho_0', '\\rho_s', 's_e', 'ln(c_{mis})', 'f_{mis}'])

    # Triangle Plot
    sns.set_style("white")
    g = plots.getSubplotPlotter(width_inch=12)
    g.triangle_plot(samples, filled=True, colors = [color], lw=[3], line_args=[{'lw':2, 'color':'k'}])
    plt.savefig(out_directory + 'Triangle_Multinest.png', dpi = 600)

    # Plot Error Region Around \\rho Derivative
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20,10))
    plt.suptitle(label, fontsize = 23, fontweight = 900)
    plt.subplot(121)
    plt.semilogx(r_rho, rho, color=color, label = 'Splashback Fit (Scipy)', linewidth = 1)
    plt.semilogx(r_rho, rho_i, color=color, label = 'Scipy Inner Profile', linewidth = 1, linestyle = '--')
    plt.semilogx(r_rho, rho_o, color=color, label = 'Scipy Outer Profile', linewidth = 1, linestyle = '-.')
    plt.semilogx(r_rho_mult, rho_mult, color="fuchsia", label = 'Splashback Fit (Multinest)', linewidth = 1)
    plt.semilogx(r_rho_mult, rho_i_mult, color="fuchsia", label = 'Multinest Inner Profile', linewidth = 1, linestyle = '--')
    plt.semilogx(r_rho_mult, rho_o_mult, color="fuchsia", label = 'Multinest Outer Profile', linewidth = 1, linestyle = '-.')
    plt.xlabel('$R  [Mpc]$', fontsize=15)
    plt.ylabel('$\\rho(R)$', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**-4)
    plt.legend(fontsize=18, loc = 'lower left')
    plt.subplot(122)
    plt.semilogx(r_rhoderiv, drho, color=color, label = 'Splashback Fit (Scipy)', linewidth = 1)
    plt.semilogx(r_rhoderiv_mult, drho_mult, color=color, label = 'Splashback Fit (Multinest)', linestyle = '--', linewidth = 1)
    plt.fill_between(r_rhoderiv,low,high,color=color, alpha=0.25)
    plt.xlim(0.1, 10.)
    plt.xlabel('$R  [Mpc]$', fontsize=15)
    plt.ylabel('$\\frac{dlog(\\rho(R))}{dlog(R)}$', fontsize=23)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_directory + 'rho_Multinest.png', dpi = 600)

    # Plot Sigmag Bestfit from Multinest
    plt.figure(figsize=(7,5))
    plt.errorbar(rperp, sigmag, yerr=sigmag_sig, capsize = 4, label=label, color=color, ls = 'none')
    plt.semilogx(rperp, Sigmag(rperp, z, best_params, h0, 1), label='Splashback Fit (Scipy)', color=color)
    plt.semilogx(rperp, Sigmag(rperp, z, best_params_mult, h0, 1), label='Splashback Fit (Multinest)', linestyle = '--', color=color)
    plt.xlabel('$R [Mpc]$', fontsize=15)
    plt.ylabel('$\Sigma_{g} [(1/Mpc)^2]$', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=14, loc='lower left')
    plt.savefig(out_directory + 'Sigmag.png', dpi = 600)

if __name__ == '__main__':
    main()
