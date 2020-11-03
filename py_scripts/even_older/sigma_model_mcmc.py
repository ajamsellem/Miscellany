import numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from scipy import integrate
from math import pi
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar, OptimizeResult
import scipy.optimize as op
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
#import emcee
h = 0.7

### parameter setting
Rmin = 0.05
Rmax = 100
nR = 150
R = np.logspace(np.log10(Rmin), np.log10(Rmax), nR)

#Rcut = bs.bisect(R,1.5*R_lam)

rlosmin = 0.01
rlosmax = 40.
nrlos = 500
rlos = np.logspace(np.log10(rlosmin), np.log10(rlosmax), nrlos)

Rsq = R**2
tem = np.tile(Rsq,(nrlos,1))
tem = tem.transpose()

rlos_sq = rlos**2
temp = np.tile(rlos_sq,(nR,1))

rsq = tem + temp
r = np.sqrt(rsq)

nphi = 50
phi = np.linspace(0,2*pi,nphi)
ndmis = 50
R_grid = np.zeros((ndmis,nphi,nR)) # 150 matrices of 50 matrices of matrices with 50 elements (with each element being 0)
for i in range(nR):
        R_grid[:,:,i] = R[i]

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


rhomax = 10000000.
r00 = 1.5

r0 = np.sqrt(rlos_sq+0.001**2)

def sigma_model(theta,rdat):
	lgrho_s, lgalpha, lgr_s, lgr_t, lgbeta, lggamma, lgrho_0, s_e, lnmis, f_mis = theta
        alpha = 10.**(lgalpha)
	beta = 10.**(lgbeta)
	gamma = 10.**(lggamma)
	mis = np.exp(lnmis+np.log(0.81)) #0.79 for 20,100 0.69 for 10, 30
	rho_s = 10.**(lgrho_s)
	rho_0 = 10.**(lgrho_0)
	r_s = 10.**(lgr_s)
	r_t = 10.**(lgr_t)
	rho_ein = rho_s * np.exp((-2./alpha)*((r/r_s)**alpha-1))
	f_trans = (1+(r/r_t)**beta)**(-gamma/beta)
	rho_coll = rho_ein * f_trans
	rho_infall = rho_0 * (1./rhomax + (r/r00)**(s_e))**(-1)
	rho = rho_coll + rho_infall

	rho_ein0 = rho_s * np.exp((-2./alpha)*((r0/r_s)**alpha-1))
	f_trans0 = (1+(r0/r_t)**beta)**(-gamma/beta)
	rho_infall0 = rho_0 * (1./rhomax + (r0/r00)**(s_e))**(-1)
	rho0 = rho_ein0*f_trans0 + rho_infall0

	sigma = 2. * integrate.simps(rho, rlos, axis=1)
	sigma0 = 2. * integrate.simps(rho0, rlos)
    # find a function that describes the relationship between R and sigma
    func = interp1d(np.hstack((0.001,R)),np.hstack((sigma0,sigma)),fill_value = "extrapolate")

    # Miscentering Corrections
    R_mis = np.sqrt(R_grid**2 + (d_mis_grid*mis)**2 + 2.*R_grid*(d_mis_grid*mis)*np.cos(phi_grid))
    sigma_tem = func(R_mis)
    sigma_temp = np.mean(sigma_tem,axis=1)
    sigma_mis = np.average(sigma_temp,weights=p_mis,axis=0)
    sigma_tot = (1.-f_mis)*sigma + f_mis*sigma_mis
	func_tot = interp1d(R,sigma_tot,kind='linear')
	return func_tot(rdat)

def sigma_model_spt(theta,rdat):
        lgrho_s, lgalpha, lgr_s, lgr_t, lgbeta, lggamma, lgrho_0, s_e, lnmis = theta
        alpha = 10.**(lgalpha)
        beta = 10.**(lgbeta)
        gamma = 10.**(lggamma)
        mis = np.exp(lnmis)
        rho_s = 10.**(lgrho_s)
        rho_0 = 10.**(lgrho_0)
        r_s = 10.**(lgr_s)
        r_t = 10.**(lgr_t)
        rho_ein = rho_s * np.exp((-2./alpha)*((r/r_s)**alpha-1))
        f_trans = (1+(r/r_t)**beta)**(-gamma/beta)
        rho_coll = rho_ein * f_trans
        rho_infall = rho_0 * (1./rhomax + (r/r00)**(s_e))**(-1)
        rho = rho_coll + rho_infall

        rho_ein0 = rho_s * np.exp((-2./alpha)*((r0/r_s)**alpha-1))
        f_trans0 = (1+(r0/r_t)**beta)**(-gamma/beta)
        rho_infall0 = rho_0 * (1./rhomax + (r0/r00)**(s_e))**(-1)
        rho0 = rho_ein0*f_trans0 + rho_infall0

        sigma = 2. * integrate.simps(rho, rlos, axis=1)
        sigma0 = 2. * integrate.simps(rho0, rlos)
        func = interp1d(np.hstack((0.001,R)),np.hstack((sigma0,sigma)),fill_value = "extrapolate")

        R_mis = np.sqrt(R_grid**2 + (d_mis_grid*mis)**2 + 2.*R_grid*(d_mis_grid*mis)*np.cos(phi_grid))
        sigma_tem = func(R_mis)
        sigma_temp = np.mean(sigma_tem,axis=1)
        sigma_mis = np.average(sigma_temp,weights=p_mis,axis=0)
        func_mis = interp1d(R,sigma_mis,kind='linear')
        return func_mis(rdat)

# Equation (24) in Chang 2018
def lnlike(theta,rdat,sig0,covinv0):
	lgrho_s, lgalpha, lgr_s, lgr_t, lgbeta, lggamma, lgrho_0, s_e, lnmis, f_mis = theta
	sig_m = sigma_model(theta,rdat)
	vec = sig_m - sig0
	like = -0.5*np.matmul(np.matmul(vec,covinv0),vec.T)
	return like

def lnlike_spt(theta,rdat,sig0,covinv0):
        lgrho_s, lgalpha, lgr_s, lgr_t, lgbeta, lggamma, lgrho_0, s_e, lnmis= theta
        sig_m = sigma_model_spt(theta,rdat)
        vec = sig_m - sig0
        like = -0.5*np.matmul(np.matmul(vec,covinv0),vec.T)
        return like

def lnlike_spt_clr(theta,rdat,sig0,covinv0):
	lgrho_s_r, lgalpha_r, lgr_s_r, lgr_t_r, lgbeta_r, lggamma_r, lgrho_0_r, s_e_r, lgrho_s_g, lgalpha_g, lgr_s_g, lgr_t_g, lgbeta_g, lggamma_g, lgrho_0_g, s_e_g, lgrho_s_b, lgalpha_b, lgr_s_b, lgr_t_b, lgbeta_b, lggamma_b, lgrho_0_b, s_e_b, lnmis = theta
	theta_r = lgrho_s_r, lgalpha_r, lgr_s_r, lgr_t_r, lgbeta_r, lggamma_r, lgrho_0_r, s_e_r, lnmis
	theta_g = lgrho_s_g, lgalpha_g, lgr_s_g, lgr_t_g, lgbeta_g, lggamma_g, lgrho_0_g, s_e_g, lnmis
	theta_b = lgrho_s_b, lgalpha_b, lgr_s_b, lgr_t_b, lgbeta_b, lggamma_b, lgrho_0_b, s_e_b, lnmis
	sig_m_r = sigma_model_spt(theta_r,rdat)
	sig_m_g = sigma_model_spt(theta_g,rdat)
	sig_m_b = sigma_model_spt(theta_b,rdat)
	sig_m = np.hstack((sig_m_r,sig_m_g,sig_m_b))
	vec = sig_m - sig0
	like = -0.5*np.matmul(np.matmul(vec,covinv0),vec.T)
	return like

def lnlike_diff(theta,rdat1,rdat2,sig0,covinv0):
	lgrho_s1, lgalpha1, lgr_s1, lgr_t1, lgbeta1, lggamma1, lgrho_01, s_e1, lnmis1, f_mis1 = theta[:10]
	lgrho_s2, lgalpha2, lgr_s2, lgr_t2, lgbeta2, lggamma2, lgrho_02, s_e2, lnmis2, f_mis2 = theta[10:]
	sig_m1 = sigma_model(theta[:10],rdat1)
	sig_m2 = sigma_model(theta[10:],rdat2)
	sig_m = np.hstack((sig_m1,sig_m2))
	vec = sig_m - sig0
	like = -0.5*np.matmul(np.matmul(vec,covinv0),vec.T)
	return like

# Some number?
def lnprior(theta):
        lgrho_s, lgalpha, lgr_s, lgr_t, lgbeta, lggamma, lgrho_0, s_e, lnmis, f_mis = theta
        if -4. < lgrho_0 < 2. and -3. < lgrho_s < 4. and np.log10(0.01) < lgr_s < np.log10(5.0) and np.log10(0.1) < lgr_t < np.log10(5.0) and 0.1 < s_e < 10. and 0.01 < f_mis < 0.99 and np.log(0.01) < lnmis < np.log(0.99):
	           return -0.5*(-1.13-lnmis)**2/0.22**2 - 0.5*(lgalpha - np.log10(0.19))**2/0.4**2 - 0.5*(lgbeta - np.log10(6.0))**2/0.4**2 - 0.5*(lggamma - np.log10(4.0))**2/0.4**2  -0.5*(f_mis-0.22)**2/0.11**2
        else:
            return -np.inf
def lnprior_spt(theta):
        lgrho_s, lgalpha, lgr_s, lgr_t, lgbeta, lggamma, lgrho_0, s_e, lnmis = theta
        if -5. < lgrho_0 < 5. and -5. < lgrho_s < 5. and np.log10(0.01) < lgr_s < np.log10(5.0) and np.log10(0.1) < lgr_t < np.log10(5.0) and 0.1 < s_e < 10. and np.log(0.01) < lnmis < np.log(0.99):
            return -0.5*(-2.68-lnmis)**2/0.8**2 - 0.5*(lgalpha - np.log10(0.23))**2/0.65**2 - 0.5*(lgbeta - np.log10(6.0))**2/0.2**2 - 0.5*(lggamma - np.log10(4.0))**2/0.2**2
        else:
            return -np.inf

def lnprior_spt_clr(theta):
	lgrho_s_r, lgalpha_r, lgr_s_r, lgr_t_r, lgbeta_r, lggamma_r, lgrho_0_r, s_e_r, lgrho_s_g, lgalpha_g, lgr_s_g, lgr_t_g, lgbeta_g, lggamma_g, lgrho_0_g, s_e_g, lgrho_s_b, lgalpha_b, lgr_s_b, lgr_t_b, lgbeta_b, lggamma_b, lgrho_0_b, s_e_b, lnmis = theta
	if -5. < lgrho_0_r < 5. and -5. < lgrho_s_r < 5. and np.log10(0.01) < lgr_s_r < np.log10(5.0) and np.log10(0.1) < lgr_t_r < np.log10(5.0) and 0.1 < s_e_r < 10. and -5. < lgrho_0_g < 5. and -5. < lgrho_s_g < 5. and np.log10(0.01) < lgr_s_g < np.log10(5.0) and np.log10(0.1) < lgr_t_g < np.log10(5.0) and 0.1 < s_e_g < 10. and -5. < lgrho_0_b < 5. and -5. < lgrho_s_b < 5. and np.log10(0.01) < lgr_s_b < np.log10(5.0) and np.log10(0.1) < lgr_t_b < np.log10(5.0) and 0.1 < s_e_b < 10. and np.log(0.01) < lnmis < np.log(0.99):
		return -0.5*(-2.68-lnmis)**2/0.8**2 - 0.5*(lgalpha_r - np.log10(0.23))**2/0.65**2 - 0.5*(lgbeta_r - np.log10(6.0))**2/0.2**2 - 0.5*(lggamma_r - np.log10(4.0))**2/0.2**2 - 0.5*(lgalpha_g - np.log10(0.23))**2/0.65**2 - 0.5*(lgbeta_g - np.log10(6.0))**2/0.2**2 - 0.5*(lggamma_g - np.log10(4.0))**2/0.2**2 - 0.5*(lgalpha_b - np.log10(0.23))**2/0.65**2 - 0.5*(lgbeta_b - np.log10(6.0))**2/0.2**2 - 0.5*(lggamma_b - np.log10(4.0))**2/0.2**2
	else:
		return -np.inf

def ln_prob(theta,rdat,sig0,covinv0):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        lnl = lnlike(theta,rdat,sig0,covinv0) +lp
        print(theta)
        print(lnl)
        return lnl

def ln_prob_spt(theta,rdat,sig0,covinv0):
        lp = lnprior_spt(theta)
        if not np.isfinite(lp):
            return -np.inf
        lnl = lnlike_spt(theta,rdat,sig0,covinv0) +lp
        print(theta)
        print(lnl)
        return lnl

def ln_prob_spt_clr(theta,rdat,sig0,covinv0):
	lp = lnprior_spt_clr(theta)
	if not np.isfinite(lp):
		return -np.inf
	lnl = lnlike_spt_clr(theta,rdat,sig0,covinv0) +lp
	print(theta)
	print(lnl)
	return lnl

def ln_prob_diff(theta,rdat1,rdat2,sig0,covinv0):
        lp1 = lnprior(theta[:10])
        if not np.isfinite(lp1):
            return -np.inf
	lp2 = lnprior(theta[10:])
	if not np.isfinite(lp2):
	    return -np.inf
        lnl = lnlike_diff(theta,rdat1,rdat2,sig0,covinv0) +lp1 + lp2
        print(theta)
        print(lnl)
        return lnl

rbin = np.logspace(np.log10(0.2),np.log10(12.),26)
#rbin = np.logspace(np.log10(0.1),np.log10(10.),16)
#rbin = np.logspace(np.log10(0.2),np.log10(10.),10)
#rbin = np.logspace(np.log10(0.1),np.log10(10.),13)
rmid = np.sqrt(rbin[1:]*rbin[:-1])
#dat225 = np.load('./output/Sig_Y3RM0-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim22.5.dat.npz')
#dat220 = np.load('./output/Sig_Y3RM60-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim22.0.dat.npz')
#dat215 = np.load('./output/Sig_Y3RM60-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim21.5.dat.npz')
#dat225 = np.load('./output/Sig_RASS-MCMF0-100_clr-magerr0.1_nR15_0.29-17.14_z0.10-0.70_absmag1_maglim22.5.dat.npz')
#dat_low = np.load('./output/Sig_Y3RM0-100_l20-200_nR15_0.14-14.29_z0.10-0.70_absmag1_maglim22.5_low.dat.npz')
#dat_high = np.load('./output/Sig_Y3RM0-100_l20-200_nR15_0.14-14.29_z0.10-0.70_absmag1_maglim22.5_high.dat.npz')
#dat_maj = np.load('./output/Sig_sptmajor_nR8_0.29-17.14_z0.25-0.70_absmag1_maglim22.5.dat.npz')
#dat_min = np.load('./output/Sig_sptminor_nR8_0.29-17.14_z0.25-0.70_absmag1_maglim22.5.dat.npz')
##dat_dim = np.load('./output/Sig_Y3RM0-100_nR12_0.14-14.29_z0.20-0.55_absmag1_maglim22.1-22.5.dat.npz')
##dat_bright = np.load('./output/Sig_Y3RM0-100_nR12_0.14-14.29_z0.20-0.55_absmag1_maglim0.0-20.8.dat.npz')
#dat_dim = np.load('./output/Sig_advact_SNRlt5.5_nR15_0.29-17.14_z0.25-0.70_maglim22.1-22.5.dat.npz')
#dat_bright = np.load('./output/Sig_advact_SNRlt5.5_nR15_0.29-17.14_z0.25-0.70_maglim0.0-20.8.dat.npz')
dat_all = np.load('./output/Sig_advact_all_SNRgt5.0_nR25_0.29-17.14_z0.25-0.70_maglim22.5.dat.npz')
dat_red = np.load('./output/Sig_advact_red_SNRgt5.0_nR25_0.29-17.14_z0.25-0.70_maglim22.5.dat.npz')
dat_green = np.load('./output/Sig_advact_green_SNRgt5.0_nR25_0.29-17.14_z0.25-0.70_maglim22.5.dat.npz')
dat_blue = np.load('./output/Sig_advact_blue_SNRgt5.0_nR25_0.29-17.14_z0.25-0.70_maglim22.5.dat.npz')
dat_clr = np.load('./output/Sig_advact_clr-cov_SNRgt5.0_nR25_0.29-17.14_z0.25-0.70_maglim22.5.dat.npz')

#xi_low = dat_low['xi']
#xi_high = dat_high['xi']
#xi_maj = dat_maj['xi']
#xi_min = dat_min['xi']
##xi_dim = dat_dim['xi']
##xi_bright = dat_bright['xi']
#Sig_lowdens = np.loadtxt('SPTz25lowdensxredmagic_absmag1_brute_clran_z0.70_maglim22.5_points.dat')[:-1]
#Sig_highdens = np.loadtxt('SPTz25highdensxredmagic_absmag1_brute_clran_z0.70_maglim22.5_points.dat')[:-1]
#Sig_dim  = dat_dim['Sigma']
#Sig_bright = dat_bright['Sigma']
Sig_all = dat_all['Sigma']
Sig_red = dat_red['Sigma']
Sig_green = dat_green['Sigma']
Sig_blue = dat_blue['Sigma']
Sig_clr = dat_clr['Sigma']

#cov_low = dat_low['cov']
#cov_high = dat_low['cov']
#cov_maj = dat_maj['cov']
#cov_min = dat_min['cov']
##cov_dim = dat_dim['cov']
##cov_bright = dat_bright['cov']
#cov_lowdens = np.loadtxt('SPTz25lowdensxredmagic_absmag1_brute_clran_z0.70_maglim22.5_cov.dat')[:-1,:-1]
#cov_highdens = np.loadtxt('SPTz25highdensxredmagic_absmag1_brute_clran_z0.70_maglim22.5_cov.dat')[:-1,:-1]
#cov_dim = dat_dim['cov']
#cov_bright = dat_bright['cov']
cov_all = dat_all['cov']
cov_red = dat_red['cov']
cov_green = dat_green['cov']
cov_blue = dat_blue['cov']
cov_clr = dat_clr['cov']

#covinv_low = np.linalg.inv(cov_low)
#covinv_high = np.linalg.inv(cov_high)
#covinv_maj = np.linalg.inv(cov_maj)
#covinv_min = np.linalg.inv(cov_min)
##covinv_dim = np.linalg.inv(cov_dim)
##covinv_bright = np.linalg.inv(cov_bright)
#covinv_lowdens = np.linalg.inv(cov_lowdens)
#covinv_highdens = np.linalg.inv(cov_highdens)
#covinv_dim = np.linalg.inv(cov_dim)
#covinv_bright = np.linalg.inv(cov_bright)
covinv_all = np.linalg.inv(cov_all)
covinv_red = np.linalg.inv(cov_red)
covinv_green = np.linalg.inv(cov_green)
covinv_blue = np.linalg.inv(cov_blue)
covinv_clr = np.linalg.inv(cov_clr)
#xi225 = dat225['xi']
#cov225 = dat225['cov']
#covinv225 = np.linalg.inv(cov225)
#xi220 = dat220['xi']
#cov220 = dat220['cov']
#covinv220 = np.linalg.inv(cov220)
#xi215 = dat215['xi']
#cov215 = dat215['cov']
#covinv215 = np.linalg.inv(cov215)

'''
errorbar(rmid,xi225,np.sqrt(np.diag(cov225)),label='Y3RM galaxies')
yscale('log')
xscale('log')
xlabel('Mpc/h')
ylabel(r'$\\xi$')
legend()
savefig('galaxy_profile_y3rm.pdf')
exit()

dat_bright = np.load('./output/Sig_Y3RM0-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim0.0-21.6.dat.npz')
xi_bright = dat_bright['xi']
cov_bright = dat_bright['cov']
covinv_bright = np.linalg.inv(cov_bright)

dat_dim  = np.load('./output/Sig_Y3RM0-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim21.6-22.5.dat.npz')
xi_dim = dat_dim['xi']
cov_dim = dat_dim['cov']
covinv_dim = np.linalg.inv(cov_dim)
'''

#dat = np.load('./output/Sig_Y3RM0-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim0.0-21.6_and_Sig_Y3RM0-100_clr-magerr0.1_nR15_0.14-14.29_z0.20-0.55_absmag1_maglim21.6-22.5.dat.npz')
#xi = dat['xi']
#cov = dat['cov']
#covinv = np.linalg.inv(cov)

#nll = lambda *args: -ln_prob(*args)
nll_single = lambda *args: -ln_prob_spt(*args)
nll = lambda *args: -ln_prob_spt_clr(*args)
#nll = lambda *args: -ln_prob_diff(*args)
#init0 = np.array([ 0.09854578, -0.7804555,  -0.57428576,  0.00550609,  0.81632162,  0.56406143, -2.0076012,   1.62481633, -1.146114384,  0.15857366])
init0 = np.array([ 1.49854578, -0.9804555,  -1.17428576,  0.06550609,  0.91632162,  0.76406143, -1.8076012,   1.62481633, -2.76114384])
#init0 = np.hstack((init0,init0))
#args225 = (rmid,xi225,covinv225)
#args220 = (rmid,xi220,covinv220)
#args215 = (rmid,xi215,covinv215)
#args225 = (rmid,xi225,covinv225)
#args_low  = (rmid,xi_low,covinv_low)
#args_high = (rmid,xi_high,covinv_high)
#args_maj = (rmid,xi_maj,covinv_maj)
#args_min = (rmid,xi_min,covinv_min)
#args_bright = (rmid,xi_bright,covinv_bright)
#args_dim = (rmid,xi_dim,covinv_dim)
#args_lowdens = (rmid,Sig_lowdens,covinv_lowdens)
#args_highdens = (rmid,Sig_highdens,covinv_highdens)
#args_dim = (rmid,Sig_dim,covinv_dim)
#args_bright = (rmid,Sig_bright,covinv_bright)
args_all = (rmid,Sig_all,covinv_all)
args_red = (rmid,Sig_red,covinv_red)
args_green = (rmid,Sig_green,covinv_green)
args_blue = (rmid,Sig_blue,covinv_blue)
args_clr = (rmid,Sig_clr,covinv_clr)
method = 'SLSQP'
#bounds = ((-3,4),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-4.0,2.0),(0.1,10.),(np.log(0.01),np.log(0.99)),(0.01,0.99))
bounds_single = ((-5,5),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-5.0,5.0),(0.1,10.),(np.log(0.01),np.log(0.99)))
bounds = ((-5,5),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-5.0,5.0),(0.1,10.),(-5,5),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-5.0,5.0),(0.1,10.),(-5,5),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-5.0,5.0),(0.1,10.),(np.log(0.01),np.log(0.99)))
#bounds = ((-3,4),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-4.0,2.0),(0.1,10.),(np.log(0.01),np.log(0.99)),(0.01,0.99),(-3,4),(-2.,2.),(-2.0,np.log10(5.)),(np.log10(0.1),np.log10(5.)),(-2.,np.log10(100.)),(-2.,np.log10(100.)),(-4.0,2.0),(0.1,10.),(np.log(0.01),np.log(0.99)),(0.01,0.99))
#result225 = op.minimize(nll, init0, args=args225, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par225 = result225.x
#print par225
#result_low = op.minimize(nll, init0, args=args_low, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_low = result_low.x
#print par_low
#result_high = op.minimize(nll, init0, args=args_high, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_high = result_high.x
#print par_high
#result_maj = op.minimize(nll, init0, args=args_maj, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_maj = result_maj.x
#print par_maj
#result_min = op.minimize(nll, init0, args=args_min, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_min = result_min.x
#print par_min
#result220 = op.minimize(nll, init0, args=args225, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par220 = result220.x
#print par220
#result215 = op.minimize(nll, init0, args=args215, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par215 = result215.x
#print par215
##result_bright = op.minimize(nll, init0, args=args_bright, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
##par_bright = result_bright.x
##print par_bright
##result_dim = op.minimize(nll, init0, args=args_dim, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
##par_dim = result_dim.x
##print par_dim
#result_lowdens = op.minimize(nll, init0, args=args_lowdens, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_lowdens = result_lowdens.x
#print par_lowdens
#result_highdens = op.minimize(nll, init0, args=args_highdens, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_highdens = result_highdens.x
#print par_highdens
#result_dim = op.minimize(nll, init0, args=args_dim, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_dim = result_dim.x
#print par_dim
#result_bright = op.minimize(nll, init0, args=args_bright, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds)
#par_bright = result_bright.x
#print par_bright

result_all = op.minimize(nll_single, init0, args=args_all, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds_single)
par_all = result_all.x
result_red = op.minimize(nll_single, init0, args=args_red, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds_single)
par_red = result_red.x
result_green = op.minimize(nll_single, init0, args=args_green, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds_single)
par_green = result_green.x
result_blue = op.minimize(nll_single, init0, args=args_blue, method = method, options = {'maxiter':1000,'ftol':1e-6}, bounds = bounds_single)
par_blue = result_blue.x

'''
init0_clr = [ 1.6677885,  -0.75202328, -0.68243948,  0.41218095,  1.18762404,  0.96983108, -0.88012186,  1.71897756, -0.5283801,  -0.87333436,  0.01985132,  0.44423692, 0.65599002,  0.58848008, -1.77927874,  1.04450833,  0.86646279, -1.19861962, -0.79251121,  0.27421627,  0.82188421,  0.62237007, -2.73865154,  1.5, -1.89564379]
result_clr = op.minimize(nll, init0_clr, args=args_clr, method = method, options = {'maxiter':1000, 'ftol':1e-6}, bounds = bounds)
par_clr = result_clr.x
'''
#result = op.minimize(nll,init0,args=args,method=method,options={'maxiter':1000,'ftol':1e-6},bounds=bounds)
#par = result.x
#print par
#exit()

'''
print 'start fitting for difference'
init = result.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(20) for i in range(128)]
sampler = emcee.EnsembleSampler(128, 20, ln_prob_diff, args=(rmid,rmid,xi,covinv),threads=32)
sampler.run_mcmc(pos,10000)
samples = sampler.flatchain
lnprob = sampler.flatlnprobability
np.savetxt('sp_params_magbin2_diff.dat',np.vstack((samples.T,lnprob)))
'''
#mcmc = np.loadtxt('sp_params_magbin2_diff.dat')
#samples = mcmc[:-1].T
#lnprob = mcmc[-1]



'''
#print 'start fitting for MCMF 22.5'
#init = result225.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(10) for i in range(128)]
#sampler225 = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi225,covinv225),threads=32)
#sampler225.run_mcmc(pos,10000)
#samples225 = sampler225.flatchain
#lnprob225 = sampler225.flatlnprobability
#np.savetxt('sp_params_MCMF_m22.5.dat',np.vstack((samples225.T,lnprob225)))
mcmc225 = np.loadtxt('sp_params_MCMF_m22.5.dat')
samples225 = mcmc225[:-1].T
lnprob225 = mcmc225[-1]
'''

#print 'start fitting for Y3RM 22.5'
#init = result225.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(10) for i in range(128)]
#sampler225 = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi225,covinv225),threads=32)
#sampler225.run_mcmc(pos,10000)
#samples225 = sampler225.flatchain
#lnprob225 = sampler225.flatlnprobability
#np.savetxt('sp_params_Y3RM_m22.5.dat',np.vstack((samples225.T,lnprob225)))
#mcmc225 = np.loadtxt('sp_params_Y3RM-RG_m22.5.dat')
#samples225 = mcmc225[:-1].T
#lnprob225 = mcmc225[-1]

'''
#print 'start fitting for Y3RM-RG 22.5'
#init = result225.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(10) for i in range(128)]
#sampler225 = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi225,covinv225),threads=32)
#sampler225.run_mcmc(pos,10000)
#samples225 = sampler225.flatchain
#lnprob225 = sampler225.flatlnprobability
#np.savetxt('sp_params_Y3RM-RG_m22.5.dat',np.vstack((samples225.T,lnprob225)))
mcmc225 = np.loadtxt('sp_params_Y3RM-RG_m22.5.dat')
samples225 = mcmc225[:-1].T
lnprob225 = mcmc225[-1]

print 'start fitting for Y3RM-RG 22.0'
init = result220.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(10) for i in range(128)]
sampler220 = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi220,covinv220),threads=32)
sampler220.run_mcmc(pos,10000)
samples220 = sampler220.flatchain
lnprob220 = sampler220.flatlnprobability
np.savetxt('sp_params_Y3RM-RG_m22.0.dat',np.vstack((samples220.T,lnprob220)))
#mcmc220 = np.loadtxt('sp_params_Y3RM-RG_m22.0.dat')
#samples220 = mcmc220[:-1].T
#lnprob220 = mcmc220[-1]

#print 'start fitting for Y3RM-RG 21.5'
#init = result215.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(10) for i in range(128)]
#sampler215 = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi215,covinv215),threads=32)
#sampler215.run_mcmc(pos,10000)
#samples215 = sampler215.flatchain
#lnprob215 = sampler215.flatlnprobability
#np.savetxt('sp_params_Y3RM-RG_m21.5.dat',np.vstack((samples215.T,lnprob215)))
mcmc215 = np.loadtxt('sp_params_Y3RM-RG_m21.5.dat')
samples215 = mcmc215[:-1].T
lnprob215 = mcmc215[-1]
'''
'''
print 'start fitting for Y3RM bright sample'
init = result_bright.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(10) for i in range(128)]
sampler_bright = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi_bright,covinv_bright),threads=32)
sampler_bright.run_mcmc(pos,10000)
samples_bright = sampler_bright.flatchain
lnprob_bright = sampler_bright.flatlnprobability
np.savetxt('sp_params_Y3RM_m0.0-21.6.dat',np.vstack((samples_bright.T,lnprob_bright)))
#mcmc_bright = np.loadtxt('sp_params_Y3RM_m0.0-21.6.dat')
#samples_bright = mcmc_bright[:-1].T
#lnprob_bright = mcmc_bright[-1]

print 'start fitting for Y3RM dim sample'
init = result_dim.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(10) for i in range(128)]
sampler_dim = emcee.EnsembleSampler(128, 10, ln_prob, args=(rmid,xi_dim,covinv_dim),threads=32)
sampler_dim.run_mcmc(pos,10000)
samples_dim = sampler_dim.flatchain
lnprob_dim = sampler_dim.flatlnprobability
np.savetxt('sp_params_Y3RM_m21.6-22.5.dat',np.vstack((samples_dim.T,lnprob_dim)))
#mcmc_dim = np.loadtxt('sp_params_Y3RM_m21.6-22.5.dat')
#samples_dim = mcmc_dim[:-1].T
#lnprob_dim = mcmc_dim[-1]
'''
'''
print 'start fitting for AdvACT bright sample'
init = result_bright.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(9) for i in range(128)]
sampler_bright = emcee.EnsembleSampler(128, 9, ln_prob_spt, args=(rmid,Sig_bright,covinv_bright),threads=32)
sampler_bright.run_mcmc(pos,10000)
samples_bright = sampler_bright.flatchain
lnprob_bright = sampler_bright.flatlnprobability
np.savetxt('sp_params_AdvACT_m0.0-20.8.dat',np.vstack((samples_bright.T,lnprob_bright)))
#mcmc_bright = np.loadtxt('sp_params_AdvACT_m0.0-20.8.dat')
#samples_bright = mcmc_bright[:-1].T
#lnprob_bright = mcmc_bright[-1]

print 'start fitting for AdvACT dim sample'
init = result_dim.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(9) for i in range(128)]
sampler_dim = emcee.EnsembleSampler(128, 9, ln_prob_spt, args=(rmid,Sig_dim,covinv_dim),threads=32)
sampler_dim.run_mcmc(pos,10000)
samples_dim = sampler_dim.flatchain
lnprob_dim = sampler_dim.flatlnprobability
np.savetxt('sp_params_AdvACT_m22.1-22.5.dat',np.vstack((samples_dim.T,lnprob_dim)))
#mcmc_dim = np.loadtxt('sp_params_AdvACT_m22.1-22.5.dat')
#samples_dim = mcmc_dim[:-1].T
#lnprob_dim = mcmc_dim[-1]
'''


#print 'start fitting for AdvACT all galaxy sample'
#init = result_all.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(9) for i in range(128)]
#sampler_all = emcee.EnsembleSampler(128, 9, ln_prob_spt, args=(rmid,Sig_all,covinv_all),threads=32)
#sampler_all.run_mcmc(pos,10000)
#samples_all = sampler_all.flatchain
#lnprob_all = sampler_all.flatlnprobability
#np.savetxt('sp_params_AdvACT_all_m22.5.dat',np.vstack((samples_all.T,lnprob_all)))
mcmc_all = np.loadtxt('sp_params_AdvACT_all_m22.5.dat')
samples_all = (mcmc_all[:-1].T)[128*3000:]
lnprob_all = mcmc_all[-1][128*3000:]


#print 'start fitting for AdvACT red galaxy sample'
#init = result_red.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(9) for i in range(128)]
#sampler_red = emcee.EnsembleSampler(128, 9, ln_prob_spt, args=(rmid,Sig_red,covinv_red),threads=32)
#sampler_red.run_mcmc(pos,10000)
#samples_red = sampler_red.flatchain
#lnprob_red = sampler_red.flatlnprobability
#np.savetxt('sp_params_AdvACT_red_m22.5.dat',np.vstack((samples_red.T,lnprob_red)))
mcmc_red = np.loadtxt('sp_params_AdvACT_red_m22.5.dat')
samples_red = (mcmc_red[:-1].T)[128*3000:]
lnprob_red = mcmc_red[-1][128*3000:]

#print 'start fitting for AdvACT green galaxy sample'
#init = result_green.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(9) for i in range(128)]
#sampler_green = emcee.EnsembleSampler(128, 9, ln_prob_spt, args=(rmid,Sig_green,covinv_green),threads=32)
#sampler_green.run_mcmc(pos,10000)
#samples_green = sampler_green.flatchain
#lnprob_green = sampler_green.flatlnprobability
#np.savetxt('sp_params_AdvACT_green_m22.5.dat',np.vstack((samples_green.T,lnprob_green)))
mcmc_green = np.loadtxt('sp_params_AdvACT_green_m22.5.dat')
samples_green = (mcmc_green[:-1].T)[128*3000:]
lnprob_green = mcmc_green[-1][128*3000:]

#print 'start fitting for AdvACT blue galaxy sample'
#init = result_blue.x
#scatter = 0.1*init
#pos = [init + scatter*np.random.randn(9) for i in range(128)]
#sampler_blue = emcee.EnsembleSampler(128, 9, ln_prob_spt, args=(rmid,Sig_blue,covinv_blue),threads=32)
#sampler_blue.run_mcmc(pos,10000)
#samples_blue = sampler_blue.flatchain
#lnprob_blue = sampler_blue.flatlnprobability
#np.savetxt('sp_params_AdvACT_blue_m22.5.dat',np.vstack((samples_blue.T,lnprob_blue)))
mcmc_blue = np.loadtxt('sp_params_AdvACT_blue_m22.5.dat')
samples_blue = (mcmc_blue[:-1].T)[128*3000:]
lnprob_blue = mcmc_blue[-1][128*3000:]

'''
print 'start fitting for AdvACT all colors simulataneously'
init = np.array(init0_clr) #result_clr.x
scatter = 0.1*init
pos = [init + scatter*np.random.randn(25) for i in range(128)]
sampler_clr = emcee.EnsembleSampler(128, 25, ln_prob_spt_clr, args=(rmid,Sig_clr,covinv_clr),threads=128)
sampler_clr.run_mcmc(pos,2000)
samples_clr = sampler_clr.flatchain
lnprob_clr = sampler_clr.flatlnprobability
np.savetxt('sp_params_AdvACT_clr_m22.5.dat',np.vstack((samples_clr.T,lnprob_clr)))
'''
#mcmc_clr = np.loadtxt('sp_params_AdvACT_clr_m22.5.dat')
#samples_clr = (mcmc_clr[:-1].T)[1000:]
#lnprob_clr = (mcmc_clr[-1])[1000:]

#ind_maxlike = np.argmax(lnprob_clr)

def profile_bestpar(params,nR):
	r = np.logspace(np.log10(0.1),np.log10(10.),nR)
	lgrho_s = params[0]
	lgalpha = params[1]
	lgr_s = params[2]
	lgr_t = params[3]
	lgbeta = params[4]
	lggamma = params[5]
	lgrho_0 = params[6]
	s_e = params[7]
	alpha = 10.**(lgalpha)
	beta = 10.**(lgbeta)
	gamma = 10.**(lggamma)
	rho_s = 10.**(lgrho_s)
	rho_0 = 10.**(lgrho_0)
	r_s = 10.**(lgr_s)
	r_t = 10.**(lgr_t)
	rho_ein = rho_s * np.exp((-2./alpha)*((r/r_s)**(alpha)-1))
	f_trans = (1+(r/r_t)**beta)**(-gamma/beta)
	rho_coll = rho_ein * f_trans
	rho_infall = rho_0 * (1./rhomax + (r/r00)**(s_e))**(-1)
	rho = rho_coll + rho_infall #+ 0.35*rho_0
	prof= (np.log(rho[2:])-np.log(rho[:-2]))/(np.log(r[2])-np.log(r[0]))
	rsp = r[1:-1][np.argmin(prof)]
	prof_coll = (np.log(rho_coll[2:])-np.log(rho_coll[:-2]))/(np.log(r[2])-np.log(r[0]))
	slp = prof_coll[np.argmin(prof)]
        return r[1:-1],prof,prof_coll,rsp,slp

#rsp_diff=[]
#nsam = 10000
#ind = np.random.choice(128*7000,nsam)
#for i in range(nsam):
#	par1 = samples[3000*128:][ind[i],:10]
#	par2 = samples[3000*128:][ind[i],10:]
#	sp1 = profile_bestpar(par1,4000)[3]
#	sp2 = profile_bestpar(par2,4000)[3]
#	rsp_diff.append(sp2-sp1)
#rsp_diff = np.array(rsp_diff)

#print np.sum(rsp_diff<0)
#exit()

#fit_bright = profile_bestpar(par_bright,4000)
#print fit_bright[3]
#fit_dim = profile_bestpar(par_dim,4000)
#print fit_dim[3]
#fit_bright = profile_bestpar(par[:10],4000)
#fit_dim = profile_bestpar(par[10:],4000)
#print fit_bright[3]
#print fit_dim[3]
#fit225 = profile_bestpar(par225,4000)
#plot(fit225[0],fit225[1],'b',label='Y3RM 22.5')

#fit_low = profile_bestpar(par_low,4000)
#plot(fit_low[0],fit_low[1],'b',label='low mag-gap, galaxies')
#plot(fit_low[0],fit_low[2],'b--')
#fit_high = profile_bestpar(par_high,4000)
#plot(fit_high[0],fit_high[1],'r',label='high mag-gap, galaxies')
#plot(fit_high[0],fit_high[2],'r--')

#fit_maj = profile_bestpar(par_maj,4000)
#plot(fit_maj[0],fit_maj[1],'b',label='major axis, galaxies')
#plot(fit_maj[0],fit_maj[2],'b--')
#fit_min = profile_bestpar(par_min,4000)
#plot(fit_min[0],fit_min[1],'r',label='minor axis, galaxies')
#plot(fit_min[0],fit_min[2],'r--')

##fit_dim = profile_bestpar(par_dim,4000)
##plot(fit_dim[0],fit_dim[1],'k',label='dimmer [22.1,22.5]')
##plot(fit_dim[0],fit_dim[2],'k--')
##fit_bright = profile_bestpar(par_bright,4000)
##plot(fit_bright[0],fit_bright[1],'b',label='brighter [0.0,20.8]')
##plot(fit_bright[0],fit_bright[2],'b--')

#fit_lowdens = profile_bestpar(par_lowdens,4000)
#plot(fit_lowdens[0],fit_lowdens[1],'k',label='low density')
#plot(fit_lowdens[0],fit_lowdens[2],'k--')
#fit_highdens = profile_bestpar(par_highdens,4000)
#plot(fit_highdens[0],fit_highdens[1],'b',label='high density')
#plot(fit_highdens[0],fit_highdens[2],'b--')

#fit_dim = profile_bestpar(par_dim,4000)
#plot(fit_dim[0],fit_dim[1],'k',label='AdvACT m_i=22.1-22.5')
#plot(fit_dim[0],fit_dim[2],'k--')
#fit_bright = profile_bestpar(par_bright,4000)
#plot(fit_bright[0],fit_bright[1],'b',label='AdvACT m_i=0.0-20.8')
#plot(fit_bright[0],fit_bright[2],'b--')


fit_all = profile_bestpar(par_all,4000)
plot(fit_all[0],fit_all[1],'k',label='AdvACT all galaxy')
plot(fit_all[0],fit_all[2],'k--')
'''
fit_red = profile_bestpar(par_red,4000)
plot(fit_red[0],fit_red[1],'r',label='AdvACT red galaxy')
plot(fit_red[0],fit_red[2],'r--')
fit_green = profile_bestpar(par_green,4000)
plot(fit_green[0],fit_green[1],'g',label='AdvACT green galaxy')
plot(fit_green[0],fit_green[2],'g--')
fit_blue = profile_bestpar(par_blue,4000)
plot(fit_blue[0],fit_blue[1],'b',label='AdvACT blue galaxy')
plot(fit_blue[0],fit_blue[2],'b--')
'''
#### HAVE TO CODE THIS

'''
ind_red_sel = np.array([0,1,2,3,4,5,6,7,24])
ind_green_sel = np.array([8,9,10,11,12,13,14,15,24])
ind_blue_sel = np.array([16,17,18,19,20,21,22,23,24])

par_red = samples_clr[ind_maxlike][ind_red_sel]
par_green = samples_clr[ind_maxlike][ind_green_sel]
par_blue = samples_clr[ind_maxlike][ind_blue_sel]
'''
fit_red = profile_bestpar(par_red,4000)
plot(fit_red[0],fit_red[1],'r',label='AdvACT red galaxy')
plot(fit_red[0],fit_red[2],'r--')
fit_green = profile_bestpar(par_green,4000)
plot(fit_green[0],fit_green[1],'g',label='AdvACT green galaxy')
plot(fit_green[0],fit_green[2],'g--')
fit_blue = profile_bestpar(par_blue,4000)
plot(fit_blue[0],fit_blue[1],'b',label='AdvACT blue galaxy')
plot(fit_blue[0],fit_blue[2],'b--')

#fit220 = profile_bestpar(par220,4000)
#plot(fit220[0],fit220[1],'r',label='Y3RM R+G 22.0')
#fit215 = profile_bestpar(par215,4000)
#plot(fit215[0],fit215[1],'r',label='Y3RM R+G 21.5')
#print fit225[3]
#print fit225[4]

#print fit_low[3], 'low'
#print fit_low[4]
#print fit_high[3], 'high'
#print fit_high[4]


#print fit_maj[3], 'major'
#print fit_maj[4]
#print fit_min[3], 'minor'
#print fit_min[4]

#print fit_dim[3], 'dimmer'
#print fit_dim[4]
#print fit_bright[3], 'brighter'
#print fit_bright[4]

#print fit_lowdens[3], 'low'
#print fit_lowdens[4]
#print fit_highdens[3], 'high'
#print fit_highdens[4]

#print fit_dim[3], 'dim'
#print fit_dim[4]
#print fit_bright[3], 'bright'
#print fit_bright[4]

print(fit_all[3], 'all')
print(fit_all[4])
print(fit_red[3], 'red')
print(fit_red[4])
print(fit_green[3], 'green')
print(fit_green[4])
print(fit_blue[3], 'blue')
print(fit_blue[4])

#print fit220[3]
##print fit220[4]
#print fit215[3]
##print fit215[4]
#errorbar(1.13,-2.5,xerr=0.07,color='g',label='Y1 rsp')
#plot(fit225[3],-2.5,'bo',label='Y3RM R+G 22.5')
#plot(fit220[3],-2.55,'go',label='Y3RM R+G 22.0')
#plot(fit215[3],-2.45,'ro',label='Y3RM R+G 21.5')
#rr = np.logspace(np.log10(0.1),np.log10(10.),1000)
#sigma_fit225 = sigma_model(par225,rr)
#sigma_fit220 = sigma_model(par220,rr)
#sigma_fit215 = sigma_model(par215,rr)
legend()
xscale('log')
#yscale('log')
ylim([-5,0])
savefig('slope_AdvACT_color.pdf')
close()
#errorbar(rmid,rmid*xi225,rmid*np.sqrt(np.diag(cov225)),color='b',label='Y3RM R+G 22.5')
#errorbar(rmid,rmid*xi220,rmid*np.sqrt(np.diag(cov220)),color='g',label='Y3RM R+G 22.0')
#errorbar(rmid,rmid*xi215,rmid*np.sqrt(np.diag(cov215)),color='r',label='Y3RM R+G 21.5')
#plot(rr,rr*sigma_fit225,'b')
#plot(rr,rr*sigma_fit220,'g')
#plot(rr,rr*sigma_fit215,'r')
#xscale('log')
#yscale('log')
#savefig('profile_RM_RG_comp.pdf')



def profile_range(params,nsteps,nR,filename,fast=False):
    if fast:
        data_range = np.load('slope_range_%s.npz'%(filename))
        low = data_range['low']
        high = data_range['high']
        low_coll = data_range['low_coll']
        high_coll = data_range['high_coll']
        r = np.logspace(np.log10(0.1),np.log10(10.),nR)
	    rsp_low = data_range['rsp_low']
	    rsp_high = data_range['rsp_high']
	    slp_low = data_range['slp_low']
	    slp_high = data_range['slp_high']
    else:
        nsample = 10000
        arr = np.random.choice(128*nsteps,nsample)
        r = np.logspace(np.log10(0.1),np.log10(10.),nR)
        prof_stack = np.zeros((nsample,nR-2))
        prof_coll_stack = np.zeros((nsample,nR-2))
	rsp_stack = np.zeros(nsample)
	slp_stack = np.zeros(nsample)
        for i in range(nsample):
                lgrho_s = params[arr[i],0]
                lgalpha = params[arr[i],1]
                lgr_s = params[arr[i],2]
                lgr_t = params[arr[i],3]
                lgbeta = params[arr[i],4]
                lggamma = params[arr[i],5]
                lgrho_0 = params[arr[i],6]
                s_e = params[arr[i],7]
                alpha = 10.**(lgalpha)
                beta = 10.**(lgbeta)
                gamma = 10.**(lggamma)
                rho_s = 10.**(lgrho_s)
                rho_0 = 10.**(lgrho_0)
                r_s = 10.**(lgr_s)
                r_t = 10.**(lgr_t)
                rho_ein = rho_s * np.exp((-2./alpha)*((r/r_s)**(alpha)-1))
                f_trans = (1+(r/r_t)**beta)**(-gamma/beta)
                rho_coll = rho_ein * f_trans
                rho_infall = rho_0 * (1./rhomax + (r/r00)**(s_e))**(-1)
                rho = rho_coll + rho_infall #+ 0.35*rho_0
                prof_stack[i] = (np.log(rho[2:])-np.log(rho[:-2]))/(np.log(r[2])-np.log(r[0]))
		rsp_stack[i] = r[1:-1][np.argmin(prof_stack[i])]
                prof_coll_stack[i] = (np.log(rho_coll[2:])-np.log(rho_coll[:-2]))/(np.log(r[2])-np.log(r[0]))
		slp_stack[i] = prof_coll_stack[i][np.argmin(prof_stack[i])]
        rho_coll_sum = np.sum(prof_coll_stack,axis=1)
        indcoll = (np.isnan(rho_coll_sum) == 0.)
        prof_coll_stack = prof_coll_stack[indcoll]
        low = np.percentile(prof_stack,16,axis=0)
        high = np.percentile(prof_stack,84,axis=0)
        low_coll = np.percentile(prof_coll_stack,16,axis=0)
        high_coll = np.percentile(prof_coll_stack,84,axis=0)
	rsp_low = np.percentile(rsp_stack,16)
	rsp_high = np.percentile(rsp_stack,84)
	slp_low = np.percentile(slp_stack,16)
	slp_high = np.percentile(slp_stack,84)
	print(np.sum(slp_stack<-3))
        np.savez('slope_range_%s'%(filename),low=low,high=high,low_coll=low_coll,high_coll=high_coll,rsp_low=rsp_low,rsp_high=rsp_high,slp_low=slp_low,slp_high=slp_high)
    return r[1:-1],low,high,low_coll,high_coll,rsp_low,rsp_high,slp_low,slp_high

close()
#rfit225,low225,high225,low_coll225,high_coll225,rsp_low225,rsp_high225,slp_low225,slp_high225 = profile_range(samples225,4000,'Y3RM_m22.5',fast=False) #y3rm == mcmf
#rfit220,low220,high220,low_coll220,high_coll220,rsp_low220,rsp_high220,slp_low220,slp_high220 = profile_range(samples220,4000,'Y3RM-RG_m22.0',fast=False)
#rfit215,low215,high215,low_coll215,high_coll215,rsp_low215,rsp_high215,slp_low215,slp_high215 = profile_range(samples215,4000,'Y3RM-RG_m21.5',fast=True)
#rfit_bright,low_bright,high_bright,low_coll_bright,high_coll_bright,rsp_low_bright,rsp_high_bright,slp_low_bright,slp_high_bright = profile_range(samples_bright,4000,'MCMF_m0.0-21.6',fast=False)
#rfit_dim,low_dim,high_dim,low_coll_dim,high_coll_dim,rsp_low_dim,rsp_high_dim,slp_low_dim,slp_high_dim = profile_range(samples_dim,4000,'MCMF_m21.6-22.5',fast=False)
#rfit_bright,low_bright,high_bright,low_coll_bright,high_coll_bright,rsp_low_bright,rsp_high_bright,slp_low_bright,slp_high_bright = profile_range(samples_bright,4000,'AdvACT_m0.0-20.8',fast=False)
#rfit_dim,low_dim,high_dim,low_coll_dim,high_coll_dim,rsp_low_dim,rsp_high_dim,slp_low_dim,slp_high_dim = profile_range(samples_dim,4000,'AdvACT_m22.1-22.5',fast=False)
#rfit_all, low_all, high_all, low_coll_all, high_coll_all, rsp_low_all, rsp_high_all, slp_low_all, slp_high_all = profile_range(samples_all[3000*128:],4000,'AdvACT_all_m22.5',fast=True)

'''
samples_red = ((samples_clr.T)[ind_red_sel]).T
samples_green = ((samples_clr.T)[ind_green_sel]).T
samples_blue = ((samples_clr.T)[ind_blue_sel]).T
'''
rfit_all, low_all, high_all, low_coll_all, high_coll_all, rsp_low_all, rsp_high_all, slp_low_all, slp_high_all = profile_range(samples_all,7000,4000,'AdvACT_all_m22.5',fast=True)
rfit_red, low_red, high_red, low_coll_red, high_coll_red, rsp_low_red, rsp_high_red, slp_low_red, slp_high_red = profile_range(samples_red,1000,4000,'AdvACT_red_m22.5',fast=True)
rfit_green, low_green, high_green, low_coll_green, high_coll_green, rsp_low_green, rsp_high_green, slp_low_green, slp_high_green = profile_range(samples_green,1000,4000,'AdvACT_green_m22.5',fast=True)
rfit_blue, low_blue, high_blue, low_coll_blue, high_coll_blue, rsp_low_blue, rsp_high_blue, slp_low_blue, slp_high_blue = profile_range(samples_blue,1000,4000,'AdvACT_blue_m22.5',fast=True)
#fill_between(rfit225,low225,high225,color='b',label='Y3RM 22.5',alpha=0.25)
#fill_between(rfit225,low_coll225,high_coll225,color='b',label='MCMF 22.5, coll',alpha=0.6)
#fill_between(rfit220,low220,high220,color='g',label='Y3RM R+G 22.0',alpha=0.25)
#fill_between(rfit220,low_coll220,high_coll220,color='g',label='Y3RM R+G 22.0, coll',alpha=0.6)
#fill_between(rfit215,low215,high215,color='r',label='Y3RM R+G 21.5',alpha=0.25)
#fill_between(rfit215,low_coll215,high_coll215,color='r',label='Y3RM R+G 21.5, coll',alpha=0.6)

#fill_between(rfit_bright,low_bright,high_bright,color='b',label='Y3RM imag=[0.0,21.6]',alpha=0.25)
#fill_between(rfit_bright,low_coll_bright,high_coll_bright,color='b',label='MCMF imag=[0.0,21.6], coll',alpha=0.6)
#errorbar(fit_bright[3],-2.0,xerr=0.5*(rsp_high_bright-rsp_low_bright),color='b')
#fill_between(rfit_dim,low_dim,high_dim,color='r',label='Y3RM imag=[21.6,22.5]',alpha=0.25)
#fill_between(rfit_dim,low_coll_dim,high_coll_dim,color='r',label='MCMF imag=[21.6,22.5], coll',alpha=0.6)
#errorbar(fit_dim[3],-2.05,xerr=0.5*(rsp_high_dim-rsp_low_dim),color='r')

#fill_between(rfit_bright,low_bright,high_bright,color='b',label='AdvACT imag=[0.0,20.8]',alpha=0.25)
#fill_between(rfit_bright,low_coll_bright,high_coll_bright,color='b',label='AdvACT imag=[0.0,20.8], coll',alpha=0.6)
#errorbar(fit_bright[3],-2.0,xerr=0.5*(rsp_high_bright-rsp_low_bright),color='b')
#fill_between(rfit_dim,low_dim,high_dim,color='r',label='AdvACT imag=[22.1,22.5]',alpha=0.25)
#fill_between(rfit_dim,low_coll_dim,high_coll_dim,color='r',label='AdvACT imag=[22.1,22.5], coll',alpha=0.6)
#errorbar(fit_dim[3],-2.05,xerr=0.5*(rsp_high_dim-rsp_low_dim),color='r')
fill_between(rfit_all,low_all,high_all,color='k',label='AdvACT all galaxies',alpha=0.6)
fill_between(rfit_all,low_coll_all,high_coll_all,color='k',label='AdvACT all galaxies, coll',alpha=0.2)
errorbar(fit_all[3],-1.45,xerr=0.5*(rsp_high_all-rsp_low_all),color='k')
fill_between(rfit_red,low_red,high_red,color='r',label='AdvACT red galaxies',alpha=0.6)
fill_between(rfit_red,low_coll_red,high_coll_red,color='r',label='AdvACT red galaxies, coll',alpha=0.2)
errorbar(fit_red[3],-1.50,xerr=0.5*(rsp_high_red-rsp_low_red),color='r')
fill_between(rfit_green,low_green,high_green,color='g',label='AdvACT green galaxies',alpha=0.6)
#fill_between(rfit_green,low_coll_green,high_coll_green,color='g',label='AdvACT green galaxies, coll',alpha=0.2)
errorbar(fit_green[3],-1.55,xerr=0.5*(rsp_high_green-rsp_low_green),color='g')
fill_between(rfit_blue,low_blue,high_blue,color='b',label='AdvACT blue galaxies',alpha=0.6)
#fill_between(rfit_blue,low_coll_blue,high_coll_blue,color='b',label='AdvACT blue galaxies, coll',alpha=0.6)
#errorbar(fit_blue[3],-2.10,xerr=0.5*(rsp_high_blue-rsp_low_blue),color='b')

#errorbar(fit225[3],-2.0,xerr=0.5*(rsp_high225-rsp_low225),color='g')
#errorbar(fit215[3],-1.95,xerr=0.5*(rsp_high215-rsp_low215),color='r')
#print fit225[3]
#print fit225[4]
#print rsp_low225,rsp_high225
#print slp_low225,slp_high225
#print rsp_low220,rsp_high220
#print slp_low220,slp_high220
#print rsp_low215,rsp_high215
#print slp_low215,slp_high215
#print rsp_low_bright,rsp_high_bright
#print slp_low_bright,slp_high_bright
#print rsp_low_dim,rsp_high_dim
#print slp_low_dim,slp_high_dim
print(rsp_low_all,rsp_high_all, 'all')
print(slp_low_all,slp_high_all)
print(rsp_low_red,rsp_high_red, 'red')
print(slp_low_red,slp_high_red)
print(rsp_low_green,rsp_high_green, 'green')
print(slp_low_green,slp_high_green)
print(rsp_low_blue,rsp_high_blue, 'blue')
#print slp_low_blue,slp_high_blue
xscale('log')
legend()
ylim([-6,0.])
xlabel(r'R$[h^{-1}{\rm Mpc}]$',size='x-large')
ylabel(r'$d\log\rho(r)/d\log r$')
tick_params(axis='both',which='both',direction='in',labelsize='x-large')
savefig('slope_fit_AdvACT_color.pdf')
close()

def fraction_3d_range(params_r,params_g,params_b,nsteps,nR,filename,fast=False):
    if fast:
        data_range = np.load('fraction_range_%s.npz'%(filename))
	r = np.logspace(np.log10(0.1),np.log10(10.),nR)
        low = data_range['low']
        high = data_range['high']
    else:
        nsample = 10000
        arr = np.random.choice(128*nsteps,nsample)
        r = np.logspace(np.log10(0.1),np.log10(10.),nR)
        low = np.zeros((nsample,3))
	high = np.zeros((nsample,3))
	frac_stack = np.zeros((nsample,nR,3))
        for i in range(nsample):
                lgrho_s_r = params_r[arr[i],0]
                lgalpha_r = params_r[arr[i],1]
                lgr_s_r = params_r[arr[i],2]
                lgr_t_r = params_r[arr[i],3]
                lgbeta_r = params_r[arr[i],4]
                lggamma_r = params_r[arr[i],5]
                lgrho_0_r = params_r[arr[i],6]
                s_e_r = params_r[arr[i],7]
                alpha_r = 10.**(lgalpha_r)
                beta_r = 10.**(lgbeta_r)
                gamma_r = 10.**(lggamma_r)
                rho_s_r = 10.**(lgrho_s_r)
                rho_0_r = 10.**(lgrho_0_r)
                r_s_r = 10.**(lgr_s_r)
                r_t_r = 10.**(lgr_t_r)
                rho_ein_r = rho_s_r * np.exp((-2./alpha_r)*((r/r_s_r)**(alpha_r)-1))
                f_trans_r = (1+(r/r_t_r)**beta_r)**(-gamma_r/beta_r)
                rho_coll_r = rho_ein_r * f_trans_r
                rho_infall_r = rho_0_r * (1./rhomax + (r/r00)**(s_e_r))**(-1)
                rho_r = rho_coll_r + rho_infall_r

		lgrho_s_g = params_g[arr[i],0]
                lgalpha_g = params_g[arr[i],1]
                lgr_s_g = params_g[arr[i],2]
                lgr_t_g = params_g[arr[i],3]
                lgbeta_g = params_g[arr[i],4]
                lggamma_g = params_g[arr[i],5]
                lgrho_0_g = params_g[arr[i],6]
                s_e_g = params_g[arr[i],7]
                alpha_g = 10.**(lgalpha_g)
                beta_g = 10.**(lgbeta_g)
                gamma_g = 10.**(lggamma_g)
                rho_s_g = 10.**(lgrho_s_g)
                rho_0_g = 10.**(lgrho_0_g)
                r_s_g = 10.**(lgr_s_g)
                r_t_g = 10.**(lgr_t_g)
                rho_ein_g = rho_s_g * np.exp((-2./alpha_g)*((r/r_s_g)**(alpha_g)-1))
                f_trans_g = (1+(r/r_t_g)**beta_g)**(-gamma_g/beta_g)
                rho_coll_g = rho_ein_g * f_trans_g
                rho_infall_g = rho_0_g * (1./rhomax + (r/r00)**(s_e_g))**(-1)
                rho_g = rho_coll_g + rho_infall_g

		lgrho_s_b = params_b[arr[i],0]
                lgalpha_b = params_b[arr[i],1]
                lgr_s_b = params_b[arr[i],2]
                lgr_t_b = params_b[arr[i],3]
                lgbeta_b = params_b[arr[i],4]
                lggamma_b = params_b[arr[i],5]
                lgrho_0_b = params_b[arr[i],6]
                s_e_b = params_b[arr[i],7]
                alpha_b = 10.**(lgalpha_b)
                beta_b = 10.**(lgbeta_b)
                gamma_b = 10.**(lggamma_b)
                rho_s_b = 10.**(lgrho_s_b)
                rho_0_b = 10.**(lgrho_0_b)
                r_s_b = 10.**(lgr_s_b)
                r_t_b = 10.**(lgr_t_b)
                rho_ein_b = rho_s_b * np.exp((-2./alpha_b)*((r/r_s_b)**(alpha_b)-1))
                f_trans_b = (1+(r/r_t_b)**beta_b)**(-gamma_b/beta_b)
                rho_coll_b = rho_ein_b * f_trans_b
                rho_infall_b = rho_0_b * (1./rhomax + (r/r00)**(s_e_b))**(-1)
                rho_b = rho_coll_b + rho_infall_b

		frac_stack[i,:,0] = rho_r/(rho_r + rho_g + rho_b)
		frac_stack[i,:,1] = rho_g/(rho_r + rho_g + rho_b)
		frac_stack[i,:,2] = rho_b/(rho_r + rho_g + rho_b)
        low = np.percentile(frac_stack,16,axis=0)
        high = np.percentile(frac_stack,84,axis=0)
        np.savez('fraction_range_%s'%(filename),low=low,high=high)
    return r, low, high

rfit_clr, low_frac, high_frac = fraction_3d_range(samples_red,samples_green,samples_blue,1000,4000,'AdvACT_clr_m22.5',fast=True)

fill_between(rfit_clr,low_frac[:,0],high_frac[:,0],color='r',label='AdvACT red galaxies',alpha=0.6)
fill_between(rfit_clr,low_frac[:,1],high_frac[:,1],color='g',label='AdvACT green galaxies',alpha=0.6)
fill_between(rfit_clr,low_frac[:,2],high_frac[:,2],color='b',label='AdvACT blue galaxies',alpha=0.6)
axvline(rsp_low_all,linestyle='--')
axvline(rsp_high_all,linestyle='--')
xscale('log')
legend()
ylim([0,1])
xlabel(r'r$[h^{-1}{\rm Mpc}]$',size='x-large')
ylabel('fraction',size='x-large')
tick_params(axis='both',which='both',direction='in',labelsize='x-large')
savefig('fraction_3d_AdvACT_color.pdf')
