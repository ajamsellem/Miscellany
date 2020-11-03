#! /home/taeshin/anaconda2/bin/python2.7
import numpy as np
import pyfits as pf
import healpy as hp
from math import pi
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy import units as u
import kmeans_radec
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from colossus.halo import *
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 70., 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.95}
cosmology.addCosmology('SimpleCosmo', params)
Cosmo = cosmology.setCosmology('SimpleCosmo')
h = Cosmo.H0/100

### DES data ###
clusters = pf.open('y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl02_catalog.fit')[1].data
#redshift and richness cut
l_t = clusters['LAMBDA_CHISQ']
z_t = clusters['Z_LAMBDA']
indcl = (l_t > 56.)*(z_t>0.25)*(z_t<0.7)

clusters = clusters[indcl]
memid_rm = clusters['MEM_MATCH_ID']
ra_rm = clusters['RA']
dec_rm = clusters['DEC']
z_rm = clusters['Z_LAMBDA']
zerr_rm = clusters['Z_LAMBDA_E']
lam_rm = clusters['LAMBDA_CHISQ']
m200_rm = (1.05*3.08e14)*(lam_rm/40.)**1.356*((1+z_rm)/1.35)**(-0.30) #mass-richness relation

#convert m200m to m500c
'''
m500_rm = np.zeros(len(m200_rm))
for i in range(len(m200_rm)):
	print i
	m500_rm_tem, _, _ = mass_adv.changeMassDefinitionCModel(m200_rm[i]*h, z_rm[i], '200m', '500c')
	m500_rm[i] = m500_rm_tem/h
np.savetxt('m500_y3rm_vl_lgt20_upp.dat',m500_rm)
'''
m500_rm = np.loadtxt('m500_y3rm_vl_lgt20_upp.dat') ##vl, full
print np.mean(m500_rm)


### DES footprint ###
y3_badregion_mask = hp.read_map('/home/taeshin/scripts/splashback/y3a2_badregions_mask_v2.0.fits.gz',nest=True)
y3_maglim_mask = hp.read_map('/home/taeshin/scripts/splashback/y3a2_gold_2_2_1_sof_nside4096_nest_i_depth.fits.gz',nest=True)
y3_foreground_mask = hp.read_map('/home/taeshin/scripts/splashback/y3a2_foreground_mask_v2.1.fits.gz',nest=True)
y3_maglim_mask[y3_badregion_mask==2] = 0.
y3_maglim_mask[y3_foreground_mask >= 2.] = 0.
ind_maglim = (y3_maglim_mask > 0.)
ft = ind_maglim
print 41253*np.sum(ft)/len(ft),'=total coverage of sky'


### SPT clusters ###
data_spt = pf.open('2500d_cluster_sample_fiducial_cosmology_withxicorr.fits')[1].data
data_spt_bocquet = pf.open('2500d_cluster_sample_Bocquet18.fits')[1].data
#print pf.open('2500d_cluster_sample_Bocquet18.fits')[1].columns
ra_spt = data_spt.field('RA')
dec_spt = data_spt.field('DEC')
z_spt = data_spt_bocquet.field('REDSHIFT')
zerr_spt = data_spt_bocquet.field('REDSHIFT_UNC')
m500_spt = data_spt_bocquet.field('M500')*1e14
m500err_spt_u = data_spt_bocquet.field('M500_uerr')*1e14
m500err_spt_u_marge = data_spt_bocquet.field('M500_marge_uerr')*1e14
m500err_spt_l = data_spt_bocquet.field('M500_lerr')*1e14
m500err_spt_l_marge = data_spt_bocquet.field('M500_marge_lerr')*1e14
thc_spt = data_spt.field('THETA_CORE')
xi_spt = data_spt.field('XI_CORRECTED')
id_spt = np.arange(len(ra_spt))

#select clusters in the DES footprint, apply the redshift cut and pick ones whose masses are estimated
theta_spt = (90.0 - dec_spt)*np.pi/180.
phi_spt = ra_spt*np.pi/180.
pix_spt = hp.ang2pix(4096, theta_spt, phi_spt, nest=True)
hp_pix_spt = hp.ang2pix(4096,pi/2.-dec_spt*pi/180.,ra_spt*pi/180.,nest=True)
ind_match_spt = (z_spt<0.7)*(z_spt>0.25)*(m500_spt!=0)*(ft[pix_spt]==1)


### ACT clusters ###
clusters_act = pf.open('E-D56Clusters.fits')[1].data
ra_act = clusters_act['RADeg']
dec_act = clusters_act['decDeg']
z_act = clusters_act['z']
zerr_act = clusters_act['zErr']
m500_act = clusters_act['M500cCal']*0.68/0.75
SNR_act = clusters_act['SNR2p4']
id_act = np.arange(len(clusters_act))

#the same selection for ACT clusters
theta_act = (90.0 - dec_act)*np.pi/180.
phi_act = ra_act*np.pi/180.
pix_act = hp.ang2pix(4096, theta_act, phi_act, nest=True)
hp_pix_act = hp.ang2pix(4096,pi/2.-dec_act*pi/180.,ra_act*pi/180.,nest=True)
ind_match_act = (z_act<0.7)*(z_act>0.25)*(m500_act!=0)*(ft[pix_act]==1)


### apply the cut
ra_spt_match = ra_spt[ind_match_spt]
dec_spt_match = dec_spt[ind_match_spt]
id_spt_match = id_spt[ind_match_spt]
z_spt_match = z_spt[ind_match_spt]
zerr_spt_match = zerr_spt[ind_match_spt]
m500_spt_match = m500_spt[ind_match_spt]
m500err_spt_u_match = m500err_spt_u[ind_match_spt]
m500err_spt_l_match = m500err_spt_l[ind_match_spt]
m500err_spt_u_marge_match = m500err_spt_u_marge[ind_match_spt]
m500err_spt_l_marge_match = m500err_spt_l_marge[ind_match_spt]
thc_spt_match = thc_spt[ind_match_spt]
xi_spt_match = xi_spt[ind_match_spt]
hp_pix_spt_match = hp_pix_spt[ind_match_spt]

ra_act_match = ra_act[ind_match_act]
dec_act_match = dec_act[ind_match_act]
z_act_match = z_act[ind_match_act]
m500_act_match = 1.0e14*m500_act[ind_match_act]
id_act_match = id_act[ind_match_act]
zerr_act_match = zerr_act[ind_match_act]
hp_pix_act_match = hp_pix_act[ind_match_act]
SNR_act_match = SNR_act[ind_match_act]
id_act_match = id_act[ind_match_act]

figure(figsize=(8,8))


### matching SPT/ACT to RM ##
cat_des = SkyCoord(ra=ra_rm*u.degree,dec=dec_rm*u.degree)
cat_spt = SkyCoord(ra=ra_spt_match*u.degree,dec=dec_spt_match*u.degree)
#cat_act = SkyCoord(ra=ra_act_match4*u.degree,dec=dec_act_match4*u.degree)
idx,d2d,d3d = cat_spt.match_to_catalog_sky(cat_des) #closest matching regardless of redshift
#idx,d2d,d3d = cat_act.match_to_catalog_sky(cat_des) #same for ACT
id_rm = np.arange(len(cat_des))

rm_id = np.zeros((len(cat_spt),5))
index_spt = np.zeros((len(cat_spt),5))
lam_spt = np.zeros((len(cat_spt),5))
z_nbr_rm = np.zeros((len(cat_spt),5))
zerr_nbr_rm = np.zeros((len(cat_spt),5))
separation = np.zeros((len(cat_spt),5))
memid_nbr_rm = np.zeros((len(cat_spt),5))
for i in range(len(cat_spt)): ##interate over the SPT clusters to find the 5 most closest RM clusters
	print i, len(cat_spt)
	separation_tem = cat_spt[i].separation(cat_des).radian ## separation
	sep_sort = np.argsort(separation_tem)  ## ascending order
	separation[i] = np.tan(separation_tem[sep_sort][:5])*(cosmo.comoving_distance(z_spt_match[i]).value)*0.7 # 5 most cloest ones in Mpc/h
	rm_id[i] = id_rm[sep_sort][:5] #corresponding properties
	lam_spt[i] = lam_rm[sep_sort][:5]
	z_nbr_rm[i] = z_rm[sep_sort][:5]
	zerr_nbr_rm[i] = zerr_rm[sep_sort][:5]
	memid_nbr_rm[i] = memid_rm[sep_sort][:5]
np.savez('match_sptxdesy3', separation = separation, rm_id = rm_id, lam_spt = lam_spt, z_nbr_rm = z_nbr_rm, zerr_nbr_rm = zerr_nbr_rm, memid_nbr_rm = memid_nbr_rm)
#exit()

#do the same for ACT
'''
des_id = np.zeros((len(cat_act),5))
lam_act = np.zeros((len(cat_act),5))
Ztile_des = np.zeros((len(cat_act),5))
Zerrtile_des = np.zeros((len(cat_act),5))
separation = np.zeros((len(cat_act),5))
memid_des = np.zeros((len(cat_act),5))
rades = np.zeros((len(cat_act),5))
decdes = np.zeros((len(cat_act),5))
for i in range(len(cat_act)):
        print i, len(cat_act)
        separation_tem = cat_act[i].separation(cat_des).radian
        sep_sort = np.argsort(separation_tem)
        separation[i] = np.tan(separation_tem[sep_sort][:5])*(cosmo.comoving_distance(z_act_match4[i]).value)*0.7
        des_id[i] = id_des[sep_sort][:5]
        lam_act[i] = LAMB[sep_sort][:5]
        Ztile_des[i] = Z[sep_sort][:5]
        Zerrtile_des[i] = Zerr[sep_sort][:5]
        memid_des[i] = memid[sep_sort][:5]
	rades[i] = ra_des[sep_sort][:5]
	decdes[i] = dec_des[sep_sort][:5]

np.savez('match_actxdes', separation = separation, des_id = des_id, lam_act = lam_act, Ztile_des = Ztile_des, Zerrtile_des = Zerrtile_des, memid_des = memid_des, rades = rades, decdes = decdes)
#exit()
'''
#load the saved
dat_match = np.load('match_sptxdesy3.npz')
separation = dat_match['separation']
rm_id = dat_match['rm_id']
lam_spt = dat_match['lam_spt']
z_nbr_rm = dat_match['z_nbr_rm']
zerr_nbr_rm = dat_match['zerr_nbr_rm']
memid_rm = dat_match['memid_nbr_rm']

'''
dat_match = np.load('match_actxdes.npz')#_snr5_z0p1-0p33.npz')
separation = dat_match['separation']
des_id = dat_match['des_id']
lam_act = dat_match['lam_act']
Ztile_des = dat_match['Ztile_des']
Zerrtile_des = dat_match['Zerrtile_des']
memid_des = dat_match['memid_des']
rades = dat_match['rades']
decdes = dat_match['decdes']
'''

ind = (np.amin(separation,axis=1)< 3.) ### select those whose closest match is smaller thatn 3Mpc/h
### YOU CAN CHANGE THIS NUMBER UP TO YOUR PREFERENCE


lambda_match = lam_spt[ind]
Z_unmatch = z_nbr_rm[~ind]
Z_match = z_nbr_rm[ind]
Zerr_match = zerr_nbr_rm[ind]
memid_match = memid_rm[ind]
z_spt_tem = z_spt_match[ind]
SNR_spt_tem = xi_spt_match[ind]
m500_spt_tem = m500_spt_match[ind]
z_spt_unmatch = z_spt_match[~ind]
zerr_spt_unmatch = zerr_spt_match[~ind]
zerr_spt_tem = zerr_spt_match[ind]
separation_match = separation[ind]
ra_spt_mat = ra_spt_match[ind]
dec_spt_mat = dec_spt_match[ind]
pix_spt_match = hp_pix_spt_match[ind]
pix_spt_unmatch = hp_pix_spt_match[~ind]
pix_spt_match = hp.nest2ring(4096,pix_spt_match)
pix_spt_unmatch = hp.nest2ring(4096,pix_spt_unmatch)

print len(z_spt_tem),'after ra/dec matching <3Mpc/h'
print len(ind)-len(z_spt_tem),'number of unmatched clusters'
'''
lambda_match = lam_act[ind]
Z_unmatch = Ztile_des[~ind]
Z_match = Ztile_des[ind]
rades_match = rades[ind]
decdes_match = decdes[ind]
Zerr_match = Zerrtile_des[ind]
memid_match = memid_des[ind]
z_act_tem = z_act_match4[ind]
z_act_unmatch = z_act_match4[~ind]
zerr_act_tem = zerr_act_match4[ind]
SNR_act_tem = SNR_act_match4[ind]
m500_act_tem = m500_act_match4[ind]
separation_match = separation[ind]
ra_act_mat = ra_act_match4[ind]
dec_act_mat = dec_act_match4[ind]
pix_act_match = hp_pix_act_match4[ind]
pix_act_match = hp.nest2ring(4096,pix_act_match)
id_act_tem = id_act_match4[ind]
#print len(z_act_tem),'after ra/dec matching <3Mpc/h'
#print separation[~ind],'closest distance of unmatched clusters'
#print Z_unmatch,'RM redshift values of angle-unmatched clusters'
#print z_act_unmatch,'ACT redshift values of angle-unmatched clusters'
'''

### the healpix map of the maximum redshift values that RM can detect the clusters; above those, there is no RM cluster
zmax_mask = pf.open('y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl50_vlim_zmask.fit')[1].data
hpix = zmax_mask['HPIX']
zmax_val = zmax_mask['ZMAX']
zmaxmask = np.zeros(12*4096**2)
zmaxmask[hpix] = zmax_val


### MATCH THE CLUSTERS!!
lam_spt = np.zeros(len(Z_match))
memid_spt = np.zeros(len(Z_match))
#index_proj_finalmatch = np.zeros(len(Z_match))
for i in range(len(Z_match)):
	z_diff_sig = (Z_match[i]-z_spt_tem[i])/np.sqrt(Zerr_match[i]**2+zerr_spt_tem[i]**2)
	if z_spt_tem[i]>zmaxmask[pix_spt_match[i]]:
		print 'redshift of this SPT cluster > maximum redshift of RM'
	separation_tem = separation_match[i]
	ind_tem = (np.abs(z_diff_sig)<5.)*(separation_tem<1.5)   ## separation is in Mpc/h, z_diff_sig is in mutiple of sigma
	lam_tem = lambda_match[i][ind_tem]
	z_tem = Z_match[i][ind_tem]
	memid_tem = memid_match[i][ind_tem]
	separation_temp = separation_tem[ind_tem]
	memid_tem = memid_match[i][ind_tem]
	print 'for the cluster at ra/dec=',ra_spt_mat[i],dec_spt_mat[i]
	print 'redshift of SPT cluster=',z_spt_tem[i]
	print 'redshift of RM cluster=',z_tem
	print z_diff_sig[ind_tem],'= delta z-',i
	print separation_temp,' separation-',i
	print lam_tem,'= lambda-',i
	if len(lam_tem)==0:
		print 'no match',i
		print z_diff_sig[~ind_tem],'delta z of unmatched-',i
		print separation_tem[~ind_tem],'separation of unmatched-',i
		print lambda_match[i][~ind_tem],'lambda of unmatched-',i
	print '------------------'
	if len(lam_tem) != 0: 
		lam_spt[i] = lam_tem[np.argmax(lam_tem)]
		memid_spt[i] = memid_tem[np.argmax(lam_tem)]
np.savez('spt-desy3_match',lam_spt=lam_spt, memid=memid_spt)
close()
#hist(lam_spt, histtype='step', color='r', range=(np.amin(lam_spt),np.amax(lam_spt)), bins=10, normed=True, label='SPT')
#hist(lambda_cl_agg, histtype='step', color='b', range=(np.amin(lam_spt),np.amax(lam_spt)), bins=10, normed=True, label='RM')
#savefig('lam_matched-cluster_RMxSPT.png')
indlam = (lam_spt != 0)
#np.savetxt('ACT-DES_match.dat',np.vstack((memid_act[indlam],lam_act[indlam])).T, header='MEM_MATCH_ID RICHNESS')
#exit()
print 'total number of matching',np.sum(indlam)
print 'total number of unmatching',len(indlam)-np.sum(indlam)
lam_spt_match = lam_spt[indlam]
z_spt_desmatch = z_spt_tem[indlam]
SNR_spt_temm = SNR_spt_tem[indlam]
#close()
#scatter(lam_spt_match,SNR_spt_temm)
#xscale('log')
#yscale('log')
#xlabel('richness')
#ylabel('SNR_SZ')
#savefig('lam-xi_lamsum.pdf')
m500_spt_temm = m500_spt_tem[indlam]
m200_spt_lam = 3.08e14*(lam_spt_match/40.)**1.356*((1+z_spt_desmatch)/1.35)**(-0.30)
m500_spt_lam = np.zeros(len(m200_spt_lam))
for i in range(len(m200_spt_lam)):
        m500_spt_lam[i], _, _ = mass_adv.changeMassDefinitionCModel(3.08e14*(lam_spt_match[i]/40.)**1.356*((1+z_spt_desmatch[i])/1.35)**(-0.30)*h, z_spt_desmatch[i], '200m', '500c')
m500_spt_lam = m500_spt_lam/h
m200_spt_avglam = 3.08e14*(np.mean(lam_spt_match)/40.)**1.356*((1+np.mean(z_spt_desmatch))/1.35)**(-0.30)
m500_spt_avglam, _, _ = mass_adv.changeMassDefinitionCModel(m200_spt_avglam*h, np.mean(z_spt_desmatch), '200m', '500c')
print 'average richness=',np.mean(lam_spt_match)
print 'average SPT mass=',np.mean(m500_spt_temm)
print 'average RM mass (mean richness)=',m500_spt_avglam/h
print 'average RM mass (mean mass)=',np.mean(m500_spt_lam)
print 'average redshift=',np.mean(z_spt_desmatch)

### I did the indexing by hand. You can automate the indexing above, if you want. 
# 0 no match
# 1 match
# 2 first + second  ## SPT clusters = possibly the sum of the first and the second RM match! 
# 3 second ## SPT clusters = the second RM match
# 4 first + third ## SPT clusters = possibly the sum of the first and the third RM match
### 3 is obvious, but 2 and 4 is totally up to you
match_flag = np.ones(len(pix_spt_match))

'''
match_flag[7] = 0
match_flag[13] = 2 
match_flag[17] = 0
match_flag[26] = 3
match_flag[49] = 2
match_flag[50] = 2
match_flag[68] = 3
match_flag[81] = 0
match_flag[83] = 2
match_flag[87] = 2
match_flag[124] = 3
match_flag[125] = 2
match_flag[135] = 0
match_flag[140] = 0
match_flag[142] = 0
match_flag[145] = 3
match_flag[154] = 2
match_flag[157] = 0 # or 4
match_flag[159] = 2
match_flag[165] = 0 # or 2/3 
match_flag[172] = 0
match_flag[181] = 0
match_flag[188] = 0
match_flag[203] = 2
match_flag[204] = 0
match_flag[211] = 0
match_flag[215] = 3
match_flag[216] = 0
match_flag[221] = 2 # or plus fourth
match_flag[223] = 0
match_flag[236] = 2
'''
match_flag = match_flag.astype(np.int64)

exit()



#print id_act_tem[45],id_act_tem[56],id_act_tem[64],id_act_tem[72],id_act_tem[74],'=ids no matching'
#print id_act_tem[13],id_act_tem[37],id_act_tem[86],id_act_tem[87],'ids redshift<zmax'
#exit()

'''
match_flag = np.ones(len(pix_act_match))
'''
#match_flag[6] = 0

#match_flag[18] = 0  #5.0
#match_flag[29] = 0

#match_flag[15] = 0  #5.5
'''
match_flag[13] = 2
match_flag[37] = 0
match_flag[41] = 0
match_flag[45] = 0
match_flag[56] = 0
match_flag[64] = 0
match_flag[72] = 0
match_flag[74] = 0
match_flag[86] = 0
match_flag[87] = 0
match_flag = match_flag.astype(np.int64)
'''
#indflag = (match_flag != 0.)
#ind_act_id = id_act_tem[indflag]
#np.savetxt('act_match_id.dat',ind_act_id,fmt='%d')
#exit()
'''
lam_act = np.zeros(len(cat_act))
memid_act = np.zeros(len(cat_act))
id_actmat = np.zeros(len(cat_act))
sep_actdes = np.zeros(len(cat_act))
ra_actdes = np.zeros(len(cat_act))
dec_actdes = np.zeros(len(cat_act))
for i in range(len(cat_act)):
#       if (i<=87): continue
        z_diff_sig = (Z_match[i]-z_act_tem[i])/np.sqrt(Zerr_match[i]**2+zerr_act_tem[i]**2)
        if z_act_tem[i]-zerr_act_tem[i]>zmaxmask[pix_act_match[i]]:
                print 'redshift of this ACT cluster > maximum redshift of RM'
        separation_tem = separation_match[i]
        ind_tem = (np.abs(z_diff_sig)<10.)*(separation_tem<3.)
        lam_tem = lambda_match[i][ind_tem]
        z_tem = Z_match[i][ind_tem]
	ra_temp = rades_match[i][ind_tem]
	dec_temp = decdes_match[i][ind_tem]
        memid_tem = memid_match[i][ind_tem]
        separation_temp = separation_tem[ind_tem]
        memid_tem = memid_match[i][ind_tem]
        print 'redshift of ACT cluster=',z_act_tem[i]
        print 'redshift of RM cluster=',z_tem
        print z_diff_sig[ind_tem],'delta z-',i
        print separation_temp,'separation-',i
        print lam_tem,'lambda-',i
	print SNR_act_tem[i],'=SNR-',i
        if len(lam_tem)==0:
                print z_diff_sig[~ind_tem],'delta z of unmatched-',i
                print separation_tem[~ind_tem],'separation of unmatched-',i
                print lambda_match[i][~ind_tem],'lambda of unmatched-',i
        print '------------------'
#        annotate('%.2f'%(z_act_tem[i]),xy=(ra_act_mat[i],dec_act_mat[i]),xytext=(ra_act_mat[i],dec_act_mat[i]))
#        plot(ra_act_mat[i],dec_act_mat[i],'bo')
#        for j in range(len(memid_tem)):
#                inddd = (memid == memid_tem[j])
#                annotate('z=%.2f\n %.2f Mpc/h\n l=%.1f'%(z_tem[j],separation_tem[j],lam_tem[j]),xy=(ra_des[inddd],dec_des[inddd]),xytext=(ra_des[inddd],dec_des[inddd]))
#                plot(ra_des[inddd],dec_des[inddd],'ro',markersize=lam_tem[j],alpha=0.5)
#        savefig('ACT_matching_%d.png'%(i))
        close()
	if match_flag[i] == 0:
                lam_act[i] = 0
		ra_actdes[i] =ra_act_mat[i]
		dec_actdes[i] = dec_act_mat[i]
		id_actmat[i] = id_act_tem[i]
        elif match_flag[i] == 1:
                lam_act[i] = lam_tem[0]
		memid_act[i] = memid_tem[0]
		id_actmat[i] = id_act_tem[i]
		sep_actdes[i] = separation_temp[0]
		dec_actdes[i] = dec_temp[0]
		ra_actdes[i] = ra_temp[0]
        elif match_flag[i] == 2:
                lam_act[i] = lam_tem[0] + lam_tem[1]
		memid_act[i] = memid_tem[0]
		id_actmat[i] = id_act_tem[i]
		sep_actdes[i] = separation_temp[0]
		dec_actdes[i] = dec_temp[0]
                ra_actdes[i] = ra_temp[0]


indlam = (lam_act != 0)
np.savetxt('ACT_RM-CG.dat',np.vstack((ra_actdes,dec_actdes)).T,header='RA_RM DEC_RM')
np.savetxt('ACT-DES_match.dat',np.vstack((id_actmat[indlam],memid_act[indlam],lam_act[indlam])).T, header='ACT ID MEM_MATCH_ID RICHNESS')
exit()
print 'total number of matching',np.sum(indlam)
lam_act_match = lam_act[indlam]
z_act_desmatch = z_act_tem[indlam]
SNR_act_temm = SNR_act_tem[indlam]
m500_act_temm = m500_act_tem[indlam]
m200_act_lam = 3.08e14*(lam_act_match/40.)**1.356*((1+z_act_desmatch)/1.35)**(-0.30)
m500_act_lam = np.zeros(len(m200_act_lam))
for i in range(len(m200_act_lam)):
	m500_act_lam[i], _, _ = mass_adv.changeMassDefinitionCModel(3.08e14*(lam_act_match[i]/40.)**1.356*((1+z_act_desmatch[i])/1.35)**(-0.30)*h, z_act_desmatch[i], '200m', '500c')
m500_act_lam = m500_act_lam/h
m200_act_avglam = 3.08e14*(np.mean(lam_act_match)/40.)**1.356*((1+np.mean(z_act_desmatch))/1.35)**(-0.30)
m500_act_avglam, _, _ = mass_adv.changeMassDefinitionCModel(m200_act_avglam*h, np.mean(z_act_desmatch), '200m', '500c')
print 'average richness=',np.mean(lam_act_match)
print 'average ACT mass=',np.mean(m500_act_temm)
print 'average RM mass (mean richness)=',m500_act_avglam/h
print 'average RM mass (mean mass)=',np.mean(m500_act_lam)
print 'average redshift=',np.mean(z_act_desmatch)

exit()
'''
#print len(lambda_match),'after redshift matching'
#print np.mean(lambda_match),'mean richness of SPT clusters'
#print Z_unmatch_z,'RM redshift values of redshift-unmatched clusters'
#print z_spt_unmatch,'SPT redshift values of redshift-unmatched clusters'
#exit()

#hist(lambda_match,bins=np.linspace(20,200,11),normed=True,alpha=0.3,label='SPT matched to RM')
#hist(m500_spt_tem,bins=np.linspace(0,2.0e15,21),normed=True,alpha=0.3,label='SPT matched to RM, SZ mass')
#hist(m500c_match,bins=np.linspace(0,2.0e15,21),normed=True,alpha=0.3,label='SPT matched to RM, with mass-richness relation')
#legend()
#savefig('mass_comp_conserv.png')

#print 'mean mass of DES side',np.mean(m500c_match)
#print 'mean mass of SPT side',np.mean(m500_spt_tem)
####



h=0.7
unc_cen = np.sqrt(1.2**2 + thc_spt_match**2)/xi_spt_match

print unc_cen
print np.mean(unc_cen)

D_l = cosmo.comoving_distance(z_spt_match).value # comiving distance
R_mis = D_l*np.tan((np.pi/180.)*unc_cen / 60.)*h #Mpc/h

sig_Rmis = np.std(R_mis)#/len(R_mis)

print 'prior on SZ miscentering=',np.log(np.mean(R_mis)),'+',np.log(np.mean(R_mis)+sig_Rmis)-np.log(np.mean(R_mis)),'-',-np.log(np.mean(R_mis)-sig_Rmis)+np.log(np.mean(R_mis))


'''mbin = np.linspace(0,2e15,21)
zbin = np.linspace(0,2,21)

index = np.arange(len(clusters)).copy()
index = index[ind_cl]

hist_m, mbin_edges, zbin_edges = np.histogram2d(m500_cl,z_cl,bins=(mbin,zbin))
hist_mspt, msptbin_edges, zsptbin_edges = np.histogram2d(m500_spt_match,z_spt_match,bins=(mbin,zbin))
index_m = np.digitize(m500_cl,mbin_edges)
index_z = np.digitize(z_cl,zbin_edges)

n = 0
for i in range(20):
    for j in range(20):
	count = int(hist_mspt[i,j])
	print i,j
	indm_tem = (index_m == i+1)
	indz_tem = (index_z == j+1)
	ind_tem = indm_tem*indz_tem
	index_tem = index[ind_tem]
	if len(index_tem) == 0:
		continue
	if count == 0:
		continue
	if len(index_tem) == 1:
		idx = index_tem
	if len(index_tem) > 1:
		idx = np.random.choice(index_tem,count,replace=False)
	if n == 0:
		idx_seq = idx
	elif i+j != 0:
		idx_seq = np.hstack((idx_seq,idx))
	n += count

print 'count=',n
idx_seq = np.sort(idx_seq)
np.savez('RM_sample-index_number-matched-to-spt',index=idx_seq)

m500_cl = np.loadtxt('m500_y3rm_vl.dat')
z_cl = clusters['Z_LAMBDA']

print np.mean(m500_cl[idx_seq])

print len(idx_seq)'''

########## galaxies ##########

hdu_gal = pf.open('/data3/taeshin/data/DES/y3gold2.2_select_clr-magerr0.1_m22.5.fits')[1].data
z = hdu_gal['z_bpz']

rc("font", size=12, **{'family': 'serif', 'serif': ['Computer Modern']})
rc("text", usetex=True)

print 'number of SPT clusters after footprint matching and z/m cut =',len(z_spt_match)
print 'number of ACT clusters after footprint matching and z/m cut =',len(z_act_match)
ax1 = subplot(211)
ax1.hist(z_spt_match,weights=np.ones_like(z_spt_match)/len(np.ones_like(z_spt_match)),histtype='step', color='r', range=(0,1.5), bins=30, label='SPT')
#hist(z_spt_match,histtype='step', color='r', range=(0,2), bins=20, lw=2, normed=True, label='SPT conservative')
ax1.hist(z_act_match,weights=np.ones_like(z_act_match)/len(np.ones_like(z_act_match)),histtype='step', color='g', range=(0,1.5), bins=30,  label='ACT')
ax1.hist(z_rm,weights=np.ones_like(z_rm)/len(np.ones_like(z_rm)),histtype='step',color='b',range=(0,1.5),bins=30, label='RM')
ax1.hist(z,weights=np.ones_like(z)/len(np.ones_like(z)),color='k',range=(0,1.5),bins=30,label='galaxies',alpha=0.1)
ax1.set_xlabel('z',fontsize=18)
ax1.set_ylabel('normalized count',fontsize=18)
ax1.tick_params(axis='both',which='both',direction='in',labelsize='large')
ax1.legend(fontsize='x-large')

ax2 = subplot(212)
m500_sim = np.loadtxt('list_tae_mdpl2_m500c_z045_43e14',unpack=True)[8]
m_tot = np.hstack((m500_sim,m500_cl*0.7,m500_spt_match*0.7,m500_act_match*0.7))
ax2.hist(m500_spt_match*0.7,weights=np.ones_like(m500_spt_match)/len(np.ones_like(m500_spt_match)),histtype='step', color='r', bins=np.logspace(np.log10(np.amin(m_tot)),np.log10(np.amax(m_tot)),16), label='SPT')
#hist(m500_spt_match,histtype='step', color='r', range=(0,2e15), bins=20, lw=2, normed=True, label='SPT conservative')
ax2.hist(m500_act_match*0.7,weights=np.ones_like(m500_act_match)/len(np.ones_like(m500_act_match)),histtype='step', color='g', bins=np.logspace(np.log10(np.amin(m_tot)),np.log10(np.amax(m_tot)),16), label='ACT')
ax2.hist(m500_cl*0.7,weights=np.ones_like(m500_cl)/len(np.ones_like(m500_cl)),histtype='step',color='b',bins=np.logspace(np.log10(np.amin(m_tot)),np.log10(np.amax(m_tot)),16), label='RM')
ax2.hist(m500_sim,weights=np.ones_like(m500_sim)/len(np.ones_like(m500_sim)),bins=np.logspace(np.log10(np.amin(m_tot)),np.log10(np.amax(m_tot)),16),color='k',histtype='step',label='simulation halos')#,alpha=0.5)
ax2.set_xlabel(r'$M_{500}\, [10^{14}\,h^{\rm -1}M_{\rm \odot}]$',fontsize=18)
ax2.set_xscale('log')
ax2.set_ylabel('normalized count',fontsize=18)
ax2.set_xticks([2e14,3e14,4e14,5e14,6e14,7e14,8e14,9e14,1e15,2e15])
ax2.set_xticklabels(['2','3','4','5','6','7','8','9','10','20'])
ax2.tick_params(axis='both',which='both',direction='in',labelsize='large')
ax2.legend(fontsize='x-large')
tight_layout()
savefig('sample_spt.pdf')
print 'mean mass of fiducial SPT clusters =',np.mean(m500_spt_match)
#print 'error of mean mass of fiducial SPT clusters =',np.sqrt(np.sum(np.square(m500err_spt_match)))/len(m500err_spt_match)
print 'mean redshift of fiducial SPT clusters =', np.mean(z_spt_match)
print len(id_spt_match),'= number of SPT clusters after all the masks'

exit()
