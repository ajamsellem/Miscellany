import numpy as np
import os
import sys
import treecorr


# this is to combine all the JK files and output the mean and covariance
name = sys.argv[1]
Njk = int(sys.argv[2])
Nz = int(sys.argv[3])
result_dir = sys.argv[4]

R = np.load(result_dir+'/Sigmag_0.npz')['R']

sum_RR = np.zeros((Nz, len(R)))
sum_DD = np.zeros((Nz, len(R)))
sum_DR = np.zeros((Nz, len(R)))
sum_RD = np.zeros((Nz, len(R)))
sum_nclust = np.zeros(Nz)
sum_nclust_ran = np.zeros(Nz)
sum_ngal = np.zeros(Nz)
sum_ngal_ran = np.zeros(Nz)
sum_area = np.zeros(Nz)
sum_ngal_jk = np.zeros(Nz)

DD_full_simple = []
for i in range(Njk):
    if os.path.exists(result_dir+'/Sigmag_'+str(i)+'.npz') == True:
        infile = np.load(result_dir+'/Sigmag_'+str(i)+'.npz')
        sum_RR += infile['rr']
        sum_DD += infile['dd']
        sum_DR += infile['dr']
        sum_RD += infile['rd']
        sum_area += infile['area']
        sum_nclust += infile['nclust']
        sum_nclust_ran += infile['nclust_ran_w']
        sum_ngal += infile['ngal']
        #print(i)
        #print(sum_ngal)
        sum_ngal_ran += infile['ngal_ran']
        sum_ngal_jk += infile['ngal_jk']
        DD_full_simple.append(sum_DD)

#print(np.sum(np.array(DD_full_simple) , axis = 0))
#print()

Sg = []
DD_full = []
RR_full = []
DR_full = []
RD_full = []
#DD_act_full = []

for i in range(Njk):
    if os.path.exists(result_dir+'/Sigmag_'+str(i)+'.npz') == True:
        infile = np.load(result_dir+'/Sigmag_'+str(i)+'.npz')

        #DD_act = infile['dd']
        DD = sum_DD - infile['dd']
        #if i <= 10.:
        #    print(DD)
        RR = sum_RR - infile['rr']
        DR = sum_DR - infile['dr']
        RD = sum_RD - infile['rd']

        RR = RR*1.0*((sum_nclust - infile['nclust'])*1.0/(sum_nclust_ran - infile['nclust_ran_w'])*(sum_ngal - infile['ngal'])*1.0/(sum_ngal_ran - infile['ngal_ran'])).reshape(Nz,1)
        DR = DR*1.0*((sum_ngal - infile['ngal'])*1.0/(sum_ngal_ran - infile['ngal_ran'])).reshape(Nz,1)
        RD = RD*1.0*((sum_nclust - infile['nclust'])*1.0/(sum_nclust_ran - infile['nclust_ran_w'])).reshape(Nz,1)
        ave_dens = (sum_ngal_jk-infile['ngal_jk'])*1.0/(sum_area-infile['area'])
        nclust = (sum_nclust - infile['nclust'])*1.0

        dens = np.sum(ave_dens*nclust)/np.sum(nclust)

        #mean = np.sum(DD-DR-RD+RR, axis=0)/np.sum(RR, axis=0)*dens
        mean = np.sum((DD-DR-RD+RR)/RR, axis=0)*dens

        Sg.append(mean)
        DD_full.append(DD)
        RR_full.append(RR)
        DR_full.append(DR)
        RD_full.append(RD)
        #DD_act_full.append(DD_act)


DD_full = np.sum(np.sum(np.array(DD_full), axis = 0), axis = 0)
RR_full = np.sum(np.sum(np.array(RR_full), axis = 0), axis = 0)
DR_full = np.sum(np.sum(np.array(DR_full), axis = 0), axis = 0)
RD_full = np.sum(np.sum(np.array(RD_full), axis = 0), axis = 0)
#DD_act_full = np.sum(np.sum(np.array(DD_act_full), axis = 0), axis = 0)
DD_full_simple = np.sum(np.sum(np.array(DD_full_simple), axis = 0), axis = 0)

#print(np.sum(np.sum(DD_full, axis = 0), axis = 0))
#print(np.sum(DD_full))
#print(np.sum(sum_nclust))
#print()
#print(R)
Sg = np.array(Sg)
C = np.cov(Sg.T)*(len(Sg)-1)
Sg_mean = np.mean(Sg, axis=0)
Sg_sig = np.sum((Sg-Sg_mean)**2, axis=0)**0.5/len(Sg)**0.5*(len(Sg)-1)**0.5

#np.savez('splashback_cov_'+str(name)+'.npz', cov=C, r_data=R, sg_mean=Sg_mean, sg_sig=Sg_sig)
#np.savez('DD_'+str(name)+'.npz', DD=DD_full, RR=RR_full, DR=DR_full, RD=RD_full, density=dens, r_data=R)
#np.savez('DD_actual_'+str(name)+'.npz', DD_actual=DD_act_full, r_data=R)
#np.savez('DD_simple_'+str(name)+'.npz', DD_simple=DD_full_simple, r_data=R)
print(DD_full.shape)
print()
print(Sg_mean)
