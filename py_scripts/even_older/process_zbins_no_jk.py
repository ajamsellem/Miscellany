
import numpy as np
import os
import sys

# this is to combine all the JK files and output the mean and covariance

name = sys.argv[1]
Nz = int(sys.argv[2])
result_dir = sys.argv[3]
#id_meta = int(sys.argv[4])

#meta = np.load('splashback_meta_info.npz')
R = np.load(result_dir+'/Sigmag_'+name+'_0.npz')['R']
Sg = []

for i in range(Nz):
    xi = []
    w = []
    ave_dens = []
    nclust = []
    infile = np.load(result_dir+'/Sigmag_'+name+'_'+str(i)+'.npz')
    xi.append(infile['xi'])
    w.append(infile['w'])
    ave_dens.append(infile['ave_dens'])
    nclust.append(infile['nclust'])
    xi = np.array(xi)
    w = np.array(w)
    ave_dens = np.array(ave_dens)
    nclust = np.array(nclust)

    dens = np.sum(ave_dens*nclust)/np.sum(nclust)
    mean = np.sum(xi*w, axis=0)/np.sum(w, axis=0)*dens

    Sg.append(mean)

Sg = np.array(Sg)
C = np.cov(Sg.T)*(len(Sg)-1)
Sg_mean = np.mean(Sg, axis=0)
Sg_sig = np.sum((Sg-Sg_mean)**2, axis=0)**0.5/len(Sg)**0.5*(len(Sg)-1)**0.5

np.savez('splashback_cov_'+str(name)+'.npz', cov=C,
         r_data=R, sg_mean=Sg_mean, sg_sig=Sg_sig)
#        h=meta['h'],
#        R_unit=meta['R_unit'],
#        Sigmag_unit=meta['Sigmag_unit'],
#        DeltaSigma_unit=meta['DeltaSigma_unit'],
#        cat=meta['cat'][id_meta],
#        mean_z=meta['mean_z'][id_meta],
#        mean_lambda=meta['mean_lambda'][id_meta],
#        mean_rlambda=meta['mean_rlambda'][id_meta],
#        mean_rlambda_com=meta['mean_rlambda_com'][id_meta],
#n_clust=meta['n_clust'][id_meta]
