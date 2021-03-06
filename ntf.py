import numpy as np
import scipy.io as sio 
import math
import matplotlib.pyplot as plt
from ntfmodel import *
from utils import *

##########################################################################-----
# Load hsi in matlab format
datapath = '../../MATLAB/ComponentAnalysisExperiments/data/'
## Max singular value of Y(:,:,k) > 1
# filename = 'h01-samson'; Lr = 12
# filename = 'h02-jasper'; Lr = 46
# filename = 'h03-urban'; Lr = 168
## Best empirical:
# filename = 'h01-samson'; Lr = 31
# filename = 'h02-jasper'; Lr = 12
# filename = 'h03-urban'; Lr = 220

## WHISPERS Paper L=(min(I,J)^2)/K
# 95*95/(156*3) = 19
# 100^2/(198*4) = 12
# 307^2/(162*4) = 145
trials = range(1)
filenames = ['usgs/synleg-e4s2-01',
    'usgs/synleg-e4s2-02',
    'usgs/synleg-e4s2-03',
    'usgs/synleg-e4s2-04']
# filenames = ['usgs/synleg-e4s2-04']
# filenames = ['usgs/synt-e4x64-03']
# filenames = ['h03-urban']
lowranks = [16]
parms = LrModelParameters()
parms.lrate = 0.001
parms.MaxDelta = 1e-8
parms.RegWeight = 0.
parms.AscWeight = 0.

AbundanceThreshold = 0.95
AbundanceFromTarget = False

#for fn,Lr in zip(filenames,lowranks):
for fn in filenames:
    for Lr in lowranks:
        matdict = sio.loadmat(datapath + fn)
        Y = matdict['hsiten']
        Sgt = matdict['Sgt']
        Sname = matdict['Sname']
        
        # R = 6
        # Sgt = np.random.uniform(size=(Y.shape[2],R))

        ### L-inf Normalized on mode-2
        Y = Y/np.max(Y)
        Yninf = np.linalg.norm(Y, ord=np.inf, axis=2, keepdims=True)
        Yn2 = np.linalg.norm(Y, ord=2, axis=2, keepdims=True)
        Y=Y/Yninf
        
        [I,J,K] = Y.shape
        [K,R] = Sgt.shape
        print(fn)
        print(f'[I,J,K]=>[{I},{J},{K}]   [Lr,R]=>[{Lr},{R}]')
        parms.prnt()

        for i in trials:
            # Instanciate Model
            model = LrModel(Y,Lr,R,seed=i,parms=parms)
            model.run_optimizer()
            # Compute endmembers using spatial components
            # and reconstructed tensor

            Sprime = get_endmembers(model, 
                AbundanceThreshold,
                norms=Yninf,
                fromtarget=AbundanceFromTarget)
            (Sprime,p) = reorder(Sprime,Sgt)
            print(f'Reorder: {p}')
            # plot_decomposition(model,Sgt,Sprime,p)
            # plt.show()

            # Compute Fully Constrained Least Squares 
            A = fcls_np(Y,Sprime)
            Agt = read_agt(matdict)
            nx = math.floor(matdict['nx'])
            mtx = compute_metrics(i,Sgt, Sprime, A, Agt)
            # plot_abundance(Agt,A,nx)
            # plot_all(Agt,A,nx,model,Sgt,Sprime,p,Sname)
            # plt.show()


####################################################
## This applies ANC after the factorization is done
####################################################
# E = model.E_np
# E1 = E/np.max(E,axis=(1,2),keepdims=True)
# E2 = E1/np.sum(E1,axis=0)
# E3 = E2 - np.min(E2,axis=(1,2),keepdims=True)
# Easc = E3/np.max(E3,axis=(1,2),keepdims=True)
# Aasc = np.reshape(Easc,(R,I*J))
# Aasc = Aasc[p,:]
# rmse = np.mean((Aasc-Agt)**2, axis=1) ** 0.5
# print(f'EXP: rmse: {rmse} avg: {np.mean(rmse):.4f}')
# plot_abundance(Agt,Aasc,nx)

# print(f'bias: {model.Cb}')

# Aref = fcls_np(Y,Sgt)
# rsme = tf.sqrt(tf.reduce_mean(tf.pow(Aref-Agt,2),axis=1))
# print(f'REF: rmse: {rsme} avg: {tf.reduce_mean(rsme):.4f}')
# plot_abundance(Agt,Aref,nx)

#########################################
# Plot Normalized inputs
#########################################
# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(hsi2rgb(Y/Ynmax), cmap=plt.cm.jet)
# plt.subplot(1,3,2)
# plt.imshow(hsi2rgb(Y/Yninf), cmap=plt.cm.jet)
# plt.subplot(1,3,3)
# y2rgb = hsi2rgb(Y/Yn2)
# y2max = np.max(Y/Yn2)
# y2rgbmax= np.max(y2rgb)
# plt.imshow(y2rgb/y2rgbmax, cmap=plt.cm.jet)
# plt.show()
