import numpy as np
import scipy.io as sio 
import math
import matplotlib.pyplot as plt
from ntfmodel import *
from utils import *

##########################################################################-----
# Load hsi in matlab format
datapath = '../../MATLAB/ComponentAnalysisExperiments/data/'
## Avg singular values that are > 1
# filename = 'h01-samson'; Lr = 8
# filename = 'h02-jasper'; Lr = 21
# filename = 'h03-urban'; Lr = 122

## Max singular value of Y(:,:,k) > 1
# filename = 'h01-samson'; Lr = 12
# filename = 'h02-jasper'; Lr = 46
# filename = 'h03-urban'; Lr = 168
## Best empirical:
# filename = 'h01-samson'; Lr = 31
# filename = 'h02-jasper'; Lr = 12
# filename = 'h03-urban'; Lr = 40
trials = range(1)
filenames = ('h01-samson','h02-jasper','h03-urban')
lowranks = (24,28,116)
lowranks = (8,40,220)
filenames = ['usgs/synt-e4x64-03']
lowranks = [8]

for fn,Lr in zip(filenames,lowranks):
    matdict = sio.loadmat(datapath + fn)
    Y = matdict['hsiten']
    Sgt = matdict['Sgt'] 
    # R = 6
    # Sgt = np.random.uniform(size=(Y.shape[2],R))

    ### L-inf Normalized on mode-2
    Ynmax = np.max(Y)
    Yninf = np.linalg.norm(Y, ord=np.inf, axis=2, keepdims=True)
    Yn2 = np.linalg.norm(Y, ord=2, axis=2, keepdims=True)
    Y=Y/Yninf

    [I,J,K] = Y.shape
    [K,R] = Sgt.shape
    print(fn)
    print(f'[I,J,K]=>[{I},{J},{K}]   [Lr,R]=>[{Lr},{R}]')
    print()
        
    for i in trials:
        # Instanciate Model
        model = LrModel(Y,Lr,R,seed=i)
        model.lrate = 0.001
        model.MaxDelta = 1e-8
        model.AscWeight = 0.01
        model.run_optimizer()
        AbundanceThreshold = 0.95
        AbundanceFromTarget = False

        # Compute endmembers using spatial components
        # and reconstructed tensor
        Sprime = get_endmembers(model, \
            AbundanceThreshold, 
            fromtarget=AbundanceFromTarget)
        (Sprime,p) = reorder(Sprime,Sgt)
        plot_decomposition(model,Sgt,Sprime,p)
        plt.show()

        # Compute Fully Constrained Least Squares 
        A = fcls_np(Y,Sprime)
        Agt = read_agt(matdict)
        nx = math.floor(matdict['nx'])
        mtx = compute_metrics(i,Sgt, Sprime, A, Agt)
        plot_abundance(Agt,A,nx)
        plt.show()

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
