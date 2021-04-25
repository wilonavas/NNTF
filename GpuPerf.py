import numpy as np
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt
from ntfmodel import *
from utils import *

# Run Parameters
# SizeList = [16,32,64,128,256,384,512]
SizeList = [32]
NemList = [4]
datapath = '../../MATLAB/ComponentAnalysisExperiments/data/usgs/'

# Model Parameters
parms = LrModelParameters()
parms.lrate = 0.001
parms.MaxDelta = 1e-8
parms.MovAvgCount = 100
parms.LimitMaxNorm = True
AbundanceThreshold = 0.95

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


for s in SizeList:
    for r in NemList:
        filename = f'sleg-e{r:d}x{s:d}'
        ds = loadmat(datapath+filename)
        ds = loadmat(datapath+"../h03-urban")
        Yin = ds['hsiten']
        Sgt = ds['Sgt']
        Sname = ds['Sname']
        [I,J,K] = Yin.shape
        [K,R] = Sgt.shape
        IorJ = np.maximum(I,J)
        Lr = int(IorJ**2/(np.minimum(IorJ,K)*R))
        
        # Pre-process inputs
        Yin = Yin/np.max(Yin)
        Ynorm = np.linalg.norm(Yin,
            ord=np.inf,axis=2,keepdims=True)
        Y = Yin/Ynorm
        # plt.imshow(hsi2rgb(Y))
        # plt.show()


#         # Run optimization
        model = LrModel(Y,Lr,R,seed=0,parms=parms)
        results = model.run_optimizer()
        (cost_vector, delta, it, et) = results
        str1 = f'[I,J,K]=>[{I},{J},{K}]  [Lr,R]=>[{Lr},{R}] '
        str2 = f'|{cost_vector[-1]:10.3e} |{delta:10.3e} '
        str3 = f'|{it:10d} |{it/et:7.1f} |{et:5.0f}'
        print(str1 + str2 + str3)
        # plt.semilogy(cost_vector)
        # plt.show()
#         # Extract Endmembers
#         Sprime = get_endmembers(model, AbundanceThreshold)
#         (Sprime,p) = reorder(Sprime,Sgt)
#         print(f'Reorder: {p}')
#         plot_decomposition(model,Sgt,Sprime,p)
#         plt.show()