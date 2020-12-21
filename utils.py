import time, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def hsi2rgb(Y):
    Yb = np.mean(Y[:,:,0:60],axis=2)
    Yg = np.mean(Y[:,:,60:120],axis=2)
    Yr = np.mean(Y[:,:,120:-1],axis=2)
    Yrgb = np.stack([Yr,Yg,Yb],axis=2)
    return Yrgb

def plot_decomposition(model, Sgt, Sprime, p):
    (K,R) = Sgt.shape
    M=5
    E = model.E_np
    E = E/np.max(E,axis=(1,2),keepdims=True)
    # E = E/np.sum(E,axis=0,keepdims=True)
    # E = E/np.max(E,axis=(1,2),keepdims=True)

    C = model.C_np
    C = C/np.max(C,axis=1,keepdims=True)
    Y = model.Y_np
    Yprime = model.Yprime
    #Cr = np.transpose(C)
    #(p,Cr) = reorder(Cr,Sgt)

    # Plot target tensor and reconstruction
    plt.subplot(M,R,1)
    plt.imshow(hsi2rgb(Y))
    plt.subplot(M,R,3)
    plt.imshow(hsi2rgb(Yprime))
    for i in range(R):
        # Plot Spatial Slab components
        plt.subplot(M,R,R+i+1)
        plt.imshow(E[p[i],:,:],cmap=plt.cm.jet)
        # Plot Ground Truth Endmembers
        plt.subplot(M,R,R*2+i+1)
        plt.plot(Sgt[:,i])
        # Plot Spectral Components
        plt.subplot(M,R,R*3+i+1)
        plt.plot(C[p[i],:])
        # Plot Reconstructed Endmembers
        plt.subplot(M,R,R*4+i+1)
        plt.plot(Sprime[:,i])
    #plt.show()

def read_agt(matdict):
    Agt = matdict['Agt']
    [R,N] = Agt.shape
    nx = math.floor(matdict['nx'])
    ny = math.floor(N/nx)
    Agt = np.reshape(Agt,(R,ny,nx))
    Agt = np.transpose(Agt,(0,2,1))
    Agt = np.reshape(Agt,(R,N))
    return(Agt)

def plot_abundance(A1, A2, nx):
    [R,N] = A1.shape
    ny = math.floor(N/nx)
    A1_img = np.reshape(A1,(R,nx,ny))
    A2_img = np.reshape(A2,(R,nx,ny))
    prows = 2
    for r in range(R):
        plt.subplot(prows,R,r+1)
        plt.imshow(A1_img[r,:,:],cmap=plt.cm.jet)
        plt.subplot(prows,R,R+r+1)
        plt.imshow(A2_img[r,:,:],cmap=plt.cm.jet)
    #plt.show()

def plot_endmembers(S1, S2):
    [K,R] = S1.shape
    prows = 2
    for r in range(R):
        plt.subplot(prows,R,r+1)
        plt.plot(S1[:,r])
        plt.subplot(prows,R,R+r+1)
        plt.plot(S2[:,r])
    #plt.show()
 
def get_endmembers(model,threshold, fromtarget=False, asc=False):
    E = model.E_np
    [R,I,J] = E.shape
    # Normalize spatial weights
    
    
    if asc:
        # E1 = E/np.max(E,axis=(1,2),keepdims=True)
        E1 = E/np.sum(E ,axis=0)
        # E1 = E1 - np.min(E1,axis=(1,2),keepdims=True)
        Em = E1/np.max(E1,axis=(1,2),keepdims=True)
    else:
        Em = E/np.max(E,axis=(1,2),keepdims=True)
        
    Yprime = model.Yprime
    Ytarget = model.Y_np
    [_,_,K] = Yprime.shape
    
    Sprime = np.zeros(shape=(K,R),dtype=np.float)
    # mask = tf.greater_equal(Em,threshold)
    mask = np.greater_equal(Em,threshold)
    # count = tf.where(mask==True,1,0)
    count = np.sum(mask,axis=(1,2))
    for r in range(R):
        print(f'Comp:{r} VectCount:{count[r]}')
        if fromtarget :
            cand_vector = tf.boolean_mask(Ytarget,mask[r,:,:])
        else :
            cand_vector = tf.boolean_mask(Yprime,mask[r,:,:])
        mean_vector = tf.reduce_mean(cand_vector,axis=0)
        Sprime[:,r] = mean_vector.numpy()
    return Sprime

def fcls_np(Yin, Sin):
    '''
    A[R,N] = fcls_np(Y[I,J,K], S[K,R])
    Numpy implementation
    fcls_np() Elapsed time: 9.566718578338623 seconds
    rmse: [0.02025357 0.0487208  0.03135294 0.05578483]
    '''
    # Sin = Sin / np.sqrt(np.sum(np.square(Sin),axis=0))
    # Yin = Yin / np.sqrt(np.sum(np.square(Yin),axis=2,keepdims=True))
    t0 = time.time()
    Sin = Sin / np.linalg.norm(Sin,axis=0,keepdims=True)
    Yin = Yin / np.linalg.norm(Yin,axis=2,keepdims=True)

    (I,J,K) = Yin.shape
    (K,R) = Sin.shape
    N = I*J
    Ymat = np.reshape(Yin,(N,K))
    A = np.zeros((R,N),dtype=np.float)

    for n in range(N):
        yn = Ymat[n,:]
        alpha_idx = [_ for _ in range(R)]
        S = Sin
        St = np.transpose(S)
        StS = np.matmul(St,S)
        StSinv = np.linalg.pinv(StS)
        #StSinvSt = np.matmul(StSinv,St)
        Styn = np.matmul(St,yn)
        als_hat = np.matmul(StSinv,Styn)
        csum = np.sum(StSinv,axis=1)
        for r in range(R):
            afcls_hat = als_hat - csum  \
                * (np.sum(als_hat)-1) / np.sum(csum)
                # * np.reciprocal(np.sum(StSinv)) \
                # * (np.sum(als_hat)-1)
            # Find negative abundances
            is_neg = np.less(afcls_hat,0)
            if np.any(is_neg):
                scaled_neg = np.where(is_neg, np.abs(afcls_hat / csum), 0)
                min_idx = np.argmax(scaled_neg)
                del alpha_idx[min_idx]
                S = Sin[:,alpha_idx]
                St = np.transpose(S)
                StS = np.matmul(St,S)
                StSinv = np.linalg.pinv(StS)
                #StSinvSt = np.matmul(StSinv,St)
                Styn = np.matmul(St,yn)
                als_hat = np.matmul(StSinv,Styn)
                csum = np.sum(StSinv,axis=1)
            else:
                break
        alpha = np.zeros(R,dtype=np.float)
        alpha[alpha_idx] = afcls_hat
        A[:,n]=alpha
        if n%1000 == 0 : 
            et = time.time()-t0
            print(f'Iter: {n}  Time: {et:.4f}',\
                end='\r', flush=True)
    print(f'fcls time: {(time.time()-t0):.2f}')
    return A

def fcls(Ynp, Snp):
    '''
    Implementation most similar to Matlab with
    all matrix operations in the inner loop
    fcls Elapsed time: 71.63096284866333 seconds
    rmse: [0.02025334 0.04872063 0.03135224 0.05578368]
    '''
    # Sin = Sin / np.sqrt(np.sum(np.square(Sin),axis=0))
    # Yin = Yin / np.sqrt(np.sum(np.square(Yin),axis=2,keepdims=True))
    Snp = Snp / np.sqrt(np.sum(np.square(Snp),axis=0))
    Ynp = Ynp / np.sqrt(np.sum(np.square(Ynp),axis=2,keepdims=True))
    Sin = tf.constant(Snp,dtype=tf.float32)
    Yin = tf.constant(Ynp,dtype=tf.float32)

    (I,J,K) = Yin.shape; N=I*J
    (K,R) = Sin.shape
    Ymat = tf.reshape(Yin,shape=(N,K))
    A = tf.Variable(tf.zeros((R,N),dtype=tf.float32))
    # Select pixel to process
    # Iterate through pixels
    t0 = time.time()
    for n in range(N):
        yn = tf.reshape(Ymat[n,:],shape=(K,1))
        alpha = np.zeros(R,dtype=np.float)
        alpha_idx = [_ for _ in range(R)]
        S = Sin
        StS = tf.matmul(S,S,transpose_a=True)
        StSinv = tf.linalg.inv(StS)
        StSinvSt = tf.matmul(StSinv,S,transpose_b=True)
            
        for r in range(R):
            als_hat = tf.matmul(StSinvSt,yn)
            csum = tf.reduce_sum(StSinv,axis=1,keepdims=True)
            afcls_hat = als_hat - csum  \
                * tf.math.reciprocal(tf.reduce_sum(StSinv)) \
                * (tf.reduce_sum(als_hat)-1)
            # Find negative abundances
            is_neg = tf.less(afcls_hat,0)
            if tf.reduce_any(is_neg):
                #scaled_neg = tf.where(afcls_hat<0, afcls_hat / csum, afcls_hat)
                scaled_neg = tf.where(afcls_hat<0, tf.abs(afcls_hat / csum), 0)
                min_idx = np.argmax(scaled_neg)
                del alpha_idx[min_idx]
                #min_idx = alpha_idx[min_idx]
                #alpha_idx.remove(min_idx)
                S = tf.gather(Sin,alpha_idx,axis=1)
                StS = tf.matmul(S,S,transpose_a=True)
                StSinv = tf.linalg.inv(StS)
                StSinvSt = tf.matmul(StSinv,S,transpose_b=True)
            
            else:
                break
        alpha[alpha_idx] = afcls_hat[:,0]
        An = A[:,n]
        An.assign(alpha)
        if n%1000 == 0 : 
            et = time.time()-t0
            print(f'Iter: {n}  Time: {et:.4f}',\
                end='\r', flush=True)
    print(f'fcls time: {(time.time()-t0):.2f}')
    return A.numpy()

def correlation(x,y,centered=False):
    ''' 
    This function computes the Pearson Correlation
    unless the optional argument centered is set to False.
    When centered=False it computes the cosine(theta), 
    where theta is the angle between the two vectors.
    '''
    # Compute Means
    if centered:
        x = x-np.mean(x,axis=0)
        y = y-np.mean(y,axis=0)
    # Normalize inputs    
    xn = x/np.linalg.norm(x,axis=0)
    yn = y/np.linalg.norm(y,axis=0)
    # Compute dot products (same as matmul)
    corr = np.tensordot(xn,yn,axes=[0,0])
    return corr

def reorder(x,y):
    [K,R] = x.shape
    c = correlation(x,y)
    p = np.zeros(R,dtype=np.int)
    for r in range(R,0,-1):
        max_idx = np.argmax(c)
        row = max_idx // R
        col = max_idx % R
        # print(f'max_idx{max_idx} r{row} c{col} i{r} corr:\n{c}')
        p[col] = row
        c[row,:]=-10; c[:,col]=-10
        # print(f'p={p}')
    xprime = x[:,p]
    return(xprime,p)
