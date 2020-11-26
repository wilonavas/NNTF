# Specify (Lr,Lr,1)-rank decomposition model
# and corresponding loss function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time

class LrModel:
    def __init__(self,target,Lr,R):
        # tf.debugging.set_log_device_placement(True)
        self.Y = tf.constant(target,dtype=tf.float32)
        [I,J,K] = target.shape
        As = [R,I,Lr]
        Bs = [R,Lr,J]
        Cs = [R,K]
        # Cbs = [R,1]
        
        self.A = self.initvar(As)
        self.B = self.initvar(Bs)
        self.C = self.initvar(Cs)
        self.apply_anc('abs')
        # self.Cb = tf.Variable(
        #    tf.random.normal(shape=Cbs, dtype=tf.float32))

        self.RegWeight  = 0.
        self.RegType    = 1.
        self.AscWeight  = 0.
        self.MaxIter    = 50000
        self.MaxDelta   = 1e-7
        self.MaxLoss    = 1e-7
        self.MovAvgCount = 10
    
    def initvar(self,shape):
        v = tf.Variable(
            # tf.random.uniform(shape=shape, dtype=tf.float32)
            tf.random.truncated_normal(shape=shape, dtype=tf.float32,
                mean=0.5, stddev=0.25)
        )
        return v
    
    @tf.function
    def __call__(self):
        Eop = tf.matmul(self.A,self.B)
        Yprime = tf.tensordot(Eop,self.C,[0,0])
        return Yprime

    @property
    def E_np(self):
        Eop = tf.matmul(self.A,self.B)
        return Eop.numpy()
    
    @property
    def C_np(self):
        return self.C.numpy()
    
    @property
    def Y_np(self):
        return self.Y.numpy()
    
    @property
    def Yprime(self):
        return self().numpy()

    @tf.function
    def loss(self):
        se = tf.math.squared_difference(self.Y,self())
        mse = tf.reduce_mean(se)
        return mse
    
    @tf.function
    def cost(self):
        tc = self.loss() 
        # + self.reg_term() + self.asc_term()
        return tc
    
    @tf.function
    def reg_term(self):
        # self.E.assign(tf.matmul(self.A,self.B))
        Eop = tf.matmul(self.A,self.B)
        N=np.prod(Eop.shape)
        reg = tf.norm(tf.abs(Eop),self.RegType)
        scale = N**(1/self.RegType)
        # reg = reg/scale * self.RegWeight 
        reg = reg * self.RegWeight 
        return reg
    
    @tf.function
    def asc_term(self):
        Eop = tf.matmul(self.A,self.B)
        N = np.prod(Eop.shape)
        Eop = tf.reduce_sum(Eop,axis=0)
        se = tf.math.squared_difference(Eop,tf.ones_like(Eop))
        # scale = N**2
        mse = tf.reduce_mean(se)*self.AscWeight
        rmse = tf.sqrt(mse)
        return mse

    def train(self, lrate=1):
        with tf.GradientTape() as t:
            current_loss = self.loss()
        (dA,dB,dC) = t.gradient(current_loss, (self.A, self.B, self.C))
        self.A.assign_sub(lrate * dA)
        self.B.assign_sub(lrate * dB)
        self.C.assign_sub(lrate * dC)
        
        return current_loss

    def train_keras(self, lrate=0.1):
        # opt = tf.keras.optimizers.SGD(learning_rate=lrate)
        opt = tf.optimizers.Adagrad(learning_rate=0.1)
        # opt = tf.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        # opt = tf.optimizers.Adam()
        variables = (self.A,self.B,self.C)
        with tf.GradientTape() as t:
            curr_cost = self.cost()
        grads = t.gradient(curr_cost,variables)
        opt.apply_gradients(zip(grads,variables))
        self.apply_anc('relu')
        return curr_cost

    def run_optimizer(self):
        t0=time.time()
        current_cost = self.cost()*2
        mvect = np.ones(self.MovAvgCount)*1e10
        prev_mavg = np.mean(mvect)
        lastpass=True
        for i in range(self.MaxIter):
            old_cost = current_cost
            current_cost = self.train_keras()
            current_loss = self.loss()
            # print(old_cost,current_cost)
            delta = old_cost - current_cost
            mvect[i % mvect.size] = delta
            mavg = np.mean(mvect)
            if i%100 == 0:
                et = time.time()-t0
                print(f'Iter {i:4}:' \
                    + f' loss: {current_loss:3.4g}' \
                    + f' cost: {current_cost:3.4g}' \
                    + f' delta: {delta:1.4g}' \
                    + f' mavg: {mavg:0.4g} time: {et:3.4g}')
            if mavg < self.MaxDelta or current_loss < self.MaxLoss:
                print(f'Model converged at iter: {i}' \
                    + f' cost: {current_cost:3.4g}' \
                    + f' delta: {delta:1.6g}' \
                    + f' mavg: {mavg:1.6g}')
                if lastpass:
                    break
                else:
                    # Replace C by reconstructed endmembers
                    # and run a litte
                    lastpass=True
                    print(f'Second Pass')
                    Sprime = get_endmembers(self,0.95)
                    self.C.assign(Sprime.transpose())
            prev_mavg = mavg
        print(f'Iter {i:4}: cost: {current_cost:3.4g}' \
                + f' delta: {delta:1.4g} time: {et:3.4g}')
        print(f'Train Time: {(time.time()-t0):.2f}')
    
    
    @tf.function
    def apply_anc(self,mode):
        ''' Abundance Nonnegativity constraint '''
        if mode=='relu':
            self.A.assign(tf.maximum(self.A,1e-15))
            self.B.assign(tf.maximum(self.B,1e-15))
            self.C.assign(tf.maximum(self.C,1e-15))
            
            # self.A.assign(tf.minimum(self.A,1))
            # self.B.assign(tf.minimum(self.B,1))
            # self.Cb.assign(tf.maximum(self.Cb,1e-15))
        elif mode=='abs':
            self.A.assign(tf.abs(self.A))
            self.B.assign(tf.abs(self.B))
            self.C.assign(tf.abs(self.C))
        elif mode=='sqr':
            self.A.assign(tf.square(self.A))
            self.B.assign(tf.square(self.B))
            self.C.assign(tf.square(self.C))
        else:
            print(f'Mode does not exist:{mode}')
            raise(ValueError)
       
    #Abundance Sum-to-one Constraint
    # def apply_asc(self):
    #     E = self.get_E()
    #     Eprime = E / tf.reduce_sum(E,axis=0)
    #     Bt = tf.linalg.lstsq(self.A,Eprime)
    #     self.B.assign(tf.transpose(Bt,[0,2,1]))

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
    plt.show()

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
    plt.show()

def plot_endmembers(S1, S2):
    [K,R] = S1.shape
    prows = 2
    for r in range(R):
        plt.subplot(prows,R,r+1)
        plt.plot(S1[:,r])
        plt.subplot(prows,R,R+r+1)
        plt.plot(S2[:,r])
    plt.show()
 
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
            print(f'Iter: {n}  Time: {et:.4f}')
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
            print(f'Iter: {n}  Time: {et:.4f}')
    print(f'fcls time: {(time.time()-t0):.2f}')
    return A.numpy()

def correlation(x,y,centered=True):
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
