# Specify (Lr,Lr,1)-rank decomposition model
# and corresponding loss function

# from operator import methodcaller
# from numpy.core.fromnumeric import transpose
from numpy.core.fromnumeric import transpose
import tensorflow as tf
import numpy as np
import time
from utils import *

class LrModel:
    def __init__(self,target,Lr,R):
        # tf.debugging.set_log_device_placement(True)
        self.Y = tf.constant(target,dtype=tf.float32)
        [I,J,K] = target.shape
        As = [R,I,Lr]
        Bs = [R,J,Lr]
        Cs = [R,K]
        
        self.A = self.initvar(As)
        self.B = self.initvar(Bs)
        self.C = self.initvar(Cs)
        self.vars = (self.A,self.B,self.C)

        self.RegWeight  = 0.
        self.RegType    = 1.
        self.AscWeight  = 0.
        self.MaxIter    = 50000
        self.MaxDelta   = 1e-8
        self.MaxLoss    = 3e-3
        self.MovAvgCount = 10
        self.opt_lrate = 0.01
        self.opt_persitence = True
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.opt_lrate)
        # self.opt = tf.keras.optimizers.SGD(learning_rate=.01)
        # self.opt = tf.optimizers.Adagrad(learning_rate=1)
        # self.opt = tf.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)

    def __call__(self):
        op1 = tf.matmul(self.A,self.B,transpose_b=True)
        # op2 = tf.maximum(op1,1e-15)
        op3 = tf.tensordot(op1,self.C,[0,0])
        # op4 = tf.maximum(op3,1e-15)
        return op3

    def initvar(self,shape):
        v = tf.Variable(
            tf.random.uniform(shape=shape, dtype=tf.float32)
            # tf.random.truncated_normal(shape=shape, dtype=tf.float32,
                # mean=0.5, stddev=0.25)
            # tf.random.truncated_normal(shape=shape, dtype=tf.float32,
                # mean=0.5, stddev=0.05)
        )
        return v
    
    @property
    def E_np(self):
        Eop = tf.matmul(self.A,self.B,transpose_b=True)
        return Eop.numpy()
        # return self.E.numpy()
    @property
    def C_np(self):
        return self.C.numpy()
    @property
    def Y_np(self):
        return self.Y.numpy()
    @property
    def Yprime(self):
        return self().numpy()
    
    def loss(self):
        se = tf.math.squared_difference(self.Y,self())
        mse = tf.reduce_mean(se)
        return mse
    
    @tf.function 
    def cost(self):
        tc = self.loss() + self.reg_term() \
            + self.asc_term()
        return tc
    
    def reg_term(self):
        # self.E.assign(tf.matmul(self.A,self.B))
        Eop = tf.matmul(self.A,self.B,transpose_b=True)
        N=np.prod(Eop.shape)
        reg = tf.norm(tf.abs(Eop),self.RegType)
        scale = N**(1/self.RegType)
        reg = reg/scale * self.RegWeight 
        reg = reg * self.RegWeight 
        return reg
    
    def asc_term(self):
        Eop = tf.matmul(self.A,self.B,transpose_b=True)
        N = np.prod(Eop.shape)
        Eop = tf.reduce_sum(Eop,axis=0)
        se = tf.math.squared_difference(Eop,tf.ones_like(Eop))
        # scale = N**2
        mse = tf.reduce_mean(se)*self.AscWeight
        rmse = tf.sqrt(mse)
        return mse

    def apply_anc(self,mode):
        ''' Abundance Nonnegativity constraint '''
        epsilon = 1e-15
        if mode=='relu':
            self.A.assign(tf.maximum(self.A,epsilon))
            self.B.assign(tf.maximum(self.B,epsilon))
            self.C.assign(tf.maximum(self.C,epsilon))
        elif mode=='reluAB':
            self.A.assign(tf.maximum(self.A,epsilon))
            self.B.assign(tf.maximum(self.B,epsilon))
        elif mode=='reluC':
            self.C.assign(tf.maximum(self.C,epsilon))
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
    
    @tf.function
    def train_keras(self,opt):
        with tf.GradientTape() as t:
            curr_cost = self.cost()
        grads = t.gradient(curr_cost,self.vars)
        opt.apply_gradients(zip(grads,self.vars))
        self.apply_anc('relu')
        # Normalize columns of A
        # self.normalize_columns_of_A()
        return curr_cost

    def normalize_columns_of_A(self):
        (a_norm, a_divs) = tf.linalg.normalize(self.A, \
             ord='euclidean', axis=1, name=None)
        b_scaled = self.B * a_divs
        self.A.assign(a_norm)
        self.B.assign(b_scaled)

    def train(self):
        lrate = self.opt_lrate
        with tf.GradientTape() as t:
            current_loss = self.loss()
        [dA,dB,dC] = t.gradient(current_loss, 
            [self.A, self.B, self.C])
        self.A.assign_sub(lrate * dA)
        self.B.assign_sub(lrate * dB)
        self.C.assign_sub(lrate * dC)
        return current_loss
        
    def run_optimizer(self):
        t0=time.time()
        et = time.time()-t0
        current_cost = self.cost().numpy()*2
        current_loss = current_cost
        mvect = np.ones(self.MovAvgCount)*1e10
        mavg_cost = np.mean(mvect)
        delta = 1e10; print_step = 10
        print(' Iter  | loss      | cost      |' \
            +' mavg      | delta     | iter/s | time')
        #for i in range(self.MaxIter):
        converged = False; i=0
        while(not converged):
            if not self.opt_persitence:
                self.opt = tf.keras.optimizers.Adam(learning_rate=self.opt_lrate)
            current_cost = self.train_keras(self.opt).numpy()
            current_loss = self.loss().numpy()  
            old_cost = mavg_cost
            mvect[i % mvect.size] = current_cost
            mavg_cost = np.mean(mvect)
            delta = old_cost - mavg_cost
            if i%print_step == 0:
                et = time.time()-t0
                fprint(current_cost, mavg_cost, i, current_loss, delta, et)
            i+=1
            converged = delta < self.MaxDelta \
                or current_loss < self.MaxLoss \
                or i > self.MaxIter 
        fprint(current_cost, mavg_cost, i, current_loss, delta, et)
        print()
        print(f'MovAvg Vector: {mvect}')

def fprint(current_cost, mavg_cost, i, current_loss, delta, et):
    print(f'{i:6} |{current_loss:10.3e}' \
        + f' |{current_cost:10.3e}' \
        + f' |{mavg_cost:10.3e}' \
        + f' |{delta:10.3e}' \
        + f' |{i/et:7.1f}' \
        + f' |{et:5.0f}', \
        end='\r', flush=True)

# @tf.function
# def train_step(model, data, labels, optimizer):
#     with tf.GradientTape() as tape:
#         loss = model.cost()
#     variables = (self.A,self.B,self.C)
#     grads = tape.gradient(loss,variables)

