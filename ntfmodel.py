# Specify (Lr,Lr,1)-rank decomposition model
# and corresponding loss function

# from operator import methodcaller
# from numpy.core.fromnumeric import transpose
from numpy.core.fromnumeric import mean, transpose
import tensorflow as tf
import numpy as np
import time
from utils import *

tf.get_logger().setLevel('ERROR')

class LrModel:
    def __init__(self,target,Lr,R,seed=0):
        # tf.debugging.set_log_device_placement(True)
        self.Y = tf.constant(target,dtype=tf.float32)
        [I,J,K] = target.shape
        As = [R,I,Lr]
        Bs = [R,J,Lr]
        Cs = [R,K]
        
        self.seed = seed
        self.A = self.initvar(As)
        self.B = self.initvar(Bs)
        self.C = self.initvar(Cs)
        self.vars = (self.A,self.B,self.C)
        
        self.RegWeight  = 0.0
        self.RegNorm    = 1.
        self.AscWeight  = 0.0
        self.MaxIter    = 50000
        self.MaxDelta   = 1e-6
        self.MaxLoss    = 1e-6
        self.MovAvgCount = 1
        self.lrate = 0.001
        self.optim_persist = True
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lrate)
        # self.opt = tf.keras.optimizers.SGD(learning_rate=.01)
        # self.opt = tf.optimizers.Adagrad(learning_rate=1)
        # self.opt = tf.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    def Eop(self):
        return tf.matmul(self.A,self.B,transpose_b=True)
        
    def __call__(self):
        op1 = self.Eop()
        op = tf.tensordot(op1,self.C,[0,0])
        self.apply_anc('relu')
        return op

    def initvar(self,shape):
        # See Glorot normal initializer on 
        # m = 0.5
        # sd = np.sqrt(2./np.sum(shape))
        # init = tf.random.truncated_normal(shape=shape, dtype=tf.float32,
        #         mean=m, stddev=sd, seed=self.seed)
        ki = tf.keras.initializers.GlorotNormal(self.seed)
        init = ki(shape)
        
        # Uniform
        # init = tf.random.uniform(shape=shape, dtype=tf.float32, seed=seed)
        v = tf.Variable(init)
        return v
    
    def loss(self):
        se = tf.math.squared_difference(self.Y,self())
        mse = tf.reduce_mean(se)
        return mse
    
    @tf.function 
    def cost(self):
        tc = self.loss() \
            + self.reg_term() + self.asc_term()
        return tc
    
    def reg_term(self):
        reg = tf.pow(tf.abs(self.Eop()),self.RegNorm)
        reg = tf.pow(tf.reduce_mean(reg),1/self.RegNorm)
        reg = reg*(self.RegWeight)
        return reg
    
    def asc_term(self):
        Esum = tf.reduce_sum(self.Eop(),axis=0)
        diff2 = tf.pow(Esum - tf.ones_like(Esum),2)
        a = tf.reduce_mean(diff2)*self.AscWeight
        # a = tf.sqrt(a)
        return a

    def apply_anc(self,mode):
        ''' Abundance Nonnegativity constraint '''
        epsilon = 1e-15
        if mode=='relu':
            self.A.assign(tf.maximum(self.A,epsilon))
            self.B.assign(tf.maximum(self.B,epsilon))
            self.C.assign(tf.maximum(self.C,epsilon))
            # self.C.assign(self.C + tf.reduce_min(self.C,axis=1,keepdims=True))
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
        with tf.GradientTape() as tape:
            tape.watch(self.vars)
            curr_cost = self.cost()
        grads = tape.gradient(curr_cost,self.vars)
        opt.apply_gradients(zip(grads,self.vars))
        return curr_cost

    def max_norm_constraint(self):
        (a_norm, a_divs) = tf.linalg.normalize(self.A, \
             ord='euclidean', axis=1)
        b_scaled = self.B * a_divs
        self.A.assign(a_norm)
        self.B.assign(b_scaled)

    def train(self):
        lrate = self.lrate
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
        delta = 1e10; print_step = 11
        #for i in range(self.MaxIter):
        converged = False; i=0
        # print_title()
        while(not converged):
            if not self.optim_persist:
                self.opt = tf.keras.optimizers.Adam(learning_rate=self.lrate)
            current_cost = self.train_keras(self.opt).numpy()
            current_loss = self.loss().numpy()  
            old_cost = mavg_cost
            mvect[i % mvect.size] = current_cost
            mavg_cost = np.mean(mvect)
            delta = old_cost - mavg_cost
            if i % print_step == 0:
                et = time.time()-t0
                print_train_step(current_cost, mavg_cost, i, current_loss, delta, et)
            i+=1
            converged = delta < self.MaxDelta \
                or current_loss < self.MaxLoss \
                or i > self.MaxIter 
        print_train_step(current_cost, mavg_cost, i, current_loss, delta, et)
        print()
        # print(f'MovAvg Vector: {mvect}')


# @tf.function
# def train_step(model, data, labels, optimizer):
#     with tf.GradientTape() as tape:
#         loss = model.cost()
#     variables = (self.A,self.B,self.C)
#     grads = tape.gradient(loss,variables)

