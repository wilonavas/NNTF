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

class LrModelParameters:
    def __init__(self):
        self.RegWeight  = 0.0
        self.RegNorm    = 1.
        self.AscWeight  = 0.0
        self.MaxIter    = 50000
        self.MaxDelta   = 1e-8
        self.MaxLoss    = 1e-8
        self.MovAvgCount = 10
        self.lrate = 0.001
        self.optim_persist = True
        self.LimitMaxNorm = True
    
    def prnt(self):
        print(f'MaxDelta: {self.MaxDelta} '
            + f'LRate: {self.lrate} '
            + f'RegNorm: {self.RegNorm} '
            + f'RegWeight: {self.RegWeight} '
            + f'AscWeight: {self.AscWeight}')

class LrModel:
    def __init__(self,target,Lr,R,seed=0,parms=LrModelParameters()):
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
        self.Ae = self.initvar([R,Lr,I])
        self.vars = (self.A,self.B,self.C)
        
        self.parms = parms
     
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.parms.lrate)
        # self.opt = tf.keras.optimizers.SGD(learning_rate=.001)
        # self.opt = tf.optimizers.Adagrad(learning_rate=0.001)
        # self.opt = tf.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    
    def __call__(self):
        self.apply_anc('relu')
        if self.parms.LimitMaxNorm: 
            self.max_norm_constraint()
        op1 = self.Eop()
        op = tf.tensordot(op1,self.C,[0,0])
        return op
    
    def Eop(self):
        return tf.matmul(self.A,self.B,transpose_b=True)
    
    def initvar(self,shape):
        # See Glorot normal initializer on 
        # m = 0.5
        # sd = np.sqrt(2./np.sum(shape))
        # init = tf.random.truncated_normal(shape=shape, dtype=tf.float32,
        #         mean=m, stddev=sd, seed=self.seed)
        # ... or use the built-in keras initializer
        ki = tf.keras.initializers.GlorotNormal(self.seed)
        # ki = tf.keras.initializers.GlorotUniform(self.seed)
        init = ki(shape)
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
        reg = tf.pow(tf.abs(self.Eop()),self.parms.RegNorm)
        reg = tf.pow(tf.reduce_mean(reg),1/self.parms.RegNorm)
        reg = reg*(self.parms.RegWeight)
        # an = tf.linalg.norm(self.A,ord=self.parms.RegNorm)/tf.cast(tf.reduce_prod(self.A.shape),dtype=tf.float32)
        # bn = tf.linalg.norm(self.B,ord=self.parms.RegNorm)/tf.cast(tf.reduce_prod(self.B.shape),dtype=tf.float32)
        # reg = self.parms.RegWeight*(an+bn)
        return reg
    
    def asc_term(self):
        Esum = tf.reduce_sum(self.Eop(),axis=0)
        diff2 = (Esum - tf.ones_like(Esum))**2
        a = tf.reduce_mean(diff2)*self.parms.AscWeight
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
            curr_cost = self.cost()
        grads = tape.gradient(curr_cost,self.vars)
        opt.apply_gradients(zip(grads,self.vars))
        return curr_cost

    def max_norm_constraint(self):
        (a_normalized, a_normals) = tf.linalg.normalize(self.A, \
             ord=np.inf, axis=1)
        b_scaled = tf.math.multiply(self.B,a_normals)
        cond = a_normals > 1
        self.A.assign(tf.where(cond, a_normalized,self.A))
        self.B.assign(tf.where(cond, b_scaled,self.B))

        # Unconditionally scale A
        # self.A.assign(a_normalized)
        # self.B.assign(b_scaled)
        
        # Normalize C
        # cmax = tf.reduce_max(self.C,axis=1,keepdims=True)
        # self.C.assign(tf.where(cmax>1,self.C/cmax,self.C))
        # self.C.assign(self.C/cmax)

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
        mvect = np.ones(self.parms.MovAvgCount)*1e10
        mavg_cost = np.mean(mvect)
        delta = 1e10; print_step = 1
        converged = False; i=0
        # print_title()
        cost_vector = []
        while(not converged):
            if not self.parms.optim_persist:
                self.opt = tf.keras.optimizers.Adam(learning_rate=self.parms.lrate)
            current_cost = self.train_keras(self.opt).numpy()
            current_loss = self.loss().numpy()  
            old_cost = mavg_cost
            mvect[i % mvect.size] = current_cost
            mavg_cost = np.mean(mvect)
            delta = old_cost - mavg_cost
            if i % print_step == 0:
                et = time.time()-t0
                print_train_step(current_cost, mavg_cost, i, current_loss, delta, et)
                # self.component_norms()
            i+=1
            converged = delta < self.parms.MaxDelta \
                or current_loss < self.parms.MaxLoss \
                or i > self.parms.MaxIter 
            cost_vector.append(current_cost)
        # Erase progress indicator
        print(" "*80,end='\r', flush=True)
        results = (cost_vector, delta, i, et)
        return(results)
        

    def component_norms(self):
        [R,K] = self.C.shape
        [I,J,K] = self.Y.shape
        # Matrizice E into Em1 => (R,IJ,1)
        Em = tf.reshape(self.Eop(),[R,-1])
        Em1 = tf.expand_dims(Em,2)
        # Expand outer dimension for outer product Cm1(R,1,K)
        Cm1 = tf.expand_dims(self.C,1)
        # Compute outer product Ym1 = E o C => Ym1(R,IJ,K)
        Ym1 = tf.matmul(Em1,Cm1)
        # Matricize Ym1(R,IJ,K) => Ycv(R,IJK)
        Ycv = tf.reshape(Ym1,[R,-1])
        # Compute Frobenius norm of each component
        Yc_norms = tf.norm(Ycv,axis=1)
        Yc_norms = Yc_norms/tf.reduce_sum(Yc_norms)
        # print("Yc_norms:")
        # print(Yc_norms)
        
        Ec_norms = tf.norm(Em,axis=1)
        Ec_norms = Ec_norms/tf.reduce_sum(Ec_norms)
        # print("Ec_norms")
        # print(Ec_norms)
        
        Yc = tf.reshape(Ym1,[R,I,J,K]) 
        return Yc.numpy(),Yc_norms.numpy(),Ec_norms.numpy()
    
class LrModel2(LrModel):
    def __init__(self,target,Lr,R,seed=0,parms=LrModelParameters()):
        # tf.debugging.set_log_device_placement(True)
        self.Y = tf.constant(target,dtype=tf.float32)
        [I,J,K] = target.shape
        As = [I,Lr,R]
        Bs = [J,Lr,R]
        Cs = [R,K]
        
        self.seed = seed
        self.A = self.initvar(As)
        self.B = self.initvar(Bs)
        self.C = self.initvar(Cs)
        self.vars = (self.A,self.B,self.C)
        
        self.parms = parms
     
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.parms.lrate)
        # self.opt = tf.keras.optimizers.SGD(learning_rate=.001)
        # self.opt = tf.optimizers.Adagrad(learning_rate=0.001)
        # self.opt = tf.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    
    def __call__(self):
        self.apply_anc('relu')
        if self.parms.LimitMaxNorm: 
            self.max_norm_constraint()
        op1 = self.Eop()
        op = tf.tensordot(op1,self.C,[2,0],name="SpatialTimesSpectral")
        return op
    
    def Eop(self):
        [_,_,R] = self.A.shape
        Alist = tf.unstack(self.A,axis=2)
        Blist = tf.unstack(self.B,axis=2)
        op=[]
        for i in range(R):
            op.append(tf.matmul(Alist[i],Blist[i], \
                transpose_b=True,name=f'SpatialSlice{i:d}'))
        E = tf.stack(op,axis=2)
        return E
    
    def initvar(self,shape):
        # See Glorot normal initializer on 
        # m = 0.5
        # sd = np.sqrt(2./np.sum(shape))
        # init = tf.random.truncated_normal(shape=shape, dtype=tf.float32,
        #         mean=m, stddev=sd, seed=self.seed)
        # ... or use the built-in keras initializer
        ki = tf.keras.initializers.GlorotNormal(self.seed)
        # ki = tf.keras.initializers.GlorotUniform(self.seed)
        init = ki(shape)
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
        reg = tf.pow(tf.abs(self.Eop()),self.parms.RegNorm)
        reg = tf.pow(tf.reduce_mean(reg),1/self.parms.RegNorm)
        reg = reg*(self.parms.RegWeight)
        # an = tf.linalg.norm(self.A,ord=self.parms.RegNorm)/tf.cast(tf.reduce_prod(self.A.shape),dtype=tf.float32)
        # bn = tf.linalg.norm(self.B,ord=self.parms.RegNorm)/tf.cast(tf.reduce_prod(self.B.shape),dtype=tf.float32)
        # reg = self.parms.RegWeight*(an+bn)
        return reg
    
    def asc_term(self):
        Esum = tf.reduce_sum(self.Eop(),axis=0)
        diff2 = (Esum - tf.ones_like(Esum))**2
        a = tf.reduce_mean(diff2)*self.parms.AscWeight
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
            curr_cost = self.cost()
        grads = tape.gradient(curr_cost,self.vars)
        opt.apply_gradients(zip(grads,self.vars))
        return curr_cost

    def max_norm_constraint(self):
        (a_normalized, a_normals) = tf.linalg.normalize(self.A, \
             ord=np.inf, axis=1)
        b_scaled = tf.math.multiply(self.B,a_normals)
        cond = a_normals > 1
        self.A.assign(tf.where(cond, a_normalized,self.A))
        self.B.assign(tf.where(cond, b_scaled,self.B))

        # Unconditionally scale A
        # self.A.assign(a_normalized)
        # self.B.assign(b_scaled)
        
        # Normalize C
        # cmax = tf.reduce_max(self.C,axis=1,keepdims=True)
        # self.C.assign(tf.where(cmax>1,self.C/cmax,self.C))
        # self.C.assign(self.C/cmax)

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
        mvect = np.ones(self.parms.MovAvgCount)*1e10
        mavg_cost = np.mean(mvect)
        delta = 1e10; print_step = 1
        converged = False; i=0
        # print_title()
        cost_vector = []
        while(not converged):
            if not self.parms.optim_persist:
                self.opt = tf.keras.optimizers.Adam(learning_rate=self.parms.lrate)
            current_cost = self.train_keras(self.opt).numpy()
            current_loss = self.loss().numpy()  
            old_cost = mavg_cost
            mvect[i % mvect.size] = current_cost
            mavg_cost = np.mean(mvect)
            delta = old_cost - mavg_cost
            if i % print_step == 0:
                et = time.time()-t0
                print_train_step(current_cost, mavg_cost, i, current_loss, delta, et)
                # self.component_norms()
            i+=1
            converged = delta < self.parms.MaxDelta \
                or current_loss < self.parms.MaxLoss \
                or i > self.parms.MaxIter 
            cost_vector.append(current_cost)
        # Erase progress indicator
        print(" "*80,end='\r', flush=True)
        results = (cost_vector, delta, i, et)
        return(results)
        