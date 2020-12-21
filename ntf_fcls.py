''' 
  Tensorflow implementations of Fully Constrained 
  Least Squares per Heinz 1999
'''
import numpy as np
import tensorflow as tf

def pinv(a, rcond=1e-15):
    s, u, v = tf.linalg.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.math.greater(s, limit)
    reciprocal = tf.where(non_zero, tf.math.reciprocal(s), tf.zeros_like(s))
    lhs = tf.matmul(v, tf.linalg.tensor_diag(reciprocal))
    return tf.matmul(lhs, u, transpose_b=True)

#####################################################
# Implementation most similar to Matlab with
# all matrix operations in the inner loop
# fcls Elapsed time: 71.63096284866333 seconds
# rmse: [0.02025334 0.04872063 0.03135224 0.05578368]
def fcls(Yin, Sin):
    Sin = Sin / tf.sqrt(tf.reduce_sum(tf.square(Sin),axis=0))
    Yin = Yin / tf.sqrt(tf.reduce_sum(tf.square(Yin),axis=0))

    (K,I,J) = Yin.shape
    (K,R) = Sin.shape
    # Yin = tf.transpose(Yin,(0,2,1))
    Ymat = tf.reshape(Yin,shape=(K,I*J))
    N = I*J
    A = np.zeros((R,N),dtype=np.float)
    # Select pixel to process
    # Iterate through pixels

    for n in range(N):
        #yn = tf.reshape(Ymat[:,n],shape=(K,1))
        yn = tf.slice(Ymat,[0,n],[-1,1])
        alpha = np.zeros(R,dtype=np.float)
        alpha_idx = [_ for _ in range(R)]
        S = Sin
        for r in range(R):
            StS = tf.matmul(S,S,transpose_a=True)
            StSinv = tf.linalg.inv(StS)
            StSinvSt = tf.matmul(StSinv,S,transpose_b=True)
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
            else:
                break
        alpha[alpha_idx] = afcls_hat[:,0]
        A[:,n]=alpha
        if n%1000 == 0 : print(n)
    return A

#####################################################
# Attempt to convert the take tf.matmul(StSinvSt,yn)
# outside of the innerloop as tf.matmul(StSinv,StYmat).
# tff Elapsed time: 69.94525814056396 seconds
# rmse: [0.02025338 0.04872065 0.0313523  0.0557838 ]
def fcls_tff(Yin, Sin):
    S0 = Sin / tf.sqrt(tf.reduce_sum(tf.square(Sin),axis=0))
    Yin = Yin / tf.sqrt(tf.reduce_sum(tf.square(Yin),axis=0))
    (K,I,J) = Yin.shape
    (K,R) = Sin.shape
    Ymat = tf.reshape(Yin,shape=(K,I*J))
    N = I*J
    A = np.zeros((R,N),dtype=np.float)
    # A = tf.constant(tf.zeros((R,N),dtype=tf.float32))
    # Select pixel to process
    # Iterate through pixels
    StS0 = tf.constant(tf.matmul(S0,S0,transpose_a=True))
    StSinv0 = tf.constant(tf.linalg.inv(StS0))
    #StSinvSt0 = tf.matmul(StSinv0,S0,transpose_b=True)
    StY0 = tf.constant(tf.matmul(S0,Ymat,transpose_a=True))
    als_hat0 = tf.constant(tf.matmul(StSinv0,StY0))
    for n in range(N):
        # yn = tf.reshape(Ymat[:,n],shape=(K,1))
        alpha = np.zeros(R,dtype=np.float)
        # alpha = tf.zeros(R,dtype=tf.float32)
        alpha_idx = [_ for _ in range(R)]
        StSinv = StSinv0
        als_hat = tf.reshape(als_hat0[:,n],shape=(R,1))
        S = S0
        for r in range(R):
            afcls_hat = als_hat \
                - tf.reduce_sum(StSinv,axis=1, keepdims=True) \
                * tf.math.reciprocal(tf.reduce_sum(StSinv)) \
                * (tf.reduce_sum(als_hat)-1)
            # Find negative abundances
            if tf.reduce_any(tf.less(afcls_hat,0)):
                csum = tf.reduce_sum(StSinv,axis=1,keepdims=True)
                scaled_neg = tf.where(afcls_hat<0, tf.abs(afcls_hat / csum), 0)
                # Identify most negative scaled alpha and remove
                # from from alpha_idx
                min_idx = np.argmax(scaled_neg)
                # del alpha_idx[min_idx]
                min_idx = alpha_idx[min_idx]
                alpha_idx.remove(min_idx)

                # StS = tf.matmul(S,S,transpose_a=True)
                StS = StS0.numpy()[:,alpha_idx]
                StS = StS[alpha_idx,:]
                StSinv = tf.linalg.inv(StS)
                S = tf.constant(S0.numpy()[:,alpha_idx])
                #StSinvSt = tf.matmul(StSinv,S,transpose_b=True)
                Yn = tf.reshape(Ymat[:,n],shape=(K,1))
                StYn = tf.matmul(S,Yn,transpose_a=True)
                als_hat = tf.matmul(StSinv,StYn)
            else:
                alpha[alpha_idx] = afcls_hat[:,0]
                break
        A[:,n]=alpha
            
        if n%1000 == 0 : print(n)
    return A

#####################################################
# Refactor fcls in order to use tensorflow while_loop
# this requires creating helper functions fcls_cond()
# and fcls_body.  Also attemped to isolate the coupute
# graph in tf_compute().  When decorating tf.compute 
# with @tf.function performace crashed.
# tff2 Elapsed time: 63.83496284484863 seconds
# rmse: [0.02025334 0.04872063 0.03135224 0.05578368]
def fcls_tff2(Yin, Sin):
    Sin = Sin / tf.sqrt(tf.reduce_sum(tf.square(Sin),axis=0))
    Yin = Yin / tf.sqrt(tf.reduce_sum(tf.square(Yin),axis=0))
    (K,I,J) = Yin.shape; N=I*J
    (K,R) = Sin.shape
    Ymat = tf.reshape(Yin,shape=(K,N))
    A = tf.Variable(tf.zeros((R,N),dtype=tf.float32))
    n = tf.Variable(0,dtype=tf.int32)
    
    t0 = time.time()
    ########################################
    # tensorflow while_loop helper funtions
    def fcls_cond(Ymat,n):
        (K,N)=Ymat.shape
        return n<N
    def fcls_body(Ymat,n):
        yn = tf.slice(Ymat,[0,n],[-1,1])
        (K,R) = Sin.shape
        alpha = tf.zeros(R,dtype=np.float)
        alpha_idx = [_ for _ in range(R)]
        S = tf.gather(Sin,alpha_idx,axis=1)
        ############################################
        # using @tf.function makes this 10 times
        # slower. It seems to regenerate the compute
        # graph on every iteration.  Need a profiler
        #@tf.function
        def tf_compute(S,yn):
            StS = tf.matmul(S,S,transpose_a=True)
            StSinv = tf.linalg.inv(StS)
            StSinvSt = tf.matmul(StSinv,S,transpose_b=True)
            als_hat = tf.matmul(StSinvSt,yn)
            csum = tf.reduce_sum(StSinv,axis=1,keepdims=True)
            afcls_hat = als_hat - csum  \
                * tf.math.reciprocal(tf.reduce_sum(StSinv)) \
                * (tf.reduce_sum(als_hat)-1)
            return (afcls_hat,csum)
        
        not_done = True
        #while not_done:
        for r in range(R) :
            (afcls_hat,csum) = tf_compute(S,yn)
            if tf.reduce_any(tf.less(afcls_hat,0)):
                scaled_neg = tf.where(afcls_hat<0, tf.abs(afcls_hat / csum), 0)
                min_idx = np.argmax(scaled_neg)
                min_idx = alpha_idx[min_idx]
                alpha_idx.remove(min_idx)
                S = tf.gather(Sin,alpha_idx,axis=1)
            else:
                #not_done=False
                break

        indices = tf.reshape(alpha_idx,shape=(len(alpha_idx),1))
        alpha = tf.scatter_nd(indices,afcls_hat,(R,1))
        #A[:,n]=alpha[:,0]
        A[:,n].assign(alpha[:,0])
        
        if n%1000 == 0 : 
            elt = time.time() - t0
            print(f'Iter:{n} Elapsed:{elt}')
        #x = (Ymat,n.assign_add(1))
        x = (Ymat,n+1)
        return x
    # end of helper functions
    ########################################
    
    x=tf.while_loop(fcls_cond,fcls_body,(Ymat,n),parallel_iterations=200)
    # for n in range(N):
    #     yn = tf.slice(Ymat,[0,n],[-1,1])
    #     x = fcls_body(yn,n)
    return A.numpy()

###########################################################
# Improved 33% by taking the slicing oustside of the fcls_body
# Implemented tf.fn_map() with no additional gain.
# Added changes that should have gone into fcls_tff4:
#  - Modified fcls_body() to take yn instead of Ymat and
#    return alpha as a 1D vector.
#  - Used tf.fn_map() to run vecorized iterations on Ymat
#    returning A .
# tff3 Elapsed time: 106.83657383918762 seconds
# rmse: [0.02025335 0.0487207  0.03135224 0.05578441]
def fcls_tff3(Yin, Sin):
    Sin = Sin / tf.sqrt(tf.reduce_sum(tf.square(Sin),axis=0))
    Yin = Yin / tf.sqrt(tf.reduce_sum(tf.square(Yin),axis=0))
    (K,I,J) = Yin.shape; N=I*J
    (K,R) = Sin.shape
    Ymat = tf.reshape(Yin,shape=(K,N))
    A = tf.Variable(tf.zeros(shape=(R,N),dtype=tf.float32))
    n = tf.Variable(tf.zeros(shape=(1),dtype=tf.int32))
    t0=time.time()
    ########################################
    # tensorflow while_loop helper funtions
    def fcls_cond(Sin,Ymat,n,A):
        (K,N)=Ymat.shape
        return n<N
    
    def fcls_body(yn):
        #yn = tf.reshape(yn,(K,1))
        alpha = tf.zeros(R,dtype=np.float32)
        alpha_idx = [_ for _ in range(R)]
        # alpha_idx = tf.constant(range(R),dtype=tf.int32)
        S = tf.gather(Sin,alpha_idx,axis=1)
        
        def tf_compute(S,yn):
            #StS = tf.matmul(S,S,transpose_a=True)
            StS = tf.tensordot(S,S,axes=[0,0])
            StSinv = tf.linalg.inv(StS)
            #StYn = tf.matmul(S,yn,transpose_a=True)
            StYn = tf.tensordot(S,yn,axes=[0,0])
            #als_hat = tf.matmul(StSinv,StYn)
            als_hat = tf.tensordot(StSinv,StYn,[1,0])
            csum = tf.reduce_sum(StSinv,axis=1)
            afcls_hat = als_hat - csum  \
                * tf.math.reciprocal(tf.reduce_sum(StSinv)) \
                * (tf.reduce_sum(als_hat)-1)
            return (afcls_hat,csum)
        
        not_done = True
        #while not_done:
        for r in range(R) :
            (afcls_hat,csum) = tf_compute(S,yn)
            if tf.reduce_any(tf.less(afcls_hat,0)):
                scaled_neg = tf.where(afcls_hat<0, tf.abs(afcls_hat / csum), 0)
                min_idx = np.argmax(scaled_neg)
                #min_idx = tf.argmax(scaled_neg)
                min_idx = alpha_idx[min_idx]
                alpha_idx.remove(min_idx)
                #alpha_idx = tf.gather(alpha_idx,min_idx)
                S = tf.gather(Sin,alpha_idx,axis=1)
            else:
                not_done=False
                break

        indices = tf.reshape(alpha_idx,shape=(len(alpha_idx),1))
        # updates = tf.reshape(afcls_hat,shape=(len(afcls_hat),1))
        # alpha = tf.scatter_nd(indices,updates,(R,1))
        alpha = tf.scatter_nd(indices,afcls_hat,[R])
        # alpha = tf.reshape(alpha,(R,))
        # A[:,n]=tf.scatter_nd(indices,afcls_hat,(R,1))
        # n.assign_add(tf.ones(shape=(1),dtype=tf.int32))
        # if n%1000 == 0 : 
        #     elt = time.time() - t0
        #     print(f'Iter:{n} Elapsed:{elt}')
        # n=n+1
        return alpha
    # end of helper functions
    ########################################

    # n = tf.Variable(0,dtype=tf.int32)
    # x=(Sin,Ymat,n,A)
    # x=tf.while_loop(fcls_cond,fcls_body,(x),parallel_iterations=200)
    # for n in range(N):
    #     yn = tf.slice(Ymat,[0,n],[-1,1])
    #     alpha = fcls_body(yn)
    #     A[:,n].assign(alpha[:,0])
    Ymat = tf.constant(tf.transpose(Ymat))
    AlphaMat = tf.map_fn(fcls_body,Ymat,parallel_iterations=1000)
    #AlphaMat = tf.vectorized_map(fcls_body,Ymat)
    A = tf.transpose(AlphaMat)
    return A.numpy()
