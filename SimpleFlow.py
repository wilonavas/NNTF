import tensorflow as tf
import numpy as np
import time

#
# This is a test to benchmark how tensorflow performance
# scales with vector size. 
#

# Create matrices with random data
Alist = []
for i in range(8,9):
    Nx = int(2**(i/2))
    # Nx = i*1024
    An = tf.random.uniform((Nx,Nx),dtype=tf.float32)
    Alist.append(An)

# Detect tensorflow version
fullver = tf.version.VERSION
ver = int(fullver.split('.')[0])
print('API Version: ',fullver)

# Do a thousand matrix multiplications
for _ in range(10000):
    if ver == 1:
        with tf.compat.v1.Session() as sess:
            # print('A=',A.eval())
            # print('B=',B.eval())
            for i,An in enumerate(Alist):
                #Amult = tf.linalg.inv(An)
                Amult = tf.matmul(An,An)
                t0 = time.time()
                multC = sess.run(Amult)
                (N,M) = multC.shape
                etime = time.time()-t0
                print('{} {} {} {}'.format(i,N,N*N,etime))
            # print('C=',C)
    elif ver == 2:
        for i,An in enumerate(Alist):
            t0 = time.time()
            #Amult = tf.linalg.inv(An)
            Amult = tf.matmul(An,An)
            (N,M) = Amult.shape
            multC = Amult.numpy()
            etime = time.time()-t0
            # multC = Amult.numpy()
            print('{} {} {} {}'.format(i,N,N*N,etime))
    else:
        print(f'API v{ver} not supported.')
