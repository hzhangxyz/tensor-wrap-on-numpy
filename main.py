import os
import numpy as np
import tensorflow as tf

print('载入程序中', end='\r')
from tnsp import SquareLattice
print('载入程序既')

print('构建网络中', end='\r')
np.random.seed(42)
sl = SquareLattice([3,3],D=4,D_c=6,scan_time=2,step_size=0.01,markov_chain_length=20)
print('构建网络既')

print('创建session中', end='\r')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
print('创建session既')

sl.grad_descent(sess)
