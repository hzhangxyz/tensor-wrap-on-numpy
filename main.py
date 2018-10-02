import numpy as np
import tensorflow as tf

from tnsp import SquareLattice
print('imported program')

print('构建网络')
sl = SquareLattice(4,4,D=2,D_c=6,scan_time=2,step_size=0.01,markov_chain_length=100)
print('构建网络成功')

print('创建session')
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
print('创建session成功')

sl.grad_descent(sess)
