import numpy as np
import tensorflow as tf

from tnsp import SquareLattice
print('imported program')

print('构建网络')
sl = SquareLattice(2,2,3,4,1,0.01,10)
print('构建网络成功')

print('创建session')
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
print('创建session成功')

sl.grad_descent(sess)
