import numpy as np
import tensorflow as tf
from spin_state import SpinState

ss = SpinState([3,3],2,3,1)
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
tf.summary.FileWriter('./train', sess.graph)
def __create_node(i, j, n, m, D):
    legs = 'lrud'
    if i == 0:
        legs = legs.replace('u', '')
    if i == n-1:
        legs = legs.replace('d', '')
    if j == 0:
        legs = legs.replace('l', '')
    if j == m-1:
        legs = legs.replace('r', '')
    return np.random.rand(*[D for i in legs])


spin = np.array([1,0,1,0,1,0,1,0,1]).reshape(3,3)
lat = [[__create_node(i,j,3,3,2) for j in range(3) ]for i in range(3)]
lat_hop = [[__create_node(i,j,3,3,2) for j in range(3) ]for i in range(3)]

feed_dict = {ss.state: spin}
for i in range(3):
    for j in range(3):
        feed_dict[ss.lat[i][j].data] = lat[i][j]
        feed_dict[ss.lat_hop[i][j].data] = lat_hop[i][j]

print("START")
E= sess.run(ss.energy, feed_dict=feed_dict)
print(E)
