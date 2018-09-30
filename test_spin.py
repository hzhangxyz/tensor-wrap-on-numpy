import numpy as np
import tensorflow as tf
from tnsp.spin_state import SpinState

L = 4
D = 5
ss = SpinState([L,L],D=D,D_c=7,scan_time=2)
print("CONSTRUCTED")
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
print("SESSION READY")
#tf.summary.FileWriter('./run', sess.graph)
print("GRAPH WROTE")
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


spin = [[ 1 if (i+j)%2==0 else 0 for j in range(L) ]for i in range(L)]
lat = [[__create_node(i,j,L,L,D) for j in range(L) ]for i in range(L)]
lat_hop = [[__create_node(i,j,L,L,D) for j in range(L) ]for i in range(L)]
print("DATA PREPARED")

feed_dict = {ss.state: spin}
for i in range(L):
    for j in range(L):
        feed_dict[ss.lat[i][j].data] = lat[i][j]
        feed_dict[ss.lat_hop[i][j].data] = lat_hop[i][j]

print("START")
E= sess.run(ss.energy, feed_dict=feed_dict)
print(E)
