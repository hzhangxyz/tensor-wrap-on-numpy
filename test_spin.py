import time
import numpy as np
import tensorflow as tf
from tnsp.spin_state import SpinState

L = 2
D = 3
ss = SpinState([L,L],D=D,D_c=7,scan_time=2)
print("CONSTRUCTED")
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.device_count['GPU']= 0
sess = tf.Session(config=config)
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
    return np.random.rand(2, *[D for i in legs])

spin = [[ 0 if (i+j)%2==0 else 1 for j in range(L) ]for i in range(L)]
np.random.seed(23)
lattice = [[__create_node(i,j,L,L,D) for j in range(L) ]for i in range(L)]
lat = [[lattice[i][j][spin[i][j]] for j in range(L)] for i in range(L)]
lat_hop = [[lattice[i][j][1-spin[i][j]] for j in range(L)] for i in range(L)]
print(lat)

feed_dict = {ss.state: spin}
for i in range(L):
    for j in range(L):
        feed_dict[ss.lat[i][j].data] = lat[i][j]
        feed_dict[ss.lat_hop[i][j].data] = lat_hop[i][j]

print("DATA PREPARED")
print("START")

E= ss(sess, spin, lat, lat_hop)
print(E)
"""
for _ in range(3):
    print('start')
    E= ss(sess, spin, lat, lat_hop)
    print(E)
    time.sleep(1)
for _ in range(3):
    print('start')
    E= sess.run((ss.energy, ss.stay_step, ss.next_index), feed_dict=feed_dict2)
    print(E)
    time.sleep(1)
"""
