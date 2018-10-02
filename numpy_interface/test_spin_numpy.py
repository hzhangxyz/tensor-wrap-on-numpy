import time
import numpy_wrap as np
import square_lattice as sl
import tensorflow as tf

L = 2
D = 3
print("CONSTRUCTED")
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.device_count['GPU']= 0
sess = tf.Session(config=config)
print("SESSION READY")
#tf.summary.FileWriter('./run', sess.graph)
print("GRAPH WROTE")
np.random.seed(23)

Lattice = sl.SquareLattice(L,L,D=D,D_c=7,scan_time=2,step_size=0,markov_chain_length=0)
print(Lattice.spin.lat)
print(Lattice.spin.cal_E_s_and_Delta_s())
#for i in Lattice

print("DATA PREPARED")
print("START")

