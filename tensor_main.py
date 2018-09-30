from spin_state import SpinState
import tensorflow as tf
SpinState([3,3],2,3,1)
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
tf.summary.FileWriter('./train', sess.graph)

