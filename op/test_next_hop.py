import tensorflow as tf
next_hop_module = tf.load_op_library('./next_hop.so')

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.device_count['GPU']= 0
sess = tf.Session(config=config)

with sess:
    print("Session Running")
    res1, res2 = next_hop_module.next_hop(tf.convert_to_tensor([0.1,0.2,0.4,-1,0.6], dtype=tf.float32))
    print(res1)
    print(res2)
    print(res1.get_shape())
    print(res2.get_shape())
    ans = sess.run([res1, res2])
    pint(ans)
