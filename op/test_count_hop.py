import tensorflow as tf
count_hop_module = tf.load_op_library('./count_hop.so')

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.device_count['GPU']= 0
sess = tf.Session(config=config)

with sess:
    print("Session Running")
    res = count_hop_module.count_hop(tf.convert_to_tensor([[1,0], [1,0], [0,1]], dtype=tf.int32), tf.convert_to_tensor([[0,0],[0,1]], dtype=tf.int32))
    print(res)
    print(res.get_shape())
    ans = res.eval()
    print(ans)
    print(type(ans))
