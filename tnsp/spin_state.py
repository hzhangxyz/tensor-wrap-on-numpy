import time
import os
import sys
import importlib.util
import numpy as np
import tensorflow as tf
from .tensor_node import Node

# 载入两个算子，count_hop和next_hop
next_hop_path = os.path.join(os.path.split(__file__)[0], 'op', 'next_hop.so')
next_hop = tf.load_op_library(next_hop_path).next_hop

# 载入配置文件，里面有哈密顿量和边的连接方式，现在边的链接方式还有问题
spec = importlib.util.spec_from_file_location("config", "./config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

get_lattice_node_leg = config.get_lattice_node_leg
Hamiltonian = config.Hamiltonian

def auxiliary_generate(length, former, current, initial, L='l', R='r', U='u', D='d', scan_time=2):
    # U to D, scan from L to R
    res = [initial[i] for i in range(length)]
    L1 = f'{L}1'
    L2 = f'{L}2'
    L3 = f'{L}3'
    R1 = f'{R}1'
    R2 = f'{R}2'
    R3 = f'{R}3'

    for t in range(scan_time):
        with tf.name_scope(f'scan_time_{t}'):
            res[0], QR_R = res[0].tensor_qr([D], [R], [R, L], restrict_mode=False)
            l = [None for i in range(length)]
            l[0] = former[0]\
                .tensor_contract(current[0], [D], [U], {R: R1}, {R: R2}, restrict_mode=False)\
                .tensor_contract(res[0], [D], [D], {}, {R: R3}, restrict_mode=False)

            for j in range(1, length-1):
                if t == 0:
                    res[j] = Node.tensor_contract(res[j], QR_R, [L], [R], restrict_mode=False)
                    res[j], QR_R = res[j].tensor_qr([L, D], [R], [R, L], restrict_mode=False)
                    l[j] = l[j-1]\
                        .tensor_contract(former[j], [R1], [L], {}, {R: R1}, restrict_mode=False)\
                        .tensor_contract(current[j], [R2, D], [L, U], {}, {R: R2}, restrict_mode=False)\
                        .tensor_contract(res[j], [R3, D], [L, D], {}, {R: R3}, restrict_mode=False)
                else:
                    tmp = l[j-1]\
                        .tensor_contract(former[j], [R1], [L], {}, {R: R1}, restrict_mode=False)\
                        .tensor_contract(current[j], [R2, D], [L, U], {}, {R: R2}, restrict_mode=False)
                    res[j] = tmp\
                        .tensor_contract(r[j+1], [R1, R2], [L1, L2], {R3: L}, {L3: R}, restrict_mode=False)
                    res[j], QR_R = res[j].tensor_qr([L, D], [R], [R, L], restrict_mode=False)  # 这里R完全不需要
                    l[j] = tmp\
                        .tensor_contract(res[j], [R3, D], [L, D], {}, {R: R3}, restrict_mode=False)

            res[length-1] = l[length-2]\
                .tensor_contract(former[length-1], [R1], [L], restrict_mode=False)\
                .tensor_contract(current[length-1], [R2, D], [L, U], {R3: L}, {}, restrict_mode=False)

            res[length-1], QR_R = res[length-1].tensor_qr([D], [L], [L, R], restrict_mode=False)
            r = [None for i in range(length)]
            r[length-1] = former[length-1]\
                .tensor_contract(current[length-1], [D], [U], {L: L1}, {L: L2}, restrict_mode=False)\
                .tensor_contract(res[length-1], [D], [D], {}, {L: L3}, restrict_mode=False)

            for j in range(length-2, 0, -1):
                tmp = r[j+1]\
                    .tensor_contract(former[j], [L1], [R], {}, {L: L1}, restrict_mode=False)\
                    .tensor_contract(current[j], [L2, D], [R, U], {}, {L: L2}, restrict_mode=False)
                res[j] = tmp\
                    .tensor_contract(l[j-1], [L1, L2], [R1, R2], {L3: R}, {R3: L}, restrict_mode=False)
                res[j], QR_R = res[j].tensor_qr([R, D], [L], [L, R], restrict_mode=False)
                r[j] = tmp\
                    .tensor_contract(res[j], [L3, D], [R, D], {}, {L: L3}, restrict_mode=False)

            res[0] = former[0]\
                .tensor_contract(current[0], [D], [U], {R: R1}, {R: R2}, restrict_mode=False)\
                .tensor_contract(r[1], [R1, R2], [L1, L2], {}, {L3: R}, restrict_mode=False)

    return res

class SpinState():

    def __init__(self, size, D, D_c, scan_time, TYPE=tf.float64):
        # 保存一些小数据
        self.size = size
        self.D = D
        self.TYPE = TYPE
        n, m = self.size

        # 生成state，lat，lat_hop的place_holder
        def gen_place_holder(i, j, prefix):
            legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
            return Node(tf.placeholder(self.TYPE, shape=[self.D for i in legs], name=f'{prefix}_{i}_{j}'), legs)
        with tf.name_scope('static'):
            with tf.name_scope('state'):
                self.state = tf.placeholder(tf.int32, [n, m], name='spin_state')
            with tf.name_scope('lat'):
                self.lat = [[gen_place_holder(i, j, prefix='lat') for j in range(m)] for i in range(n)]
            with tf.name_scope('lat_hop'):
                self.lat_hop = [[gen_place_holder(i, j, prefix='lat_hop') for j in range(m)] for i in range(n)]
            with tf.name_scope('misc'):
                self.random_num = tf.placeholder(tf.float64, shape=[2], name='random_num')
                self.one = tf.convert_to_tensor(1, dtype=self.TYPE, name='one')

        # 保存另外一些小数据
        self.D_c = D_c
        self.scan_time = scan_time

        # 生成辅助矩阵，并计算ws和其他所有东西
        with tf.name_scope('aux'):
            self.__auxiliary()
        with tf.name_scope('w_s'):
            self.cal_w_s()
        with tf.name_scope('cal_e_s_and_delta_s_and_markov_hop'):
            self.cal_E_s_and_Delta_s_and_markov_hop()

    def cal_w_s(self):
        n, m = self.size

        self.w_s = Node(self.one)
        for j in range(0, m):
            self.w_s = self.w_s\
                .tensor_contract(self.UpToDown[n-2][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                .tensor_contract(self.lat[n-1][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)
        self.w_s = self.w_s.data
        assert self.w_s != 0., "w_s == 0"

    def cal_E_s_and_Delta_s_and_markov_hop(self):
        # E_s=\sum_{s'} W(s')/W(s) H_{ss'}
        # H current allow:
        # OK       OK
        #    OK OK
        #    XX XX
        # XX       XX

        # 初始化三个输出，能量，梯度，markov的pool
        n, m = self.size
        E_s = []
        Delta_s = [[None for j in range(m)] for i in range(n)]  # 下面每个点记录一下
        markov = []

        # 横向j j+1
        for i in range(n):
            with tf.name_scope(f'mpo_h_{i}'):
                # 计算单列的辅助矩阵
                l = [None for j in range(m)]
                l[-1] = Node(self.one)
                r = [None for j in range(m)]
                r[0] = Node(self.one)

                with tf.name_scope('aux'):
                    for j in range(0, m-1):
                        l[j] = l[(j-1) % m]\
                            .tensor_contract(self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                            .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)\
                            .tensor_contract(self.DownToUp[(i+1) % n][j], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)
                    for j in range(m-1, 0, -1):
                        r[j] = r[(j+1) % m]\
                            .tensor_contract(self.UpToDown[(i-1) % n][j], ['l1'], ['r'], {}, {'l': 'l1'}, restrict_mode=False)\
                            .tensor_contract(self.lat[i][j], ['l2', 'd'], ['r', 'u'], {}, {'l': 'l2'}, restrict_mode=False)\
                            .tensor_contract(self.DownToUp[(i+1) % n][j], ['l3', 'd'], ['r', 'u'], {}, {'l': 'l3'}, restrict_mode=False)

                # 计算 delta, 填入lat_hop就是hop概率
                with tf.name_scope('grad'):
                    for j in range(m):
                        Delta_s[i][j] = Node.tensor_contract(
                            Node.tensor_contract(l[(j-1) % m], self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {'r2': 'l'}, {'r': 'r1', 'd': 'u'}, restrict_mode=False),
                            Node.tensor_contract(r[(j+1) % m], self.DownToUp[(i+1) % n][j], ['l3'], ['r'], {'l2': 'r'}, {'l': 'l3', 'u': 'd'}, restrict_mode=False),
                            ['r1', 'r3'], ['l1', 'l3'], restrict_mode=False)
                        wss = Delta_s[i][j].tensor_contract(self.lat_hop[i][j], ['l','r','u','d'], ['l', 'r', 'u', 'd'], restrict_mode=False).data
                        markov.append((wss*wss)/(self.w_s*self.w_s))

                # 计算Es
                with tf.name_scope('H_ss'):
                    for j in range(m-1):
                        H = Hamiltonian(i,j,i,j+1)
                        get_res = lambda :(l[(j-1) % m]
                                           .tensor_contract(self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)
                                           .tensor_contract(self.lat_hop[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)
                                           .tensor_contract(self.DownToUp[(i+1) % n][j], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)
                                           .tensor_contract(self.UpToDown[(i-1) % n][(j+1) % m], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)
                                           .tensor_contract(self.lat_hop[i][(j+1) % m], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)
                                           .tensor_contract(self.DownToUp[(i+1) % n][(j+1) % m], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)
                                           .tensor_contract(r[(j+2) % m], ['r1', 'r2', 'r3'], ['l1', 'l2', 'l3'], restrict_mode=False)).data
                        def if_can_hop():
                            tmp = H[1,1] #tf.convert_to_tensor(H[1,1], dtype=self.TYPE)
                            if H[1,2] != 0.0:
                                res = get_res()
                                tmp += res * H[1,2] / self.w_s
                            else:
                                pass
                            return tmp

                        def if_cannot_hop():
                            tmp = H[0,0] #tf.convert_to_tensor(H[0,0], dtype=self.TYPE)
                            if H[0,3] != 0.0:
                                res = get_res()
                                tmp += res * H[0,3] / self.w_s
                            else:
                                pass
                            return tmp
                        e_s_to_append = tf.cond(tf.not_equal(self.state[i][j], self.state[i][j+1]), if_can_hop, if_cannot_hop)
                        E_s.append(e_s_to_append)
        # 纵向i i+1
        for j in range(m):
            with tf.name_scope(f'mpo_v_{j}'):
                # 计算单列的辅助矩阵
                u = [None for i in range(n)]
                u[-1] = Node(self.one)
                d = [None for i in range(n)]
                d[0] = Node(self.one)

                with tf.name_scope('aux'):
                    for i in range(0, n-1):
                        u[i] = u[(i-1) % n]\
                            .tensor_contract(self.LeftToRight[i][(j-1) % m], ['d1'], ['u'], {}, {'d': 'd1'}, restrict_mode=False)\
                            .tensor_contract(self.lat[i][j], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)\
                            .tensor_contract(self.RightToLeft[i][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)
                    for i in range(n-1, 0, -1):
                        d[i] = d[(i+1) % n]\
                            .tensor_contract(self.LeftToRight[i][(j-1) % m], ['u1'], ['d'], {}, {'u': 'u1'}, restrict_mode=False)\
                            .tensor_contract(self.lat[i][j], ['u2', 'r'], ['d', 'l'], {}, {'u': 'u2'}, restrict_mode=False)\
                            .tensor_contract(self.RightToLeft[i][(j+1) % m], ['u3', 'r'], ['d', 'l'], {}, {'u': 'u3'}, restrict_mode=False)
                # 计算Es
                with tf.name_scope('H_ss'):
                    for i in range(n-1):
                        H = Hamiltonian(i,j,i+1,j)
                        get_res = lambda :(u[(i-1) % n]
                                           .tensor_contract(self.LeftToRight[i][(j-1) % m], ['d1'], ['u'], {}, {'d': 'd1'}, restrict_mode=False)
                                           .tensor_contract(self.lat_hop[i][j], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)
                                           .tensor_contract(self.RightToLeft[i][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)
                                           .tensor_contract(self.LeftToRight[(i+1) % n][(j-1) % m], ['d1'], 'u', {}, {'d': 'd1'}, restrict_mode=False)
                                           .tensor_contract(self.lat_hop[(i+1) % n][j], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)
                                           .tensor_contract(self.RightToLeft[(i+1) % n][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)
                                           .tensor_contract(d[(i+2) % n], ['d1', 'd2', 'd3'], ['u1', 'u2', 'u3'], restrict_mode=False)).data
                        def if_can_hop():
                            # H(1,1)
                            tmp = H[1,1] #tf.convert_to_tensor(H[1,1], dtype=self.TYPE)
                            # H(1,2)
                            if H[1,2] != 0.0:
                                res = get_res()
                                tmp += res * H[1,2] / self.w_s
                            else:
                                pass
                            return tmp

                        def if_cannot_hop():
                            # H(0,0)
                            tmp = H[0,0] #tf.convert_to_tensor(H[0,0], dtype=self.TYPE)
                            # H(0,3)
                            if H[0,3] != 0.0:
                                res = get_res()
                                tmp += res * H[0,3] / self.w_s
                            else:
                                pass
                            return tmp
                        e_s_to_append = tf.cond(tf.not_equal(self.state[i][j], self.state[i+1][j]), if_can_hop, if_cannot_hop)
                        E_s.append(e_s_to_append)

        with tf.name_scope('markov'):
            self.stay_step, self.next_index = next_hop(markov, self.random_num)

        with tf.name_scope('res'):
            for i in range(n):
                for j in range(m):
                    Delta_s[i][j] = tf.div(Delta_s[i][j].tensor_transpose(get_lattice_node_leg(i, j, n, m)).data, self.w_s, name=f'grad_{i}_{j}')
            self.energy = tf.reduce_sum(E_s, name='e_s')
            self.grad = Delta_s

    def __call__(self, sess, state, lat, lat_hop, random_num):
        n, m = self.size
        feed_dict = {self.state: state, self.random_num: random_num}
        for i in range(n):
            for j in range(m):
                feed_dict[self.lat[i][j].data] = lat[i][j]
                feed_dict[self.lat_hop[i][j].data] = lat_hop[i][j]
        return sess.run({"energy": self.energy, "grad": self.grad, "step": self.stay_step, "next": self.next_index}, feed_dict=feed_dict)

    def __auxiliary(self):
        self.__auxiliary_up_to_down()
        self.__auxiliary_down_to_up()
        self.__auxiliary_left_to_right()
        self.__auxiliary_right_to_left()

    def __auxiliary_up_to_down(self):
        n, m = self.size
        with tf.name_scope("UpToDown"):
            self.UpToDown = [None for i in range(n)]
            self.UpToDown[0] = [self.lat[0][j] for j in range(m)]
            for i in range(1, n-1):
                with tf.name_scope(f"UpToDown_{i}"):
                    initial = [None for j in range(m)]
                    for j in range(m):
                        legs = get_lattice_node_leg(i, j, n, m).replace('u','').replace('d','')
                        initial[j] = Node(tf.random_uniform([self.D, *[self.D_c for l in legs]], dtype=self.TYPE), legs=['d', *legs])
                    self.UpToDown[i] = auxiliary_generate(m, self.UpToDown[i-1], self.lat[i], initial, L='l', R='r', U='u', D='d', scan_time=self.scan_time)
            self.UpToDown[n-1] = [Node(self.one) for j in range(m)]

    def __auxiliary_down_to_up(self):
        n, m = self.size
        with tf.name_scope("DownToUp"):
            self.DownToUp = [None for i in range(n)]
            self.DownToUp[n-1] = [self.lat[n-1][j] for j in range(m)]
            for i in range(n-2, 0, -1):
                with tf.name_scope(f"DownToUp_{i}"):
                    initial = [None for j in range(m)]
                    for j in range(m):
                        legs = get_lattice_node_leg(i, j, n, m).replace('d','').replace('u','')
                        initial[j] = Node(tf.random_uniform([self.D, *[self.D_c for l in legs]], dtype=self.TYPE), legs=['u', *legs])
                    self.DownToUp[i] = auxiliary_generate(m, self.DownToUp[i+1], self.lat[i], initial, L='l', R='r', U='d', D='u', scan_time=self.scan_time)
            self.DownToUp[0] = [Node(self.one) for j in range(m)]

    def __auxiliary_left_to_right(self):
        n, m = self.size
        with tf.name_scope("LeftToRight"):
            self.LeftToRight = [None for j in range(m)]
            self.LeftToRight[0] = [self.lat[i][0] for i in range(n)]
            for j in range(1, m-1):
                with tf.name_scope(f"LeftToRight_{j}"):
                    initial = [None for j in range(n)]
                    for i in range(n):
                        legs = get_lattice_node_leg(i, j, n, m).replace('l','').replace('r','')
                        initial[i] = Node(tf.random_uniform([self.D, *[self.D_c for l in legs]], dtype=self.TYPE), legs=['r', *legs])
                    self.LeftToRight[j] = auxiliary_generate(n, self.LeftToRight[j-1], [self.lat[t][j] for t in range(n)], initial,
                                                             L='u', R='d', U='l', D='r', scan_time=self.scan_time)
            self.LeftToRight[m-1] = [Node(self.one) for i in range(n)]
            tmp = self.LeftToRight
            self.LeftToRight = [[tmp[j][i] for j in range(m)] for i in range(n)]

    def __auxiliary_right_to_left(self):
        n, m = self.size
        with tf.name_scope("RightToLeft"):
            self.RightToLeft = [None for j in range(m)]
            self.RightToLeft[m-1] = [self.lat[i][m-1] for i in range(n)]
            for j in range(m-2, 0, -1):
                with tf.name_scope(f"RightToLeft_{j}"):
                    initial = [None for j in range(n)]
                    for i in range(n):
                        legs = get_lattice_node_leg(i, j, n, m).replace('r','').replace('l','')
                        initial[i] = Node(tf.random_uniform([self.D, *[self.D_c for l in legs]], dtype=self.TYPE), legs=['l', *legs])
                    self.RightToLeft[j] = auxiliary_generate(n, self.RightToLeft[j+1], [self.lat[t][j] for t in range(n)], initial,
                                                             L='u', R='d', U='r', D='l', scan_time=self.scan_time)
            self.RightToLeft[0] = [Node(self.one) for i in range(n)]
            tmp = self.RightToLeft
            self.RightToLeft = [[tmp[j][i] for j in range(m)] for i in range(n)]
