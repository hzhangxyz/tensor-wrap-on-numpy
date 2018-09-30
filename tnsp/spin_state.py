import time
import os
import sys
import numpy as np
import tensorflow as tf
from tensor_node import Node


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
            res[0], QR_R = res[0].tensor_qr([D], [R], [R, L])
            l = [None for i in range(length)]
            l[0] = former[0]\
                .tensor_contract(current[0], [D], [U], {R: R1}, {R: R2})\
                .tensor_contract(res[0], [D], [D], {}, {R: R3})

            for j in range(1, length-1):
                if t == 0:
                    res[j] = Node.tensor_contract(res[j], QR_R, [L], [R])
                    res[j], QR_R = res[j].tensor_qr([L, D], [R], [R, L])
                    l[j] = l[j-1]\
                        .tensor_contract(former[j], [R1], [L], {}, {R: R1})\
                        .tensor_contract(current[j], [R2, D], [L, U], {}, {R: R2})\
                        .tensor_contract(res[j], [R3, D], [L, D], {}, {R: R3})
                else:
                    tmp = l[j-1]\
                        .tensor_contract(former[j], [R1], [L], {}, {R: R1})\
                        .tensor_contract(current[j], [R2, D], [L, U], {}, {R: R2})
                    res[j] = tmp\
                        .tensor_contract(r[j+1], [R1, R2], [L1, L2], {R3: L}, {L3: R})
                    res[j], QR_R = res[j].tensor_qr([L, D], [R], [R, L])  # 这里R完全不需要
                    l[j] = tmp\
                        .tensor_contract(res[j], [R3, D], [L, D], {}, {R: R3})

            res[length-1] = l[length-2]\
                .tensor_contract(former[length-1], [R1], [L])\
                .tensor_contract(current[length-1], [R2, D], [L, U], {R3: L}, {})

            res[length-1], QR_R = res[length-1].tensor_qr([D], [L], [L, R])
            r = [None for i in range(length)]
            r[length-1] = former[length-1]\
                .tensor_contract(current[length-1], [D], [U], {L: L1}, {L: L2})\
                .tensor_contract(res[length-1], [D], [D], {}, {L: L3})

            for j in range(length-2, 0, -1):
                tmp = r[j+1]\
                    .tensor_contract(former[j], [L1], [R], {}, {L: L1})\
                    .tensor_contract(current[j], [L2, D], [R, U], {}, {L: L2})
                res[j] = tmp\
                    .tensor_contract(l[j-1], [L1, L2], [R1, R2], {L3: R}, {R3: L})
                res[j], QR_R = res[j].tensor_qr([R, D], [L], [L, R])
                r[j] = tmp\
                    .tensor_contract(res[j], [L3, D], [R, D], {}, {L: L3})

            res[0] = former[0]\
                .tensor_contract(current[0], [D], [U], {R: R1}, {R: R2})\
                .tensor_contract(r[1], [R1, R2], [L1, L2], {}, {L3: R})

    return res


def get_lattice_node_leg(i, j, n, m):
    legs = 'lrud'
    if i == 0:
        legs = legs.replace('u', '')
    if i == n-1:
        legs = legs.replace('d', '')
    if j == 0:
        legs = legs.replace('l', '')
    if j == m-1:
        legs = legs.replace('r', '')
    return legs


class SpinState():

    def __gen_markov_chain_pool(self):
        n, m = self.size
        pool = []
        for i in range(n):
            for j in range(m):
                if j != m-1:
                    if self[i][j] != self[i][j+1]:
                        pool.append([i, j, i, j+1])
                if i != n-1:
                    if self[i][j] != self[i+1][j]:
                        pool.append([i, j, i+1, j])
        return pool

    def markov_chain_hop(self):
        pool = self.__gen_markov_chain_pool()
        choosed = pool[np.random.randint(len(pool))]
        alter = type(self)(self.lattice, self)
        alter[choosed[0]][choosed[1]] = 1 - alter[choosed[0]][choosed[1]]
        alter[choosed[2]][choosed[3]] = 1 - alter[choosed[2]][choosed[3]]
        alter.fresh()
        alter_pool = alter.__gen_markov_chain_pool()
        possibility = (alter.cal_w_s()**2)/(self.cal_w_s()**2)*len(pool)/len(alter_pool)
        # print("possi",possibility)
        if possibility > np.random.rand():
            return alter
        else:
            return self

    def __init__(self, size, D, D_c, scan_time, TYPE=tf.float32):
        self.size = size
        self.D = D
        self.TYPE = tf.float32
        n, m = self.size

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
        self.D_c = D_c
        self.scan_time = scan_time

        with tf.name_scope('aux'):
            self.__auxiliary()
        with tf.name_scope('w_s'):
            self.cal_w_s()
        with tf.name_scope('cal_e_s_and_delta_s'):
            self.cal_E_s_and_Delta_s()

    def cal_w_s(self):
        n, m = self.size

        self.w_s = Node(1.)
        for j in range(0, m):
            self.w_s = self.w_s\
                .tensor_contract(self.UpToDown[n-2][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                .tensor_contract(self.lat[n-1][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)
        self.w_s = self.w_s.data
        assert self.w_s != 0., "w_s == 0"

    def cal_E_s_and_Delta_s(self):
        """
        E_s=\sum_{s'} W(s')/W(s) H_{ss'}
        第一部分:对角
        第二部分:交换
        """
        n, m = self.size
        E_s_diag = 0.
        for i in range(n):
            for j in range(m):
                if j != m-1:
                    E_s_diag += 1 if self.state[i][j] == self.state[i][j+1] else -1  # 哈密顿量
                if i != n-1:
                    E_s_diag += 1 if self.state[i][j] == self.state[i+1][j] else -1
        E_s_non_diag = []  # 为相邻两个交换后的w(s)之和
        Delta_s = [[None for j in range(m)] for i in range(n)]  # 下面每个点记录一下
        # 横向j j+1
        for i in range(n):
            with tf.name_scope(f'mpo_h_{i}'):
                l = [None for j in range(m)]
                l[-1] = Node(1.)
                r = [None for j in range(m)]
                r[0] = Node(1.)

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
                # 计算 delta
                with tf.name_scope('grad'):
                    for j in range(m):
                        Delta_s[i][j] = Node.tensor_contract(
                            Node.tensor_contract(l[(j-1) % m], self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {'r2': 'l'}, {'r': 'r1', 'd': 'u'}, restrict_mode=False),
                            Node.tensor_contract(r[(j+1) % m], self.DownToUp[(i+1) % n][j], ['l3'], ['r'], {'l2': 'r'}, {'l': 'l3', 'u': 'd'}, restrict_mode=False),
                            ['r1', 'r3'], ['l1', 'l3'], restrict_mode=False)

                # 计算Es
                with tf.name_scope('H_ss'):
                    for j in range(m-1):
                        if self.state[i][j] != self.state[i][j+1]:
                            E_s_non_diag.append((l[(j-1) % m]\
                                .tensor_contract(self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                                .tensor_contract(self.lat_hop[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)\
                                .tensor_contract(self.DownToUp[(i+1) % n][j], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)\
                                .tensor_contract(self.UpToDown[(i-1) % n][(j+1) % m], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                                .tensor_contract(self.lat_hop[i][(j+1) % m], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)\
                                .tensor_contract(self.DownToUp[(i+1) % n][(j+1) % m], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)\
                                .tensor_contract(r[(j+2) % m], ['r1', 'r2', 'r3'], ['l1', 'l2', 'l3'], restrict_mode=False)).data * 2 / self.w_s)  # 哈密顿量
        # 纵向i i+1
        for j in range(m):
            with tf.name_scope(f'mpo_v_{j}'):
                u = [None for i in range(n)]
                u[-1] = Node(1.)
                d = [None for i in range(n)]
                d[0] = Node(1.)

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
                """
                # 计算 delta
                if cal_grad:
                    for i in range(n):
                        tmp = np.tensor_contract(
                            np.tensor_contract(u[(i-1) % n], self.LeftToRight[i][(j-1)%m], ['d1'], ['u'], {'d2': 'u'}, {'d': 'd1', 'r': 'l'}, restrict_mode=False),
                            np.tensor_contract(d[(i+1) % n], self.RightToLeft[i][(j+1)%m], ['u3'], ['d'], {'u2': 'd'}, {'u': 'u3', 'l': 'r'}, restrict_mode=False),
                            ['d1', 'd3'], ['u1', 'u3'], restrict_mode=False) / self.cal_w_s()
                        print(np.max(abs((tmp - Delta_s[i][j])/Delta_s[i][j])))
                """
                # 计算Es
                with tf.name_scope('H_ss'):
                    for i in range(n-1):
                        if self.state[i][j] != self.state[i+1][j]:
                            E_s_non_diag.append((u[(i-1) % n]\
                                .tensor_contract(self.LeftToRight[i][(j-1) % m], ['d1'], ['u'], {}, {'d': 'd1'}, restrict_mode=False)\
                                .tensor_contract(self.lat_hop[i][j], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)\
                                .tensor_contract(self.RightToLeft[i][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)\
                                .tensor_contract(self.LeftToRight[(i+1) % n][(j-1) % m], ['d1'], 'u', {}, {'d': 'd1'}, restrict_mode=False)\
                                .tensor_contract(self.lat_hop[(i+1) % n][j], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)\
                                .tensor_contract(self.RightToLeft[(i+1) % n][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)\
                                .tensor_contract(d[(i+2) % n], ['d1', 'd2', 'd3'], ['u1', 'u2', 'u3'], restrict_mode=False)).data * 2 / self.w_s)  # 哈密顿量

        #E_s = E_s_diag + E_s_non_diag
        with tf.name_scope('res'):
            for i in range(n):
                for j in range(m):
                    Delta_s[i][j] = tf.div(Delta_s[i][j].tensor_transpose(get_lattice_node_leg(i, j, n, m)).data, self.w_s, name=f'grad_{i}_{j}')
            E_s = sum(E_s_non_diag) + E_s_diag
            self.energy = tf.multiply(E_s, 0.25, name='e_s')
            self.grad = Delta_s

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
                        if j == 0:
                            initial[j] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['d', 'r'])
                        elif j == m-1:
                            initial[j] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['d', 'l'])
                        else:
                            initial[j] = Node(tf.random_uniform([self.D, self.D_c, self.D_c], dtype=self.TYPE), legs=['d', 'l', 'r'])
                    self.UpToDown[i] = auxiliary_generate(m, self.UpToDown[i-1], self.lat[i], initial, L='l', R='r', U='u', D='d', scan_time=self.scan_time)
            self.UpToDown[n-1] = [Node(1.) for j in range(m)]

    def __auxiliary_down_to_up(self):
        n, m = self.size
        with tf.name_scope("DownToUp"):
            self.DownToUp = [None for i in range(n)]
            self.DownToUp[n-1] = [self.lat[n-1][j] for j in range(m)]
            for i in range(n-2, 0, -1):
                with tf.name_scope(f"DownToUp_{i}"):
                    initial = [None for j in range(m)]
                    for j in range(m):
                        if j == 0:
                            initial[j] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['u', 'r'])
                        elif j == m-1:
                            initial[j] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['u', 'l'])
                        else:
                            initial[j] = Node(tf.random_uniform([self.D, self.D_c, self.D_c], dtype=self.TYPE), legs=['u', 'l', 'r'])
                    self.DownToUp[i] = auxiliary_generate(m, self.DownToUp[i+1], self.lat[i], initial, L='l', R='r', U='d', D='u', scan_time=self.scan_time)
            self.DownToUp[0] = [Node(1.) for j in range(m)]

    def __auxiliary_left_to_right(self):
        n, m = self.size
        with tf.name_scope("LeftToRight"):
            self.LeftToRight = [None for j in range(m)]
            self.LeftToRight[0] = [self.lat[i][0] for i in range(n)]
            for j in range(1, m-1):
                with tf.name_scope(f"LeftToRight_{j}"):
                    initial = [None for j in range(n)]
                    for i in range(n):
                        if i == 0:
                            initial[i] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['r', 'd'])
                        elif i == n-1:
                            initial[i] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['r', 'u'])
                        else:
                            initial[i] = Node(tf.random_uniform([self.D, self.D_c, self.D_c], dtype=self.TYPE), legs=['r', 'd', 'u'])
                    self.LeftToRight[j] = auxiliary_generate(n, self.LeftToRight[j-1], [self.lat[t][j] for t in range(n)], initial,
                                                         L='u', R='d', U='l', D='r', scan_time=self.scan_time)
            self.LeftToRight[m-1] = [Node(1.) for i in range(n)]
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
                        if i == 0:
                            initial[i] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['l', 'd'])
                        elif i == n-1:
                            initial[i] = Node(tf.random_uniform([self.D, self.D_c], dtype=self.TYPE), legs=['l', 'u'])
                        else:
                            initial[i] = Node(tf.random_uniform([self.D, self.D_c, self.D_c], dtype=self.TYPE), legs=['l', 'd', 'u'])
                    self.RightToLeft[j] = auxiliary_generate(n, self.RightToLeft[j+1], [self.lat[t][j] for t in range(n)], initial,
                                                         L='u', R='d', U='r', D='l', scan_time=self.scan_time)
            self.RightToLeft[0] = [Node(1.) for i in range(n)]
            tmp = self.RightToLeft
            self.RightToLeft = [[tmp[j][i] for j in range(m)] for i in range(n)]