import time
import os
import sys
import numpy as np
import tensorflow as tf
from .tensor_node import Node
from .spin_state import SpinState


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
    # 每个各点方向固定的吧


class SquareLattice():

    def __create_node(self, i, j):
        legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
        return np.random.rand(2, *[self.D for i in legs])

    def __check_shape(self, input, i, j):
        legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
        output = np.zeros([2, *[self.D for i in legs]])
        output[tuple([slice(i) for i in input.shape])] += input
        return output

    def __init__(self, n, m, D, D_c, scan_time, step_size, markov_chain_length, load_from=None, save_prefix="run", step_print=100):
        self.size = n, m
        self.D = D
        if load_from != None and not os.path.exists(load_from):
            print(f"{load_from} not found")
            load_from = None
        self.load_from = load_from

        if self.load_from == None:
            self.lattice = [[self.__create_node(i, j) for j in range(m)] for i in range(n)]
        else:
            prepare = np.load(load_from)
            print(f'{load_from} loaded')
            self.lattice = [[self.__check_shape(prepare[f'node_{i}_{j}'], i, j) for j in range(m)] for i in range(n)]
            self.prepare = prepare
        # 载入lattice信息

        self.D_c = D_c
        self.scan_time = scan_time

        self.markov_chain_length = markov_chain_length
        self.step_size = step_size

        """
        self.Hamiltonian = np.tensor(
            np.array([1, 0, 0, 0, 0, -1, 2, 0, 0, 2, -1, 0, 0, 0, 0, 1])
            .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])/4.
        self.Identity = np.tensor(
            np.identity(4)
            .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])
        """

        self.step_print = step_print
        # 保存参数

        self.TYPE=tf.float32
        self.spin_model = SpinState(size=self.size, D=self.D, D_c=self.D_c, scan_time=self.scan_time, TYPE=self.TYPE)

        def default_spin():
            return np.array([[1 if (i+j)%2==0 else 0 for j in range(m)] for i in range(n)])
        if self.load_from == None:
            self.spin = default_spin()
        else:
            if f'spin' in prepare:
                self.spin = prepare['spin']
            else:
                self.spin = default_spin()
        # 准备spin state

        """
        self.env_v = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.env_h = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        for i in range(n):
            self.env_h[i][m-1] = np.array(1)
        for j in range(m):
            self.env_v[n-1][j] = np.array(1)
        if self.load_from != None and "env_v" in self.prepare and "env_h" in self.prepare:
            for i in range(n):
                for j in range(m):
                    if i != n-1:
                        self.env_v[i][j][:len(self.prepare["env_v"][i][j])] = self.prepare["env_v"][i][j]
                    if j != m-1:
                        self.env_h[i][j][:len(self.prepare["env_h"][i][j])] = self.prepare["env_h"][i][j]
        # 准备su用的环境
        """

        self.save_prefix = time.strftime(f"{save_prefix}/%Y%m%d%H%M%S", time.gmtime())
        print(f"save_prefix={self.save_prefix}")
        os.makedirs(self.save_prefix, exist_ok=True)
        if self.load_from is not None:
            os.symlink(os.path.realpath(self.load_from), f'{self.save_prefix}/load_from')
        file = open(f'{self.save_prefix}/parameter', 'w')
        file.write(str(sys.argv))
        file.close()
        split_name = os.path.split(self.save_prefix)
        if os.path.exists(f'{split_name[0]}/last'):
            os.remove(f'{split_name[0]}/last')
        os.symlink(split_name[1], f'{split_name[0]}/last')
        # 文件记录

    def save(self, **prepare):
        # prepare 里需要有name
        n, m = self.size
        for i in range(n):
            for j in range(m):
                prepare[f'node_{i}_{j}'] = self.lattice[i][j]
                prepare[f'legs_{i}_{j}'] = ['p', *get_lattice_node_leg(i, j, self.size[0], self.size[1])]
        np.savez_compressed(f'{self.save_prefix}/{prepare["name"]}.npz', **prepare)
        if os.path.exists(f'{self.save_prefix}/bak.npz'):
            os.remove(f'{self.save_prefix}/bak.npz')
        if os.path.exists(f'{self.save_prefix}/last.npz'):
            os.rename(f'{self.save_prefix}/last.npz', f'{self.save_prefix}/bak.npz')
        os.symlink(f'{prepare["name"]}.npz', f'{self.save_prefix}/last.npz')
        # spin state 通过参数传进来, 因为他需要mpi gather

    def grad_descent(self, sess):
        n, m = self.size
        t = 0
        self.sess = sess
        file = open(f'{self.save_prefix}/GM.log', 'w')
        while True:
            energy, grad = self.markov_chain()
            for i in range(n):
                for j in range(m):
                    self.lattice[i][j] -= self.step_size*grad[i][j]
            self.save(energy=energy, name=f'GM.{t}', spin=self.spin)
            file.write(f'{t} {energy}\n')
            file.flush()
            print(t, energy)
            t += 1

    def markov_chain(self):
        n, m = self.size
        sum_E_s = tf.zeros([], dtype=self.TYPE)
        sum_Delta_s = [[np.zeros(self.lattice[i][j].shape) for j in range(m)]for i in range(n)]
        Prod = [[np.zeros(self.lattice[i][j].shape) for j in range(m)]for i in range(n)]
        for markov_step in range(self.markov_chain_length):
            print('markov chain', markov_step, '/', self.markov_chain_length, end='\r')
            feed_dict = {self.spin_model.state:self.spin}
            for i in range(n):
                for j in range(m):
                    feed_dict[self.spin_model.lat[i][j].data] = self.lattice[i][j][self.spin[i][j]]
                    feed_dict[self.spin_model.lat_hop[i][j].data] = self.lattice[i][j][1-self.spin[i][j]]
            E_s, Delta_s = self.sess.run((self.spin_model.energy, self.spin_model.grad), feed_dict=feed_dict)
            sum_E_s += E_s
            for i in range(n):
                for j in range(m):
                    sum_Delta_s[i][j][self.spin[i][j]] += Delta_s[i][j]
                    Prod[i][j][self.spin[i][j]] += E_s * Delta_s[i][j]
            #self.spin = self.spin.markov_chain_hop()

        Grad = [[2.*Prod[i][j]/(self.markov_chain_length) -
                 2.*sum_E_s*sum_Delta_s[i][j]/(self.markov_chain_length)**2 for j in range(m)] for i in range(n)]
        Energy = sum_E_s/(self.markov_chain_length*n*m)
        return Energy, Grad

    def pre_heating(self, step):
        for markov_step in range(step):
            self.spin = self.spin.markov_chain_hop()

    def accurate_energy(self):
        n, m = self.size
        psi = np.tensor(1.)
        for i in range(n):
            for j in range(m):
                psi = psi.tensor_contract(self[i][j], ['r', f'd{j}'], ['l', 'u'], {}, {'d': f'd{j}', 'p': f'p_{i}_{j}'}, restrict_mode=False)
        Hpsi = psi*0
        for i in range(n):
            for j in range(m-1):
                Hpsi += psi.tensor_contract(self.Hamiltonian,
                                            [f'p_{i}_{j}', f'p_{i}_{j+1}'], ['p1', 'p2'], {}, {'P1': f'p_{i}_{j}', 'P2': f'p_{i}_{j+1}'}, restrict_mode=False)
        for i in range(n-1):
            for j in range(m):
                Hpsi += psi.tensor_contract(self.Hamiltonian,
                                            [f'p_{i}_{j}', f'p_{i+1}_{j}'], ['p1', 'p2'], {}, {'P1': f'p_{i}_{j}', 'P2': f'p_{i+1}_{j}'}, restrict_mode=False)
        return psi.tensor_contract(Hpsi, psi.legs, psi.legs)/psi.tensor_contract(psi, psi.legs, psi.legs)/n/m

    def itebd(self, accurate=False):
        n, m = self.size
        self.__pre_itebd_done_restore()  # 载入后第一次前需要还原
        self.Evolution = self.Identity - self.step_size * self.Hamiltonian
        file = open(f'{self.save_prefix}/SU.log', 'w')
        t = 0
        while True:
            print('simple update', t % self.step_print, '/', self.step_print, end='\r')
            self.__itebd_once_h(0)
            self.__itebd_once_h(1)
            self.__itebd_once_v(0)
            self.__itebd_once_v(1)
            self.__itebd_once_v(1)
            self.__itebd_once_v(0)
            self.__itebd_once_h(1)
            self.__itebd_once_h(0)
            if t % self.step_print == 0 and t != 0:
                self.__pre_itebd_done()
                print("\033[K", end='\r')
                if accurate:
                    energy = self.accurate_energy().tolist()
                else:
                    energy = self.markov_chain(cal_grad=False)[0]
                print("\033[K", end='\r')
                file.write(f'{t} {energy}\n')
                print(t, energy)
                file.flush()
                self.save(env_v=self.env_v, env_h=self.env_h, energy=energy, name=f'SU.{t}')
                self.__pre_itebd_done_restore()
            t += 1

    def __itebd_once_h(self, base):
        n, m = self.size
        for i in range(n):
            for j in range(base, m-1, 2):
                # j,j+1
                self[i][j]\
                    .tensor_multiple(self.env_v[i-1][j], 'u', restrict_mode=False)\
                    .tensor_multiple(self.env_h[i][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(self.env_v[i][j], 'd', restrict_mode=False)
                self[i][j+1]\
                    .tensor_multiple(self.env_v[i-1][j+1], 'u', restrict_mode=False)\
                    .tensor_multiple(self.env_h[i][j+1], 'r', restrict_mode=False)\
                    .tensor_multiple(self.env_v[i][j+1], 'd', restrict_mode=False)

                tmp_left, r1 = self[i][j].tensor_qr(
                    ['u', 'l', 'd'], ['r', 'p'], ['r', 'l'], restrict_mode=False)
                tmp_right, r2 = self[i][j+1].tensor_qr(
                    ['u', 'r', 'd'], ['l', 'p'], ['l', 'r'], restrict_mode=False)
                r1.tensor_multiple(self.env_h[i][j], 'r')
                big = np.tensor_contract(r1, r2, ['r'], ['l'], {'p': 'p1'}, {'p': 'p2'})
                big = big.tensor_contract(self.Evolution, ['p1', 'p2'], ['p1', 'p2'])
                big /= np.linalg.norm(big)
                u, s, v = big.tensor_svd(['l', 'P1'], ['r', 'P2'], ['r', 'l'], full_matrices=False)

                thisD = min(self.D, len(s))
                self.env_h[i][j] = s[:thisD]
                self[i][j] = u[:, :, :thisD]\
                    .tensor_contract(tmp_left, ['l'], ['r'], {'P1': 'p'})
                self[i][j+1] = v[:thisD, :, :]\
                    .tensor_contract(tmp_right, ['r'], ['l'], {'P2': 'p'})
                legs = self[i][j+1].legs
                self[i][j+1] = self[i][j+1].tensor_transpose([*legs[1], *legs[0], *legs[2:]])

                self[i][j]\
                    .tensor_multiple(1/self.env_v[i-1][j], 'u', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i][j], 'd', restrict_mode=False)
                self[i][j+1]\
                    .tensor_multiple(1/self.env_v[i-1][j+1], 'u', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j+1], 'r', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i][j+1], 'd', restrict_mode=False)

    def __itebd_once_v(self, base):
        n, m = self.size
        for i in range(base, n-1, 2):
            for j in range(m):
                # i,i+1
                self[i][j]\
                    .tensor_multiple(self.env_h[i][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(self.env_v[i-1][j], 'u', restrict_mode=False)\
                    .tensor_multiple(self.env_h[i][j], 'r', restrict_mode=False)
                self[i+1][j]\
                    .tensor_multiple(self.env_h[i+1][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(self.env_v[i+1][j], 'd', restrict_mode=False)\
                    .tensor_multiple(self.env_h[i+1][j], 'r', restrict_mode=False)

                tmp_up, r1 = self[i][j].tensor_qr(
                    ['l', 'u', 'r'], ['d', 'p'], ['d', 'u'], restrict_mode=False)
                tmp_down, r2 = self[i+1][j].tensor_qr(
                    ['l', 'd', 'r'], ['u', 'p'], ['u', 'd'], restrict_mode=False)
                r1.tensor_multiple(self.env_v[i][j], 'd')
                big = np.tensor_contract(r1, r2, ['d'], ['u'], {'p': 'p1'}, {'p': 'p2'})
                big = big.tensor_contract(self.Evolution, ['p1', 'p2'], ['p1', 'p2'])
                big /= np.linalg.norm(big)
                u, s, v = big.tensor_svd(['u', 'P1'], ['d', 'P2'], ['d', 'u'], full_matrices=False)
                thisD = min(self.D, len(s))
                self.env_v[i][j] = s[:thisD]
                self[i][j] = u[:, :, :thisD]\
                    .tensor_contract(tmp_up, ['u'], ['d'], {'P1': 'p'})
                self[i+1][j] = v[:thisD, :, :]\
                    .tensor_contract(tmp_down, ['d'], ['u'], {'P2': 'p'})
                legs = self[i+1][j].legs
                self[i+1][j] = self[i+1][j].tensor_transpose([*legs[1], *legs[0], *legs[2:]])

                self[i][j]\
                    .tensor_multiple(1/self.env_h[i][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i-1][j], 'u', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j], 'r', restrict_mode=False)
                self[i+1][j]\
                    .tensor_multiple(1/self.env_h[i+1][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i+1][j], 'd', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i+1][j], 'r', restrict_mode=False)

    def __pre_itebd_done(self):
        n, m = self.size
        for i in range(n):
            for j in range(m):
                self[i][j]\
                    .tensor_multiple(self.env_v[i][j], 'd', restrict_mode=False)\
                    .tensor_multiple(self.env_h[i][j], 'r', restrict_mode=False)\


    def __pre_itebd_done_restore(self):
        n, m = self.size
        for i in range(n):
            for j in range(m):
                self[i][j]\
                    .tensor_multiple(1/self.env_v[i][j], 'd', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j], 'r', restrict_mode=False)\

