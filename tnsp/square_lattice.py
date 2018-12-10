import time
import os
import sys
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from .tensor_node import Node
from .spin_state import SpinState, get_lattice_node_leg


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
#print("mpi info:", mpi_rank, "/", mpi_size)

class SquareLattice():

    def __create_node(self, i, j):
        legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
        return np.random.rand(2, *[self.D for i in legs])

    def __check_shape(self, input, i, j):
        legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
        if input.shape != (2, *[self.D for i in legs]):
            print('Warning: shape of node incorrect at',i,j,legs,input.shape,(2, *[self.D for i in legs]))
        #output[tuple([slice(i) for i in input.shape])] += input
        output = input
        return output

    def __init__(self, size, D, D_c, scan_time, step_size, markov_chain_length, load_from=None, save_prefix="run"):
        n, m = self.size = size
        self.D = D
        if load_from != None and not os.path.exists(load_from):
            print(f"{load_from} not found")
            load_from = None
        self.load_from = load_from

        if self.load_from == None:
            self.lattice = [[self.__create_node(i, j) for j in range(m)] for i in range(n)]
        else:
            self.prepare = np.load(load_from)
            if mpi_rank==0:
                print(f'{load_from} loaded')
                self.lattice = [[self.__check_shape(self.prepare[f'node_{i}_{j}'], i, j) for j in range(m)] for i in range(n)]
            else:
                self.lattice = None
            self.lattice = mpi_comm.bcast(self.lattice, root=0)
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

        # 保存参数

        self.TYPE = tf.float32
        self.spin_model = SpinState(size=self.size, D=self.D, D_c=self.D_c, scan_time=self.scan_time, TYPE=self.TYPE)

        def default_spin():
            return np.array([[0 if (i+j) % 2 == 0 else 1 for j in range(m)] for i in range(n)])
        if self.load_from == None:
            self.spin = default_spin()
        else:
            if f'spin_{mpi_rank}' in self.prepare:
                self.spin = self.prepare[f'spin_{mpi_rank}']
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

        if mpi_rank == 0:
            self.save_prefix = time.strftime(f"{save_prefix}/%Y%m%d%H%M%S", time.gmtime())
            print(f"save_prefix={self.save_prefix}")
            os.makedirs(self.save_prefix, exist_ok=True)
            if self.load_from is not None:
                os.symlink(os.path.relpath(os.path.realpath(self.load_from), self.save_prefix),
                           f'{self.save_prefix}/load_from')
            file = open(f'{self.save_prefix}/parameter', 'w')
            file.write(f'mpirun -n {mpi_size} python {" ".join(sys.argv)}')
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
                #prepare[f'legs_{i}_{j}'] = ['p', *get_lattice_node_leg(i, j, self.size[0], self.size[1])]
        np.savez_compressed(f'{self.save_prefix}/{prepare["name"]}.npz', **prepare)
        if os.path.exists(f'{self.save_prefix}/bak.npz'):
            os.remove(f'{self.save_prefix}/bak.npz')
        if os.path.exists(f'{self.save_prefix}/last.npz'):
            os.rename(f'{self.save_prefix}/last.npz', f'{self.save_prefix}/bak.npz')
        os.symlink(f'{prepare["name"]}.npz', f'{self.save_prefix}/last.npz')

    def grad_descent(self, sess):
        n, m = self.size
        t = 0
        self.sess = sess
        if mpi_rank == 0:
            file = open(f'{self.save_prefix}/GM.log', 'w')
        while True:
            energy, grad = self.markov_chain()
            gather_spin = mpi_comm.gather(np.array(self.spin), root=0)
            if mpi_rank == 0:
                total_l = np.array(0.)
                for i in range(n):
                    for j in range(n):
                        total_l += np.sum(grad[i][j] * grad[i][j])
                total_l = np.sqrt(total_l/(n*m))
                for i in range(n):
                    for j in range(m):
                        self.lattice[i][j] -= self.step_size*grad[i][j]/total_l

                spin_dict = {}
                for i, s in enumerate(gather_spin):
                    spin_dict[f'spin_{i}'] = s
                self.save(energy=energy, name=f'GM.{t}', **spin_dict)
                file.write(f'{t} {energy}\n')
                file.flush()
                print(t, energy)
            for i in range(n):
                for j in range(m):
                    self.lattice[i][j] = mpi_comm.bcast(self.lattice[i][j], root=0)
            t += 1
            #if t==20:
            #    break

    def markov_chain(self):
        n, m = self.size
        real_step = np.array(0.)
        sum_E_s = np.array(0.)
        sum_Delta_s = [[np.zeros(self.lattice[i][j].shape) for j in range(m)]for i in range(n)]
        Prod = [[np.zeros(self.lattice[i][j].shape) for j in range(m)]for i in range(n)]
        #E_s_list = []
        #E_s_file = open(f'{self.save_prefix}/Es.log','a')
        for markov_step in range(self.markov_chain_length):
            print('markov chain', markov_step, '/', self.markov_chain_length, end='\r')
            lat = [[self.lattice[i][j][self.spin[i][j]] for j in range(m)] for i in range(n)]
            lat_hop = [[self.lattice[i][j][1-self.spin[i][j]] for j in range(m)] for i in range(n)]
            res = self.spin_model(self.sess, self.spin, lat, lat_hop)
            real_step += res["step"]
            #for i in range(int(res['step'])):
            #    E_s_list.append(res['energy'])
            sum_E_s += res["energy"]*res["step"]
            for i in range(n):
                for j in range(m):
                    sum_Delta_s[i][j][self.spin[i][j]] += res["grad"][i][j]*res["step"]
                    Prod[i][j][self.spin[i][j]] += res["grad"][i][j]*res["step"]*res["energy"]
            next_index = res["next"]
            hop_i = next_index // m
            hop_j = next_index % m
            self.spin[hop_i][hop_j] = 1 - self.spin[hop_i][hop_j]
            """
            if res["next"] < n*(m-1):
                next_index = res["next"]
                hop_i = next_index // (m-1)
                hop_j = next_index % (m-1)
                self.spin[hop_i][hop_j] = 1 - self.spin[hop_i][hop_j]
                self.spin[hop_i][hop_j+1] = 1 - self.spin[hop_i][hop_j+1]
            else:
                next_index = res["next"] - n*(m-1)
                hop_j = next_index // (n-1)
                hop_i = next_index % (n-1)
                self.spin[hop_i][hop_j] = 1 - self.spin[hop_i][hop_j]
                self.spin[hop_i+1][hop_j] = 1 - self.spin[hop_i+1][hop_j]
            """
        print('\033[K', end='\r')
        #for i in E_s_list:
        #    E_s_file.write(f'{i} ')
        #E_s_file.write('\n')
        #E_s_file.close()

        real_step = mpi_comm.reduce(real_step, root=0)
        sum_E_s = mpi_comm.reduce(sum_E_s, root=0)
        for i in range(n):
            for j in range(m):
                Prod[i][j] = mpi_comm.reduce(Prod[i][j], root=0)
                sum_Delta_s[i][j] = mpi_comm.reduce(sum_Delta_s[i][j], root=0)
        if mpi_rank == 0:
            print("%5.2f"%(real_step/(mpi_size*self.markov_chain_length)), end=' ')
            Grad = [[(2.*Prod[i][j]/(real_step) -
                      2.*sum_E_s*sum_Delta_s[i][j]/(real_step*real_step))/(n*m) for j in range(m)] for i in range(n)]
            Energy = sum_E_s/(real_step*n*m)
            return Energy, Grad
        else:
            return None, None
