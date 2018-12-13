import time
import os
import sys
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from .tensor_node import Node
from .spin_state import SpinState, config

get_lattice_node_leg = config.get_lattice_node_leg
default_spin = config.default_spin
seed = config.seed

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
#print("mpi info:", mpi_rank, "/", mpi_size)

class SquareLattice():

    def __create_node(self, i, j):
        legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
        return np.random.rand(2, *[self.D for i in legs]) * 2 - 1 # 不知为何，-1,1之间随机数的话，能量在0附近，0,1间会大于0

    def __check_shape(self, input, i, j):
        legs = get_lattice_node_leg(i, j, self.size[0], self.size[1])
        if input.shape != (2, *[self.D for i in legs]):
            print('Warning: shape of node incorrect at',i,j,legs,input.shape,(2, *[self.D for i in legs]))
        output = input
        return output

    def __root_print(self, *args, **kw):
        if mpi_rank==0:
            print(*args, **kw)

    def __init__(self, size, D, D_c, scan_time, step_size, markov_chain_length, load_from=None, save_prefix="run"):
        # 保存各类参数
        np.random.seed(seed+mpi_rank)
        tf.set_random_seed(seed+mpi_rank+mpi_size)
        n, m = self.size = size
        self.D = D
        if load_from == None:
            self.load_from = None
        else:
            if not os.path.exists(load_from):
                self.load_from = None
                self.__root_print(f"{load_from} not found")
            else:
                self.load_from = load_from
                if mpi_rank==0:
                    self.prepare = np.load(load_from)
                self.__root_print(f"{load_from} loaded")

        # 载入lattice信息
        if mpi_rank==0:
            if self.load_from == None:
                self.lattice = [[self.__create_node(i, j) for j in range(m)] for i in range(n)]
            else:
                self.lattice = [[self.__check_shape(self.prepare[f'node_{i}_{j}'], i, j) for j in range(m)] for i in range(n)]
        else:
            self.lattice = None
        self.lattice = mpi_comm.bcast(self.lattice, root=0)

        self.D_c = D_c
        self.scan_time = scan_time

        self.markov_chain_length = markov_chain_length
        self.step_size = step_size

        self.TYPE = tf.float64
        self.spin_model = SpinState(size=self.size, D=self.D, D_c=self.D_c, scan_time=self.scan_time, TYPE=self.TYPE)

        # 生成或者载入spin构形
        if self.load_from == None:
            self.spin = default_spin(n, m)
        else:
            if mpi_rank == 0:
                spin_to_scatter = [self.prepare[f'spin_{i}'] if f'spin_{i}' in self.prepare else default_spin(n, m) for i in range(mpi_size)]
            else:
                spin_to_scatter = None
            self.spin = mpi_comm.scatter(spin_to_scatter, root=0)

        # 文件记录
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

    def save(self, **prepare):
        # prepare 里需要有name
        n, m = self.size
        for i in range(n):
            for j in range(m):
                prepare[f'node_{i}_{j}'] = self.lattice[i][j]
        np.savez_compressed(f'{self.save_prefix}/{prepare["name"]}.npz', **prepare)
        if os.path.exists(f'{self.save_prefix}/bak.npz'):
            os.remove(f'{self.save_prefix}/bak.npz')
        if os.path.exists(f'{self.save_prefix}/last.npz'):
            os.rename(f'{self.save_prefix}/last.npz', f'{self.save_prefix}/bak.npz')
        os.symlink(f'{prepare["name"]}.npz', f'{self.save_prefix}/last.npz')

    def grad_descent(self, sess):
        # 载入格子大小，建立session，文件输出这一些杂事
        n, m = self.size
        t = 0
        self.sess = sess
        if mpi_rank == 0:
            file = open(f'{self.save_prefix}/GM.log', 'w')
        while True:
            # 每个核各跑一根markov chain
            energy, grad = self.markov_chain()

            # 统计构形信息，用来保存到文件
            gather_spin = mpi_comm.gather(np.array(self.spin), root=0)

            if mpi_rank == 0:
                # 梯度下降
                grad_norm = np.array(0.)
                for i in range(n):
                    for j in range(m):
                        tmp = np.max(np.abs(grad[i][j]))
                        if tmp > grad_norm:
                            grad_norm = tmp
                for i in range(n):
                    for j in range(m):
                        self.lattice[i][j] -= self.step_size*grad[i][j]/grad_norm

                """
                pool = []
                for i in range(n):
                    for j in range(m):
                        pool = np.append(pool, grad[i][j].reshape(-1))
                med = np.median(np.abs(pool))
                for i in range(n):
                    for j in range(m):
                        delta = (grad[i][j] > med) * np.random.rand(*grad[i][j].shape) -\
                            (grad[i][j] < -med) * np.random.rand(*grad[i][j].shape)
                        self.lattice[i][j] -= self.step_size*delta
                """

                # 文件保存，包括自旋构形
                spin_dict = {}
                for i, s in enumerate(gather_spin):
                    spin_dict[f'spin_{i}'] = s
                self.save(energy=energy, name=f'GM.{t}', **spin_dict)
                file.write(f'{t} {energy}\n')
                file.flush()
                print(t, energy)

            # 将波函数广播回去
            for i in range(n):
                for j in range(m):
                    self.lattice[i][j] = mpi_comm.bcast(self.lattice[i][j], root=0)
            t += 1

    def markov_chain(self):
        # 载入格子大小
        n, m = self.size

        # 四个求和项目，step，能量，梯度，能量梯度积
        real_step = np.array(0.)
        sum_E_s = np.array(0.)
        sum_Delta_s = [[np.zeros(self.lattice[i][j].shape) for j in range(m)]for i in range(n)]
        Prod = [[np.zeros(self.lattice[i][j].shape) for j in range(m)]for i in range(n)]

        for markov_step in range(self.markov_chain_length):
            print('markov chain', markov_step, '/', self.markov_chain_length, end='\r')

            # 准备送给TF的数据
            lat = [[self.lattice[i][j][self.spin[i][j]] for j in range(m)] for i in range(n)]
            lat_hop = [[self.lattice[i][j][1-self.spin[i][j]] for j in range(m)] for i in range(n)]

            # 调用TF
            res = self.spin_model(self.sess, self.spin, lat, lat_hop, np.random.rand(2))

            # 累加四个变量
            real_step += res["step"]
            sum_E_s += res["energy"]*res["step"]
            for i in range(n):
                for j in range(m):
                    sum_Delta_s[i][j][self.spin[i][j]] += res["grad"][i][j]*res["step"]
                    Prod[i][j][self.spin[i][j]] += res["grad"][i][j]*res["step"]*res["energy"]

            # hop构形
            next_index = res["next"]
            hop_i = next_index // m
            hop_j = next_index % m
            self.spin[hop_i][hop_j] = 1 - self.spin[hop_i][hop_j]

        print('\033[K', end='\r')

        # 通过MPI对折四个变量求和
        real_step = mpi_comm.reduce(real_step, root=0)
        sum_E_s = mpi_comm.reduce(sum_E_s, root=0)
        for i in range(n):
            for j in range(m):
                Prod[i][j] = mpi_comm.reduce(Prod[i][j], root=0)
                sum_Delta_s[i][j] = mpi_comm.reduce(sum_Delta_s[i][j], root=0)

        # 最后一点处理并返回
        if mpi_rank == 0:
            print("%5.2f"%(real_step/(mpi_size*self.markov_chain_length)), end=' ')
            Grad = [[(2.*Prod[i][j]/(real_step) -
                      2.*sum_E_s*sum_Delta_s[i][j]/(real_step*real_step))/(n*m) for j in range(m)] for i in range(n)]
            Energy = sum_E_s/(real_step*n*m)
            return Energy, Grad
        else:
            return None, None
