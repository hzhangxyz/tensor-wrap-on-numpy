import time
import os
import numpy_wrap as np
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
print("mpi info:", mpi_rank, "/", mpi_size)


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
        res[0], QR_R = res[0].tensor_qr([D], [R], [R, L])
        l = [None for i in range(length)]
        l[0] = former[0]\
            .tensor_contract(current[0], [D], [U], {R: R1}, {R: R2})\
            .tensor_contract(res[0], [D], [D], {}, {R: R3})

        for j in range(1, length-1):
            if t == 0:
                res[j] = np.tensor_contract(res[j], QR_R, [L], [R])
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


class SquareLattice(list):

    def __create_node(self, i, j):
        legs = 'lrud'
        if i == 0:
            legs = legs.replace('u', '')
        if i == self.size[0]-1:
            legs = legs.replace('d', '')
        if j == 0:
            legs = legs.replace('l', '')
        if j == self.size[1]-1:
            legs = legs.replace('r', '')
        return np.tensor(np.random.rand(2, *[self.D for i in legs]), legs=['p', *legs])

    def __new__(self, n, m, D, D_c, scan_time, *args, **kw):
        obj = super().__new__(SquareLattice)
        obj.size = n, m
        obj.D = D

        return obj

    def __check_shape(self, input, i, j):
        legs = 'lrud'
        if i == 0:
            legs = legs.replace('u', '')
        if i == self.size[0]-1:
            legs = legs.replace('d', '')
        if j == 0:
            legs = legs.replace('l', '')
        if j == self.size[1]-1:
            legs = legs.replace('r', '')
        assert input.tensor_transpose(['p', *legs]).shape == (2, *[self.D for i in legs]), 'input save data not match shape'
        return input

    def __init__(self, n, m, D, D_c, scan_time, step_size, markov_chain_length, load_from=None, save_prefix="run"):
        if load_from == None or not os.path.exists(load_from):
            if mpi_rank == 0:
                tmp = [[self.__create_node(i, j) for j in range(m)] for i in range(n)]
            else:
                tmp = None
            super().__init__(mpi_comm.bcast(tmp, root=0))  # random init
            self.D_c = D_c
            self.scan_time = scan_time
            self.spin = SpinState(self, spin_state=None)

            self.load_from = None
        else:
            prepare = np.load(load_from)
            super().__init__([[self.__check_shape(np.tensor(prepare[f'node_{i}_{j}'], legs=prepare[f'legs_{i}_{j}']), i, j)
                for j in range(m)] for i in range(n)])  # random init
            self.D_c = D_c
            self.scan_time = scan_time
            if f'spin' in prepare and len(prepare['spin']) > mpi_rank:
                self.spin = SpinState(self, spin_state=prepare['spin'][mpi_rank])
            else:
                self.spin = SpinState(self, spin_state=None)

            self.load_from = load_from

        if mpi_rank == 0:
            self.save_prefix = time.strftime(f"{save_prefix}/%Y%m%d%H%M%S", time.gmtime())
            os.makedirs(self.save_prefix, exist_ok=True)
            if self.load_from is not None:
                os.symlink(os.path.realpath(self.load_from), f'{self.save_prefix}/load_from')
            file = open(f'{self.save_prefix}/parameter', 'w')
            file.write(f'n={n}\nm={m}\nD={D}\nD_c={D_c}\nscan_time={scan_time}\nstep_size={step_size}\nmarkov_chain_length={markov_chain_length}\n')
            file.close()
            split_name = os.path.split(self.save_prefix)
            if os.path.exists(f'{split_name[0]}/last'):
                os.remove(f'{split_name[0]}/last')
            os.symlink(split_name[1], f'{split_name[0]}/last')

        self.markov_chain_length = markov_chain_length
        self.step_size = step_size

        self.env_v = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.env_h = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        for i in range(n):
            self.env_h[i][m-1] = np.array(1)
        for j in range(m):
            self.env_v[n-1][j] = np.array(1)
        self.Hamiltonian = np.tensor(
            np.array([1, 0, 0, 0, 0, -1, 2, 0, 0, 2, -1, 0, 0, 0, 0, 1])
            .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])/4.
        self.Identity = np.tensor(
            np.identity(4)
            .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])

    def save(self, **prepare):
        n, m = self.size
        for i in range(n):
            for j in range(m):
                prepare[f'node_{i}_{j}'] = self[i][j]
                prepare[f'legs_{i}_{j}'] = self[i][j].legs
        if "t" not in prepare:
            prepare["t"] = "SU"
        np.savez_compressed(f'{self.save_prefix}/{prepare["t"]}.npz', **prepare)
        if os.path.exists(f'{self.save_prefix}/bak.npz'):
            os.remove(f'{self.save_prefix}/bak.npz')
        if os.path.exists(f'{self.save_prefix}/last.npz'):
            os.rename(f'{self.save_prefix}/last.npz', f'{self.save_prefix}/bak.npz')
        os.symlink(f'{prepare["t"]}.npz', f'{self.save_prefix}/last.npz')

    def grad_descent(self):
        n, m = self.size
        t = 0
        if mpi_rank == 0:
            file = open(f'{self.save_prefix}/log', 'w')
        while True:
            energy, grad = self.markov_chain()
            gather_spin = mpi_comm.gather(np.array(self.spin), root=0)
            if mpi_rank == 0:  # mpi
                spin = np.array(gather_spin, dtype=int)
                for i in range(n):
                    for j in range(m):
                        self[i][j] -= self.step_size*grad[i][j]
                self.save(energy=energy, t=t, spin=spin)
                file.write(f'{t} {energy}\n')
                file.flush()
                print(t, energy)
            for i in range(n):
                for j in range(m):
                    tmp = self[i][j]
                    self[i][j] = mpi_comm.bcast(self[i][j], root=0)
            t += 1

    def markov_chain(self):
        n, m = self.size
        self.spin.fresh()
        sum_E_s = np.tensor(0.)
        sum_Delta_s = [[np.tensor(np.zeros(self[i][j].shape), self[i][j].legs) for j in range(m)]for i in range(n)]
        Prod = [[np.tensor(np.zeros(self[i][j].shape), self[i][j].legs) for j in range(m)]for i in range(n)]
        for markov_step in range(self.markov_chain_length):
            E_s, Delta_s = self.spin.cal_E_s_and_Delta_s()
            sum_E_s += E_s
            for i in range(n):
                for j in range(m):
                    sum_Delta_s[i][j][self.spin[i][j]] += Delta_s[i][j]
                    Prod[i][j][self.spin[i][j]] += E_s * Delta_s[i][j]
            self.spin = self.spin.markov_chain_hop()

        sum_E_s = mpi_comm.reduce(sum_E_s, root=0)
        for i in range(n):
            for j in range(m):
                sum_Delta_s[i][j] = mpi_comm.reduce(sum_Delta_s[i][j], root=0)
                Prod[i][j] = mpi_comm.reduce(Prod[i][j], root=0)
        if mpi_rank == 0:
            Grad = [[2.*Prod[i][j]/(self.markov_chain_length*mpi_size) -
                     2.*sum_E_s*sum_Delta_s[i][j]/(self.markov_chain_length*mpi_size)**2 for j in range(m)] for i in range(n)]
            Energy = sum_E_s/(self.markov_chain_length*mpi_size*n*m)
            return Energy, Grad
        else:
            return None, None

    def pre_heating(self, step):
        for markov_step in range(step):
            self.spin = self.spin.markov_chain_hop()

    def accurate_energy(self):
        n, m = self.size
        psi = np.tensor(1.)
        for i in range(n):
            for j in range(m):
                psi = psi.tensor_contract(self[i][j],['r',f'd{j}'],['l','u'],{},{'d':f'd{j}','p':f'p_{i}_{j}'},restrict_mode=False)
        Hpsi = psi*0
        for i in range(n):
            for j in range(m-1):
                Hpsi += psi.tensor_contract(self.Hamiltonian,
                        [f'p_{i}_{j}',f'p_{i}_{j+1}'],['p1','p2'],{},{'P1':f'p_{i}_{j}','P2':f'p_{i}_{j+1}'},restrict_mode=False)
        for i in range(n-1):
            for j in range(m):
                Hpsi += psi.tensor_contract(self.Hamiltonian,
                        [f'p_{i}_{j}',f'p_{i+1}_{j}'],['p1','p2'],{},{'P1':f'p_{i}_{j}','P2':f'p_{i+1}_{j}'},restrict_mode=False)
        return psi.tensor_contract(Hpsi,psi.legs,psi.legs)/psi.tensor_contract(psi,psi.legs,psi.legs)/n/m

    def itebd(self, delta, accurate=False):
        n, m = self.size
        if mpi_rank != 0:
            return
        self.Evolution = self.Identity - delta * self.Hamiltonian
        file = open(f'{self.save_prefix}/SU', 'w')
        t = 0
        while True:
            self.__itebd_once_h(0)
            self.__itebd_once_h(1)
            self.__itebd_once_v(0)
            self.__itebd_once_v(1)
            if t%1 == 0:
                self.__pre_itebd_done()
                print('itebd',t,end=' ')
                if accurate:
                    energy_save = self.accurate_energy().tolist(), #self.markov_chain()[0].tolist()
                    file.write(f'{t} {energy_save[0]}\n')
                    print(*energy_save)
                else:
                    energy_save = self.markov_chain()[0]
                    file.write(f'{t} {energy_save}\n')
                    print(energy_save)
                file.flush()
                self.__pre_itebd_done_restore()
            t += 1
        self.__itebd_done()
        if accurate:
            energy_save = self.accurate_energy()
        else:
            energy_save = self.markov_chain()[0]

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
                u, s, v = big.tensor_svd(['l', 'P1'], ['r', 'P2'], ['r', 'l'])

                thisD = min(self.D,len(s))
                self.env_h[i][j] = s[:thisD]
                self[i][j] = u[:, :, :thisD]\
                    .tensor_contract(tmp_left, ['l'], ['r'], {'P1': 'p'})
                self[i][j+1] = v[:thisD, :, :]\
                    .tensor_contract(tmp_right, ['r'], ['l'], {'P2': 'p'})
                legs = self[i][j+1].legs
                self[i][j+1] = self[i][j+1].tensor_transpose([*legs[1],*legs[0],*legs[2:]])

                self[i][j]\
                    .tensor_multiple(1/self.env_v[i-1][j], 'u', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i][j], 'd', restrict_mode=False)
                self[i][j+1]\
                    .tensor_multiple(1/self.env_v[i-1][j+1], 'u', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j+1], 'r', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i][j+1], 'd', restrict_mode=False)

    def __itebd_once_v(self,base):
        n, m = self.size
        for i in range(base,n-1,2):
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
                u, s, v = big.tensor_svd(['u', 'P1'], ['d', 'P2'], ['d', 'u'])
                thisD = min(self.D,len(s))
                self.env_h[i][j] = s[:thisD]
                self[i][j] = u[:, :, :thisD]\
                    .tensor_contract(tmp_up, ['u'], ['d'], {'P1': 'p'})
                self[i+1][j] = v[:thisD, :, :]\
                    .tensor_contract(tmp_down, ['d'], ['u'], {'P2': 'p'})
                legs = self[i+1][j].legs
                self[i+1][j] = self[i+1][j].tensor_transpose([*legs[1],*legs[0],*legs[2:]])

                self[i][j]\
                    .tensor_multiple(1/self.env_h[i][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i-1][j], 'u', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i][j], 'r', restrict_mode=False)
                self[i+1][j]\
                    .tensor_multiple(1/self.env_h[i+1][j-1], 'l', restrict_mode=False)\
                    .tensor_multiple(1/self.env_v[i+1][j], 'd', restrict_mode=False)\
                    .tensor_multiple(1/self.env_h[i+1][j], 'r', restrict_mode=False)


    def __itebd_done(self):
        n, m = self.size
        self.__pre_itebd_done()
        self.env_v = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.env_h = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.save()

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

class SpinState(list):

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

    def __new__(cls, lattice, *args, **kw):
        obj = super().__new__(SpinState)
        obj.size = lattice.size
        obj.D = lattice.D
        return obj

    def fresh(self):
        # 在spin或者lattice变化时call 之
        self.D_c = self.lattice.D_c
        self.scan_time = self.lattice.scan_time

        self.lat = [[self.lattice[i][j][self[i][j]] for j in range(self.size[1])] for i in range(self.size[0])]
        self.flag_up_to_down = False
        self.flag_down_to_up = False
        self.flag_left_to_right = False
        self.flag_right_to_left = False
        self.w_s = None

        self.res = None

    def __init__(self, lattice, spin_state=None):
        if spin_state is not None:
            super().__init__([[int(spin_state[i][j]) for j in range(self.size[1])] for i in range(self.size[0])])
        else:
            super().__init__([[(i+j) % 2 for j in range(self.size[1])] for i in range(self.size[0])])  # !!
        self.lattice = lattice
        self.fresh()

    def cal_w_s(self):
        if self.w_s is not None:
            return self.w_s
        n, m = self.size

        self.__auxiliary_up_to_down()
        self.w_s = np.tensor(1.)
        for j in range(0, m):
            self.w_s = self.w_s\
                .tensor_contract(self.UpToDown[n-2][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                .tensor_contract(self.lat[n-1][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)
        assert self.w_s != 0., "w_s == 0"
        return self.w_s

    def test_w_s(self):
        n, m = self.size
        self.__auxiliary()
        res = []

        self.w_s = np.tensor(1.)
        for j in range(0, m):
            self.w_s = self.w_s\
                .tensor_contract(self.UpToDown[n-2][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                .tensor_contract(self.lat[n-1][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)
        res.append(self.w_s)

        self.w_s = np.tensor(1.)
        for j in range(0, m):
            self.w_s = self.w_s\
                .tensor_contract(self.DownToUp[1][j], ['r2'], ['l'], {}, {'r': 'r2'}, restrict_mode=False)\
                .tensor_contract(self.lat[0][j], ['r1', 'u'], ['l', 'd'], {}, {'r': 'r1'}, restrict_mode=False)
        res.append(self.w_s)

        self.w_s = np.tensor(1.)
        for i in range(0, n):
            self.w_s = self.w_s\
                .tensor_contract(self.LeftToRight[i][m-2], ['d1'], ['u'], {}, {'d': 'd1'}, restrict_mode=False)\
                .tensor_contract(self.lat[i][m-1], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)
        res.append(self.w_s)

        self.w_s = np.tensor(1.)
        for i in range(0, n):
            self.w_s = self.w_s\
                .tensor_contract(self.RightToLeft[i][1], ['d2'], ['u'], {}, {'d': 'd2'}, restrict_mode=False)\
                .tensor_contract(self.lat[i][0], ['d1', 'l'], ['u', 'r'], {}, {'d': 'd1'}, restrict_mode=False)
        res.append(self.w_s)

        return res

    def cal_E_s_and_Delta_s(self):
        """
        E_s=\sum_{s'} W(s')/W(s) H_{ss'}
        第一部分:对角
        第二部分:交换
        """
        if self.res is not None:
            return self.res
        n, m = self.size
        self.__auxiliary()
        E_s_diag = 0.
        for i in range(n):
            for j in range(m):
                if j != m-1:
                    E_s_diag += 1 if self[i][j] == self[i][j+1] else -1  # 哈密顿量
                if i != n-1:
                    E_s_diag += 1 if self[i][j] == self[i+1][j] else -1
        E_s_non_diag = 0.  # 为相邻两个交换后的w(s)之和
        Delta_s = [[None for j in range(m)] for i in range(n)]  # 下面每个点记录一下
        # 横向j j+1
        for i in range(n):
            l = [None for j in range(m)]
            l[-1] = np.tensor(1.)
            r = [None for j in range(m)]
            r[0] = np.tensor(1.)

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
            for j in range(m):
                Delta_s[i][j] = np.tensor_contract(
                    np.tensor_contract(l[(j-1) % m], self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {'r2': 'l'}, {'r': 'r1', 'd': 'u'}, restrict_mode=False),
                    np.tensor_contract(r[(j+1) % m], self.DownToUp[(i+1) % n][j], ['l3'], ['r'], {'l2': 'r'}, {'l': 'l3', 'u': 'd'}, restrict_mode=False),
                    ['r1', 'r3'], ['l1', 'l3'], restrict_mode=False) / self.cal_w_s()
            # 计算Es
            for j in range(m-1):
                if self[i][j] != self[i][j+1]:
                    E_s_non_diag += l[(j-1) % m]\
                        .tensor_contract(self.UpToDown[(i-1) % n][j], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                        .tensor_contract(self.lattice[i][j][1-self[i][j]], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)\
                        .tensor_contract(self.DownToUp[(i+1) % n][j], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)\
                        .tensor_contract(self.UpToDown[(i-1) % n][(j+1) % m], ['r1'], ['l'], {}, {'r': 'r1'}, restrict_mode=False)\
                        .tensor_contract(self.lattice[i][(j+1) % m][1-self[i][(j+1) % m]], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'}, restrict_mode=False)\
                        .tensor_contract(self.DownToUp[(i+1) % n][(j+1) % m], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'}, restrict_mode=False)\
                        .tensor_contract(r[(j+2) % m], ['r1', 'r2', 'r3'], ['l1', 'l2', 'l3'], restrict_mode=False) * 2 / self.cal_w_s()  # 哈密顿量
        # 纵向i i+1
        for j in range(m):
            u = [None for i in range(n)]
            u[-1] = np.tensor(1.)
            d = [None for i in range(n)]
            d[0] = np.tensor(1.)

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
            for i in range(n-1):
                if self[i][j] != self[i+1][j]:
                    E_s_non_diag += u[(i-1) % n]\
                        .tensor_contract(self.LeftToRight[i][(j-1) % m], ['d1'], ['u'], {}, {'d': 'd1'}, restrict_mode=False)\
                        .tensor_contract(self.lattice[i][j][1-self[i][j]], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)\
                        .tensor_contract(self.RightToLeft[i][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)\
                        .tensor_contract(self.LeftToRight[(i+1) % n][(j-1) % m], ['d1'], 'u', {}, {'d': 'd1'}, restrict_mode=False)\
                        .tensor_contract(self.lattice[(i+1) % n][j][1-self[(i+1) % n][j]], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'}, restrict_mode=False)\
                        .tensor_contract(self.RightToLeft[(i+1) % n][(j+1) % m], ['d3', 'r'], ['u', 'l'], {}, {'d': 'd3'}, restrict_mode=False)\
                        .tensor_contract(d[(i+2) % n], ['d1', 'd2', 'd3'], ['u1', 'u2', 'u3'], restrict_mode=False) * 2 / self.cal_w_s()  # 哈密顿量

        E_s = E_s_diag + E_s_non_diag
        self.res = 0.25*E_s, Delta_s
        return self.res

    def __auxiliary(self):
        self.__auxiliary_up_to_down()
        self.__auxiliary_down_to_up()
        self.__auxiliary_left_to_right()
        self.__auxiliary_right_to_left()

    def __auxiliary_up_to_down(self):
        if self.flag_up_to_down:
            return
        n, m = self.size
        self.flag_up_to_down = True
        self.UpToDown = [None for i in range(n)]
        self.UpToDown[0] = [self.lat[0][j] for j in range(m)]
        for i in range(1, n-1):
            initial = [None for j in range(m)]
            for j in range(m):
                if j == 0:
                    initial[j] = np.tensor(np.random.rand(self.D, self.D_c), legs=['d', 'r'])
                elif j == m-1:
                    initial[j] = np.tensor(np.random.rand(self.D, self.D_c), legs=['d', 'l'])
                else:
                    initial[j] = np.tensor(np.random.rand(self.D, self.D_c, self.D_c), legs=['d', 'l', 'r'])
            self.UpToDown[i] = auxiliary_generate(m, self.UpToDown[i-1], self.lat[i], initial, L='l', R='r', U='u', D='d', scan_time=self.scan_time)
        self.UpToDown[n-1] = [np.tensor(1.) for j in range(m)]

    def __auxiliary_down_to_up(self):
        if self.flag_down_to_up:
            return
        n, m = self.size
        self.flag_down_to_up = True
        self.DownToUp = [None for i in range(n)]
        self.DownToUp[n-1] = [self.lat[n-1][j] for j in range(m)]
        for i in range(n-2, 0, -1):
            initial = [None for j in range(m)]
            for j in range(m):
                if j == 0:
                    initial[j] = np.tensor(np.random.rand(self.D, self.D_c), legs=['u', 'r'])
                elif j == m-1:
                    initial[j] = np.tensor(np.random.rand(self.D, self.D_c), legs=['u', 'l'])
                else:
                    initial[j] = np.tensor(np.random.rand(self.D, self.D_c, self.D_c), legs=['u', 'l', 'r'])
            self.DownToUp[i] = auxiliary_generate(m, self.DownToUp[i+1], self.lat[i], initial, L='l', R='r', U='d', D='u', scan_time=self.scan_time)
        self.DownToUp[0] = [np.tensor(1.) for j in range(m)]

    def __auxiliary_left_to_right(self):
        if self.flag_left_to_right:
            return
        n, m = self.size
        self.flag_left_to_right = True
        self.LeftToRight = [None for j in range(m)]
        self.LeftToRight[0] = [self.lat[i][0] for i in range(n)]
        for j in range(1, m-1):
            initial = [None for j in range(n)]
            for i in range(n):
                if i == 0:
                    initial[i] = np.tensor(np.random.rand(self.D, self.D_c), legs=['r', 'd'])
                elif i == n-1:
                    initial[i] = np.tensor(np.random.rand(self.D, self.D_c), legs=['r', 'u'])
                else:
                    initial[i] = np.tensor(np.random.rand(self.D, self.D_c, self.D_c), legs=['r', 'd', 'u'])
            self.LeftToRight[j] = auxiliary_generate(n, self.LeftToRight[j-1], [self.lat[t][j] for t in range(n)], initial,
                                                     L='u', R='d', U='l', D='r', scan_time=self.scan_time)
        self.LeftToRight[m-1] = [np.tensor(1.) for i in range(n)]
        tmp = self.LeftToRight
        self.LeftToRight = [[tmp[j][i] for j in range(m)] for i in range(n)]

    def __auxiliary_right_to_left(self):
        if self.flag_right_to_left:
            return
        n, m = self.size
        self.flag_right_to_left = True
        self.RightToLeft = [None for j in range(m)]
        self.RightToLeft[m-1] = [self.lat[i][m-1] for i in range(n)]
        for j in range(m-2, 0, -1):
            initial = [None for j in range(n)]
            for i in range(n):
                if i == 0:
                    initial[i] = np.tensor(np.random.rand(self.D, self.D_c), legs=['l', 'd'])
                elif i == n-1:
                    initial[i] = np.tensor(np.random.rand(self.D, self.D_c), legs=['l', 'u'])
                else:
                    initial[i] = np.tensor(np.random.rand(self.D, self.D_c, self.D_c), legs=['l', 'd', 'u'])
            self.RightToLeft[j] = auxiliary_generate(n, self.RightToLeft[j+1], [self.lat[t][j] for t in range(n)], initial,
                                                     L='u', R='d', U='r', D='l', scan_time=self.scan_time)
        self.RightToLeft[0] = [np.tensor(1.) for i in range(n)]
        tmp = self.RightToLeft
        self.RightToLeft = [[tmp[j][i] for j in range(m)] for i in range(n)]
