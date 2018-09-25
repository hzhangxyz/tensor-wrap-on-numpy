import numpy_wrap as np
from square_lattice import SquareLattice


class ITEBD(SquareLattice):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.env_v = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.env_h = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        for i in range(n):
            self.env_h[i][m-1] = np.array(1)
        for j in range(m):
            self.env_v[n-1][m] = np.array(1)
        self.Hamiltonian = np.tensor(
            np.array([1, 0, 0, 0, 0, -1, 2, 0, 0, 2, -1, 0, 0, 0, 0, 1])
            .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])/4.
        self.Identity = np.tensor(
            np.identity(4)
            .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])

    def itebd(self, step, delta, end=False, energy=True):
        self.Evolution = self.Identity - delta * self.Hamiltonian
        for t in range(step):
            print("itebd",t)
            self.itebd_once_h(0)
            self.itebd_once_h(1)
            self.itebd_once_v(0)
            self.itebd_once_v(1)
        if end:
            self.itebd_done()
            if energy:
                print(self.markov_chain()[0])

    def itebd_once_h(self, base):
        for i in range(n):
            for j in range(base, m-1, 2):
                # j,j+1
                self.lattice[i][j]\
                    .tensor_multiple(self.env_v[i-1][j], 'u')\
                    .tensor_multiple(self.env_h[i][j-1], 'l')\
                    .tensor_multiple(self.env_v[i][j], 'd')
                self.lattice[i][j+1]\
                    .tensor_multiple(self.env_v[i-1][j+1], 'u')\
                    .tensor_multiple(self.env_h[i][j+1], 'r')\
                    .tensor_multiple(self.env_v[i][j+1], 'd')
                tmp_left, r1 = self.lattice[i][j].tensor_qr(
                    ['u', 'l', 'd'], ['r', 'p'], ['r', 'l'])
                tmp_right, r2 = self.lattice[i][j].tensor_qr(
                    ['u', 'r', 'd'], ['l', 'p'], ['l', 'r'])
                r1.tensor_multiple(self.env_h[i][j], 'r')
                big = np.tensor_contract(r1, r2, ['r'], ['l'], {'p': 'p1'}, {'p': 'p2'})
                big = big.tensor_contract(self.Evolution, ['p1', 'p2'], ['p1', 'p2'])
                u, s, v = big.tensor_svd(['l', 'P1'], ['r', 'P2'], ['r', 'l'])
                self.env_h[i][j] = s[:self.D]
                self.lattice[i][j] = u[:, :, :self.D]\
                    .tensor_contract(tmp_left, ['l'], ['r'], {'P1': 'p'})
                self.lattice[i][j+1] = v[:, :, :self.D]\
                    .tensor_contract(tmp_right, ['r'], ['l'], {'P2': 'p'})
                self.lattice[i][j]\
                    .tensor_multiple(1/self.env_v[i-1][j], 'u')\
                    .tensor_multiple(1/self.env_h[i][j-1], 'l')\
                    .tensor_multiple(1/self.env_v[i][j], 'd')
                self.lattice[i][j+1]\
                    .tensor_multiple(1/self.env_v[i-1][j+1], 'u')\
                    .tensor_multiple(1/self.env_h[i][j+1], 'r')\
                    .tensor_multiple(1/self.env_v[i][j+1], 'd')

    def itebd_once_v(self):
        for i in range(n-1):
            for j in range(m):
                # i,i+1
                self.lattice[i][j]\
                    .tensor_multiple(self.env_h[i][j-1], 'l')\
                    .tensor_multiple(self.env_v[i-1][j], 'u')\
                    .tensor_multiple(self.env_h[i][j], 'r')
                self.lattice[i+1][j]\
                    .tensor_multiple(self.env_h[i+1][j-1], 'l')\
                    .tensor_multiple(self.env_v[i+1][j], 'd')
                .tensor_multiple(self.env_h[i+1][j], 'r')\
                    tmp_up, r1 = self.lattice[i][j].tensor_qr(
                    ['l', 'u', 'r'], ['d', 'p'], ['d', 'u'])
                tmp_down, r2 = self.lattice[i][j].tensor_qr(
                    ['l', 'd', 'r'], ['u', 'p'], ['u', 'd'])
                r1.tensor_multiple(self.env_v[i][j], 'd')
                big = np.tensor_contract(r1, r2, ['d'], ['u'], {'p': 'p1'}, {'p': 'p2'})
                big = big.tensor_contract(self.Evolution, ['p1', 'p2'], ['p1', 'p2'])
                u, s, v = big.tensor_svd(['u', 'P1'], ['d', 'P2'], ['d', 'u'])
                self.env_h[i][j] = s[:self.D]
                self.lattice[i][j] = u[:, :, :self.D]\
                    .tensor_contract(tmp_up, ['u'], ['d'], {'P1': 'p'})
                self.lattice[i+1][j] = v[:, :, :self.D]\
                    .tensor_contract(tmp_down, ['d'], ['u'], {'P2': 'p'})
                self.lattice[i][j]\
                    .tensor_multiple(1/self.env_h[i][j-1], 'l')\
                    .tensor_multiple(1/self.env_v[i-1][j], 'u')\
                    .tensor_multiple(1/self.env_h[i][j], 'r')
                self.lattice[i+1][j]\
                    .tensor_multiple(1/self.env_h[i+1][j-1], 'l')\
                    .tensor_multiple(1/self.env_v[i+1][j], 'd')
                .tensor_multiple(1/self.env_h[i+1][j], 'r')\


    def itebd_done(self):
        self.env_v = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.env_h = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        for i in range(n-1):
            for j in range(m-1):
                self.lattice[i][j]\
                    .tensor_multiple(self.env_v[i][j], 'd')\
                    .tensor_multiple(self.env_h[i][j], 'r')\
                    self.spin.fresh()
