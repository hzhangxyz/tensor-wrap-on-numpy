import numpy_wrap as np
from square_lattice import SquareLattice as sl

class ITEBD(sl):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.env_v = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        self.env_h = [[np.ones(self.D) for j in range(m)] for i in range(n)]
        for i in range(n):
            self.env_h[i][m-1] = np.array(1)
        for j in range(m):
            self.env_v[n-1][m] = np.array(1)
        self.Hamiltonian = np.tensor(
                np.array([1,0,0,0,0,-1,2,0,0,2,-1,0,0,0,0,1])
                .reshape([2,2,2,2]), legs=['p1','p2','P1','P2'])/4.
        self.Identity = np.tensor(
                np.identity(4)
                .reshape([2,2,2,2]), legs=['p1','p2','P1','P2'])

    def itebd(self, step, delta, end=False):
        self.Evolution = self.Identity - delta * self.Hamiltonian
        for t in range(step):
            self.itebd_once_h(0)
            self.itebd_once_h(1)
            self.itebd_once_v(0)
            self.itebd_once_v(1)
        if end:
            self.itebd_done()

    def itebd_once_h(self, base):
        for i in range(n):
            for j in range(base,m-1,2):
                # j,j+1
                # [i,j]
                self.lattice[i][j]\
                        .tensor_multiple(self.env_v[i-1][j],'u')\
                        .tensor_multiple(self.env_h[i][j-1],'l')\
                        .tensor_multiple(self.env_v[i][j],'d')
                self.lattice[i][j], r1 = self.lattice[i][j].tensor_qr(
                        ['u','l','d'],['r','p'],['r','l'])
                self.lattice[i][j+1]\
                        .tensor_multiple(self.env_v[i-1][j+1],'u')\
                        .tensor_multiple(self.env_h[i][j+1],'r')\
                        .tensor_multiple(self.env_v[i][j+1],'d')
                self.lattice[i][j+1], r2 = self.lattice[i][j].tensor_qr(
                        ['u','r','d'],['l','p'],['l','r'])
                big = np.tensor_contract(r1, r2, ['r'],['l'],{'p':'p1'},{'p':'p2'})
                big = big.tensor_contract(self.Evolution, ['p1','p2'],['p1','p2'])

    def itebd_once_v(self):
        for i in range(n-1):
            for j in range(m):
                pass

    def itebd_done(self):
        pass

