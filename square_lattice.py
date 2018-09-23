import numpy_wrap as np

class SqaureLattice():

    def __create_node(self, i, j):
        legs = "lrud"
        if i==0:
            legs = legs.replace("l","")
        if i==self.size[0]-1:
            legs = legs.replace("r","")
        if j==0:
            legs = legs.replace("u","")
        if j==self.size[1]-1:
            legs = legs.replace("d","")
        return np.tensor(np.random.rand(
            2,*[self.D for i in legs]),legs=['p',*legs])

    def __init__(self, n, m, D):
        self.size = (n,m)
        self.D = D
        self.lattice = [[self.__create_node(i,j)
            for j in range(m)] for i in range(n)]
        self.spin = SpinState([self])
        #self.markov_chain_length = 1000
        #self.hamiltonian = np.tenor(np.reshape([1,0,0,0,0,-1,2,0,0,2,-1,0,0,0,0,1],[2,2,2,2]),legs=["p1","p2","P1","P2"])

    def markov_chain(self, markov_chain_length):
        sum_E_s = 0
        # sum_Delta_s =
        for i in range(markov_chain_length):
            E_s = self.spin.cal_E_s()
            Delta_s = self.spin.cal_E_s()
            sum_E_s += E_s
            Delta_s += Delta_s
            Prod += E_s * Delta_s
            new_spin = self.spin.hop()
            if check(new_spin):
                self.spin = new_spin
        Grad = 2*Prod - 2*sum_E_s*sum_Delta_s
        Energy = sum_E_s
        return Energy, Grad

class SpinState(list):

    def __new__(cls, lattice, spin_state=None):
        obj = super().__new__(SpinState)
        return obj

    def __init__(self, lattice, spin_state=None):
        if spin_state:
            super().__init__(spin_state)
        else:
            super().__init__([[(i+j)%2
                for j in range(self.size[1])]
                for i in range(self.size[0])]) #!!
        obj.size = lattice[0].size
        obj.lattice = lattice[0].lattice
        obj.D = lattice[0].D
        # this 3 attr above is read only in SpinState class
        self.lat = [[self.lattice[i][j][self.spin[i][j]]
                for j in range(self.size[1])]
                for i in range(self.size[0])]
        self.auxiliary()

    def auxiliary(self):
        # up to down
        self.UpToDown = [[None for j in range(m)] for i in range(n)]
        for j in range(m):
            self.UpToDown[0][j] = self.lat[0][j]
        for i in range(1,n-1):
            #self.UpToDown[i-1], self.lat[i]
            for j in range(m):
                self.UpToDown[i][j] = self.UpToDown[i-1][j] #
            self.UpToDown[i][0], r = self.UpToDown[i][0].tensor_qr(
                    ['d'],['r'],['r','l'])
            l = [np.tensor_contract(self.UpToDown[i-1][0],self.lat[i],
                    ['d'],['u'],{"r":"r1"},{"r":"r2"})]
            for j in range(1,m-1):
                self.UpToDown[i][j] = np.tensor_contract(
                        self.UpToDown[i], r, ['l'], ['r'])
                self.UpToDown[i][j], r = self.UpToDown[i][j].tensor_qr(
                        ['l','d'],['r'],['r','l'])
                l_tmp = l[-1]
                np.tensor_contract(l_tmp,self.lat)
            self.UpToDown[i][m-1]


    def cal_E_s(self):
        """
        E_s=\sum_{s'} W(s')/W(s) H_{ss'}
        第一部分:对角
        第二部分:交换
        """
        Es = 0
        for i in range(n):
            for j in range(m):
                if j!=m-1:
                    Es += 1 if self.spin[i][j] == self.spin[i][j] else -1

    def cal_Delta_s(self):
        pass
