import numpy_wrap as np

class SqaureLattice():

    def __create_node(self, i, j):
        legs = "lrud"
        if i==0:
            legs = legs.replace("l","")
        if i==n-1:
            legs = legs.replace("r","")
        if j==0:
            legs = legs.replace("u","")
        if j==m-1:
            legs = legs.replace("d","")
        return np.tensor(np.random.rand(
            2,*[self.D for i in legs]),legs=['p',*legs])

    def __init__(self, n, m, D):
        self.size = (n,m)
        self.D = D
        self.lattice = [[self.__create_node(i,j)
            for j in range(m)] for i in range(n)]
        self.spin = SpinState(self)
        #self.markov_chain_length = 1000
        #self.hamiltonian = np.tenor(np.reshape([1,0,0,0,0,-1,2,0,0,2,-1,0,0,0,0,1],[2,2,2,2]),legs=["p1","p2","P1","P2"])
    def markov_chain(self, markov_chain_length):
        sum_E_s = 0
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



class SpinState():

    def __init__(self, lattice):
        n, m = lattice.size
        self.lattice = lattice
        self.spin = [[(i+j)%2 for j in range(m)] for i in range(n)] #!!!

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
