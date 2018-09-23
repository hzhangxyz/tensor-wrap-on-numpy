import numpy_wrap as np

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
                res[j], QR_R = res[j].tensor_qr([L, D], [R], [R, L]) # 这里R完全不需要, 而且这里还没cut
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


class SqaureLattice():

    def __create_node(self, i, j):
        legs = "lrud"
        if i == 0:
            legs = legs.replace("l", "")
        if i == self.size[0]-1:
            legs = legs.replace("r", "")
        if j == 0:
            legs = legs.replace("u", "")
        if j == self.size[1]-1:
            legs = legs.replace("d", "")
        return np.tensor(np.random.rand(2, *[self.D for i in legs]), legs=['p', *legs])

    def __init__(self, n, m, D):
        self.size = (n, m)
        self.D = D
        self.lattice = [[self.__create_node(i, j) for j in range(m)] for i in range(n)]
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
            super().__init__([[(i+j) % 2 for j in range(self.size[1])] for i in range(self.size[0])])  # !!
        obj.size = lattice[0].size
        obj.lattice = lattice[0].lattice
        obj.D = lattice[0].D
        # this 3 attr above is read only in SpinState class
        self.lat = [[self.lattice[i][j][self.spin[i][j]] for j in range(self.size[1])] for i in range(self.size[0])]
        self.flag_up_to_down = False
        self.flag_down_to_up = False
        self.w_s = None

    def cal_w_s(self):
        if self.w_s is not None:
            return self._ws
        self.auxiliary_up_to_down()
        self.w_s = np.tensor_contract(self.UpToDown[n-2][0],self.lat[n-1][0],['d'],['u'],{'r':'r1'},{'r':'r2'})
        for j in range(1,m):
            self.w_s = self.w_s\
                    .tensor_contract(self.UpToDown[n-2][j], ['r1'],['l'],{},{'r':'r1'})\
                    .tensor_contract(self.lat[n-1][j], ['r2','d'],['l','u'],{},{'r','r2'})
        return self.w_s

    def cal_E_s_and_Delta_s(self):
        """
        E_s=\sum_{s'} W(s')/W(s) H_{ss'}
        第一部分:对角
        第二部分:交换
        """
        self.auxiliary()
        E_s_diag = 0.
        for i in range(n):
            for j in range(m):
                if j != m-1:
                    E_s_diag += 1 if self.spin[i][j] == self.spin[i][j+1] else -1
                if i != n-1:
                    E_s_diag += 1 if self.spin[i][j] == self.spin[i+1][j] else -1
        E_s_non_diag = 0. # 为相邻两个交换后的w(s)之和
        #横向j j+1
        for i in range(n):
            for j in range(m):
                pass
        #纵向i i+1

        E_s = E_s_diag + 2*E_s_non_diag/self.cal_w_s()
        return E_s, Delta_s

    def auxiliary(self):
        self.auxiliary_up_to_down()
        self.auxiliary_down_to_up()

    def auxiliary_up_to_down(self):
        if self.flag_up_to_down:
            return
        self.flag_up_to_down = True
        # up to down
        self.UpToDown = [[None for j in range(m)] for i in range(n)]
        for j in range(m):
            self.UpToDown[0][j] = self.lat[0][j]
        for i in range(1, n-1):
            #self.UpToDown[i-1], self.lat[i]
            for j in range(m):
                self.UpToDown[i][j] = self.UpToDown[i-1][j]
                """这样初始化真的好么"""

            scan_time = 2
            for t in range(scan_time):
                self.UpToDown[i][0], QR_R = self.UpToDown[i][0].tensor_qr(['d'], ['r'], ['r', 'l'])
                l = [None for i in range(m)]
                l[0] = self.UpToDown[i-1][0]\
                    .tensor_contract(self.lat[i][0], ['d'], ['u'], {'r': 'r1'}, {'r': 'r2'})\
                    .tensor_contract(self.UpToDown[i][0], ['d'], ['d'], {}, {'r': 'r3'})

                for j in range(1, m-1):
                    if t == 0:
                        self.UpToDown[i][j] = np.tensor_contract(self.UpToDown[i][j], QR_R, ['l'], ['r'])
                    else:
                        self.UpToDown[i][j] = l[j-1]\
                            .tensor_contract(self.UpToDown[i-1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                            .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})\
                            .tensor_contract(self.r[j+1], ['r1', 'r2'], ['l1', 'l2'], {'r3': 'l', 'd': 'u'}, {'l3': 'r'})
                    self.UpToDown[i][j], QR_R = self.UpToDown[i][j].tensor_qr(['l', 'd'], ['r'], ['r', 'l'])
                    l[j] = l[j-1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                        .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})\
                        .tensor_contract(self.UpToDown[i][j], ['r3', 'd'], ['l', 'd'], {}, {'r': 'r3'})

                self.UpToDown[i][m-1] = l[m-2]\
                    .tensor_contract(self.UpToDown[i-1][m-1], ['r1'], ['l'])\
                    .tensor_contract(self.lat[i][m-1], ['r2', 'd'], ['l', 'u'], {'r3': 'l'}, {'d': 'u'})

                self.UpToDown[i][m-1], QR_R = self.UpToDown[i][m-1].tensor_qr(['d'], ['l'], ['l', 'r'])
                r = [None for i in range(m)]
                r[m-1] = self.UpToDown[i-1][m-1]\
                    .tensor_contract(self.lat[i][m-1], ['d'], ['u'], {'l': 'l1'}, {'l': 'l2'})\
                    .tensor_contract(self.UpToDown[i][m-1], ['d'], ['d'], {}, {'l': 'l3'})

                for j in range(m-2, 0, -1):
                    self.UpToDown[i][j] = l[j-1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                        .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})\
                        .tensor_contract(self.r[j+1], ['r1', 'r2'], ['l1', 'l2'], {'r3': 'l', 'd': 'u'}, {'l3': 'r'})
                    self.UpToDown[i][j], QR_R = self.UpToDown[i][j].tensor_qr(['r', 'd'], ['l'], ['l', 'r'])
                    r[j] = r[j+1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['l1'], ['r'], {}, {'l': 'l1'})\
                        .tensor_contract(self.lat[i][j], ['l2', 'd'], ['r', 'u'], {}, {'l': 'l2'})\
                        .tensor_contract(self.UpToDown[i][j], ['l3', 'd'], ['r', 'd'], {}, {'l': 'l3'})

                self.UpToDown[i][0] = self.UpToDown[i-1][0]\
                    .tensor_contract(self.lat[i][0], ['d'], ['u'], {'r': 'r1'}, {'r': 'r2'})\
                    .tensor_contract(self.r[1], ['r1', 'r2'], ['l1', 'l2'], {'d': 'u'}, {'l3': 'r'})

    def auxiliary_down_to_up(self):
        if self.flag_down_to_up:
            return
        self.flag_down_to_up = True
        # down to up
        self.DownToUp = [[None for j in range(m)] for i in range(n)]
        for j in range(m):
            self.DownToUp[n-1][j] = self.lat[n-1][j]
        for i in range(n-2, 0, -1):
            #self.DownToUp[i+1], self.lat[i]
            for j in range(m):
                self.DownToUp[i][j] = self.DownToUp[i+1][j]

            scan_time = 2
            for t in range(scan_time):
                self.DownToUp[i][0], QR_R = self.DownToUp[i][0].tensor_qr(['u'], ['r'], ['r', 'l'])
                l = [None for i in range(m)]
                l[0] = self.DownToUp[i+1][0]\
                    .tensor_contract(self.lat[i][0], ['u'], ['d'], {'r': 'r1'}, {'r': 'r2'})\
                    .tensor_contract(self.DownToUp[i][0], ['u'], ['u'], {}, {'r': 'r3'})

                for j in range(1, m-1):
                    if t == 0:
                        self.DownToUp[i][j] = np.tensor_contract(self.DownToUp[i][j], QR_R, ['l'], ['r'])
                    else:
                        self.DownToUp[i][j] = l[j-1]\
                            .tensor_contract(self.DownToUp[i+1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                            .tensor_contract(self.lat[i][j], ['r2', 'u'], ['l', 'd'], {}, {'r': 'r2'})\
                            .tensor_contract(self.r[j+1], ['r1', 'r2'], ['l1', 'l2'], {'r3': 'l', 'u': 'd'}, {'l3': 'r'})
                    self.DownToUp[i][j], QR_R = self.DownToUp[i][j].tensor_qr(['l', 'u'], ['r'], ['r', 'l'])
                    l[j] = l[j-1]\
                        .tensor_contract(self.DownToUp[i+1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                        .tensor_contract(self.lat[i][j], ['r2', 'u'], ['l', 'd'], {}, {'r': 'r2'})\
                        .tensor_contract(self.DownToUp[i][j], ['r3', 'u'], ['l', 'u'], {}, {'r': 'r3'})

                self.DownToUp[i][m-1] = l[m-2]\
                    .tensor_contract(self.DownToUp[i+1][m-1], ['r1'], ['l'])\
                    .tensor_contract(self.lat[i][m-1], ['r2', 'u'], ['l', 'd'], {'r3': 'l'}, {'u': 'd'})

                self.DownToUp[i][m-1], QR_R = self.DownToUp[i][m-1].tensor_qr(['u'], ['l'], ['l', 'r'])
                r = [None for i in range(m)]
                r[m-1] = self.DownToUp[i+1][m-1]\
                    .tensor_contract(self.lat[i][m-1], ['u'], ['d'], {'l': 'l1'}, {'l': 'l2'})\
                    .tensor_contract(self.DownToUp[i][m-1], ['u'], ['u'], {}, {'l': 'l3'})

                for j in range(m-2, 0, -1):
                    self.DownToUp[i][j] = l[j-1]\
                        .tensor_contract(self.DownToUp[i+1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                        .tensor_contract(self.lat[i][j], ['r2', 'u'], ['l', 'd'], {}, {'r': 'r2'})\
                        .tensor_contract(self.r[j+1], ['r1', 'r2'], ['l1', 'l2'], {'r3': 'l', 'u': 'd'}, {'l3': 'r'})
                    self.DownToUp[i][j], QR_R = self.DownToUp[i][j].tensor_qr(['r', 'u'], ['l'], ['l', 'r'])
                    r[j] = r[j+1]\
                        .tensor_contract(self.DownToUp[i+1][j], ['l1'], ['r'], {}, {'l': 'l1'})\
                        .tensor_contract(self.lat[i][j], ['l2', 'u'], ['r', 'd'], {}, {'l': 'l2'})\
                        .tensor_contract(self.DownToUp[i][j], ['l3', 'u'], ['r', 'u'], {}, {'l': 'l3'})

                self.DownToUp[i][0] = self.DownToUp[i+1][0]\
                    .tensor_contract(self.lat[i][0], ['u'], ['d'], {'r': 'r1'}, {'r': 'r2'})\
                    .tensor_contract(self.r[1], ['r1', 'r2'], ['l1', 'l2'], {'u': 'd'}, {'l3': 'r'})

