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


class SquareLattice():

    def __create_node(self, i, j):
        legs = "lrud"
        if i == 0:
            legs = legs.replace("u", "")
        if i == self.size[0]-1:
            legs = legs.replace("d", "")
        if j == 0:
            legs = legs.replace("l", "")
        if j == self.size[1]-1:
            legs = legs.replace("r", "")
        return np.tensor(np.random.rand(2, *[self.D for i in legs]), legs=['p', *legs])

    def __init__(self, n, m, D, D_c, scan_time=2):
        self.size = (n, m)
        self.D = D
        self.D_c = D
        self.scan_time = scan_time
        self.lattice = [[self.__create_node(i, j) for j in range(m)] for i in range(n)]
        self.spin = SpinState(self)
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

    def __new__(cls, lattice, *args, **kw):
        obj = super().__new__(SpinState)
        obj.size = lattice.size
        return obj

    def __init__(self, lattice, scan_time=2, spin_state=None):
        if spin_state:
            super().__init__(spin_state)
        else:
            super().__init__([[(i+j) % 2 for j in range(self.size[1])] for i in range(self.size[0])])  # !!
        self.lattice = lattice.lattice
        self.D = lattice.D
        self.D_c = lattice.D_c
        self.scan_time = lattice.scan_time
        # these attr above is read only in SpinState class
        self.lat = [[self.lattice[i][j][self[i][j]] for j in range(self.size[1])] for i in range(self.size[0])]
        self.flag_up_to_down = False
        self.flag_down_to_up = False
        self.flag_left_to_right = False
        self.flag_right_to_left = False
        self.w_s = None

        self.auxiliary()
        print(self.cal_w_s())

    def cal_w_s(self):
        if self.w_s is not None:
            return self._ws
        n, m = self.size

        self.auxiliary_up_to_down()
        self.w_s = np.tensor_contract(self.UpToDown[n-2][0], self.lat[n-1][0], ['d'], ['u'], {'r': 'r1'}, {'r': 'r2'})
        for j in range(1, m):
            self.w_s = self.w_s\
                .tensor_contract(self.UpToDown[n-2][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                .tensor_contract(self.lat[n-1][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})

        if True:
            return self.w_s
        print(self.w_s)

        self.auxiliary()
        self.w_s = np.tensor_contract(self.DownToUp[1][0], self.lat[0][0], ['u'], ['d'], {'r': 'r2'}, {'r': 'r1'})
        for j in range(1, m):
            self.w_s = self.w_s\
                .tensor_contract(self.DownToUp[1][j], ['r2'], ['l'], {}, {'r': 'r2'})\
                .tensor_contract(self.lat[0][j], ['r1', 'u'], ['l', 'd'], {}, {'r': 'r1'})
        print(self.w_s)

        self.w_s = np.tensor_contract(self.LeftToRight[0][m-2], self.lat[0][m-1], ['r'], ['l'], {'d': 'd1'}, {'d': 'd2'})
        for i in range(1, n):
            self.w_s = self.w_s\
                .tensor_contract(self.LeftToRight[i][m-2], ['d1'], ['u'], {}, {'d': 'd1'})\
                .tensor_contract(self.lat[i][m-1], ['d2', 'r'], ['u', 'l'], {}, {'d': 'd2'})
        print(self.w_s)

        self.w_s = np.tensor_contract(self.RightToLeft[0][1], self.lat[0][0], ['l'], ['r'], {'d': 'd2'}, {'d': 'd1'})
        for i in range(1, n):
            self.w_s = self.w_s\
                .tensor_contract(self.RightToLeft[i][1], ['d2'], ['u'], {}, {'d': 'd2'})\
                .tensor_contract(self.lat[i][0], ['d1', 'l'], ['u', 'r'], {}, {'d': 'd1'})
        print(self.w_s)

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
                    E_s_diag += 1 if self[i][j] == self[i][j+1] else -1
                if i != n-1:
                    E_s_diag += 1 if self[i][j] == self[i+1][j] else -1
        E_s_non_diag = 0.  # 为相邻两个交换后的w(s)之和
        Delta_s = [[None for j in range(m)] for i in range(n)]  # 下面每个点记录一下
        # 横向j j+1
        for i in range(n):
            l = [None for j in range(m)]
            r = [None for j in range(m)]
            if i == 0:
                l[0] = self.lat[i, 0]\
                    .tensor_contract(self.DownToUp[i+1, 0], ['d'], ['u'], {'r': 'r2'}, {'r': 'r3'})
                for j in range(1, m-1):
                    l[j] = l[j-1]\
                        .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})\
                        .tensor_contract(self.DownToUp[i+1][j], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'})
                r[m-1] = self.lat[i][m-1]\
                    .tensor_contract(self.DownToUp[i+1][m-1], ['d'], ['u'], {'l': 'l2'}, {'l': 'l3'})
                for j in range(m-2, 0, -1):
                    r[j] = r[j+1]\
                        .tensor_contract(self.lat[i][j], ['l2', 'd'], ['r', 'u'], {}, {'l': 'l2'})\
                        .tensor_contract(self.DownToUp[i+1][j], ['l3', 'd'], ['r', 'u'], {}, {'r': 'r3'})
            elif i == n-1:
                l[0] = self.UpToDown[i-1, 0]\
                    .tensor_contract(self.lat[i, 0], ['d'], ['u'], {'r': 'r1'}, {'r': 'r2'})
                for j in range(1, m-1):
                    l[j] = l[j-1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                        .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})
                r[m-1] = self.UpToDown[i-1, m-1]\
                    .tensor_contract(self.lat[i][m-1], ['d'], ['u'], {'l': 'l1'}, {'l': 'l2'})
                for j in range(m-2, 0, -1):
                    r[j] = r[j+1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['l1'], ['r'], {}, {'l': 'l1'})\
                        .tensor_contract(self.lat[i][j], ['l2', 'd'], ['r', 'u'], {}, {'l': 'l2'})
            else:
                l[0] = self.UpToDown[i-1, 0]\
                    .tensor_contract(self.lat[i, 0], ['d'], ['u'], {'r': 'r1'}, {'r': 'r2'})\
                    .tensor_contract(self.DownToUp[i+1, 0], ['d'], ['u'], {}, {'r': 'r3'})
                for j in range(1, m-1):
                    l[j] = l[j-1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['r1'], ['l'], {}, {'r': 'r1'})\
                        .tensor_contract(self.lat[i][j], ['r2', 'd'], ['l', 'u'], {}, {'r': 'r2'})\
                        .tensor_contract(self.DownToUp[i+1][j], ['r3', 'd'], ['l', 'u'], {}, {'r': 'r3'})
                r[m-1] = self.UpToDown[i-1, m-1]\
                    .tensor_contract(self.lat[i][m-1], ['d'], ['u'], {'l': 'l1'}, {'l': 'l2'})\
                    .tensor_contract(self.DownToUp[i+1][m-1], ['d'], ['u'], {}, {'l': 'l3'})
                for j in range(m-2, 0, -1):
                    r[j] = r[j+1]\
                        .tensor_contract(self.UpToDown[i-1][j], ['l1'], ['r'], {}, {'l': 'l1'})\
                        .tensor_contract(self.lat[i][j], ['l2', 'd'], ['r', 'u'], {}, {'l': 'l2'})\
                        .tensor_contract(self.DownToUp[i+1][j], ['l3', 'd'], ['r', 'u'], {}, {'r': 'r3'})
            for j in range(m):
                if i == 0:
                    if j == 0:
                        Delta_s[i][j] = self.DownToUp[1][0].tensor_contract(r[1], ['r'], ['l3'], {'u': 'd'}, {'l2': 'r'})
                    elif j == m-1:
                        Delta_s[i][j] = self.DownToUp[1][m-1].tensor_contract(l[m-2], ['l'], ['r3'], {'u': 'd'}, {'r2': 'l'})
                    else:
                        Delta_s[i][j] = l[j-1]\
                            .tensor_contract(self.DownToUp[1][j], ['r3'], ['l'], {'r2': 'l'}, {'u': 'd', 'r': 'r3'})\
                            .tensor_contract(r[j+1], ['r3'], ['l3'], {}, {'l2': 'r'})
                elif i == n-1:
                    if j == 0:
                        Delta_s[i][j] = self.UpToDown[n-2][0].tensor_contract(r[1], ['r'], ['l1'], {'d': 'u'}, {'l2': 'r'})
                    elif j == m-1:
                        Delta_s[i][j] = self.UpToDown[n-2][m-1].tensor_contract(l[m-2], ['l'], ['r1'], {'d': 'u'}, {'r2': 'l'})
                    else:
                        pass
                else:
                    if j == 0:
                        pass
                    elif j == m-1:
                        pass
                    else:
                        Delta_s[i][j] = np.tensor_contract(
                            np.tensor_contract(l[j-1], self.UpToDown[i-1][j], ['r1'], ['l'], {'r2': 'l'}, {'r': 'r1', 'd': 'u'}),
                            np.tensor_contract(r[j+1], self.DownToUp[i+1][j], ['l3'], ['r'], {'l2': 'r'}, {'l': 'l3', 'u': 'd'}),
                            ['r1', 'r3'], ['l1', 'l3'])

        # 纵向i i+1

        E_s = E_s_diag + 2.*E_s_non_diag/self.cal_w_s()  # 2是哈密顿量里的
        Delta_s = Delta_s/self.cal_w_s()
        return E_s, Delta_s

    def auxiliary(self):
        self.auxiliary_up_to_down()
        self.auxiliary_down_to_up()
        self.auxiliary_left_to_right()
        self.auxiliary_right_to_left()

    def auxiliary_up_to_down(self):
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

    def auxiliary_down_to_up(self):
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

    def auxiliary_left_to_right(self):
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
            self.LeftToRight[j] = auxiliary_generate(m, self.LeftToRight[j-1], [self.lat[t][j] for t in range(n)], initial,
                                                     L='u', R='d', U='l', D='r', scan_time=self.scan_time)
        self.LeftToRight[m-1] = [None for i in range(n)]
        tmp = self.LeftToRight
        self.LeftToRight = [[tmp[j][i] for j in range(m)] for i in range(n)]

    def auxiliary_right_to_left(self):
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
            self.RightToLeft[j] = auxiliary_generate(m, self.RightToLeft[j+1], [self.lat[t][j] for t in range(n)], initial,
                                                     L='u', R='d', U='r', D='l', scan_time=self.scan_time)
        self.RightToLeft[0] = [None for i in range(n)]
        tmp = self.RightToLeft
        self.RightToLeft = [[tmp[j][i] for j in range(m)] for i in range(n)]
