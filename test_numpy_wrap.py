import unittest
import numpy_wrap as np
import pickle

class TestNumpyWrap(unittest.TestCase):
    def test_tensor_basic(self):
        A = np.tensor(np.arange(6).reshape([2,3]),legs=['leg1','leg2'])
        assert A[1,0] == 3
        B = A.tensor_transpose(['leg2','leg1'])
        assert B[1,0] == 1
        assert repr(B).find('leg') != -1
        C = pickle.loads(pickle.dumps(B))
        assert (C == B).all()
        D = C[1]
        assert D.legs == ['leg1']
        E = np.tensor(np.arange(9).reshape([3,3]),legs=['A','B'])
        D = np.tensor(np.arange(9).reshape([3,3]),legs=['B','A'])
        F = np.tensor(np.arange(9).reshape([3,3]),legs=['A','B'])
        assert (E-D!=0).any()
        assert (E+(-F)==0).all()
        G = D + 2 - 1
        G += 1
        G -= 1
        assert (G-D==1).all()
        E -= D
        assert (E!=0).any()
        D += -F
        assert (D!=0).any()
        H = np.tensor(1) * A * 1
        H.rename_legs({'leg1':'l'})
        assert H.legs[0] == 'l'
        H.rename_legs({'leg1':'l', 'leg2':'r'}, restrict_mode=False)
        assert H.legs[0] == 'l'
        assert H.legs[1] == 'r'

    def test_tensor_contract(self):
        A = np.tensor(np.arange(6).reshape([2,3]),legs=['a','b'])
        B = np.tensor(np.arange(6).reshape([3,2]),legs=['c','d'])
        C = np.tensor_contract(A, B, ['b'], ['c'], {'a':'p'})
        assert C.legs == ['p','d']
        assert C.shape == (2,2)
        C = np.tensor_contract(A, B, ['b','x'], ['c','d'], restrict_mode=False)
        assert C.legs == ['a','d']
        assert C[0,0] == 10

        D = np.tensor(np.arange(4).reshape([2,2]),legs='ab')
        D.tensor_multiple([0,1],'a')
        assert D[0][0] == 0
        assert D[0][1] == 0
        assert D[1][0] != 0
        assert D[1][1] != 0
        D = D+1
        D.tensor_multiple([0,1],'b')
        assert D[0][0] == 0
        assert D[0][1] != 0
        assert D[1][0] == 0
        assert D[1][1] != 0
        D = D+2
        E = D
        D.tensor_multiple([0,1],'c', restrict_mode=False)
        assert (E==D).all()

    def test_tensor_svd_and_qr(self):
        A = np.tensor(np.arange(6).reshape([2,3]),legs=['a','b'])
        u, s, v = A.tensor_svd(['a'],['b'],['r','l'],full_matrices=False)
        assert (np.dot(u,u.transpose()) - np.identity(2) < 0.0001).all()
        assert ( v.tensor_contract(v,['b'],['b'], {'l':'L'}) - np.identity(2) < 0.0001).all()
        u.tensor_multiple(s,'r')
        assert (u.tensor_contract(v,['r'],['l']) - A.tensor_transpose(['b','a']) < 0.0001).all()

        A = np.tensor(np.arange(6).reshape([2,3]),legs=['a','b'])
        q, r = A.tensor_qr(['a'],['b'],['r','l'])
        assert (np.dot(q,q.transpose()) - np.identity(2) < 0.0001).all()
        assert (q.tensor_contract(r,['r'],['l']) - A.tensor_transpose(['b','a']) < 0.0001).all()

if __name__ == '__main__':
    unittest.main()
