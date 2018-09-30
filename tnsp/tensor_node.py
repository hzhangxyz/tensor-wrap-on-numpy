import numpy as np
import tensorflow as tf

class Node(object):
    """
    支持的操作
    - getitem
    - 数乘
    - 相同shape相加
    - neg

    转置

    - contract
    - svd
    - qr
    """
    def rename_legs(self, legs_dict, restrict_mode=True):
        for i, j in legs_dict.items():
            if str(i) in self.legs:
                self.legss[self.legs.index(i)] = j
            else:
                if restrict_mode:
                    raise Exception("leg not found")
        self.check_legs()
        return self

    def __init__(self, data, legs=[], *args, **kw):
        self.data = tf.convert_to_tensor(data, *args, **kw)
        self.legs = [legs[i] for i in range(len(self.shape))]
        self.check_legs()

    @property
    def shape(self):
        return self.data.shape

    def check_legs(self):
        assert len(set(self.legs)) == len(self.legs), "repeated legs name"

    def __repr__(self):
        return f'{self.data.__repr__()}\nlegs[{self.legs}]'

    def __str__(self):
        return f'{self.data.__str__()}\nlegs[{self.legs}]'

    """
    def __getitem__(self, arg):
        res = self.data.__getitem__(arg)
        if not isinstance(arg, tuple):
            arg = (arg,)
        index_to_del = [i for i, j in enumerate(arg) if not isinstance(j, slice)]
        res_legs = [j for i, j in enumerate(self.legs) if i not in index_to_del]
        return Node(res, res_legs)
    """

    """
    def __add__(self, arg):
        if isinstance(arg, Node):
            arg = arg.tensor_transpose(self.legs).data
        return Node(self.data + arg, self.legs)
    """

    """
    def __sub__(self, arg):
        if isinstance(arg, Node):
            arg = arg.tensor_transpose(self.legs).data
        return Node(self.data - arg, self.legs)

    def __neg__(self):
        return Node(-self.data, self.legs)
    """

    """
    def __mul__(self, arg):
        if isinstance(arg, Node):
            if len(self.legs) == 0 and len(arg.legs) != 0:
                return Node(self.data*arg.data, arg.legs)
            else:
                return Node(self.data*arg.data, self.legs)
        else:
            return Node(self.data*arg, self.legs)
    """

    """
    def __truediv__(self, arg):
        if isinstance(arg, Node):
            assert len(arg.shape) == 0
            return Node(self.data/arg.data, self.legs)
        else:
            return Node(self.data/arg, self.legs)
    """

    def tensor_transpose(self, legs, name_scope='tensor_transpose'):
        with tf.name_scope(name_scope):
            res = tf.transpose(self.data, [self.legs.index(i) for i in legs])
            return Node(res, legs)

    def tensor_contract(self, tensor, legs1, legs2, legs_dict1={}, legs_dict2={}, restrict_mode=True):
        tensor1 = self
        tensor2 = tensor
        order1 = []
        order2 = []
        correct_legs1 = []
        correct_legs2 = []
        for i, j in zip(legs1, legs2):
            if i in tensor1.legs and j in tensor2.legs:
                order1.append(tensor1.legs.index(i))
                order2.append(tensor2.legs.index(j))
                correct_legs1.append(i)
                correct_legs2.append(j)
            else:
                if restrict_mode:
                    raise Exception("leg not match in contract")
        legs = [j if j not in legs_dict1 else legs_dict1[j]
                for j in tensor1.legs if j not in correct_legs1] +\
               [j if j not in legs_dict2 else legs_dict2[j]
                for j in tensor2.legs if j not in correct_legs2]
        res = tf.tensordot(tensor1.data, tensor2.data, [order1, order2], name='tensor_contract')
        return Node(res, legs)

    def tensor_multiple(self, arr, leg, restrict_mode=True):
        if leg not in self.legs:
            if restrict_mode:
                raise Exception("leg not match in multiple")
        else:
            self.data = tf.tensordot(self.data, arr, [[self.legs.index(leg)], [0]], name='tensor_multiple')
        return self

    def tensor_svd(self, legs1, legs2, new_legs, restrict_mode=True, name_scope='tensor_svd', *args, **kw):
        with tf.name_scope(name_scope):
            assert set(legs1) | set(legs2) >= set(self.legs) and set(legs1) & set(legs2) == set(), "svd legs not correct"
            if restrict_mode:
                assert set(legs1) | set(legs2) == set(self.legs), "svd legs not correct"
            legs1 = [i for i in self.legs if i in legs1]
            legs2 = [i for i in self.legs if i in legs2]
            transposed = self.tensor_transpose([*legs1, *legs2])
            size1 = np.prod(transposed.shape[:len(legs1)], dtype=int)
            size2 = np.prod(transposed.shape[len(legs1):], dtype=int)
            tensor1, env, tensor2 = tf.svd(tf.reshape(transposed.data, [size1, size2]), *args, **kw)
            assert tensor1.shape[0] == size1
            assert tensor2.shape[0] == size2
            tensor1 = tf.reshape(tensor1, [*transposed.shape[:len(legs1)], -1])
            tensor2 = tf.reshape(tensor2, [*transposed.shape[len(legs1):], -1])
            if not isinstance(new_legs, list):
                new_legs = [new_legs, new_legs]
            return Node(tensor1, [*legs1, new_legs[0]]), env, Node(tensor2, [*legs2, new_legs[1]])

    def tensor_qr(self, legs1, legs2, new_legs, restrict_mode=True, name_scope='tensor_qr', *args, **kw):
        with tf.name_scope(name_scope):
            assert set(legs1) | set(legs2) >= set(self.legs) and set(legs1) & set(legs2) == set(), "qr legs not correct"
            if restrict_mode:
                assert set(legs1) | set(legs2) == set(self.legs), "qr legs not correct"
            legs1 = [i for i in self.legs if i in legs1]
            legs2 = [i for i in self.legs if i in legs2]
            transposed = self.tensor_transpose([*legs1, *legs2])
            size1 = np.prod(transposed.data.shape[:len(legs1)], dtype=int)
            size2 = np.prod(transposed.data.shape[len(legs1):], dtype=int)
            tensor1, tensor2 = tf.qr(tf.reshape(transposed.data, [size1, size2]), *args, **kw)
            assert tensor1.shape[0] == size1
            assert tensor2.shape[-1] == size2
            tensor1 = tf.reshape(tensor1, [*transposed.data.shape[:len(legs1)], -1])
            tensor2 = tf.reshape(tensor2, [-1, *transposed.data.shape[len(legs1):]])
            if not isinstance(new_legs, list):
                new_legs = [new_legs, new_legs]
            return Node(tensor1, [*legs1, new_legs[0]]), Node(tensor2, [new_legs[1], *legs2])
