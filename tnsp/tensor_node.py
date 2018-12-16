import numpy as np
import tensorflow as tf


class Node(object):
    """node, 有legs信息的tensorflow tensor.

    tensor信息存储在data里, legs信息存储在legs里.

    仅仅支持以下操作
    - transpose
    - contract
    - scalar
    - multiple
    - svd
    - qr
    """
    def assert_legs_dict(self, legs_dict, restrict_mode=True):
        assert isinstance(legs_dict, dict), 'legs_dict should be dict'
        for i, j in legs_dict.items():
            assert isinstance(i, str), 'legs_dict keys should be str'
            assert isinstance(j, str), 'legs_dict values should be str'
        if restrict_mode:
            for i in legs_dict:
                if not i in self.legs:
                    raise Exception(f'leg {i} not found')
    
    def assert_legs(self, legs, restrict_mode=True, part=True):
        assert isinstance(legs, list), 'legs array should be a list'
        for i in legs:
                assert isinstance(i, str), 'legs should be str'
        assert len(set(legs)) == len(legs), 'repeated legs name' 
        if part:
            if restrict_mode:
                for i in legs:
                    assert i in self.legs, 'legs not found'
        else:
            assert len(legs) == len(self.shape), 'number of legs should equal to dims of tensor'
            if hasattr(self,"legs"):
                assert set(legs) == set(self.legs), 'legs should be permutation of node legs'

    def rename_legs(self, legs_dict, restrict_mode=True):
        """重命名node的legs.

        参数为重命名用的dict, 应该是str->str的dict.
        """
        self.assert_legs_dict(legs_dict, restrict_mode)

        legs = [legs_dict[i] if i in legs_dict else i for i in self.legs]
        return Node(self.data, legs)

    def __init__(self, data, legs=[]):
        """创建一个node

        输入一个tensorflow的tensor与legs数组.
        """
        assert isinstance(data, tf.Tensor), 'data should be a tensor'
        self.data = data
        self.assert_legs(legs, part=False)
        self.legs = legs[:]

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f'{self.data.__repr__()}\nlegs[{self.legs}]'

    def __str__(self):
        return f'{self.data.__str__()}\nlegs[{self.legs}]'

    def tensor_transpose(self, legs, name='tensor_transpose'):
        """转置一个node.

        输入转置后的legs顺序作为参数, legs必须是self.legs的置换.
        """
        self.assert_legs(legs, part=False)

        with tf.name_scope(name):
            res = tf.transpose(self.data, [self.legs.index(i) for i in legs])
            return Node(res, legs)

    def tensor_contract(self, tensor, legs1, legs2, legs_dict1={}, legs_dict2={}, restrict_mode=True, name='tensor_contract'):
        """缩并两个张量.

        参数依次为两个node, 需要缩并的对应legs, 需要顺便重命名的其他legs.
        """
        assert isinstance(self, Node), 'contract parameter is not node'
        assert isinstance(tensor, Node), 'contract parameter is not node'
        self.assert_legs(legs1, restrict_mode)
        tensor.assert_legs(legs2, restrict_mode)
        self.assert_legs_dict(legs_dict1, restrict_mode)
        tensor.assert_legs_dict(legs_dict2, restrict_mode)
        assert len(legs1) == len(legs2), 'contract legs number differ'
        for i in legs_dict1:
            assert not i in legs1, 'element of legs_dict should not be in contract legs'
        for i in legs_dict2:
            assert not i in legs2, 'element of legs_dict should not be in contract legs'

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
        legs = [j if j not in legs_dict1 else legs_dict1[j]
                for j in tensor1.legs if j not in correct_legs1] +\
               [j if j not in legs_dict2 else legs_dict2[j]
                for j in tensor2.legs if j not in correct_legs2]
        res = tf.tensordot(tensor1.data, tensor2.data, [order1, order2], name=name)
        return Node(res, legs)

    def tensor_scalar(self, scalar, name='tensor_scalar'):
        """整个tensor乘上一个数."""
        assert isinstance(scalar, tf.Tensor), 'number to multiple should be convert to tensor first'
        assert len(scalar.shape) == 0, 'number should be dimension 0 tensor'

        data = tf.multiply(self.data, scalar, name=name)
        return Node(data, self.legs)

    def tensor_multiple(self, arr, leg, restrict_mode=True, name='tensor_multiple'):
        """在这个tensor的一个方向上乘上一个向量."""
        self.assert_legs([leg], restrict_mode)
        assert isinstance(arr, tf.Tensor), 'array to multiple should be convert to tensor first'
        assert len(arr.shape) == 1, 'array should be dimension 1 tensor'

        if leg not in self.legs:
            data = self.data
        else:
            data = tf.tensordot(self.data, arr, [[self.legs.index(leg)], [0]], name=name)
        return Node(data, self.legs)

    def tensor_svd(self, legs1, legs2, new_legs, restrict_mode=True, name='tensor_svd', *args, **kw):
        """对node做svd分解.

        参数依次是U侧leg, V侧leg, 和新产生的leg名称."""
        with tf.name_scope(name):
            assert set(legs1) | set(legs2) >= set(self.legs) and set(legs1) & set(legs2) == set(), 'svd legs not correct'
            if restrict_mode:
                assert set(legs1) | set(legs2) == set(self.legs), 'svd legs not correct'
            legs1 = [i for i in self.legs if i in legs1]
            legs2 = [i for i in self.legs if i in legs2]
            transposed = self.tensor_transpose([*legs1, *legs2])
            size1 = np.prod(transposed.shape[:len(legs1)], dtype=int)
            size2 = np.prod(transposed.shape[len(legs1):], dtype=int)
            env, tensor1, tensor2 = tf.svd(tf.reshape(transposed.data, [size1, size2]), *args, **kw)
            assert tensor1.shape[0] == size1
            assert tensor2.shape[0] == size2
            tensor1 = tf.reshape(tensor1, [*transposed.shape[:len(legs1)], -1])
            tensor2 = tf.reshape(tensor2, [*transposed.shape[len(legs1):], -1])
            if not isinstance(new_legs, list):
                new_legs = [new_legs, new_legs]
            return Node(tensor1, [*legs1, new_legs[0]]), env, Node(tensor2, [*legs2, new_legs[1]])

    def tensor_qr(self, legs1, legs2, new_legs, restrict_mode=True, name='tensor_qr', *args, **kw):
        """对node做qr分解.

        参数依次是Q侧leg, R侧leg, 和新产生的leg名称."""
        with tf.name_scope(name):
            assert set(legs1) | set(legs2) >= set(self.legs) and set(legs1) & set(legs2) == set(), 'qr legs not correct'
            if restrict_mode:
                assert set(legs1) | set(legs2) == set(self.legs), 'qr legs not correct'
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
