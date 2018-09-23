from numpy import *

class tensor(ndarray):
    def __new__(cls, input_array, legs=None):
        obj = asarray(input_array).view(cls)
        obj.legs = [str(legs[i]) for i in range(obj.ndim) ]
        assert len(set(obj.legs)) == len(obj.legs), "repeated legs name"
        return obj

    def __repr__(self):
        return f'{super().__repr__()}\nlegs({self.legs})'

    def __array_finalize__(self, obj):
        if obj is None:
            return
        legs = getattr(obj, 'legs', None)
        if legs is None:
            self.legs = legs
            return
        self.legs = [str(legs[i]) for i in range(self.ndim) ]
        assert len(set(self.legs)) == len(self.legs), "repeated legs name"

    def __getitem__(self, args):
        res = super().__getitem__(args)
        if not isinstance(res, ndarray):
            return res
        if isinstance(args, int):
            args = (args,)
        legs_to_del = [i for i,j in enumerate(args) if isinstance(j, int)]
        tmp_legs = [j for i,j in enumerate(self.legs) if i not in legs_to_del]
        res.set_legs(tmp_legs)
        return res

    def set_legs(self, legs):
        self.legs = [str(legs[i]) for i in range(self.ndim) ]
        assert len(set(self.legs)) == len(self.legs), "repeated legs name"
        return self

    def rename_legs(self, legs_dict):
        for i, j in legs_dict.items():
            self.legs[self.legs.index(str(i))] = str(j)
        assert len(set(self.legs)) == len(self.legs), "repeated legs name"
        return self

    def tensor_transpose(self, legs):
        res = self.transpose([self.legs.index(str(i)) for i in legs])
        res.set_legs(legs)
        return res

    def tensor_contract(self, tensor, legs1, legs2,
            legs_dict1=None, legs_dict2=None):
        tensor1 = self
        tensor2 = tensor
        legs_dict1 = {} if legs_dict1 is None else legs_dict1
        legs_dict2 = {} if legs_dict2 is None else legs_dict2
        order1 = [tensor1.legs.index(str(i)) for i in legs1]
        order2 = [tensor2.legs.index(str(i)) for i in legs2]
        legs = [j if j not in legs_dict1 else legs_dict1[j] \
                for j in tensor1.legs if j not in legs1] +\
                [j if j not in legs_dict2 else legs_dict2[j] \
                for j in tensor2.legs if j not in legs2]
        res = tensordot(tensor1, tensor2, [order1, order2])
        return self.__class__(res, legs=legs)

    """
    def __matmul__(self, b):
        res = super().dot(b)
        if self.ndim is not 2 or b.ndim is not 2:
            raise Exception("ambiguous dot")
        res.rename_legs({self.legs[1]:b.legs[1]})
        return res
    """

    """
    def inverse(self):
        res = linalg.inv(self)
        pass
    """

    def tensor_svd(self, legs1, legs2, new_legs):
        assert set(legs1) | set(legs2) == set(self.legs), "svd legs not correct"
        transposed = self.tensor_transpose([*legs1, *legs2])
        size1 = prod(self.shape[:len(legs1)])
        size2 = prod(self.shape[len(legs1):])
        tensor1, env, tensor2 = linalg.svd(transposed.reshape(
            [size1, size2]))
        tensor1.resize([*transposed.shape[:len(legs1)],tensor1.size//size1])
        tensor2.resize([*transposed.shape[len(legs1):],tensor2.size//size2])
        tensor1.set_legs([*legs1,new_legs])
        tensor2.set_legs([*legs2,new_legs])
        return tensor1, env, tensor2

    """
    def tensor_svd_cut():
        pass
    """

    def tensor_qr(self, legs1, legs2, new_legs):
        assert set(legs1) | set(legs2) == set(self.legs), "svd legs not correct"
        transposed = self.tensor_transpose([*legs1, *legs2])
        size1 = prod(self.shape[:len(legs1)])
        size2 = prod(self.shape[len(legs1):])
        tensor1, tensor2 = linalg.qr(transposed.reshape(
            [size1, size2]))
        tensor1.resize([*transposed.shape[:len(legs1)],tensor1.size//size1])
        tensor2.resize([*transposed.shape[len(legs1):],tensor2.size//size2])
        tensor1.set_legs([*legs1,new_legs])
        tensor2.set_legs([*legs2,new_legs])
        return tensor1, tensor2

tensor_transpose = tensor.tensor_transpose
tensor_contract = tensor.tensor_contract
tensor_svd = tensor.tensor_svd
tensor_qr = tensor.tensor_qr
