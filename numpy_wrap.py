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
        print("legs",legs)
        self.legs = [str(legs[i]) for i in range(self.ndim) ]
        assert len(set(self.legs)) == len(self.legs), "repeated legs name"

    def set_legs(self, legs):
        self.legs = [str(legs[i]) for i in range(self.ndim) ]
        assert len(set(self.legs)) == len(self.legs), "repeated legs name"
        return self

    def rename_legs(self, legs_dict):
        for i, j in legs_dict.items():
            try:
                self.legs[self.legs.index(str(i))] = str(j)
            except Exception as e:
                raise Exception("error when rename legs") from e
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

    def tensor_svd():
        pass
