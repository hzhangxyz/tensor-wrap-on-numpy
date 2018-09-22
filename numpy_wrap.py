from numpy import *

class tensor(ndarray):
    def __new__(cls, input_array, legs=None):
        obj = asarray(input_array).view(cls)
        obj.legs = [str(legs[i]) for i in range(obj.ndim) ]
        assert len(set(obj.legs)) == len(obj.legs), "repeated legs name"
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.legs = getattr(obj, 'legs', None)

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
