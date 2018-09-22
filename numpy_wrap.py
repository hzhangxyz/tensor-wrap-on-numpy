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
        pass
