import square_lattice as sl
import numpy_wrap as np

a = sl.SquareLattice(4, 6, 4, 8, 2)

b = list(map(lambda x:x.tolist(),a.spin.test_w_s()))
print(b)
print(np.std(b)/np.mean(b))

lat = a.spin.lat
e_s, delta_s = a.spin.cal_E_s_and_Delta_s()

for i in range(4):
    for j in range(4):
        assert lat[i][j].tensor_transpose(delta_s[i][j].legs).shape == delta_s[i][j].shape

print(e_s)
