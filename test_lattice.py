import square_lattice as sl
import numpy_wrap as np

a = sl.SquareLattice(4, 6, 6, 8, 2)

b = list(map(lambda x:x.tolist(),a.spin.test_w_s()))
print(b)
print(np.std(b)/np.mean(b))

lat = a.spin.lat
e_s, delta_s = a.spin.cal_E_s_and_Delta_s()

for i in range(4):
    for j in range(4):
        print(lat[i][j].legs,delta_s[i][j].legs)
        print(lat[i][j].shape,delta_s[i][j].shape)
