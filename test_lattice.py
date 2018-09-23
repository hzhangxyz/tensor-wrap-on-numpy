import square_lattice as sl

a = sl.SquareLattice(4, 4, 4, 8)

print(a.spin.cal_E_s_and_Delta_s())
