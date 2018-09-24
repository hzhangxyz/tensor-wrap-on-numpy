import square_lattice as sl
import numpy_wrap as np

a = sl.SquareLattice(2, 2, D=4, D_c=8, scan_time=2, step_size=0.01, markov_chain_length=100)

a.grad_descent()
