import square_lattice as sl
import numpy_wrap as np

a = sl.SquareLattice(4, 6, 6, 8, 2)

e, grad = a.markov_chain()
print(e)
#print(grad)

