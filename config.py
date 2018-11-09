import numpy as np

def get_lattice_node_leg(i, j, n, m):
    legs = 'lrud'
    if i == 0:
        legs = legs.replace('u', '')
    if i == n-1:
        legs = legs.replace('d', '')
    if j == 0:
        legs = legs.replace('l', '')
    if j == m-1:
        legs = legs.replace('r', '')
    return legs

Hamiltonian = lambda a,b,c,d: np.array([[ 1, 0, 0, 0],
                        [ 0,-1, 2, 0],
                        [ 0, 2,-1, 0],
                        [ 0, 0, 0, 1]])/4.
