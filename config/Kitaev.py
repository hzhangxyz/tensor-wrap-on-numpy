# Kitaev Model

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
        """
    if (i+j) % 2 == 0:
        legs = legs.replace('l', '')
    else:
        legs = legs.replace('r', '')
        """
    return legs

Sz = np.array([[ 1, 0, 0, 0],
               [ 0,-1, 0, 0],
               [ 0, 0,-1, 0],
               [ 0, 0, 0, 1]])/4.
Sx = np.array([[ 0, 0, 0, 1],
               [ 0, 0, 1, 0],
               [ 0, 1, 0, 0],
               [ 1, 0, 0, 0]])/4.
Sy = np.array([[ 0, 0, 0,-1],
               [ 0, 0, 1, 0],
               [ 0, 1, 0, 0],
               [-1, 0, 0, 0]])/4.
Zero = np.array([[ 0, 0, 0, 0],
                 [ 0, 0, 0, 0],
                 [ 0, 0, 0, 0],
                 [ 0, 0, 0, 0]])/4.

def Hop(a,b,c,d):
    if a == c and (a+b) % 2 == 1:
        return False
    else:
        return True

def Hamiltonian(a,b,c,d):
    if a == c:
        if (a+b) % 2 == 0:
            return Sz
        else:
            return Zero
    else:
        if (a+b) % 2 == 0:
            return Sy
        else:
            return Sx

def default_spin(n, m):
    return np.array([[0 if (i+j) % 2 == 0 else 1 for j in range(m)] for i in range(n)])

np.random.seed(42)
