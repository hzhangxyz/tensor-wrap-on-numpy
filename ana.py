import sys
import os
import numpy as np
from matplotlib import pyplot as plt

name = "./run/last/"
data = []
index = None
while True:
    print(f'{name}/log')
    tmp = np.loadtxt(f'{name}/log')
    data.append(tmp[:index])
    if not os.path.exists(f'{name}/load_from'):
        break
    name, index = os.path.split(os.path.realpath(f'{name}/load_from'))
    index = int(index[:-4])

data = np.concatenate(data[::-1])[:,1]

plt.plot(data)
plt.axis([0, len(data), -0.5, 0.5])
plt.show()
