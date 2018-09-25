#!/usr/bin/env python
import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt

first_name = name = "./run/last/"
data = []
index = None
while os.path.exists(f'{name}/load_from'):
    name, index = os.path.split(os.path.realpath(f'{name}/load_from'))
    index = int(index[:-4])
    print(f'{name}/log')
    tmp = np.loadtxt(f'{name}/log')
    data.append(tmp[:index])

data = np.concatenate(data[::-1])[:,1]

def handle_close(evt):
    exit()

plt.ion()
fig = plt.figure()
fig.canvas.mpl_connect('close_event', handle_close)
ax = fig.add_subplot(111)
#ax.axis([0, len(data), -0.6, 0.5])
now_data = np.loadtxt(f'{first_name}/log')[:,1]
line, = ax.plot([*data,*now_data])
base_len = len(now_data)
while True:
    now_data = np.loadtxt(f'{first_name}/log')[:,1]
    delta = len(now_data) - base_len
    line.set_ydata([*data[delta:],*now_data])
    plt.pause(1)
