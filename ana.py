#!/usr/bin/env python
import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt

name = "./run/last/"

def handle_close(evt):
    exit()

plt.ion()
fig = plt.figure()
fig.canvas.mpl_connect('close_event', handle_close)
ax = fig.add_subplot(111)
#ax.axis([0, len(data), -0.6, 0.5])
data = np.loadtxt(f'{name}/SU.log')[:,1]
line, = ax.plot(data)
base_len = len(data)
while True:
    data = np.loadtxt(f'{name}/SU.log')[:,1]
    line.set_ydata(data[-base_len:])
    plt.pause(1)
