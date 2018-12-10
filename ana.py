import os
import numpy as np
from matplotlib import pyplot as plt

def getAll(base):
    thisData = np.loadtxt(f'{base}/GM.log')
    label = " " .join(np.loadtxt(f"{base}/parameter",dtype=str))
    if os.path.exists(f'{base}/load_from'):
        loadFrom = os.path.realpath(f'{base}/load_from')
        nextBase, offSetFile = os.path.split(loadFrom)
        if not os.path.exists(f'{nextBase}/GM.log'):
            return [(0, thisData, label)]
        offSet = int(offSetFile[3:-4])
        result = getAll(nextBase)
        result.append((offSet+result[-1][0]+1, thisData, label))
        return result
    return [(0, thisData, label)]

def getPoint(base):
    allData = getAll(base)
    data = [(i[1].T[0]+i[0], i[1].T[1], i[2]) for i in allData]
    return data

def plotIt(ax, base, acc=-0.16580050716890718,lim=(1.001,0.975),size=1):
    data = getPoint(base)
    [ax.plot(i[0], i[1] ,'.',markersize=size, label=i[2]) for i in data]
    ax.legend(loc='upper right', markerscale=10/size)
    ax.axhline(y=acc)
    ax.axhline(y=acc*0.999)
    ax.set_ylim(acc*lim[0],acc*lim[1])