import os
import numpy as np
from matplotlib import pyplot as plt

def getAll(base):
    thisData = np.loadtxt(f'{base}/GM.log')
    if os.path.exists(f'{base}/load_from'):
        loadFrom = os.path.realpath(f'{base}/load_from')
        nextBase, offSetFile = os.path.split(loadFrom)
        offSet = int(offSetFile[3:-4])
        result = getAll(nextBase)
        result.append((offSet+result[-1][0]+1, thisData))
        return result
    return [(0, thisData)]

def getPoint(base):
    allData = getAll(base)
    data = [i[1].T+[[i[0]],[0]] for i in allData]
    return data

def plotIt(base, acc=-0.16580050716890718,lim=(1.001,0.975),size=1):
    data = getPoint(base)
    fig, ax = plt.subplots(figsize=(18, 12))
    _ = [ax.plot(*i,'.',markersize=size) for i in data]
    ax.axhline(y=acc)
    _ = ax.set_ylim(acc*lim[0],acc*lim[1])