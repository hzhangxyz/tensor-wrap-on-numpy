from square_lattice import auxiliary_generate as ag
import numpy_wrap as np

former1 = np.tensor(np.arange(4*4).reshape([4,4]),legs=['d','r'])
former2 = np.tensor(np.arange(4*4*4).reshape([4,4,4])+10,legs=['d','l','r'])
former3 = np.tensor(np.arange(4*4).reshape([4,4])+20,legs=['d','l'])

former=[former1, former2, former3]


current1 = np.tensor(np.arange(4*4*4).reshape([4,4,4]),legs=['d','r','u'])
current2 = np.tensor(np.arange(4*4*4*4).reshape([4,4,4,4])+10,legs=['d','l','r','u'])
current3 = np.tensor(np.arange(4*4*4).reshape([4,4,4])+20,legs=['d','l','u'])

current=[current1, current2, current3]

initial1 = np.tensor(np.arange(4*8).reshape([4,8]),legs=['d','r'])
initial2 = np.tensor(np.arange(4*8*8).reshape([4,8,8])+10,legs=['d','l','r'])
initial3 = np.tensor(np.arange(4*8).reshape([4,8])+20,legs=['d','l'])

initial=[initial1, initial2, initial3]

res = ag(3,former,current,initial,scan_time=1)

print(res)
