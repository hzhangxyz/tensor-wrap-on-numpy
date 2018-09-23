from square_lattice import auxiliary_generate as ag
import numpy_wrap as np

former1 = np.tensor(np.arange(4*4).reshape([4, 4]), legs=['d', 'r'])
former2 = np.tensor(np.arange(4*4*4).reshape([4, 4, 4])+10, legs=['d', 'l', 'r'])
former3 = np.tensor(np.arange(4*4*4).reshape([4, 4, 4])+10, legs=['d', 'l', 'r'])
former4 = np.tensor(np.arange(4*4*4).reshape([4, 4, 4])+10, legs=['d', 'l', 'r'])
former5 = np.tensor(np.arange(4*4*4).reshape([4, 4, 4])+10, legs=['d', 'l', 'r'])
former6 = np.tensor(np.arange(4*4).reshape([4, 4])+20, legs=['d', 'l'])

former = [former1, former2, former3, former4, former5, former6]


current1 = np.tensor(np.arange(4*4*4).reshape([4, 4, 4]), legs=['d', 'r', 'u'])
current2 = np.tensor(np.arange(4*4*4*4).reshape([4, 4, 4, 4])+10, legs=['d', 'l', 'r', 'u'])
current3 = np.tensor(np.arange(4*4*4*4).reshape([4, 4, 4, 4])+10, legs=['d', 'l', 'r', 'u'])
current4 = np.tensor(np.arange(4*4*4*4).reshape([4, 4, 4, 4])+10, legs=['d', 'l', 'r', 'u'])
current5 = np.tensor(np.arange(4*4*4*4).reshape([4, 4, 4, 4])+10, legs=['d', 'l', 'r', 'u'])
current6 = np.tensor(np.arange(4*4*4).reshape([4, 4, 4])+20, legs=['d', 'l', 'u'])

current = [current1, current2, current3, current4, current5, current6]

initial1 = np.tensor(np.arange(4*8).reshape([4, 8]), legs=['d', 'r'])
initial2 = np.tensor(np.arange(4*8*8).reshape([4, 8, 8])+10, legs=['d', 'l', 'r'])
initial3 = np.tensor(np.arange(4*8*8).reshape([4, 8, 8])+10, legs=['d', 'l', 'r'])
initial4 = np.tensor(np.arange(4*8*8).reshape([4, 8, 8])+10, legs=['d', 'l', 'r'])
initial5 = np.tensor(np.arange(4*8*8).reshape([4, 8, 8])+10, legs=['d', 'l', 'r'])
initial6 = np.tensor(np.arange(4*8).reshape([4, 8])+20, legs=['d', 'l'])

initial = [initial1, initial2, initial3, initial4, initial5, initial6]

res = ag(6, former, current, initial, scan_time=4)

print(res)
print([i.shape for i in res])
