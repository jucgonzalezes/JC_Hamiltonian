from Solver import *
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time

start = time()

g = 1.
wc = 1_000.
wx = 1_000.

t = 50.

am = np.arange(10, 1000, 247.5)
wm = np.arange(0.1, 100, 24.975)

print(am)
print(wm)

with open('.log', 'r') as f:
    file = f.read().splitlines()

root = os.getcwd()
print(root)

run_num = eval(file[-1])+1

path = 'Run_{}'.format(run_num)
os.makedirs(path)
os.chdir(path)

for amplitude in am:
    path1 = 'am_{}'.format(amplitude)
    os.mkdir(path1)
    os.chdir(path1)
    for frequency in wm:
        sol = Solver.solve(frequency, amplitude, wc, g, wx, t)

        plt.plot(sol[2], sol[0])
        plt.plot(sol[2], sol[1])
        plt.savefig('{}.png'.format(frequency))
        plt.close()
        plt.plot(sol[2], sol[3])
        plt.savefig('Trace_{}.png'.format(frequency))

        plt.close()
    os.chdir('..')

os.chdir('..')

with open('.log', 'a+') as f:
    f.write('{}/n'.format(run_num))

end=time()

print('Elapsed time: {}'.format(end-start))

