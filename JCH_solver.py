from Solver import *
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time

start = time()

g = 1.
wc = 1_000.
wx = 1_000.

t = 100.

with open('.log', 'r') as f:
    file = f.read().splitlines()

run_num = eval(file[-1])+1

path = f'Run_{run_num}'
os.makedirs(path)
os.chdir(path)

ami, amf = 10, 1000
fmi, fmf = 0.01, 10

dam = 10
dfm = 0.01

am = ami
fm = fmi

while am <= amf:
    path1 = f'am_{am:03}'
    os.mkdir(path1)
    os.chdir(path1)
    while fm <= fmf:
        print(fm)
        sol = Solver.solve(fm, am, wc, g, wx, t)

        plt.plot(sol[2], sol[0])
        plt.plot(sol[2], sol[1])
        plt.savefig(f'{fm:.2f}.png')
        plt.close()
        plt.plot(sol[2], sol[3])
        plt.savefig(f'Trace_{fm:.2f}.png')

        plt.close()

        fm += dfm
        print(fm)
    am += dam
    fm = fmi

    os.chdir('..')

os.chdir('..')

# with open('.log', 'a+') as f:
#     f.write('{}/n'.format(run_num))

end=time()

print('Elapsed time: {}'.format(end-start))



