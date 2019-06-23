from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Solver:
    y0, t0 = [0.0 + 0.0j, 1.0 + 0.0j], 0.0

    def __init__(self, wm=None, am=None, wc=None, g=None, wx=None, nsteps=8_000., time=20.):
        self.wm = wm
        self.am = am
        self.wc = - 1.0j * wc
        self.g = - 1.0j * g
        self.wx = - 1.0j * wx
        self.t = time
        self.nsteps = nsteps

        self.wt = {'coeff1': lambda x: self.time_dep(self.wm * x)}

    @staticmethod
    def time_dep(t):
        return np.sin(t)

    def f(self, t, y, k, l, m):
        return [(k - self.am * 1.0j * self.wt['coeff1'](t)) * y[0] + l * y[1], l * y[0] + m * y[1]]

    def jac(self, t, k, l, m):
        return [[(k - self.am * 1.0j * self.wt['coeff1'](t)), l], [l, m]]

    def integrate(self):
        r = ode(self.f, self.jac).set_integrator('zvode', method='adams',
                                                 with_jacobian=True, rtol=1e-16, nsteps=1_000)
        r.set_initial_value(self.y0, self.t0).set_f_params(self.wc, self.g,
                                                           self.wx).set_jac_params(self.wc, self.g, self.wx)
        t1 = self.t
        dt = t1/self.nsteps

        c1 = [np.abs(np.power(self.y0[0], 2))]
        c2 = [np.abs(np.power(self.y0[1], 2))]
        tf = [self.t0]

        pbar = tqdm(total=self.t)

        while r.successful() and r.t < t1:
            r.integrate(r.t + dt)
            c1.append(np.abs(r.y[0] ** 2))
            c2.append(np.abs(r.y[1] ** 2))
            tf.append(r.t)
            pbar.update(dt)

        pbar.close()

        trace = [c1[j] + c2[j] for j in range(len(c1))]

        return c1, c2, tf, trace


def solve(wm, am, wc, g, wx, t):
    stps = 8_000.

    item = [[0, 0], [0, 0], [0, 0]]

    while item[2][-1] < t:
        s = Solver(wm=wm, am=am, wc=wc, g=g, wx=wx, time=t, nsteps=stps)
        item = s.integrate()

        stps += 2_000

    return item

'''
Wm = 0.1
Am = 250.
Wc = 1_000.
G = 1.
Wx = 1_000.
T = 250.

sol = solve(Wm, Am, Wc, G, Wx, T)

plt.plot(sol[2], sol[0])
plt.plot(sol[2], sol[1])
plt.show()

plt.plot(sol[2], sol[3])
plt.show()
'''