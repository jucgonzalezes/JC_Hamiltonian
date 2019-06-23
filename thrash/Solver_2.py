# Integración con los coeficientes dependientes del tiempo

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

y0, t0 = [0.0+0.0j, 1.0+0.0j], 0.0


wm = 2. * np.pi * 0.1
Am = 2. * np.pi * 250.


def time_dep(t):
    return np.sin(t)


wt = {'coeff1': lambda x: time_dep(wm*x)}


def f(t, y, k, l, m):
    return [(k - Am*wm*1.0j * wt['coeff1'](t))*y[0] + l*y[1], l*y[0]+m*y[1]]


def jac(t, y, k, l, m):
    return [[(k - Am*wm*1.0j * wt['coeff1'](t)), l], [l, m]]


wc = - 2. * np.pi * 1.0j * 1000.
g = - 2. * np.pi * 1.0j * 1.
wx = - 2. * np.pi * 1.0j * 1000.

r = ode(f, jac).set_integrator('zvode', method='adams', with_jacobian=True, rtol=1e-16)
r.set_initial_value(y0, t0).set_f_params(wc, g, wx).set_jac_params(wc, g, wx)
t1 = 50.
dt = t1/10_000.

C1 = [np.abs(np.power(y0[0], 2))]
C2 = [np.abs(np.power(y0[1], 2))]
tf = [t0]

print(C1, C2)

while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    # print("{}, {}".format(r.t, r.y))
    C1.append(np.abs(r.y[0]**2))
    C2.append(np.abs(r.y[1]**2))
    tf.append(r.t)
    print(r.t)

# tf = np.arange(t0, t1+dt, dt)
# tf = np.arange(t0,2803,dt*100)
Traza = [C1[j]+C2[j] for j in range(len(C1))]

print(len(tf))
print(len(C1))

# print(np.linspace(0, 3, num=4))

plt.plot(tf, C1)
plt.plot(tf, C2)
# plt.show()

plt.savefig('Calibracion.png')

plt.plot(tf, Traza)
plt.ylim([0, 1.1])
plt.show()

print(Traza[-1])
