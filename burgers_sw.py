import numpy as np
import matplotlib.pyplot as plt

def l_operator(f, u, dx, nu):
# NB u is defined at integral index
# f[i-1/2] = (f[i-1] + f[i]) / 2
# f[i+1/2] = (f[i] + f[i+1]) / 2
# f[i+1/2] - f[i-1/2] = (f[i+1] - f[i-1]) / 2
    l = np.zeros_like(u)
    l[1:-1] = f[1:-1] - 0.5 * nu / dx * (u[2:] - u[0:-2])
    return l


def step(u, dx, dt, nu):
    u1 = np.zeros_like(u)
    u2 = np.zeros_like(u)
    f = 0.5 * u**2
    l1 = l_operator(f, u, dx, nu)
    u1[1:-1] = u[1:-1] - dt * (l1[1:-1] - l1[0:-2]) / dx
    u1[0] = u[0]
    u1[-1] = u[-1]
    f = 0.5 * u**2
    l2 = l_operator(f, u1, dx, nu)
    u2[1:-1] = u1[1:-1] - dt * (l2[2:] - l2[1:-1]) / dx
    u2[0] = u[0]
    u2[-1] = u[-1]
    return 0.5 * (u1 + u2)


if __name__ == "__main__":
    n = 81
    nu = 0.05
    dt = 0.0125
    tmax = 5.0
    tsave = 0.25
    dx = 0.05
    xmax = dx * ((n-1) / 2)
    x = np.linspace(-xmax, xmax, n)
    
    u = np.zeros(n)
    u[0] = 1.0
    nt = int(tmax / dt) + 1

    print("n={} nu={} dt={:7.3e} tmax={} tsave={} nt={}".format(n, nu, dt, tmax, tsave, nt))
    
    np.savetxt("x.txt".format(0), x)
    np.savetxt("u{:05d}.txt".format(0), u)
    fig, ax = plt.subplots()
    for k in range(nt):
        if k * dt % tsave == 0:
            print("step {:05d}".format(k))
            np.savetxt("u{:05d}.txt".format(k), u)
            ax.plot(x, u)
        u = step(u, dx, dt, nu)
    fig.savefig("u.png")
