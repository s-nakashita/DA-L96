import numpy as np
import matplotlib.pyplot as plt

# Reference: Marchant and Smyth 2002

def derivative(p, dx):
    f = np.zeros_like(p)
    f[2:-2] = -3.0 * p[2:-2] * (p[3:-1] - p[1:-3]) / dx \
        + (0.5 * p[4:] - p[3:-1] + p[1:-3] - 0.5 * p[:-4]) / dx**3
    return df


def step(u, dx, dt):
    a = derivative(u, dx)
    b = derivative(u + 0.5 * a)
    c = derivative(u + 0.5 * b)
    d = derivative(u + c)
    return u + (a + 2 * b + 2 * c + d) / 6


if __name__ == "__main__":
    n = 1011
    nu = 1.0
    dt = 0.002
    tmax = 2.0
    tsave = 0.5
    reynolds_number = 1.0

    x = np.linspace(-np.pi, np.pi, n)
    dx = x[1] - x[0]
    u = -reynolds_number * np.sin(x)
    umax = np.amax(np.abs(u))
    c = umax * dt / dx
    nt = int(tmax / dt) + 1

    print("n={} nu={} dt={:7.3e} tmax={} tsave={}".format(n, nu, dt, tmax, tsave))
    print("R={} dx={:7.3e} umax={} c={} nt={}".format(reynolds_number, dx, umax, c, nt))

    np.savetxt("x.txt".format(0), x)
    np.savetxt("u{:05d}.txt".format(0), u)
    fig, ax = plt.subplots()
    for k in range(nt):
        print("step {:05d}".format(k))
        if k * dt % tsave == 0:
                np.savetxt("u{:05d}.txt".format(k), u)
                ax.plot(x, u)
        u = step(u, dx, dt, nu)
    fig.savefig("u.png")
