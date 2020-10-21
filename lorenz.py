import numpy as np

def l96(x, F):
    l = np.zeros_like(x)
    l = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + F
    return l

def step(xa, h, F):
    k1 = l96(xa, F)
    k1 = k1 * h

    k2 = l96(xa+k1/2, F)
    k2 = k2 * h

    k3 = l96(xa+k2/2, F)
    k3 = k3 * h

    k4 = l96(xa+k3, F)
    k4 = k4 * h

    return xa + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0

if __name__ == "__main__":
    n = 40
    F = 8.0
    h = 0.05

    x0 = np.ones(n)*F
    x0[19] += 0.001*F
    tmax = 2.0
    nt = int(tmax/h) + 1

    for k in range(nt):
        x0 = step(x0, h, F)
        #x0 = x
    print(x0)
    