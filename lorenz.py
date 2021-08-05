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
    import matplotlib.pyplot as plt
    n = 80
    F = 2.0
    h = 0.05 / 6

    nk = 2.0
    ix = np.arange(n)
    x0 = np.cos(2.0*np.pi*nk*ix/n) * F
    #x0 = np.ones(n)*F
    #x0[19] += 0.001*F
    plt.plot(x0)
    plt.show()
    plt.close()

    tmax = 25.0
    nt = int(tmax/h)
    
    t = []
    X = []
    t.append(0.0)
    X.append(x0)

    for k in range(nt):
        x0 = step(x0, h, F)
        #x0 = x
        X.append(x0)
        t.append((k+1)*h)
    #print(x0)
    t.append(nt*h)
    xs = np.arange(n+1)
    X = np.array(X).reshape(len(t)-1,len(xs)-1)
    plt.pcolor(xs,t,X,cmap='RdBu_r')
    plt.colorbar()
    plt.xlabel('site')
    plt.ylabel('time')
    plt.savefig("contour{:3.1f}_N{}.jpg".format(F, n))
    plt.close()
    plt.plot(X[-1,:])
    plt.show()