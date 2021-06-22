import matplotlib.pyplot as plt
import numpy as np
import obs
# xf : initial(forecast) ensemble
# xf_ : initial(forecast) control or ensemble mean
# xa : analysis ensemble
# xa_ : analysis control or ensemble mean
# y : observation
# sig : observation error standard deviation
# htype : dictionary of observation type and perturbation
# var : experiment setting (optimization method etc.)
def plot_wind(xf, xf_, xa, xa_, y, sig, ytype, htype, var):
    theta = np.linspace(0.0, 2.0*np.pi, 360)
    ind_speed = ytype.index("speed")
    if y.size > 1:
        ind_u = ytype.index("u")
    rmin = y[ind_speed] - sig
    rmax = y[ind_speed] + sig
    xmin = rmin*np.cos(theta)
    ymin = rmin*np.sin(theta)
    xmax = rmax*np.cos(theta)
    ymax = rmax*np.sin(theta)
    
    x = np.arange(-5,11)
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(xf[0,:], xf[1,:], s=5)
    ax[0].scatter(xf_[0], xf_[1], s=30, marker='^')
    #ax[0].quiver(xf_[0], xf_[1], -djx[0], -djx[1], angles="xy", color="red", scale_units="xy", scale=5.0)
    ax[0].set_xlabel("u")
    ax[0].set_ylabel("v")
    ax[0].set_aspect("equal")
    ax[0].set_xticks(x[::5])
    ax[0].set_yticks(x[::5])
    ax[0].set_xticks(x, minor=True)
    ax[0].set_yticks(x, minor=True)
    ax[0].grid(which="major")
    ax[0].grid(which="minor", linestyle="dashed")
    ax[0].set_title("initial ensemble")

    ax[1].plot(xmin, ymin, color="black", linestyle="dashed")
    ax[1].plot(xmax, ymax, color="black", linestyle="dashed")
    if y.size > 1:
        umin = y[ind_u] - sig
        umax = y[ind_u] + sig
        ax[1].axvline(x=umin, color="black", linestyle="dashed")
        ax[1].axvline(x=umax, color="black", linestyle="dashed")
    ax[1].scatter(xa[0,:], xa[1,:], s=10, marker='*')#, zorder=2.5)
    ax[1].scatter(xa_[0], xa_[1], s=30, marker='^')#, zorder=2.5)
    ax[1].set_xlabel("u")
    ax[1].set_ylabel("v")
    ax[1].set_aspect("equal")
    ax[1].set_xticks(x[::5])
    ax[1].set_yticks(x[::5])
    ax[1].set_xticks(x, minor=True)
    ax[1].set_yticks(x, minor=True)
    ax[1].grid(which="major")
    ax[1].grid(which="minor", linestyle="dashed")
    ax[1].set_title(htype["perturbation"] + " analysis")
    #ax[1].set_title(r"etkf analysis $\mathbf{H}_i \delta \mathbf{X}^f$")
    fig.tight_layout()
    fig.savefig("{}_speed_{}.png".format(htype["perturbation"], var))
    plt.close()

    sb = np.zeros((y.size, xf.shape[1]))
    for i in range(y.size):
        op = ytype[i]
        sb[i] = obs.h_operator(xf, op)
    sb -= y[:, None] #[0]
    vlim = 5.0 #max(np.max(sa), -np.min(sa))
    if y.size <= 1:
        plt.hist(sb[0], bins=100, density=True, range=(-vlim,vlim))
        plt.title(r"$y-H(x^f_i)$")
        plt.savefig("{}_init_hist_{}.png".format(htype["perturbation"], var))
        plt.close()
    else:
        fig, ax = plt.subplots(nrows=2)
        for i in range(y.size):
            ax[i].hist(sb[i], bins=100, density=True, range=(-vlim, vlim))
            ax[i].set_title(ytype[i])
        fig.suptitle(r"$y-H(x^f_i)$")
        fig.tight_layout()
        fig.savefig("{}_init_hist_{}.png".format(htype["perturbation"], var))
        plt.close()
    for i in range(y.size):
        print(ytype[i])
        print("O-B mean={}".format(np.mean(sb[i])))
        stdv = np.sqrt(np.mean(sb[i]**2) - np.mean(sb[i])**2)
        print("O-B stdv={}".format(stdv))

    sa = np.zeros((y.size, xa.shape[1]))
    for i in range(y.size):
        op = ytype[i]
        sa[i] = obs.h_operator(xa, op)
    sa -= y[:, None] #[0]
    vlim = 3.0 #max(np.max(sa), -np.min(sa))
    if y.size <= 1:
        plt.hist(sa[0], bins=100, density=True, range=(-vlim,vlim))
        plt.title(r"$y-H(x^a_i)$")
        plt.savefig("{}_hist_{}.png".format(htype["perturbation"], var))
        plt.close()
    else:
        fig, ax = plt.subplots(nrows=2)
        for i in range(y.size):
            ax[i].hist(sa[i], bins=100, density=True, range=(-vlim, vlim))
            ax[i].set_title(ytype[i])
        fig.suptitle(r"$y-H(x^a_i)$")
        fig.tight_layout()
        fig.savefig("{}_hist_{}.png".format(htype["perturbation"], var))
        plt.close()
    for i in range(y.size):
        print(ytype[i])
        print("O-A mean={}".format(np.mean(sa[i])))
        stdv = np.sqrt(np.mean(sa[i]**2) - np.mean(sa[i])**2)
        print("O-A stdv={}".format(stdv))

    if y.size > 1:
        ut = y[ind_u]
        st = y[ind_speed]
        vtp = np.sqrt(st**2 - ut**2)
        vtm = -vtp 
        print(vtp, vtm)
        tan = xa[0] / xa[1]
        fig, ax = plt.subplots()
        ax.hist(tan, bins=100, density=True, range=(-vlim,vlim))
        ax.axvline(x=1.0/vtp, color="red", linestyle="dashed")
        ax.axvline(x=1.0/vtm, color="red", linestyle="dashed")
        ax.set_title(r"$\tan \theta$")
        fig.savefig("{}_tan_{}.png".format(htype["perturbation"], var))
        plt.close()