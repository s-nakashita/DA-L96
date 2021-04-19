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
def plot_wind(xf, xf_, xa, xa_, y, sig, htype, var):
    theta = np.linspace(0.0, 2.0*np.pi, 360)
    rmin = y[0] - sig
    rmax = y[0] + sig
    xmin = rmin*np.cos(theta)
    ymin = rmin*np.sin(theta)
    xmax = rmax*np.cos(theta)
    ymax = rmax*np.sin(theta)
    umin = y[1] - sig
    umax = y[1] + sig

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

    ax[1].plot(xmin, ymin, color="black")
    ax[1].plot(xmax, ymax, color="black")
    ax[1].axvline(x=umin, color="black", linestyle="dashed")
    ax[1].axvline(x=umax, color="black", linestyle="dashed")
    ax[1].scatter(xa[0,:], xa[1,:], s=10, marker='*', zorder=2.5)
    ax[1].scatter(xa_[0], xa_[1], s=30, marker='^', zorder=2.5)
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

    sa = obs.h_operator(xa, htype["operator"])
    sa -= y[0]
    vlim = 3.0 #max(np.max(sa), -np.min(sa))
    plt.hist(sa[0], bins=100, density=True, range=(-vlim,vlim))
    plt.title(r"$y-H(x^a_i)$")
    plt.savefig("{}_hist_{}.png".format(htype["perturbation"], var))
    plt.close()