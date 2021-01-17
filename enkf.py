import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs


#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger(__name__)


def analysis(xf, xf_, y, sig, dx, htype, infl=False, loc=False, tlm=True, \
    save_dh=False, model="z08", icycle=0):
    op = htype["operator"]
    da = htype["perturbation"]
    JH = obs.dhdx(xf_, op)
    R = np.eye(y.size)*sig*sig
    rinv = np.eye(y.size)/sig/sig
    nmem = xf.shape[1]
    dxf = xf - xf_[:,None]
    if tlm:
        dy = JH @ dxf
    else:
        dy = obs.h_operator(xf, op) - obs.h_operator(xf_, op)[:, None]
    if save_dh:
        print("save dxf")
        np.save("{}_dxf_{}_{}_cycle{}.npy".format(model, op, da, icycle), dxf)
        np.save("{}_dhdx_{}_{}_cycle{}.npy".format(model, op, da, icycle), JH)
        np.save("{}_dy_{}_{}_cycle{}.npy".format(model, op, da, icycle), dy)
        ob = y - obs.h_operator(xf_, op)
        np.save("{}_d_{}_{}_cycle{}.npy".format(model, op, da, icycle), ob)
    print("save_dh={} cycle{}".format(save_dh, icycle))

    # inflation parameter
    if op == "linear":
        alpha = 1.2
    elif op == "quadratic" or op == "quadratic-nodiff":
        #alpha = 1.35
        alpha = 1.2
    elif op == "cubic" or op == "cubic-nodiff":
        #alpha = 1.65
        alpha = 1.2
    if infl:
        if da != "letkf":
            print("==inflation==, alpha={}".format(alpha))
            dxf *= alpha
    pf = dxf @ dxf.T / (nmem-1)
    if save_dh:
        print("save pf")
        np.save("{}_pf_{}_{}_cycle{}.npy".format(model, op, da, icycle), pf)
    #if loc: # B-localization
    #    if da == "etkf":
    #        print("==B-localization==")
    #        dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=xf_.size)
    #        pf = pf * l_mat
    d = y - obs.h_operator(xf_, op)
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if da == "etkf":
        #if tlm:
        #    K1 = pf @ JH.T
        #    K2 = JH @ pf @ JH.T + R
        #    #K = pf @ JH.T @ la.inv(JH @ pf @ JH.T + R)
        #else:
        """
        K1 = dxf @ dy.T / (nmem-1)
        K2 = dy @ dy.T / (nmem-1) + R
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
        K2inv = la.inv(K2)
        K = K1 @ K2inv
        if save_dh:
            print("save K")
            np.save("{}_K_{}_{}_cycle{}.npy".format(model, op, da, icycle), K)
            np.save("{}_K1_{}_{}_cycle{}.npy".format(model, op, da, icycle), K1)
            np.save("{}_K2_{}_{}_cycle{}.npy".format(model, op, da, icycle), K2)
            np.save("{}_K2i_{}_{}_cycle{}.npy".format(model, op, da, icycle), K2inv)
        if loc: # K-localization
            print("==K-localization==")
            dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=y.size)
            #print(l_mat)
            K = K * l_mat
            #print(K)
            if save_dh:
                np.save("{}_Kloc_{}_{}_cycle{}.npy".format(model, op, da, icycle), K)
        #xa_ = xf_ + K @ d
        if save_dh:
            print("save increment")
            np.save("{}_dx_{}_{}_cycle{}.npy".format(model, op, da, icycle), K@d)
        #TT = la.inv( np.eye(nmem) + dy.T @ la.inv(R) @ dy / (nmem-1) )
        #lam, v = la.eigh(TT)
        #print("eigenvalue(sqrt)={}".format(np.sqrt(lam)))
        #D = np.diag(np.sqrt(lam))
        #T = v @ D
        """        
        A = (nmem-1)*np.eye(nmem) + dy.T @ rinv @ dy
        #TT = la.inv( np.eye(nmem) + dy.T @ rinv @ dy / (nmem-1) )
        lam, v = la.eigh(A)
        print("eigenvalue(sqrt)={}".format(np.sqrt(lam)))
        Dinv = np.diag(1.0/lam)
        TT = v @ Dinv @ v.T
        T = np.sqrt(nmem-1) * v @ np.sqrt(Dinv) @ v.T

        K = dxf @ TT @ dy.T @ rinv
        if save_dh:
            print("save K")
            np.save("{}_K_{}_{}_cycle{}.npy".format(model, op, da, icycle), K)
        if loc: # K-localization
            print("==K-localization==")
            dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=y.size)
            #print(l_mat)
            K = K * l_mat
            #print(K)
            if save_dh:
                np.save("{}_Kloc_{}_{}_cycle{}.npy".format(model, op, da, icycle), K)
        if save_dh:
            print("save increment")
            np.save("{}_dx_{}_{}_cycle{}.npy".format(model, op, da, icycle), K@d)
        xa_ = xf_ + K @ d

        dxa = dxf @ T
        xa = dxa + xa_[:,None]
        if save_dh:
            print("save ua")
            ua = np.zeros((xa_.size,nmem+1))
            ua[:,0] = xa_
            ua[:,1:] = xa
            np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, da, icycle), ua)
            #print("xa_={}".format(xa_))
            #print("xa={}".format(xa))
        pa = dxa@dxa.T/(nmem-1)

    elif da=="po":
        Y = np.zeros((y.size,nmem))
        np.random.seed(514)
        err = np.random.normal(0, scale=sig, size=Y.size)
        err_ = np.mean(err.reshape(Y.shape), axis=1)
        Y = y[:,None] + err.reshape(Y.shape)
        d_ = y + err_ - obs.h_operator(xf_, op)
        if tlm:
            K = pf @ JH.T @ la.inv(JH @ pf @ JH.T + R)
        else:
            K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
        if loc:
            print("==localization==")
            dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=y.size)
            K = K * l_mat
        xa_ = xf_ + K @ d_
        if tlm:
            xa = xf + K @ (Y - JH @ xf)
            pa = (np.eye(xf_.size) - K @ JH) @ pf
        else:
            HX = obs.h_operator(xf, op)
            xa = xf + K @ (Y - HX)
            pa = pf - K @ dy @ dxf.T / (nmem-1)

    elif da=="srf":
        I = np.eye(xf_.size)
        #p0 = np.zeros_like(pf)
        #p0 = pf[:,:]
        dx0 = np.zeros_like(dxf)
        dx0 = dxf[:,:]
        dy0 = np.zeros_like(dy)
        dy0 = dy[:,:]
        x0_ = np.zeros_like(xf_)
        x0_ = xf_[:]
        d0 = np.zeros_like(d)
        d0 = d[:]
        if loc:
            print("==localization==")
            dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=y.size)
        for i in range(y.size):
            #hrow = JH[i].reshape(1,-1)
            dyi = dy0[i,:].reshape(1,-1)
            #d1 = hrow @ p0 @ hrow.T + sig*sig
            d1 = dyi @ dyi.T + sig*sig
            #k1 = p0 @ hrow.T /d1
            k1 = dx0 @ dyi.T /d1
            k1_ = k1 / (1.0 + sig/np.sqrt(d1))
            if loc:
                k1_ = k1_ * l_mat[:,i].reshape(k1_.shape)
            #xa_ = x0_.reshape(k1_.shape) + k1_ * (y[i] - hrow@x0_)
            xa_ = x0_.reshape(k1_.shape) + k1_ * d0[i]
            #dxa = (I - k1_@hrow) @ dx0
            dxa = dx0 - k1_ @ dyi

            x0_ = xa_[:]
            dx0 = dxa[:,:]
            dy0 = obs.h_operator(x0_+dx0, op) - obs.h_operator(x0_, op)
            d0 = y[:,None] - obs.h_operator(x0_, op)
            #p0 = pa[:,:]
            print(x0_.shape)
            print(dx0.shape)
            print(dy0.shape)
            print(d0.shape)
            #print(p0.shape)
        pa = dxa@dxa.T/(nmem-1)
        xa = dxa + xa_
        xa_ = np.squeeze(xa_)

    elif da=="letkf":
        #r0 = dx * 10.0 # local observation max distance
        sigma = 7.5
        #r0 = 2.0 * np.sqrt(10.0/3.0) * sigma
        r0 = 100.0 # all
        if loc:
            r0 = 5.0
        nx = xf_.size
        dist, l_mat = loc_mat(sigma, nx, ny=y.size)
        print(dist[0])
        xa = np.zeros_like(xf)
        xa_ = np.zeros_like(xf_)
        dxa = np.zeros_like(dxf)
        E = np.eye(nmem)
        hx = obs.h_operator(xf_, op)
        if infl:
            print("==inflation==")
            E /= alpha
        for i in range(nx):
            far = np.arange(y.size)
            far = far[dist[i]>r0]
            print("number of assimilated obs.={}".format(y.size - len(far)))
            yi = np.delete(y,far)
            if tlm:
                Hi = np.delete(JH,far,axis=0)
                di = yi - Hi @ xf_
                dyi = Hi @ dxf
            else:
                hxi = np.delete(hx,far)
                di = yi - hxi
                dyi = np.delete(dy,far,axis=0)
            Ri = np.delete(R,far,axis=0)
            Ri = np.delete(Ri,far,axis=1)
            if loc:
                print("==localization==")
                diagR = np.diag(Ri)
                l = np.delete(l_mat[i],far)
                Ri = np.diag(diagR/l)
            R_inv = la.inv(Ri)
            
            A = (nmem-1)*E + dyi.T @ R_inv @ dyi
            lam,v = la.eigh(A)
            D_inv = np.diag(1.0/lam)
            pa_ = v @ D_inv @ v.T
            
            xa_[i] = xf_[i] + dxf[i] @ pa_ @ dyi.T @ R_inv @ di
            sqrtPa = v @ np.sqrt(D_inv) @ v.T * np.sqrt(nmem-1)
            dxa[i] = dxf[i] @ sqrtPa
            xa[i] = np.full(nmem,xa_[i]) + dxa[i]
        pa = dxa@dxa.T/(nmem-1)
    if save_dh:
        print("save pa")
        np.save("{}_pa_{}_{}_cycle{}.npy".format(model, op, da, icycle), pa)

    innv = y - obs.h_operator(xa_, op)
    p = innv.size
    G = JH @ pf @ JH.T + R 
    chi2 = innv.T @ la.inv(G) @ innv / p
    ds = dof(dy, sig, nmem)
    print(ds)
    #if len(innv.shape) > 1:
    #    Rsim = innv.reshape(innv.size,1) @ d[None,:]
    #else:
    #    Rsim = innv[:,None] @ d[None,:]
    #chi2 = np.mean(np.diag(Rsim)) / sig / sig

    return xa, xa_, pa, chi2, ds

def loc_mat(sigma, nx, ny):
    dist = np.zeros((nx,ny))
    l_mat = np.zeros_like(dist)
    for j in range(nx):
        for i in range(ny):
            dist[j,i] = min(abs(j-i),nx-abs(j-i))
    d0 = 2.0 * np.sqrt(10.0/3.0) * sigma
    l_mat = np.exp(-dist**2/(2.0*sigma**2))
    l_mat[dist>d0] = 0
    return dist, l_mat 

def dof(dy, sig, nmem):
    zmat = dy / sig
    u, s, vt = la.svd(zmat)
    ds = np.sum(s**2/(1.0+s**2))/(nmem-1)
    return ds