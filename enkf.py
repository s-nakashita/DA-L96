import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs
#from obs2 import Obs

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')


def analysis(xf, xf_, y, sig, dx, htype, infl=False, loc=False, tlm=True, infl_parm=1.1, \
    save_dh=False, model="z08", icycle=0):
    op = htype["operator"]
    da = htype["perturbation"]
    ga = htype["gamma"]
    #obs = Obs(op, sig)
    JH = obs.dhdx(xf_, op, ga)
    #JH = obs.dh_operator(y[:,0], xf_)
    R = np.eye(y.size)*sig*sig
    rinv = np.eye(y.size)/sig/sig
    #R, rmat, rinv = obs.set_r(y.shape[0])
    nmem = xf.shape[1]
    dxf = xf - xf_[:,None]
    logger.debug(dxf.shape)
    if tlm:
        dy = JH @ dxf
        #dy = np.zeros((y.size,nmem))
        #for i in range(nmem):
        #    jhi = obs.dhdx(xf[:,i], op, ga)
        #    dy[:,i] = jhi @ dxf[:,i]
    else:
        #dy = obs.h_operator(xf, op, ga) - np.mean(obs.h_operator(xf, op, ga), axis=1)[:, None]
        #dy = obs.h_operator(y[:,0], xf) - np.mean(obs.h_operator(y[:,0], xf), axis=1)[:, None]
        dy = obs.h_operator(xf, op, ga) - obs.h_operator(xf_, op, ga)[:, None]
    d = y - np.mean(obs.h_operator(xf, op, ga), axis=1)
    #d = y[:,1] - np.mean(obs.h_operator(y[:,0], xf), axis=1)
    #d = y - obs.h_operator(xf_, op, ga)
    
    hes = np.eye(nmem) + dy.T @ rinv @ dy
    condh = la.cond(hes)
    # inflation parameter
    """
    if op == "linear" or op == "test":
        alpha = 1.2
    elif op == "quadratic" or op == "quadratic-nodiff":
        #alpha = 1.35
        alpha = 1.2
    elif op == "cubic" or op == "cubic-nodiff":
        #alpha = 1.65
        alpha = 1.2
    """
    alpha = infl_parm
    if infl:
        if da != "letkf" and da != "etkf":
            logger.info("==inflation==, alpha={}".format(alpha))
            dxf *= alpha
    pf = dxf @ dxf.T / (nmem-1)
    if save_dh:
        #logger.info("save pf")
        np.save("{}_pf_{}_{}_cycle{}.npy".format(model, op, da, icycle), pf)
    #if loc: # B-localization
    #    if da == "etkf":
    #        print("==B-localization==")
    #        dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=xf_.size)
    #        pf = pf * l_mat
    if save_dh:
        #logger.info("save dxf")
        np.save("{}_dxf_{}_{}_cycle{}.npy".format(model, op, da, icycle), dxf)
        np.save("{}_dhdx_{}_{}_cycle{}.npy".format(model, op, da, icycle), JH)
        np.save("{}_dh_{}_{}_cycle{}.npy".format(model, op, da, icycle), dy)
        np.save("{}_d_{}_{}_cycle{}.npy".format(model, op, da, icycle), d)
    logger.info("save_dh={} cycle{}".format(save_dh, icycle))

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
        if infl:
            logger.info("==inflation==, alpha={}".format(alpha))
            A = np.eye(nmem) / alpha 
        else:
            A = np.eye(nmem)      
        A = (nmem-1)*A + dy.T @ rinv @ dy
        #TT = la.inv( np.eye(nmem) + dy.T @ rinv @ dy / (nmem-1) )
        lam, v = la.eigh(A)
        #print("eigenvalue(sqrt)={}".format(np.sqrt(lam)))
        Dinv = np.diag(1.0/lam)
        TT = v @ Dinv @ v.T
        T = v @ np.sqrt((nmem-1)*Dinv) @ v.T

        K = dxf @ TT @ dy.T @ rinv
        if save_dh:
            #logger.info("save K")
            np.save("{}_K_{}_{}_cycle{}.npy".format(model, op, da, icycle), K)
        if loc: # K-localization
            logger.info("==K-localization==")
            dist, l_mat = loc_mat(sigma=4.0, nx=xf_.size, ny=y.size)
            #print(l_mat)
            K = K * l_mat
            #print(K)
            if save_dh:
                np.save("{}_Kloc_{}_{}_cycle{}.npy".format(model, op, da, icycle), K)
        if save_dh:
            #logger.info("save increment")
            np.save("{}_dx_{}_{}_cycle{}.npy".format(model, op, da, icycle), K@d)
        xa_ = xf_ + K @ d

        dxa = dxf @ T
        xa = dxa + xa_[:,None]
        pa = dxa@dxa.T/(nmem-1)

    elif da=="po":
        Y = np.zeros((y.size,nmem))
        #Y = np.zeros((y.shape[0],nmem))
        #np.random.seed(514)
        #err = np.random.normal(0, scale=sig, size=Y.size)
        #err_ = np.mean(err.reshape(Y.shape), axis=1)
        err = np.zeros_like(Y)
        for j in range(nmem):
            err[:, j] = np.random.normal(0.0, scale=sig, size=err.shape[0])
        err_ = np.mean(err, axis=1)
        stdv = np.sqrt(np.mean((err-err_[:, None])**2, axis=1))
        logger.debug("err mean {}".format(err_))
        logger.debug("err stdv {}".format(stdv))
        #Y = y[:,None] + err.reshape(Y.shape)
        Y = y[:,None] + err
        #Y = y[:,1].reshape(-1,1) + err.reshape(Y.shape)
        #d_ = y + err_ - obs.h_operator(xf_, op, ga)
        d_ = d #+ err_
        if tlm:
            K = pf @ JH.T @ la.inv(JH @ pf @ JH.T + R)
        else:
            K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
        if loc:
            logger.info("==localization==")
            dist, l_mat = loc_mat(sigma=4.0, nx=xf_.size, ny=y.size)
            K = K * l_mat
        xa_ = xf_ + K @ d_
        if tlm:
            xa = xf + K @ (Y - JH @ xf)
            pa = (np.eye(xf_.size) - K @ JH) @ pf
        else:
            HX = obs.h_operator(xf, op, ga)
            #HX = obs.h_operator(y[:,0], xf)
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
            logger.info("==localization==")
            dist, l_mat = loc_mat(sigma=4.0, nx=xf_.size, ny=y.size)
        for i in range(y.size):
        #for i in range(y.shape[0]):
            #hrow = JH[i].reshape(1,-1)
            dyi = dy0[i,:].reshape(1,-1)
            #d1 = hrow @ p0 @ hrow.T + sig*sig
            d1 = dyi @ dyi.T/(nmem-1) + sig*sig
            #k1 = p0 @ hrow.T /d1
            k1 = dx0 @ dyi.T /d1/(nmem-1)
            k1_ = k1 / (1.0 + sig/np.sqrt(d1))
            if loc:
                k1_ = k1_ * l_mat[:,i].reshape(k1_.shape)
            #xa_ = x0_.reshape(k1_.shape) + k1_ * (y[i] - hrow@x0_)
            #xa_ = x0_.reshape(k1_.shape) + k1_ * d0[i]
            xa_ = x0_.reshape(k1.shape) + k1 * d0[i]
            #dxa = (I - k1_@hrow) @ dx0
            dxa = dx0 - k1_ @ dyi

            x0_ = xa_[:]
            dx0 = dxa[:,:]
            x0 = x0_ + dx0
            if tlm:
                logger.debug(x0_.shape)
                logger.debug(obs.dhdx(np.squeeze(x0_), op, ga).shape)
                logger.debug(dx0.shape)
                dy0 = obs.dhdx(np.squeeze(x0_), op, ga) @ dx0
                #dy0 = obs.dh_operator(y[:,0], x0_) @ dx0
                #dy0 = np.zeros((y.size,nmem))
                #for i in range(nmem):
                #    jhi = obs.dhdx(x0[:,i], op, ga)
                #    dy0[:,i] = jhi @ dx0[:,i]
            else:
                dy0 = obs.h_operator(x0, op, ga) - np.mean(obs.h_operator(x0, op, ga), axis=1)[:, None]
                #dy0 = obs.h_operator(y[:,0], x0) - np.mean(obs.h_operator(y[:,0], x0), axis=1)[:, None]
                #dy0 = obs.h_operator(x0, op, ga) - obs.h_operator(x0_, op, ga)#[:, None]
            d0 = y - np.mean(obs.h_operator(x0, op, ga), axis=1)
            #d0 = y[:,1] - np.mean(obs.h_operator(y[:,0], x0), axis=1)
            #d0 = y - np.squeeze(obs.h_operator(x0_, op, ga))
            #p0 = pa[:,:]
            #print(x0_.shape)
            #print(dx0.shape)
            #print(dy0.shape)
            #print(d0.shape)
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
        #dist, l_mat = loc_mat(sigma, nx, ny=y.shape[0])
        logger.debug(dist[0])
        xa = np.zeros_like(xf)
        xa_ = np.zeros_like(xf_)
        dxa = np.zeros_like(dxf)
        E = np.eye(nmem)
        #hx = obs.h_operator(xf_, op, ga)
        hx = np.mean(obs.h_operator(xf, op, ga), axis=1)
        #hx = np.mean(obs.h_operator(y[:,0], xf), axis=1)
        if infl:
            logger.info("==inflation==")
            E /= alpha
        for i in range(nx):
            far = np.arange(y.size)
            #far = np.arange(y.shape[0])
            far = far[dist[i]>r0]
            logger.info("number of assimilated obs.={}".format(y.size - len(far)))
            #print("number of assimilated obs.={}".format(y.shape[0] - len(far)))
            #yi = np.delete(y,far)
            #if tlm:
            #    Hi = np.delete(JH,far,axis=0)
            #    di = yi - Hi @ xf_
            #    dyi = Hi @ dxf
            #else:
            #    hxi = np.delete(hx,far)
            #    di = yi - hxi
            dyi = np.delete(dy,far,axis=0)
            di = np.delete(d,far)
            Ri = np.delete(R,far,axis=0)
            Ri = np.delete(Ri,far,axis=1)
            if loc:
                logger.info("==localization==")
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
        #logger.info("save pa")
        np.save("{}_pa_{}_{}_cycle{}.npy".format(model, op, da, icycle), pa)
        np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, da, icycle), xa)
        
    #innv = y - obs.h_operator(xa_, op, ga)
    innv = y - np.mean(obs.h_operator(xa, op, ga), axis=1)
    #innv = y[:,1] - np.mean(obs.h_operator(y[:,0], xa), axis=1)
    p = innv.size
    G = dy @ dy.T + R 
    chi2 = innv.T @ la.inv(G) @ innv / p
    ds = dof(dy, sig, nmem)
    logger.info("ds={}".format(ds))
    #if len(innv.shape) > 1:
    #    Rsim = innv.reshape(innv.size,1) @ d[None,:]
    #else:
    #    Rsim = innv[:,None] @ d[None,:]
    #chi2 = np.mean(np.diag(Rsim)) / sig / sig

    return xa, xa_, pa, chi2, ds, condh

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