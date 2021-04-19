import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import obs
from var import analysis as analysis_var
from mlef05 import precondition

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, method="CG", cgtype=None,
        maxiter=None, maxrest=20, restart=False,
        disp=False, save_hist=False, save_dh=False, 
        infl=False, loc = False, infl_parm=1.0, model="z08", icycle=0):
    op = htype["operator"]
    pt = htype["perturbation"]
    ga = htype["gamma"]

    nmem = xf.shape[1]
    spf = (xf - xc[:, None])/np.sqrt(nmem-1)
    pf = spf @ spf.T
    logger.debug("spf shape={}".format(spf.shape))
    logger.debug("pf shape={}".format(pf.shape))
    dpf = np.ones(pf.shape[0])*1e-10
    for j in range(len(dpf)):
        if np.diag(pf)[j] > 1e-10:
            dpf[j] = np.diag(pf)[j]
    binv = np.diag(1.0/dpf)
    #u, s, vt = la.svd(spf)
    #binv = u[:, :nmem] @ np.diag(1.0/s**2) @ u[:, :nmem].T

    xa, chi2 = analysis_var(xc, binv, y, rinv, htype, 
        gtol=gtol, method=method, cgtype=cgtype, 
        maxiter=maxiter, disp=disp, 
        save_hist=save_hist, save_dh=save_dh, 
        model=model, icycle=icycle)

    dh = obs.dhdx(xa, op, ga) @ spf
    zmat = rmat @ dh
    tmat, heinv = precondition(zmat)
    spa = spf @ tmat 
    pa = spa @ spa.T
    ua = xa[:, None] + spa * np.sqrt(nmem-1)
    if save_dh:
        logger.info("save ua")
        #ua = np.zeros((xa.size,spf.shape[1]+1))
        #ua[:,0] = xa
        #ua[:,1:] = xa[:, None] + spa
        np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ua)
    
    #return xa, spa, chi2
    return ua, pa, chi2