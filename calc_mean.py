import sys
import numpy as np

vname = sys.argv[1]
na = int(sys.argv[2])
ivar = sys.argv[3]
nmax = int(sys.argv[4])
#if vname == "chi":
#    emean = np.zeros(na)
#else:
#    emean = np.zeros(na+1)
for count in range(1,nmax+1):
    e = np.loadtxt("{}{}_{}.txt".format(vname, ivar, count))
    if count == 1:
        emean = np.zeros_like(e)
    emean += e
emean /= nmax
np.savetxt("{}{}_mean.txt".format(vname,ivar), emean)