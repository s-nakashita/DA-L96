import numpy as np
import sys

obs_s = float(sys.argv[1])
#iobs = -int(np.log10(obs_s))
iobs = int(obs_s*1e4)
print(str(iobs).zfill(4))