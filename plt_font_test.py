import matplotlib.pyplot as plt
import numpy as np
from numpy import random

t = np.arange(21)
step = np.linspace(0, 20, 401)
obs = random.normal(0.1, scale=0.1, size=t.size)
xa = 0.1*np.exp(0.11*np.abs(t-20.0))
xf = np.zeros_like(xa)
xf[0] = 2.5
g = np.array([1.0,0.9,0.9,0.8,0.8,0.8,0.7,0.7,0.7,0.7,0.6,0.6,0.6,0.6,0.6,0.5,0.5,0.5,0.5,0.5])
for i in range(1,21):
    xf[i] = g[i-1] + xa[i-1]

fig, ax = plt.subplots()
lf =np.zeros(11)
dx = np.linspace(0, 1, 11)
for i in range(20):
    lf = g[i]*dx**2 + xa[i]
    ax.plot(dx+t[i], lf, linestyle='solid',color="tab:blue" )
ax.plot(t, xf, marker='^', linestyle='None' ,label="forecast")
ax.plot(t, xa, marker='o', linestyle='None' ,label="analysis")
ax.plot(t, obs, marker='x', linestyle='None' ,label="obs")
print(ax.get_xlim())
ax.set_xlim(-0.1,2.5)
print(ax.get_xlim())
ax.set_ylim(-0.1,3.0)
ax.set_xticks(t[:3])
ax.set_xticks(step[:60], minor=True)

#ax.set_xlabel('time')
params = {'legend.fontsize': 16, 
          'axes.labelsize': 16,
          'legend.handlelength': 3}

plt.rcParams.update(params)
#ax.set_aspect(1.5)
#ax.legend(loc='upper right')#, bbox_to_anchor=(0.0,2.0))

plt.tight_layout()
fig.savefig("DAimage1.png")
plt.show()