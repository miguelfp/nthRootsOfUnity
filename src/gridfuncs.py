# coding: utf-8

import numpy as np
from scipy.special import erf

# Stretching function
def stretching(n, dn0, dn1, ns, ws, we, maxs):
    ne = ns + np.log(dn1/dn0)/np.log(1+maxs)
    
    s=np.array([maxs*0.25*(1+erf(6*(x-ns)/(ws)))*(1-erf(6*(x-ne)/we)) for x in range(n)])

    f_=np.empty(s.shape); f_[0] = dn0
    for k in range(1,len(f_)):
      f_[k] = f_[k-1]*(1+s[k])
    f=np.empty(s.shape);  f[0] = 0.0
    for k in range(1,len(f)):
      f[k] = f[k-1] + f_[k]

    return f, f_, s

