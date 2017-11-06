import numpy as np
from gridfuncs import stretching

# Numerical grid for the fluid
_s1 = stretching(256+128, 0.0036, 0.03, int(0.55/0.0036), 16, 16, 0.04)[0]
_s2 = stretching(256+128, 0.0036, 0.03, int(0.55/0.0036), 16, 16, 0.04)[0]
x = np.concatenate([-_s2[::-1], _s1[1:]])
_s = np.linspace(0, 0.46468, 130)
y = np.concatenate([-_s[::-1], _s[1:]]) 

n, m = len(y)-1, len(x)-1

dy, dx = np.diff(y), np.diff(x)
dxmin = min(np.min(dx), np.min(dy))

yp, xp = 0.5*(y[1:] + y[:-1]), 0.5*(x[1:] + x[:-1])
dyp, dxp = np.diff(yp), np.diff(xp)
pshape = (n, m)

yu, xu = yp, x[1:-1]
ushape = (n, m-1)

yv, xv = y[:-1], xp
vshape = (n, m)

# Immersed boundary
(_xi0, _eta0) = np.loadtxt('E3RotorB512.txt', unpack=True)
_xi0, _eta0 = _xi0[:-1] - 0.5, _eta0[:-1] - 0.037
_sang = 56.6*np.pi/180.
xi  = _xi0*np.cos(_sang) - _eta0*np.sin(_sang)
eta = _xi0*np.sin(_sang) + _eta0*np.cos(_sang)
l = xi.size
ds = np.sqrt((xi[1]-xi[0])**2 + (eta[1]-eta[0])**2)*np.ones(l)
uB = np.zeros_like(xi)
vB = np.zeros_like(xi)

iRe = 1/700.0
dt = 0.5 * min(dxmin**2/iRe, dxmin)

# Tip speed ratio
Ux_Ut = 0.41

Ut = 1/np.sqrt(Ux_Ut**2 + 1)
Ux = Ux_Ut/np.sqrt(Ux_Ut**2 + 1)

print("Ux/Utip = ", Ux_Ut)
print("AoA =", 180/np.pi*np.arctan(1/Ux_Ut))
print("Re = ", 1/iRe)
print("dt = ", dt)

# Boundary conditions
uE, uW = Ux*np.ones(n), Ux*np.ones(n)
vE, vW = Ut*np.ones(n), Ut*np.ones(n)
