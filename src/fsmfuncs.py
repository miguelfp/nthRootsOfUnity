# coding: utf-8

import numpy as np
import scipy.sparse as sp

# fractional step method functions with Dirichlet boundary conditions along the
# x and y axes.

# Gradient operator and boundary terms
def gradient (dxp, dyp):
    n, m = len(dyp)+1, len(dxp)+1
    
    G = sp.vstack([sp.kron(sp.eye(n), sp.diags([-1, 1], [0, 1], (m-1, m) )), 
                   sp.kron(sp.diags([-1, 1], [0, 1], (n-1, n) ), sp.eye(m))])

    DuW = sp.kron(sp.eye(n), sp.diags([-1], [0], (m, 1)))
    DuE = sp.kron(sp.eye(n), sp.diags([1], [-m+1], (m, 1)))

    DvS = sp.kron(sp.diags([-1], [0], (n, 1)), sp.eye(m))
    DvN = sp.kron(sp.diags([1], [-n+1], (n, 1)), sp.eye(m))
    
    return G, DuW, DuE, DvS, DvN


# Laplacian operator and boundary terms
def laplacian_hat (dx, dy, dxp, dyp):
    n, m = len(dy), len(dx)
    
    # First u
    gux = sp.diags([-1/dx, 1/dx], [0, 1], (m, m+1))
    guxx = sp.diags([-1/dxp, 1/dxp], [0, 1], (m-1, m)).dot(gux)
    Lux = sp.kron(sp.eye(n), guxx.dot(sp.diags([1], [-1], (m+1, m-1))))
    Lux0 = sp.kron(sp.eye(n), guxx.dot(sp.diags([1], [0], (m+1, 1))))
    Lux1 = sp.kron(sp.eye(n), guxx.dot(sp.diags([1], [-m], (m+1, 1))))

    dyp_ = np.concatenate([0.5*dy[:1], dyp, 0.5*dy[-1:]])
    guy = sp.diags([-1/dyp_, 1/dyp_], [0, 1], (n+1, n+2))
    dy_ = np.concatenate([0.75*dy[:1], dy[1:-1], 0.75*dy[-1:]])
    guyy = sp.diags([-1/dy_, 1/dy_], [0, 1], (n, n+1)).dot(guy)

    Luy = sp.kron(guyy.dot(sp.diags([1], [-1], (n+2, n))), sp.eye(m-1))
    Luy0 = sp.kron(guyy.dot(sp.diags([1], [0], (n+2, 1))), sp.eye(m-1))
    Luy1 = sp.kron(guyy.dot(sp.diags([1], [-(n+1)], (n+2, 1))), sp.eye(m-1))

    Lu = Lux + Luy
    
    # Then v
    gvy = sp.diags([-1/dy, 1/dy], [0, 1], (n, n+1))
    gvyy = sp.diags([-1/dyp, 1/dyp], [0, 1], (n-1, n)).dot(gvy)
    Lvy = sp.kron(gvyy.dot(sp.diags([1], [-1], (n+1, n-1))), sp.eye(m))
    Lvy0 = sp.kron(gvyy.dot(sp.diags([1], [0], (n+1, 1))), sp.eye(m))
    Lvy1 = sp.kron(gvyy.dot(sp.diags([1], [-n], (n+1, 1))), sp.eye(m))

    dxp_ = np.concatenate([0.5*dx[:1], dxp, 0.5*dx[-1:]])
    gvx = sp.diags([-1/dxp_, 1/dxp_], [0, 1], (m+1, m+2))
    dx_ = np.concatenate([0.75*dx[:1], dx[1:-1], 0.75*dx[-1:]])
    gvxx = sp.diags([-1/dx_, 1/dx_], [0, 1], (m, m+1)).dot(gvx)

    Lvx = sp.kron(sp.eye(n-1), gvxx.dot(sp.diags([1], [-1], (m+2, m))))
    Lvx0 = sp.kron(sp.eye(n-1), gvxx.dot(sp.diags([1], [0], (m+2, 1))))
    Lvx1 = sp.kron(sp.eye(n-1), gvxx.dot(sp.diags([1], [-(m+1)], (m+2, 1))))

    Lv = Lvx + Lvy
    
    L = sp.block_diag([Lu, Lv])

    return L, Lux0, Lux1, Luy0, Luy1, Lvx0, Lvx1, Lvy0, Lvy1


# Weight matrices
def weight (dx, dy):
    n, m = len(dy), len(dx)
    
    R = sp.block_diag([sp.kron(sp.diags(dy, 0), sp.eye(m-1)),
                       sp.kron(sp.eye(n-1), sp.diags(dx, 0))])
    iR = sp.block_diag([sp.kron(sp.diags(1/dy, 0), sp.eye(m-1)),
                       sp.kron(sp.eye(n-1), sp.diags(1/dx, 0))])
    
    return R, iR

# Mass matrix

# Mass matrix
def mass_hat (dxp, dyp):
    n, m = len(dyp)+1, len(dxp)+1
    Iy = np.concatenate([[0.75,], np.ones(n-2), [0.75,]])
    Ix = np.concatenate([[0.75,], np.ones(m-2), [0.75,]])

    Mh = sp.block_diag([sp.kron(sp.diags(Iy, 0), sp.diags(dxp, 0)),
                        sp.kron(sp.diags(dyp, 0), sp.diags(Ix, 0))])
    iMh = sp.block_diag([sp.kron(sp.diags(1/Iy, 0), sp.diags(1/dxp, 0)),
                         sp.kron(sp.diags(1/dyp, 0), sp.diags(1/Ix, 0))])
    
    return Mh, iMh


# Advection terms
def advection_hat(dx, dy, dxp, dyp, iRq, uS, uN, uW, uE, vS, vN, vW, vE):
    n, m = len(dy), len(dx)
    
    u = iRq[:n*(m-1)].reshape((n, m-1))
    v = iRq[n*(m-1):].reshape((n-1, m))
    
    Nu = np.zeros_like(u)
    Nv = np.zeros_like(v)
    
    u2 = u**2
    u2c = np.zeros((n, m), dtype=u.dtype)
    u2c[:,0]=0.5*(uW**2+u2[:,0])
    u2c[:,1:-1]=0.5*(u2[:,1:]+u2[:,:-1])
    u2c[:,-1]=0.5*(uE**2+u2[:,-1])
    
    Nu = np.diff(u2c, axis=1)/dxp
    
    v2 = v**2
    v2c = np.zeros((n, m), dtype=v.dtype)
    v2c[0,:]=0.5*(vS**2+v2[0,:])
    v2c[1:-1,:]=0.5*(v2[1:,:]+v2[:-1,:])
    v2c[-1,:]=0.5*(vN**2+v2[-1,:])
    
    Nv = np.diff(v2c, axis=0)/dyp[:,np.newaxis]
    
    uv = 0.25*(u[1:,:] + u[:-1,:])*(v[:,1:] + v[:,:-1])
    uvS = 0.5*uS*(vS[1:]+vS[:-1])
    uvN = 0.5*uN*(vN[1:]+vN[:-1])
    
    uvW = 0.5*vW*(uW[1:]+uW[:-1])
    uvE = 0.5*vE*(uE[1:]+uE[:-1])
    
    Nu += np.diff(np.vstack([uvS, uv, uvN]), axis=0)/dy[:,np.newaxis]
    Nv += np.diff(np.hstack([uvW[:,np.newaxis], uv, uvE[:,np.newaxis]]), axis=1)/dx
    
    return Nu.ravel(), Nv.ravel()

