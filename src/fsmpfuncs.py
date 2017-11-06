# coding: utf-8

import numpy as np
import scipy.sparse as sp

# fractional step method functions with Dirichlet boundary conditions along the
# x axis and periodic boundary conditions along the y axis.

# Gradient operator and boundary terms
def gradient (dxp, dyp):
    n, m = len(dyp)+1, len(dxp)+1
    
    G = sp.vstack([sp.kron(sp.eye(n), sp.diags([-1, 1], [0, 1], (m-1, m) )), 
                   sp.kron(sp.diags([-1, 1, -1], [-1, 0, n-1], (n, n) ), sp.eye(m))])

    DuW = sp.kron(sp.eye(n), sp.diags([-1], [0], (m, 1)))
    DuE = sp.kron(sp.eye(n), sp.diags([1], [-m+1], (m, 1)))

    return G, DuW, DuE


# Laplacian operator and boundary terms
def laplacian_hat (dx, dy, dxp, dyp):
    n, m = len(dy), len(dx)
    
    # First u
    gux = sp.diags([-1/dx, 1/dx], [0, 1], (m, m+1))
    guxx = sp.diags([-1/dxp, 1/dxp], [0, 1], (m-1, m)).dot(gux)
    Lux = sp.kron(sp.eye(n), guxx.dot(sp.diags([1], [-1], (m+1, m-1))))
    Lux0 = sp.kron(sp.eye(n), guxx.dot(sp.diags([1], [0], (m+1, 1))))
    Lux1 = sp.kron(sp.eye(n), guxx.dot(sp.diags([1], [-m], (m+1, 1))))

    dyp_ = np.concatenate([0.5*(dy[:1]+dy[-1:]), dyp])
    guy = sp.diags([-1/dyp_, 1/dyp_, -1/dyp_], [-1, 0, n-1], (n, n))
    guyy = sp.diags([1/dy, -1/dy, 1/dy], [-n+1, 0, 1], (n, n)).dot(guy)
    Luy = sp.kron(guyy, sp.eye(m-1))
    
    Lu = Lux + Luy
    
    # Then v
    gvy = sp.diags([1/dy, -1/dy, 1/dy], [-n+1, 0, 1], (n, n))
    gvyy = sp.diags([-1/dyp_, 1/dyp_, -1/dyp_], [-1, 0, n-1], (n, n)).dot(gvy)
    Lvy = sp.kron(gvyy, sp.eye(m))

    dxp_ = np.concatenate([0.5*dx[:1], dxp, 0.5*dx[-1:]])
    gvx = sp.diags([-1/dxp_, 1/dxp_], [0, 1], (m+1, m+2))
    dx_ = np.concatenate([0.75*dx[:1], dx[1:-1], 0.75*dx[-1:]])
    gvxx = sp.diags([-1/dx_, 1/dx_], [0, 1], (m, m+1)).dot(gvx)

    Lvx = sp.kron(sp.eye(n), gvxx.dot(sp.diags([1], [-1], (m+2, m))))
    Lvx0 = sp.kron(sp.eye(n), gvxx.dot(sp.diags([1], [0], (m+2, 1))))
    Lvx1 = sp.kron(sp.eye(n), gvxx.dot(sp.diags([1], [-(m+1)], (m+2, 1))))

    Lv = Lvx + Lvy
    
    L = sp.block_diag([Lu, Lv])

    return L, Lux0, Lux1, Lvx0, Lvx1


# Weight matrices
def weight (dx, dy):
    n, m = len(dy), len(dx)
    
    R = sp.block_diag([sp.kron(sp.diags(dy, 0), sp.eye(m-1)),
                       sp.kron(sp.eye(n), sp.diags(dx, 0))])
    iR = sp.block_diag([sp.kron(sp.diags(1/dy, 0), sp.eye(m-1)),
                       sp.kron(sp.eye(n), sp.diags(1/dx, 0))])
    
    return R, iR


# Mass matrix
def mass_hat (dx, dy, dxp, dyp):
    n, m = len(dyp)+1, len(dxp)+1
    Iy = np.concatenate([np.ones(n)])
    Ix = np.concatenate([[0.75,], np.ones(m-2), [0.75,]])
    
    dyp_ = np.concatenate([0.5*(dy[:1]+dy[-1:]), dyp])

    Mh = sp.block_diag([sp.kron(sp.diags(Iy, 0), sp.diags(dxp, 0)),
                        sp.kron(sp.diags(dyp_, 0), sp.diags(Ix, 0))])
    iMh = sp.block_diag([sp.kron(sp.diags(1/Iy, 0), sp.diags(1/dxp, 0)),
                         sp.kron(sp.diags(1/dyp_, 0), sp.diags(1/Ix, 0))])
    
    return Mh, iMh


# Advection
def advection_hat(dx, dy, dxp, dyp, iRq, uW, uE, vW, vE):
    n, m = len(dy), len(dx)
    
    u = iRq[:n*(m-1)].reshape((n, m-1))
    v = iRq[n*(m-1):].reshape((n, m))
    
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
    v2c[:-1,:]=0.5*(v2[:-1,:]+v2[1:,:])
    v2c[-1,:]=0.5*(v2[0,:]+v2[-1,:])
    
    Nv = np.empty_like(v2, dtype=v.dtype)
    Nv[1:, :] = np.diff(v2c, axis=0)/dyp[:,np.newaxis]
    Nv[0, :] = (v2c[0,:]-v2c[-1,:])/(0.5*(dy[0]+dy[-1]))
    
    uv = 0.25*np.vstack([u[:1,:] + u[-1:,:], u[1:,:] + u[:-1,:]])*(v[:,1:] + v[:,:-1])
    
    uvW = 0.5*vW*np.concatenate([[uW[0]+uW[-1],], uW[1:]+uW[:-1]])
    uvE = 0.5*vE*np.concatenate([[uE[0]+uE[-1],], uE[1:]+uE[:-1]])
    
    Nu += np.diff(np.vstack([uv, uv[:1,:]]), axis=0)/dy[:,np.newaxis]
    Nv += np.diff(np.hstack([uvW[:,np.newaxis], uv, uvE[:,np.newaxis]]), axis=1)/dx
    
    return Nu.ravel(), Nv.ravel()

