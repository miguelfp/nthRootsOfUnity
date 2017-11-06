# coding: utf-8
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from fsmpfuncs import advection_hat 

def linearize_advection_hat (dx, dy, dxp, dyp, U0, uW, uE, vW, vE, test=False):
    n, m = len(dy), len(dx)
    Nu0, Nv0 = advection_hat(dx, dy, dxp, dyp, U0, uW, uE, vW, vE)
    h = 1e-8j

    # Tiling width
    sw = 3
    if (n%3)!=0:
        raise ValueError ("Dimension n must be divisible by 3: n=%d, n/3 = %f"%(n, n/3))

    # aux vectors for building the block matrices that form N
    # N = [Nuu, Nuv]
    #     [Nvu, Nvv]
    data_Nuu, row_Nuu, col_Nuu = np.asarray([]), np.asarray([]), np.asarray([])
    data_Nvu, row_Nvu, col_Nvu = np.asarray([]), np.asarray([]), np.asarray([])
    data_Nuv, row_Nuv, col_Nuv = np.asarray([]), np.asarray([]), np.asarray([])
    data_Nvv, row_Nvv, col_Nvv = np.asarray([]), np.asarray([]), np.asarray([])

    # stencils
    suu = [[0, -1], [0, 0], [0, 1], [-1, 0], [1, 0]]
    svu = [[0, 0], [0, 1], [1, 0], [1, 1]]
    svv = [[0, -1], [0, 0], [0, 1], [-1, 0], [1, 0]]
    suv = [[0, -1], [-1, -1], [-1, 0], [0, 0]]

    # obtain Nuu and Nvu
    v=np.zeros((n, m))
    for idxj in range(sw):
      for idxi in range(sw):
        u=np.zeros((n, m-1))
        u[idxj::sw, idxi::sw] = 1

        idx = np.where(u)
        tmpidx = np.arange(u.size).reshape(u.shape)[idx]

        uidx = -np.ones((n, m-1), dtype=int)
        for suuk in suu:
            jj, ii = idx[0]+suuk[0], idx[1]+suuk[1]
            
            jj[jj == -1] = n-1
            jj[jj == n] = 0
            
            mask = (0<=jj)*(jj<n)*(0<=ii)*(ii<(m-1))
            uidx[jj[mask], ii[mask]] = tmpidx[mask]

        vidx = -np.ones((n, m), dtype=int)
        for svuk in svu:
            jj, ii = idx[0]+svuk[0], idx[1]+svuk[1]
            
            jj[jj == -1] = n-1
            jj[jj == n] = 0
            
            mask = (0<=jj)*(jj<n)*(0<=ii)*(ii<m)
            vidx[jj[mask], ii[mask]] = tmpidx[mask]

        U = np.concatenate([u.ravel(), v.ravel()])
        α=la.norm(U, np.inf)
        Nu1, Nv1 = advection_hat(dx, dy, dxp, dyp, U0+h/α*U, uW, uE, vW, vE)
        Nu, Nv = α*((Nu1-Nu0)/h).real, α*((Nv1-Nv0)/h).real

        mask = uidx.ravel()>=0
        row_Nuu = np.concatenate([row_Nuu, np.arange(u.size)[mask]])
        col_Nuu = np.concatenate([col_Nuu, uidx.ravel()[mask]])
        data_Nuu = np.concatenate([data_Nuu, Nu[mask]])

        mask = vidx.ravel()>=0
        row_Nvu = np.concatenate([row_Nvu, np.arange(v.size)[mask]])
        col_Nvu = np.concatenate([col_Nvu, vidx.ravel()[mask]])
        data_Nvu = np.concatenate([data_Nvu, Nv[mask]])

    # obtain Nuv and Nvv
    u=np.zeros((n, m-1))
    for idxj in range(sw):
      for idxi in range(sw):
        v=np.zeros((n, m))
        v[idxj::sw, idxi::sw] = 1

        idx = np.where(v)
        tmpidx = np.arange(v.size).reshape(v.shape)[idx]

        vidx = -np.ones((n, m), dtype=int)
        for svvk in svv:
            jj, ii = idx[0]+svvk[0], idx[1]+svvk[1]
            
            jj[jj == -1] = n-1
            jj[jj == n] = 0
            
            mask = (0<=jj)*(jj<n)*(0<=ii)*(ii<m)
            vidx[jj[mask], ii[mask]] = tmpidx[mask]

        uidx = -np.ones((n, m-1), dtype=int)
        for suvk in suv:
            jj, ii = idx[0]+suvk[0], idx[1]+suvk[1]
            
            jj[jj == -1] = n-1
            jj[jj == n] = 0
            
            mask = (0<=jj)*(jj<n)*(0<=ii)*(ii<(m-1))
            uidx[jj[mask], ii[mask]] = tmpidx[mask]

        U = np.concatenate([u.ravel(), v.ravel()])
        α=la.norm(U, np.inf)
        Nu1, Nv1 = advection_hat(dx, dy, dxp, dyp, U0+h/α*U, uW, uE, vW, vE)
        Nu, Nv = α*((Nu1-Nu0)/h).real, α*((Nv1-Nv0)/h).real

        mask = uidx.ravel()>=0
        row_Nuv = np.concatenate([row_Nuv, np.arange(u.size)[mask]])
        col_Nuv = np.concatenate([col_Nuv, uidx.ravel()[mask]])
        data_Nuv = np.concatenate([data_Nuv, Nu[mask]])

        mask = vidx.ravel()>=0
        row_Nvv = np.concatenate([row_Nvv, np.arange(v.size)[mask]])
        col_Nvv = np.concatenate([col_Nvv, vidx.ravel()[mask]])
        data_Nvv = np.concatenate([data_Nvv, Nv[mask]])

    Nuu = sp.coo_matrix((data_Nuu, (row_Nuu, col_Nuu)), shape=(u.size,u.size))
    Nvu = sp.coo_matrix((data_Nvu, (row_Nvu, col_Nvu)), shape=(v.size,u.size))
    Nuv = sp.coo_matrix((data_Nuv, (row_Nuv, col_Nuv)), shape=(u.size,v.size))
    Nvv = sp.coo_matrix((data_Nvv, (row_Nvv, col_Nvv)), shape=(v.size,v.size))

    Nh = sp.bmat([[Nuu, Nuv],[Nvu, Nvv]]).tocsr()
    
    if test:
        # First random u    
        u=np.random.random((n,m-1))
        v=np.zeros((n, m))

        U = np.concatenate([u.ravel(), v.ravel()])
        α=la.norm(U, np.inf)

        Nu1, Nv1 = advection_hat(dx, dy, dxp, dyp, U0+h/α*U, uW, uE, vW, vE)
        NU1 = np.concatenate([α*((Nu1-Nu0)/h).real, α*((Nv1-Nv0)/h).real])
        NU2 = Nh.dot(U)
        NUerr = la.norm(NU2-NU1)/la.norm(NU1)

        # Then random v
        u=np.zeros((n,m-1))
        v=np.random.random((n, m))

        U = np.concatenate([u.ravel(), v.ravel()])
        α=la.norm(U, np.inf)
        Nu1, Nv1 = advection_hat(dx, dy, dxp, dyp, U0+h/α*U, uW, uE, vW, vE)

        NU1 = np.concatenate([α*((Nu1-Nu0)/h).real, α*((Nv1-Nv0)/h).real])
        NU2 = Nh.dot(U)
        NVerr = la.norm(NU2[u.size:]-NU1[u.size:])/la.norm(NU1[u.size:])
        
        if NUerr > 1e-13 or NVerr > 1e-13:
            raise ValueError("Linearization check failed: Nuerr = %f and Nverr = %f" % (NUerr, NVerr))
        
    return Nh

