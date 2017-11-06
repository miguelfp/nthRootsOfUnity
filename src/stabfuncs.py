import numpy as np
from petsc4py import PETSc

Print = PETSc.Sys.Print

def load_gep(fA, fB, adjoint=False):
    A = PETSc.Mat().load(PETSc.Viewer().createBinary(fA, 'r'))
    B = PETSc.Mat().load(PETSc.Viewer().createBinary(fB, 'r'))

    if adjoint:
       A.transpose().conjugate()
       B.transpose().conjugate()

    return A, B

def load_gep_n(fA1, fA2, fA3, fB1, fB2, fB3, j, n, adjoint=False):
    dtheta = 2*np.pi/n
    
    rj = np.exp(1j*j*dtheta)

    A = PETSc.Mat().load(PETSc.Viewer().createBinary(fA1, 'r'))

    A2 = PETSc.Mat().load(PETSc.Viewer().createBinary(fA2, 'r'))
    A.axpy(rj, A2)
    del A2

    A3 = PETSc.Mat().load(PETSc.Viewer().createBinary(fA3, 'r'))
    A.axpy(1/rj, A3)
    del A3

    B = PETSc.Mat().load(PETSc.Viewer().createBinary(fB1, 'r'))

    B2 = PETSc.Mat().load(PETSc.Viewer().createBinary(fB2, 'r'))
    B.axpy(rj, B2)
    del B2

    B3 = PETSc.Mat().load(PETSc.Viewer().createBinary(fB3, 'r'))
    B.axpy(1/rj, B3)
    del B3

    if adjoint:
       A.transpose().conjugate()
       B.transpose().conjugate()

    return A, B

def load_qep(fM, fC, fK, adjoint=False):
   M = PETSc.Mat().load(PETSc.Viewer().createBinary(fM, 'r'))
   C = PETSc.Mat().load(PETSc.Viewer().createBinary(fC, 'r'))
   K = PETSc.Mat().load(PETSc.Viewer().createBinary(fK, 'r'))

   if adjoint:
      M.transpose().conjugate()
      C.transpose().conjugate()
      K.transpose().conjugate()

   return M, C, K

def load_qep_n(fM1, fM2, fM3, fC1, fC2, fC3, fK1, fK2, fK3, j, n, adjoint=False):
   dtheta = 2*np.pi/n
   
   rj = np.exp(1j*j*dtheta)
   
   M = PETSc.Mat().load(PETSc.Viewer().createBinary(fM1, 'r'))

   M2 = PETSc.Mat().load(PETSc.Viewer().createBinary(fM2, 'r'))
   M.axpy(rj, M2)
   del M2

   M3 = PETSc.Mat().load(PETSc.Viewer().createBinary(fM3, 'r'))
   M.axpy(1/rj, M3)
   del M3
   
   C = PETSc.Mat().load(PETSc.Viewer().createBinary(fC1, 'r'))

   C2 = PETSc.Mat().load(PETSc.Viewer().createBinary(fC2, 'r'))
   C.axpy(rj, C2)
   del C2

   C3 = PETSc.Mat().load(PETSc.Viewer().createBinary(fC3, 'r'))
   C.axpy(1/rj, C3)
   del C3
   
   K = PETSc.Mat().load(PETSc.Viewer().createBinary(fK1, 'r'))

   K2 = PETSc.Mat().load(PETSc.Viewer().createBinary(fK2, 'r'))
   K.axpy(rj, K2)
   del K2

   K3 = PETSc.Mat().load(PETSc.Viewer().createBinary(fK3, 'r'))
   K.axpy(1/rj, K3)
   del K3

   if adjoint:
      M.transpose().conjugate()
      C.transpose().conjugate()
      K.transpose().conjugate()
   
   return M, C, K
   

class ResolventOperator(object):
    def __init__(self, iR, I, scale_left, scale_right):
        self.iR = iR
        self.I = I

        self.scale_left = scale_left.copy()
        self.scale_right = scale_right.copy()

        self.iRH = self.iR.copy()
        self.iRH.transpose()
        self.iRH.conjugate()

        self.R = PETSc.KSP().create()
        self.R.setOperators(self.iR)
        self.R.setType('preonly')

        self.RH = PETSc.KSP().create()
        self.RH.setOperators(self.iRH)
        self.RH.setType('preonly')

        pc = self.R.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

        pc = self.RH.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

    def mult(self, A, x, y):
        f, q = self.I.getVecRight(), self.I.getVecRight()
        self.I.multTranspose(x*self.scale_left, f)
        self.R.solve(f, q)
        self.I.mult(q, y)
        y*=self.scale_right

    def multHermitian(self, A, x, y):
        f, q = self.I.getVecRight(), self.I.getVecRight()
        self.I.multTranspose(x*self.scale_right, f)
        self.RH.solve(f, q)
        self.I.mult(q, y)
        y *= self.scale_left

class MatrixExponentialEuler(object):
    def __init__(self, A, B, I, scale_left, scale_right, its):
        self.A = A
        self.B = B
        self.I = I
        self.its = its

        self.scale_left = scale_left.copy()
        self.scale_right = scale_right.copy()

        self.AH = self.A.copy()
        self.AH.transpose()
        self.AH.conjugate()

        self.BH = self.B.copy()
        self.BH.transpose()
        self.BH.conjugate()

        self.iB = PETSc.KSP().create()
        self.iB.setOperators(self.B)
        self.iB.setType('preonly')

        self.iBH = PETSc.KSP().create()
        self.iBH.setOperators(self.BH)
        self.iBH.setType('preonly')

        pc = self.iB.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

        pc = self.iBH.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

    def mult(self, A, x, y):
        q = self.I.getVecRight()
        self.I.multTranspose(x*self.scale_left, q)
        for k in range(self.its):
            self.iB.solve(self.A(q), q)
        self.I.mult(q, y)
        y*=self.scale_right

    def multHermitian(self, A, x, y):
        q, qaux = self.I.getVecRight(), self.I.getVecRight()
        self.I.multTranspose(x*self.scale_right, q)
        for k in range(self.its):
            self.iBH.solve(q, qaux)
            self.AH(qaux, q)

        self.I.mult(q, y)
        y *= self.scale_left

class MatrixExponentialAB2(object):
    def __init__(self, M, C, K, I, scale_left, scale_right, steps):
        self.M = M
        self.C = C
        self.K = K
        self.I = I
        self.steps = steps

        self.scale_left = scale_left.copy()
        self.scale_right = scale_right.copy()

        self.MH = self.M.copy()
        self.MH.transpose()
        self.MH.conjugate()

        self.CH = self.C.copy()
        self.CH.transpose()
        self.CH.conjugate()

        self.KH = self.K.copy()
        self.KH.transpose()
        self.KH.conjugate()

        self.iM = PETSc.KSP().create()
        self.iM.setOperators(self.M)
        self.iM.setType('preonly')

        self.iMH = PETSc.KSP().create()
        self.iMH.setOperators(self.MH)
        self.iMH.setType('preonly')

        pc = self.iM.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

        pc = self.iMH.getPC()
        pc.setType('lu')
        pc.setFactorSolverPackage('mumps')

    def mult(self, A, x, y):
        q0, q1  = self.I.getVecRight(), self.I.getVecRight()
        q2, aux = self.I.getVecRight(), self.I.getVecRight()

        self.I.multTranspose(x*self.scale_left, q0)
        self.iM.solve(-self.C(q0) - self.K(q0), q1)

        for k in range(1, self.steps):
            self.iM.solve(-self.C(q1) - self.K(q0), q2)
            aux = q0
            q0 = q1
            q1 = q2
            q2 = aux
            
        self.I.mult(q1, y)
        y*=self.scale_right

    def multHermitian(self, A, x, y):
        q0, q1  = self.I.getVecRight(), self.I.getVecRight()
        q2, aux = self.I.getVecRight(), self.I.getVecRight()

        self.I.multTranspose(x*self.scale_right, aux)

        self.iMH.solve(aux, q0)

        self.iMH.solve(self.CH(-q0), q1)

        for k in range(1, self.steps-1):
            self.iMH.solve(-self.CH(q1)-self.KH(q0), q2)

            aux = q0
            q0 = q1
            q1 = q2
            q2 = aux

        q2 = -self.CH(q1) -self.KH(q1+q0)

        aux = q0
        q0 = q1
        q1 = q2
        q2 = aux

        self.I.mult(q1, y)
        y *= self.scale_left

