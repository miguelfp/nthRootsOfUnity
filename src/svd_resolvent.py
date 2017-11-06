import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc

import numpy as np

from stabfuncs import *

opts = PETSc.Options()
omega = opts.getScalar('omega').real
noWrite = opts.getBool('no_write', False)
prefix = opts.getString('prefix', '')
folder = opts.getString('folder', '.')
ab2 = opts.getBool('ab2', False)

j = opts.getInt('j', 0)
n = opts.getInt('n', 0)
dt = opts.getScalar('dt', 1.0)

# Load common auxiliary matrices first
I = PETSc.Mat().load(PETSc.Viewer().createBinary(folder+'/I.dat', 'r'))

Mh12 = PETSc.Vec().load(PETSc.Viewer().createBinary(folder+'/Mh12.dat', 'r'))
Rh12 = PETSc.Vec().load(PETSc.Viewer().createBinary(folder+'/Rh12.dat', 'r'))

# Load operators
if ab2:
  if n==0:
      Print("periodic case")
      M, iR, K = load_qep(folder+'/M.dat', folder+'/C.dat', folder+'/K.dat')
  else:
      Print("n-periodic case")
      M, iR, K = load_qep_n(folder+'/M1.dat', folder+'/M2.dat', folder+'/M3.dat', 
                            folder+'/C1.dat', folder+'/C2.dat', folder+'/C3.dat',
                            folder+'/K1.dat', folder+'/K2.dat', folder+'/K3.dat', j, n)
  iR.axpy(np.exp(1j*omega*dt), M)
  iR.axpy(np.exp(-1j*omega*dt), K)
else:
  if n==0:
      Print("periodic case")
      iR, B = load_gep(folder+'/A.dat', folder+'/B.dat')
  else:
      Print("n-periodic case")
      iR, B = load_gep_n(folder+'/A1.dat', folder+'/A2.dat', folder+'/A3.dat', 
                         folder+'/B1.dat', folder+'/B2.dat', folder+'/B3.dat', j, n)
  iR.axpy(-np.exp(1j*omega*dt), B)
  iR.scale(-1)

if np.abs(omega*dt)>=1e-4:
     iR.scale((np.exp(1j*omega*dt)-1)/(1j*omega*dt))
else:
     iR.scale(1 + 0.5*(1j*omega*dt))  

ctx = ResolventOperator(iR, I, Mh12, Mh12/Rh12)

S = SLEPc.SVD()
S.create()

MhRsize = (I.getSize()[0], I.getSize()[0])
MhR = PETSc.Mat().createPython((MhRsize, MhRsize), ctx)
MhR.setUp()

S.setOperator(MhR)
S.setFromOptions()

S.solve()

Print = PETSc.Sys.Print

Print( "******************************" )
Print( "*** SLEPc Solution Results ***" )
Print( "******************************\n" )
if noWrite:
    Print("Singular vectors will not be written")
else:
    Print("Singular vectors will be written")

svd_type = S.getType()
Print( "Solution method: %s" % svd_type )

its = S.getIterationNumber()
Print( "Number of iterations of the method: %d" % its )

nsv, ncv, mpd = S.getDimensions()
Print( "Number of requested singular values: %d" % nsv )

tol, maxit = S.getTolerances()
Print( "Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit) )

nconv = S.getConverged()
Print( "Number of converged approximate singular triplets %d" % nconv )

if nconv > 0:
  v, u = MhR.getVecs()
  q, f = I.getVecRight(), I.getVecRight()
  Print()
  Print("  w     j    n    i      sigma     residual norm ")
  Print("-----  ---  ---  ---  ----------  ---------------")

  for i in range(nconv):
    sigma = S.getSingularTriplet(i, u, v)
    error = S.computeError(i)

    Print( "%05.2f  %03d  %03d  %03d   %6f     %12g" % (omega, j, n, i, sigma, error) )

    if not noWrite:
        v/=Mh12*Rh12
        PETSc.Viewer().createBinary('%sj%03d_n%03d_w%05.2f_f_i%03d.dat' % (prefix, j, n, omega, i), 'w')(v)
                                                                                                
        v*=Mh12*Mh12                                                                            
        I.multTranspose(v, f)                                                                   
        ctx.R.solve(f, q)                                                                       
        PETSc.Viewer().createBinary('%sj%03d_n%03d_w%05.2f_q_i%03d.dat' % (prefix, j, n, omega, i), 'w')(q)

  Print()
