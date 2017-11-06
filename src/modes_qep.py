import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

from stabfuncs import *

Print = PETSc.Sys.Print

# Load options from database
opts = PETSc.Options()
adjoint = opts.getBool('adjoint', False)
noWrite = opts.getBool('no_write', False)
folder = opts.getString('folder', '.')
prefix = opts.getString('prefix', '')
j = opts.getInt('j', 0)
n = opts.getInt('n', 0)

# Build operators
if n==0:
    Print("Regular operators")
    M, C, K = load_qep(folder+'/M.dat', folder+'/C.dat', folder+'/K.dat', adjoint)
else:
    Print("Circulant operators")
    M, C, K = load_qep_n(folder+'/M1.dat', folder+'/M2.dat', folder+'/M3.dat', 
                         folder+'/C1.dat', folder+'/C2.dat', folder+'/C3.dat',
                         folder+'/K1.dat', folder+'/K2.dat', folder+'/K3.dat', j, n, adjoint)

# set NullSpace if required
if j==0:
    basis = [PETSc.Vec().load(PETSc.Viewer().createBinary(folder+'/Mk.dat', 'r'))]
    nullsp = PETSc.NullSpace().create(False, basis)
    M.setNullSpace(nullsp)

# Solve QEP
Q = SLEPc.PEP().create()
Q.setOperators([K, C, M])

Q.setProblemType(SLEPc.PEP.ProblemType.GENERAL)

Q.setFromOptions()
Q.solve()

Print()
Print("******************************")
Print("*** SLEPc Solution Results ***")
Print("******************************")
Print()
if adjoint:
    Print("ADJOINT MODES")
    Print("    Note: This program returns Mh·w.")
    Print("    If you need w, multiply by Mh^(-1) at postprocessing")
else:
    Print("DIRECT MODES")
Print()
if noWrite:
    Print("Eigenvectors will not be written")
else:
    Print("Eigenvectors will be written")
Print()


its = Q.getIterationNumber()
Print("Number of iterations of the method: %d" % its)

sol_type = Q.getType()
Print("Solution method: %s" % sol_type)

nev, ncv, mpd = Q.getDimensions()
Print("Number of requested eigenvalues: %d" % nev)

tol, maxit = Q.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

nconv = Q.getConverged()
Print("Number of converged eigenpairs %d" % nconv)

if nconv > 0:
    # Create the results vectors
    q = M.getVecRight()

    Print()
    Print(" j   n   i       real part      imaginary part       error    ")
    Print("--- --- --- ------------------------------------ -------------")

    for i in range(nconv):
        k = Q.getEigenpair(i, q)
        error = Q.computeError(i)

        Print("%03d %03d %03d %.15f%+.15fj  %12g" % (j, n, i, k.real, k.imag, error))

        if not noWrite:
          if not adjoint:
             PETSc.Viewer().createBinary('%sj%03d_n%03d_v_i%03d.dat' % (prefix, j, n, i), 'w')(q)
          else:                                           
             MHq = M(q)                                   
             PETSc.Viewer().createBinary('%sj%03d_n%03d_w_i%03d.dat' % (prefix, j, n, i), 'w')(MHq)

    Print()
