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
    A, B = load_gep(folder+'/A.dat', folder+'/B.dat', adjoint)
else:
    Print("Circulant operators")
    A, B = load_gep_n(folder+'/A1.dat', folder+'/A2.dat', folder+'/A3.dat', 
                      folder+'/B1.dat', folder+'/B2.dat', folder+'/B3.dat', j, n, adjoint)

# set NullSpace if required
if j==0:
    basis = [PETSc.Vec().load(PETSc.Viewer().createBinary(folder+'/Bk.dat', 'r'))]
    nullsp = PETSc.NullSpace().create(False, basis)
    B.setNullSpace(nullsp)

# Solve GEP
E = SLEPc.EPS().create()

E.setOperators(A, B)
E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

if j==0:
    E.setDeflationSpace(basis)

E.setFromOptions()
E.solve()

Print()
Print("******************************")
Print("*** SLEPc Solution Results ***")
Print("******************************")
Print()
if adjoint:
    Print("ADJOINT MODES")
    Print("    Note: This program returns Mw")
    Print("    If you need w, multiply by Mh^(-1) at postprocessing")
else:
    Print("DIRECT MODES")
Print()
if noWrite:
    Print("Eigenvectors will not be written")
else:
    Print("Eigenvectors will be written")
Print()


its = E.getIterationNumber()
Print("Number of iterations of the method: %d" % its)

eps_type = E.getType()
Print("Solution method: %s" % eps_type)

nev, ncv, mpd = E.getDimensions()
Print("Number of requested eigenvalues: %d" % nev)

tol, maxit = E.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

nconv = E.getConverged()
Print("Number of converged eigenpairs %d" % nconv)

if nconv > 0:
    # Create the results vectors
    q = A.getVecRight()

    Print()
    Print(" j   n   i       real part      imaginary part       error    ")
    Print("--- --- --- ------------------------------------ -------------")

    for i in range(nconv):
        k = E.getEigenpair(i, q)
        error = E.computeError(i)

        Print("%03d %03d %03d %.15f%+.15fj  %12g" % (j, n, i, k.real, k.imag, error))

        if not noWrite:
          if not adjoint:
             PETSc.Viewer().createBinary('%sj%03d_n%03d_v_i%03d.dat' % (prefix, j, n, i), 'w')(q)
          else:                                          
             BHq = B(q)                                  
             PETSc.Viewer().createBinary('%sj%03d_n%03d_w_i%03d.dat' % (prefix, j, n, i), 'w')(BHq)

    Print()
