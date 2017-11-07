# nthRootsOfUnity

This software demonstrates the techniques presented in the paper entitled
"Stability analysis for *n*-periodic arrays of fluid systems" by P.J. Schmid,
M. Fosas de Pando &amp; N. Peake (Phys. Rev. Fluids, 2017). 

It consists of a flow solver based on the projection-based immersed boundary
method by Taira & Colonius (J. Comp. Phys., 2007), the implementation of the
linearized equations and a set of tools that illustrate the analysis of flows
using the nth roots-of-unity technique. Eigenvalue problems are solved using
the SLEPc eigenvalue solver and PETSc.

Examples and test cases are provided in the examples/ and tests/ folders,
respectively, as Jupyter notebooks. The implementation of the flow solver,
linearization and the interface to eigenvalue solvers are located in src/. The
most recent version of this software will always be available
[here](https://doi.org/10.5281/zenodo.1040159).

Dependencies:

 + Python 3.6+
 + numpy 1.13+
 + scipy 0.19+
 + matplotlib 2.1+
 + jupyter 4.3+
 + petsc4py 3.7+ (with complex scalars and MUMPS support)
 + slepc4py 3.7+ (with complex scalars and MUMPS support)

When using this software, make sure the src/ and $PETSC_DIR/bin folders are
added to the PYTHONPATH environment variable. If you are using bash, this can
be accomplished executing the following command:

$ export PYTHONPATH=$PWD/src:$PETSC_DIR/bin

We gratefully acknowledge financial support from MINECO through the *Programa
Estatal de Retos de I+D+i Orientada a los Retos de la Sociedad*, grant number
DPI2016-75777-R AEI/FEDER, UE.
