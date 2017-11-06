#!/bin/bash

export J=6
export N=13

export OPTS="-st_pc_factor_mat_solver_package mumps -pep_target 1 -pep_nev 256"\
" -st_type sinvert -st_ksp_type preonly -st_pc_type lu -pep_tol 1e-14"\
" -pep_monitor -folder ../1-ExtractMatrices -j $J -n $N"

export JSTR=$(printf %03d $J)
export NSTR=$(printf %03d $N)

set +ex

pushd 2-Spectrum
mpirun -n 1 python -u ../../src/modes_qep.py $OPTS -pep_target 0.99999+0.0108i | tee v_j${JSTR}_n${NSTR}_log.txt 2>&1
grep -P '^\d+' v_j${JSTR}_n${NSTR}_log.txt > v_j${JSTR}_n${NSTR}_spectrum.txt
popd

set -ex
