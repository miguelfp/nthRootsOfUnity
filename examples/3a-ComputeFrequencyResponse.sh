#!/bin/bash

export J=6
export N=13

export JSTR=$(printf %03d $J)
export NSTR=$(printf %03d $N)

set +ex

pushd 3-FrequencyResponse
for OMEGA in `seq -w 0 0.5 10`; do
    mpirun -n 1 python -u ../../src/svd_resolvent.py -omega $omega -folder ../1-ExtractMatrices -dt 0.0018 -ab2 -j $J -n $N -svd_monitor -svd_tol 1e-8 | tee j${JSTR}_n${NSTR}_w${OMEGA}_log.txt
done

cat j${JSTR}_n${NSTR}_w*_log.txt | awk '$4==000' > j${JSTR}_n${NSTR}_optimal_gains.txt

popd

set -ex
