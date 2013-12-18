#!/bin/bash

ulimit -s unlimited

PROB_DIM=10000000

for i in 1 2 4 8;
do
   export OMP_NUM_THREADS=$i
   time /export/home/ncit-cluster/stud/m/mihai.barbulescu/liquid_dsp/examples/fft_example -n $PROB_DIM 2>&1
done

