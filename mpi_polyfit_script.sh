#!/bin/bash

ulimit -s unlimited 
module load libraries/openmpi-1.6-gcc-4.6.3

echo Running with n=1000,2000,5000,10000 with 2 processes 1>&2

for i in 1000 2000 5000 10000;
do
    time mpirun -np 2 /export/home/ncit-cluster/stud/m/mihai.barbulescu/liquid_dsp/examples/mpi_lagrange_polyfit
done

echo Running with n=1000,2000,5000,10000 with 4 processes 1>&2

for i in 1000 2000 5000 10000;
do
    time mpirun -np 4 /export/home/ncit-cluster/stud/m/mihai.barbulescu/liquid_dsp/examples/mpi_lagrange_polyfit
done



