# ALM-pbc-ground-state

Exact numerical solution of the ground state of the one dimensional anharmonic Larkin Model with periodic bounday conditions. 

* CUDA code generates a sample from a seed.
* Thrust vectors and algorithms.

Compile with:

nvcc  -arch=sm_75 alm.cu --extended-lambda -o a.out -DANHN=2 -DSIZEL=1024
