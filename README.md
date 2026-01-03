# ALM-pbc-ground-state

Exact numerical solution of the one dimensional anharmonic Larkin Model with periodic bounday conditions. 

* CUDA code to solve the 1d ground state.
* Thrust vectors and algorithms.
  
 nvcc  -arch=sm_75 alm.cu --extended-lambda -o a.out -DANHN=2 -DSIZEL=1024
