CXX = nvcc

ANHN?=2
SIZEL?=1024

INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/include 
FLAGS = --expt-extended-lambda -lcufft -std=c++17 -arch=sm_75 \
-gencode arch=compute_61,code=sm_61 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_75,code=sm_75 \

LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/lib64 

PARAMS = -DANHN=$(ANHN) -DSIZEL=$(SIZEL)

alm: 	main.cu Makefile
	$(CXX) $(FLAGS) $(PARAMSEW) main.cu -o alm $(LDFLAGS) $(INCLUDES) $(PARAMS) 

update_git:
	git add *.cu Makefile README.md ; git commit -m "program update"; git push

clean:
	rm -f alm
