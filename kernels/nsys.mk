# Shared flags for *_nsys.bin targets. Override via: make SM=89 nsys
NVCC      ?= nvcc
SM        ?= 80
CUDA_HOME ?= /usr/local/cuda
NVTX_INC  ?= -I$(CUDA_HOME)/include
CFLAGS    ?= -O3 -std=c++17 -arch=sm_$(SM) $(NVTX_INC)
LDFLAGS   ?= -lnvToolsExt

# Usage in a per-dir Makefile:
#   include ../nsys.mk
#   nsys: mykernel_nsys.bin
#   mykernel_nsys.bin: mykernel_nsys.cu
#   	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
#   clean:
#   	rm -f *_nsys.bin
