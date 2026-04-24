INCLUDE_DIRS=-I ../../third-party/cutlass/include -I ../../third-party/cutlass/tools/util/include
ARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89
ARCHS_89=-gencode arch=compute_89,code=sm_89
DEFAULT_FLAGS=-O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
DEFAULT_FLAGS_89=-O2 $(ARCHS_89) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas
default:
	nvcc hgemm_mma_swizzle.cu -o hgemm_mma_swizzle.bin $(DEFAULT_FLAGS)
	nvcc mat_trans_swizzle.cu -o mat_trans_swizzle.bin $(DEFAULT_FLAGS)
	nvcc mma_simple_swizzle.cu -o mma_simple_swizzle.bin $(DEFAULT_FLAGS)
hgemm_89:
	nvcc hgemm_mma_swizzle.cu -o hgemm_mma_swizzle.89.bin $(DEFAULT_FLAGS_89)
mma_89:
	nvcc mma_simple_swizzle.cu -o mma_simple_swizzle.89.bin $(DEFAULT_FLAGS_89)
mat_89:
	nvcc mat_trans_swizzle.cu -o mat_trans_swizzle.89.bin $(DEFAULT_FLAGS_89)
clean:
	rm -rf *.bin

# --- nsys profiling target (appended) ---
SM_NSYS   ?= 80
NSYS_CFLAGS=-O3 -std=c++17 -arch=sm_$(SM_NSYS) -I$(CUDA_HOME)/include
NSYS_LDFLAGS=-lnvToolsExt
nsys: swizzle_nsys.bin
swizzle_nsys.bin: swizzle_nsys.cu
	nvcc $(NSYS_CFLAGS) $< -o $@ $(NSYS_LDFLAGS)
