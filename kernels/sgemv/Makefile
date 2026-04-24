include ../nsys.mk
nsys: sgemv_nsys.bin
sgemv_nsys.bin: sgemv_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-k32: sgemv_nsys_k32.bin
sgemv_nsys_k32.bin: sgemv_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMV_K32 $< -o $@ $(LDFLAGS)
nsys-k128-f32x4: sgemv_nsys_k128_f32x4.bin
sgemv_nsys_k128_f32x4.bin: sgemv_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMV_K128_F32X4 $< -o $@ $(LDFLAGS)
nsys-k16: sgemv_nsys_k16.bin
sgemv_nsys_k16.bin: sgemv_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMV_K16 $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
