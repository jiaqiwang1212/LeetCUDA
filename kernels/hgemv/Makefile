include ../nsys.mk
nsys: hgemv_nsys.bin
hgemv_nsys.bin: hgemv_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-k32: hgemv_nsys_k32.bin
hgemv_nsys_k32.bin: hgemv_nsys.cu
	$(NVCC) $(CFLAGS) -DHGEMV_K32 $< -o $@ $(LDFLAGS)
nsys-k128-f16x4: hgemv_nsys_k128_f16x4.bin
hgemv_nsys_k128_f16x4.bin: hgemv_nsys.cu
	$(NVCC) $(CFLAGS) -DHGEMV_K128_F16X4 $< -o $@ $(LDFLAGS)
nsys-k16: hgemv_nsys_k16.bin
hgemv_nsys_k16.bin: hgemv_nsys.cu
	$(NVCC) $(CFLAGS) -DHGEMV_K16 $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
