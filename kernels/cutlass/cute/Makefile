include ../../nsys.mk
nsys: cutlass_nsys.bin
cutlass_nsys.bin: cutlass_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
