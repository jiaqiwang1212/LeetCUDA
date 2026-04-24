include ../nsys.mk
nsys: ws_hgemm_nsys.bin
ws_hgemm_nsys.bin: ws_hgemm_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
