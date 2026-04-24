include ../nsys.mk
nsys: softmax_nsys.bin
softmax_nsys.bin: softmax_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
