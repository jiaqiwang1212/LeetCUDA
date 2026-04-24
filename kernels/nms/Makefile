include ../nsys.mk
nsys: nms_nsys.bin
nms_nsys.bin: nms_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-nms: nms_nsys_nms.bin
nms_nsys_nms.bin: nms_nsys.cu
	$(NVCC) $(CFLAGS) -DNMS $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
