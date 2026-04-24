include ../nsys.mk

SRC = histogram_nsys.cu

nsys: histogram_nsys.bin
histogram_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-i32: histogram_nsys_i32.bin
histogram_nsys_i32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHISTOGRAM_I32 $< -o $@ $(LDFLAGS)

nsys-i32x4: histogram_nsys_i32x4.bin
histogram_nsys_i32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHISTOGRAM_I32X4 $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
