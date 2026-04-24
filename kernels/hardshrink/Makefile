include ../nsys.mk

SRC = hardshrink_nsys.cu

nsys: hardshrink_nsys.bin
hardshrink_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: hardshrink_nsys_f32.bin
hardshrink_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSHRINK_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: hardshrink_nsys_f32x4.bin
hardshrink_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSHRINK_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: hardshrink_nsys_f16.bin
hardshrink_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSHRINK_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: hardshrink_nsys_f16x2.bin
hardshrink_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSHRINK_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: hardshrink_nsys_f16x8.bin
hardshrink_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSHRINK_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: hardshrink_nsys_f16x8_pack.bin
hardshrink_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSHRINK_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
