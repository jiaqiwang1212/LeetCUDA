include ../nsys.mk

SRC = swish_nsys.cu

nsys: swish_nsys.bin
swish_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: swish_nsys_f32.bin
swish_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSWISH_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: swish_nsys_f32x4.bin
swish_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSWISH_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: swish_nsys_f16.bin
swish_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSWISH_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: swish_nsys_f16x2.bin
swish_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSWISH_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: swish_nsys_f16x8.bin
swish_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSWISH_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: swish_nsys_f16x8_pack.bin
swish_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSWISH_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
