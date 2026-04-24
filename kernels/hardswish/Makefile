include ../nsys.mk

SRC = hardswish_nsys.cu

nsys: hardswish_nsys.bin
hardswish_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: hardswish_nsys_f32.bin
hardswish_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSWISH_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: hardswish_nsys_f32x4.bin
hardswish_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSWISH_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: hardswish_nsys_f16.bin
hardswish_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSWISH_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: hardswish_nsys_f16x2.bin
hardswish_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSWISH_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: hardswish_nsys_f16x8.bin
hardswish_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSWISH_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: hardswish_nsys_f16x8_pack.bin
hardswish_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DHARDSWISH_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
