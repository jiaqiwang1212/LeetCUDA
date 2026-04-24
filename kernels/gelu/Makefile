include ../nsys.mk

SRC = gelu_nsys.cu

nsys: gelu_nsys.bin
gelu_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: gelu_nsys_f32.bin
gelu_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DGELU_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: gelu_nsys_f32x4.bin
gelu_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DGELU_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: gelu_nsys_f16.bin
gelu_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DGELU_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: gelu_nsys_f16x2.bin
gelu_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DGELU_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: gelu_nsys_f16x8.bin
gelu_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DGELU_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: gelu_nsys_f16x8_pack.bin
gelu_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DGELU_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
