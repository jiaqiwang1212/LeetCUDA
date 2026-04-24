include ../nsys.mk

SRC = elu_nsys.cu

nsys: elu_nsys.bin
elu_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: elu_nsys_f32.bin
elu_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELU_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: elu_nsys_f32x4.bin
elu_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELU_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: elu_nsys_f16.bin
elu_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELU_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: elu_nsys_f16x2.bin
elu_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELU_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: elu_nsys_f16x8.bin
elu_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELU_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: elu_nsys_f16x8_pack.bin
elu_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELU_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
