include ../nsys.mk

SRC = relu_nsys.cu

nsys: relu_nsys.bin
relu_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: relu_nsys_f32.bin
relu_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DRELU_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: relu_nsys_f32x4.bin
relu_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DRELU_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: relu_nsys_f16.bin
relu_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DRELU_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: relu_nsys_f16x2.bin
relu_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DRELU_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: relu_nsys_f16x8.bin
relu_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DRELU_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: relu_nsys_f16x8_pack.bin
relu_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DRELU_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
