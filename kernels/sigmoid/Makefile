include ../nsys.mk

SRC = sigmoid_nsys.cu

nsys: sigmoid_nsys.bin
sigmoid_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: sigmoid_nsys_f32.bin
sigmoid_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSIGMOID_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: sigmoid_nsys_f32x4.bin
sigmoid_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSIGMOID_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: sigmoid_nsys_f16.bin
sigmoid_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSIGMOID_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: sigmoid_nsys_f16x2.bin
sigmoid_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSIGMOID_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: sigmoid_nsys_f16x8.bin
sigmoid_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSIGMOID_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: sigmoid_nsys_f16x8_pack.bin
sigmoid_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DSIGMOID_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
