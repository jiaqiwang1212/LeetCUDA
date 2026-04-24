include ../nsys.mk

SRC = elementwise_nsys.cu

nsys: elementwise_nsys.bin
elementwise_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: elementwise_nsys_f32.bin
elementwise_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELEMENTWISE_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: elementwise_nsys_f32x4.bin
elementwise_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELEMENTWISE_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: elementwise_nsys_f16.bin
elementwise_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELEMENTWISE_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: elementwise_nsys_f16x2.bin
elementwise_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELEMENTWISE_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8: elementwise_nsys_f16x8.bin
elementwise_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELEMENTWISE_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: elementwise_nsys_f16x8_pack.bin
elementwise_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DELEMENTWISE_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
