include ../nsys.mk

SRC = dot_product_nsys.cu

nsys: dot_product_nsys.bin
dot_product_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: dot_product_nsys_f32.bin
dot_product_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DDOT_PROD_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: dot_product_nsys_f32x4.bin
dot_product_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DDOT_PROD_F32X4 $< -o $@ $(LDFLAGS)

nsys-f16: dot_product_nsys_f16.bin
dot_product_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DDOT_PROD_F16 $< -o $@ $(LDFLAGS)

nsys-f16x2: dot_product_nsys_f16x2.bin
dot_product_nsys_f16x2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DDOT_PROD_F16X2 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: dot_product_nsys_f16x8_pack.bin
dot_product_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DDOT_PROD_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
