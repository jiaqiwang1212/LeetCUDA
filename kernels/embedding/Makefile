include ../nsys.mk

SRC = embedding_nsys.cu

nsys: embedding_nsys.bin
embedding_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: embedding_nsys_f32.bin
embedding_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DEMBEDDING_F32 $< -o $@ $(LDFLAGS)

nsys-f32x4: embedding_nsys_f32x4.bin
embedding_nsys_f32x4.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DEMBEDDING_F32X4 $< -o $@ $(LDFLAGS)

nsys-f32x4-pack: embedding_nsys_f32x4_pack.bin
embedding_nsys_f32x4_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DEMBEDDING_F32X4_PACK $< -o $@ $(LDFLAGS)

nsys-f16: embedding_nsys_f16.bin
embedding_nsys_f16.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DEMBEDDING_F16 $< -o $@ $(LDFLAGS)

nsys-f16x8: embedding_nsys_f16x8.bin
embedding_nsys_f16x8.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DEMBEDDING_F16X8 $< -o $@ $(LDFLAGS)

nsys-f16x8-pack: embedding_nsys_f16x8_pack.bin
embedding_nsys_f16x8_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DEMBEDDING_F16X8_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
