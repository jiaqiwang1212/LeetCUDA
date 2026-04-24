include ../nsys.mk

SRC = rope_nsys.cu

nsys: rope_nsys.bin
rope_nsys.bin: $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

nsys-f32: rope_nsys_f32.bin
rope_nsys_f32.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DROPE_F32 $< -o $@ $(LDFLAGS)

nsys-f32-v2: rope_nsys_f32_v2.bin
rope_nsys_f32_v2.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DROPE_F32_V2 $< -o $@ $(LDFLAGS)

nsys-f32x4-pack: rope_nsys_f32x4_pack.bin
rope_nsys_f32x4_pack.bin: $(SRC)
	$(NVCC) $(CFLAGS) -DROPE_F32X4_PACK $< -o $@ $(LDFLAGS)

clean:
	rm -f *_nsys.bin
