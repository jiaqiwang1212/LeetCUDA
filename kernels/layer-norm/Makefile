include ../nsys.mk
nsys: layer_norm_nsys.bin
layer_norm_nsys.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-f32: layer_norm_nsys_f32.bin
layer_norm_nsys_f32.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F32 $< -o $@ $(LDFLAGS)
nsys-f32x4: layer_norm_nsys_f32x4.bin
layer_norm_nsys_f32x4.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F32X4 $< -o $@ $(LDFLAGS)
nsys-f16-f16: layer_norm_nsys_f16_f16.bin
layer_norm_nsys_f16_f16.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F16_F16 $< -o $@ $(LDFLAGS)
nsys-f16x2-f16: layer_norm_nsys_f16x2_f16.bin
layer_norm_nsys_f16x2_f16.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F16X2_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-f16: layer_norm_nsys_f16x8_f16.bin
layer_norm_nsys_f16x8_f16.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F16X8_F16 $< -o $@ $(LDFLAGS)
nsys-f16-f32: layer_norm_nsys_f16_f32.bin
layer_norm_nsys_f16_f32.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F16_F32 $< -o $@ $(LDFLAGS)
nsys-f16x8-pack-f16: layer_norm_nsys_f16x8_pack_f16.bin
layer_norm_nsys_f16x8_pack_f16.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F16X8_PACK_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-pack-f32: layer_norm_nsys_f16x8_pack_f32.bin
layer_norm_nsys_f16x8_pack_f32.bin: layer_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DLAYERNORM_F16X8_PACK_F32 $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
