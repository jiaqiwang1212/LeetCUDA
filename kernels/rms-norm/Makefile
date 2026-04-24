include ../nsys.mk
nsys: rms_norm_nsys.bin
rms_norm_nsys.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-f32: rms_norm_nsys_f32.bin
rms_norm_nsys_f32.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F32 $< -o $@ $(LDFLAGS)
nsys-f32x4: rms_norm_nsys_f32x4.bin
rms_norm_nsys_f32x4.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F32X4 $< -o $@ $(LDFLAGS)
nsys-f16-f16: rms_norm_nsys_f16_f16.bin
rms_norm_nsys_f16_f16.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16_F16 $< -o $@ $(LDFLAGS)
nsys-f16x2-f16: rms_norm_nsys_f16x2_f16.bin
rms_norm_nsys_f16x2_f16.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16X2_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-f16: rms_norm_nsys_f16x8_f16.bin
rms_norm_nsys_f16x8_f16.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16X8_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-f32: rms_norm_nsys_f16x8_f32.bin
rms_norm_nsys_f16x8_f32.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16X8_F32 $< -o $@ $(LDFLAGS)
nsys-f16-f32: rms_norm_nsys_f16_f32.bin
rms_norm_nsys_f16_f32.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16_F32 $< -o $@ $(LDFLAGS)
nsys-f16x8-pack-f16: rms_norm_nsys_f16x8_pack_f16.bin
rms_norm_nsys_f16x8_pack_f16.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16X8_PACK_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-pack-f32: rms_norm_nsys_f16x8_pack_f32.bin
rms_norm_nsys_f16x8_pack_f32.bin: rms_norm_nsys.cu
	$(NVCC) $(CFLAGS) -DRMSNORM_F16X8_PACK_F32 $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
