include ../nsys.mk
nsys: block_all_reduce_nsys.bin
block_all_reduce_nsys.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-f32-f32: block_all_reduce_nsys_f32_f32.bin
block_all_reduce_nsys_f32_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F32_F32 $< -o $@ $(LDFLAGS)
nsys-f32x4-f32: block_all_reduce_nsys_f32x4_f32.bin
block_all_reduce_nsys_f32x4_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F32X4_F32 $< -o $@ $(LDFLAGS)
nsys-f16-f16: block_all_reduce_nsys_f16_f16.bin
block_all_reduce_nsys_f16_f16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F16_F16 $< -o $@ $(LDFLAGS)
nsys-f16-f32: block_all_reduce_nsys_f16_f32.bin
block_all_reduce_nsys_f16_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F16_F32 $< -o $@ $(LDFLAGS)
nsys-f16x2-f32: block_all_reduce_nsys_f16x2_f32.bin
block_all_reduce_nsys_f16x2_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F16X2_F32 $< -o $@ $(LDFLAGS)
nsys-f16x2-f16: block_all_reduce_nsys_f16x2_f16.bin
block_all_reduce_nsys_f16x2_f16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F16X2_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-pack-f16: block_all_reduce_nsys_f16x8_pack_f16.bin
block_all_reduce_nsys_f16x8_pack_f16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F16X8_PACK_F16 $< -o $@ $(LDFLAGS)
nsys-f16x8-pack-f32: block_all_reduce_nsys_f16x8_pack_f32.bin
block_all_reduce_nsys_f16x8_pack_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_F16X8_PACK_F32 $< -o $@ $(LDFLAGS)
nsys-bf16-bf16: block_all_reduce_nsys_bf16_bf16.bin
block_all_reduce_nsys_bf16_bf16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_BF16_BF16 $< -o $@ $(LDFLAGS)
nsys-bf16-f32: block_all_reduce_nsys_bf16_f32.bin
block_all_reduce_nsys_bf16_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_BF16_F32 $< -o $@ $(LDFLAGS)
nsys-bf16x2-bf16: block_all_reduce_nsys_bf16x2_bf16.bin
block_all_reduce_nsys_bf16x2_bf16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_BF16X2_BF16 $< -o $@ $(LDFLAGS)
nsys-bf16x2-f32: block_all_reduce_nsys_bf16x2_f32.bin
block_all_reduce_nsys_bf16x2_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_BF16X2_F32 $< -o $@ $(LDFLAGS)
nsys-bf16x8-pack-bf16: block_all_reduce_nsys_bf16x8_pack_bf16.bin
block_all_reduce_nsys_bf16x8_pack_bf16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_BF16X8_PACK_BF16 $< -o $@ $(LDFLAGS)
nsys-bf16x8-pack-f32: block_all_reduce_nsys_bf16x8_pack_f32.bin
block_all_reduce_nsys_bf16x8_pack_f32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_BF16X8_PACK_F32 $< -o $@ $(LDFLAGS)
nsys-fp8-e4m3-f16: block_all_reduce_nsys_fp8_e4m3_f16.bin
block_all_reduce_nsys_fp8_e4m3_f16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_FP8_E4M3_F16 $< -o $@ $(LDFLAGS)
nsys-fp8-e5m2-f16: block_all_reduce_nsys_fp8_e5m2_f16.bin
block_all_reduce_nsys_fp8_e5m2_f16.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_FP8_E5M2_F16 $< -o $@ $(LDFLAGS)
nsys-i8-i32: block_all_reduce_nsys_i8_i32.bin
block_all_reduce_nsys_i8_i32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_I8_I32 $< -o $@ $(LDFLAGS)
nsys-i8x16-pack-i32: block_all_reduce_nsys_i8x16_pack_i32.bin
block_all_reduce_nsys_i8x16_pack_i32.bin: block_all_reduce_nsys.cu
	$(NVCC) $(CFLAGS) -DREDUCE_I8X16_PACK_I32 $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
