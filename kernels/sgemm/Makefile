include ../nsys.mk
nsys: sgemm_nsys.bin
sgemm_nsys.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-naive-f32: sgemm_nsys_naive_f32.bin
sgemm_nsys_naive_f32.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_NAIVE_F32 $< -o $@ $(LDFLAGS)
nsys-sliced-k-f32: sgemm_nsys_sliced_k_f32.bin
sgemm_nsys_sliced_k_f32.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_SLICED_K_F32 $< -o $@ $(LDFLAGS)
nsys-t-8x8-f32x4: sgemm_nsys_t_8x8_f32x4.bin
sgemm_nsys_t_8x8_f32x4.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_T_8X8_F32X4 $< -o $@ $(LDFLAGS)
nsys-t-8x8-f32x4-bcf: sgemm_nsys_t_8x8_f32x4_bcf.bin
sgemm_nsys_t_8x8_f32x4_bcf.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_T_8X8_F32X4_BCF $< -o $@ $(LDFLAGS)
nsys-t-8x8-f32x4-bcf-offset: sgemm_nsys_t_8x8_f32x4_bcf_offset.bin
sgemm_nsys_t_8x8_f32x4_bcf_offset.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_T_8X8_F32X4_BCF_OFFSET $< -o $@ $(LDFLAGS)
nsys-t-8x8-f32x4-bcf-dbuf: sgemm_nsys_t_8x8_f32x4_bcf_dbuf.bin
sgemm_nsys_t_8x8_f32x4_bcf_dbuf.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_T_8X8_F32X4_BCF_DBUF $< -o $@ $(LDFLAGS)
nsys-t-8x8-f32x4-bcf-dbuf-offset: sgemm_nsys_t_8x8_f32x4_bcf_dbuf_offset.bin
sgemm_nsys_t_8x8_f32x4_bcf_dbuf_offset.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_T_8X8_F32X4_BCF_DBUF_OFFSET $< -o $@ $(LDFLAGS)
nsys-wmma-tf32-stage2: sgemm_nsys_wmma_tf32_stage2.bin
sgemm_nsys_wmma_tf32_stage2.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_WMMA_TF32_STAGE2 $< -o $@ $(LDFLAGS)
nsys-wmma-tf32-stage3: sgemm_nsys_wmma_tf32_stage3.bin
sgemm_nsys_wmma_tf32_stage3.bin: sgemm_nsys.cu
	$(NVCC) $(CFLAGS) -DSGEMM_WMMA_TF32_STAGE3 $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
