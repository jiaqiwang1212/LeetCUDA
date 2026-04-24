include ../nsys.mk
nsys: mat_transpose_nsys.bin
mat_transpose_nsys.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
nsys-f32-col2row: mat_transpose_nsys_f32_col2row.bin
mat_transpose_nsys_f32_col2row.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32_COL2ROW $< -o $@ $(LDFLAGS)
nsys-f32-row2col: mat_transpose_nsys_f32_row2col.bin
mat_transpose_nsys_f32_row2col.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32_ROW2COL $< -o $@ $(LDFLAGS)
nsys-f32x4-col2row: mat_transpose_nsys_f32x4_col2row.bin
mat_transpose_nsys_f32x4_col2row.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_COL2ROW $< -o $@ $(LDFLAGS)
nsys-f32x4-row2col: mat_transpose_nsys_f32x4_row2col.bin
mat_transpose_nsys_f32x4_row2col.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_ROW2COL $< -o $@ $(LDFLAGS)
nsys-f32-col2row2d: mat_transpose_nsys_f32_col2row2d.bin
mat_transpose_nsys_f32_col2row2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32_COL2ROW2D $< -o $@ $(LDFLAGS)
nsys-f32-row2col2d: mat_transpose_nsys_f32_row2col2d.bin
mat_transpose_nsys_f32_row2col2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32_ROW2COL2D $< -o $@ $(LDFLAGS)
nsys-f32-diagonal2d: mat_transpose_nsys_f32_diagonal2d.bin
mat_transpose_nsys_f32_diagonal2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32_DIAGONAL2D $< -o $@ $(LDFLAGS)
nsys-f32x4-col2row2d: mat_transpose_nsys_f32x4_col2row2d.bin
mat_transpose_nsys_f32x4_col2row2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_COL2ROW2D $< -o $@ $(LDFLAGS)
nsys-f32x4-row2col2d: mat_transpose_nsys_f32x4_row2col2d.bin
mat_transpose_nsys_f32x4_row2col2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_ROW2COL2D $< -o $@ $(LDFLAGS)
nsys-f32x4-shared-col2row2d: mat_transpose_nsys_f32x4_shared_col2row2d.bin
mat_transpose_nsys_f32x4_shared_col2row2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_SHARED_COL2ROW2D $< -o $@ $(LDFLAGS)
nsys-f32x4-shared-row2col2d: mat_transpose_nsys_f32x4_shared_row2col2d.bin
mat_transpose_nsys_f32x4_shared_row2col2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_SHARED_ROW2COL2D $< -o $@ $(LDFLAGS)
nsys-f32x4-shared-bcf-col2row2d: mat_transpose_nsys_f32x4_shared_bcf_col2row2d.bin
mat_transpose_nsys_f32x4_shared_bcf_col2row2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_SHARED_BCF_COL2ROW2D $< -o $@ $(LDFLAGS)
nsys-f32x4-shared-bcf-row2col2d: mat_transpose_nsys_f32x4_shared_bcf_row2col2d.bin
mat_transpose_nsys_f32x4_shared_bcf_row2col2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_SHARED_BCF_ROW2COL2D $< -o $@ $(LDFLAGS)
nsys-f32x4-shared-bcf-merge-write-row2col2d: mat_transpose_nsys_f32x4_shared_bcf_merge_write_row2col2d.bin
mat_transpose_nsys_f32x4_shared_bcf_merge_write_row2col2d.bin: mat_transpose_nsys.cu
	$(NVCC) $(CFLAGS) -DMAT_TRANS_F32X4_SHARED_BCF_MERGE_WRITE_ROW2COL2D $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
