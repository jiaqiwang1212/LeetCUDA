include ../nsys.mk
nsys: flash_attn_nsys.bin
flash_attn_nsys.bin: flash_attn_nsys.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)
clean:
	rm -f *_nsys.bin
