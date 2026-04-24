# HGEMV

## 0x00 说明

包含以下内容：

- [X] hgemv_k32_f16_kernel
- [X] hgemv_k128_f16x4_kernel
- [X] hgemv_k16_f16_kernel
- [X] hgemv_f16_cute_kernel
- [X] hgemv_f16x8_cute_kernel
- [X] hgemv_tensor_core_cute_kernel
- [X] PyTorch bindings


## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 hgemv.py
```

输出:

```bash
--------------------------------------------------------------------------------
   out_k32f16: [15.609375, 2.15234375, -10.9296875], time:0.00324011ms
out_k128f16x4: [15.609375, 2.15625, -10.9296875], time:0.00322700ms
out_hgemv_f16_cute: [15.609375, 2.15234375, -10.9296875], time:0.00318646ms
out_hgemv_f16x8_cute: [15.609375, 2.16015625, -10.9375], time:0.00323176ms
out_hgemv_tensor_core_cute: [15.6171875, 2.15625, -10.9375], time:0.00531912ms
   out_f16_th: [15.6171875, 2.15429688, -10.9375], time:0.00889659ms
--------------------------------------------------------------------------------
   out_k16f16: [-6.69140625, -7.2265625, -6.4921875], time:0.00339985ms
out_hgemv_f16_cute: [-6.69140625, -7.2265625, -6.4921875], time:0.00323296ms
out_hgemv_f16x8_cute: [-6.6875, -7.2265625, -6.4921875], time:0.00319839ms
out_hgemv_tensor_core_cute: [-6.6875, -7.22265625, -6.4921875], time:0.00305891ms
   out_f16_th: [-6.69140625, -7.2265625, -6.4921875], time:0.00872254ms
--------------------------------------------------------------------------------
```

## Nsys 性能分析

使用 `make` 编译并通过 Nsight Systems 进行性能分析：

```bash
# 编译所有变体（默认）
make nsys
nsys profile --stats=true ./hgemv_nsys.bin

# 编译单个变体
make nsys-k128-f16x4
nsys profile --stats=true ./hgemv_nsys_k128_f16x4.bin
```

支持的编译目标：

| Make 目标 | 编译宏 | 说明 |
|-----------|--------|------|
| `make nsys` | 全部 | 所有变体顺序执行 |
| `make nsys-k32` | `-DHGEMV_K32` | K=32 基础版本 |
| `make nsys-k128-f16x4` | `-DHGEMV_K128_F16X4` | K=128 half4 向量化 |
| `make nsys-k16` | `-DHGEMV_K16` | K=16 版本 |
