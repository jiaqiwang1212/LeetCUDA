# SGEMV

## 0x00 说明

包含以下内容：

- [X] sgemv_k32_f32_kernel
- [X] sgemv_k128_f32x4_kernel
- [X] sgemv_k16_f32_kernel
- [X] PyTorch bindings

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 sgemv.py
```

输出:

```bash
--------------------------------------------------------------------------------
   out_k32f32: [-0.49123383, -13.83110714, -9.43372917], time:0.00372529ms
out_k128f32x4: [-0.49123383, -13.83110905, -9.43372917], time:0.00376225ms
   out_f32_th: [-0.49123335, -13.83110809, -9.43372917], time:0.00836253ms
--------------------------------------------------------------------------------
   out_k16f32: [-1.54411626, 9.39481068, 1.68226683], time:0.00364780ms
   out_f32_th: [-1.54411626, 9.39480972, 1.68226647], time:0.00812173ms
--------------------------------------------------------------------------------
```

## Nsys 性能分析

使用 `make` 编译并通过 Nsight Systems 进行性能分析：

```bash
# 编译所有变体（默认）
make nsys
nsys profile --stats=true ./sgemv_nsys.bin

# 编译单个变体
make nsys-k128-f32x4
nsys profile --stats=true ./sgemv_nsys_k128_f32x4.bin
```

支持的编译目标：

| Make 目标 | 编译宏 | 说明 |
|-----------|--------|------|
| `make nsys` | 全部 | 所有变体顺序执行 |
| `make nsys-k32` | `-DSGEMV_K32` | K=32 基础版本 |
| `make nsys-k128-f32x4` | `-DSGEMV_K128_F32X4` | K=128 float4 向量化 |
| `make nsys-k16` | `-DSGEMV_K16` | K=16 版本 |
