# 直方图统计

## 0x00 说明

包含以下内容：

- [X] histogram_i32_kernel
- [X] histogram_i32x4_kernel(int4向量化版本)
- [X] PyTorch bindings


## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 histogram.py
```

输出:

```bash
--------------------------------------------------------------------------------
h_i32   0: 1000
h_i32   1: 1000
h_i32   2: 1000
h_i32   3: 1000
h_i32   4: 1000
h_i32   5: 1000
h_i32   6: 1000
h_i32   7: 1000
h_i32   8: 1000
h_i32   9: 1000
--------------------------------------------------------------------------------
h_i32x4 0: 1000
h_i32x4 1: 1000
h_i32x4 2: 1000
h_i32x4 3: 1000
h_i32x4 4: 1000
h_i32x4 5: 1000
h_i32x4 6: 1000
h_i32x4 7: 1000
h_i32x4 8: 1000
h_i32x4 9: 1000
--------------------------------------------------------------------------------
```

## Nsys 性能分析

使用 `make` 编译并通过 Nsight Systems 进行性能分析：

```bash
# 编译所有变体（默认）
make nsys
nsys profile --stats=true ./histogram_nsys.bin

# 编译单个变体
make nsys-i32x4
nsys profile --stats=true ./histogram_nsys_i32x4.bin
```

支持的编译目标：

| Make 目标 | 编译宏 | 说明 |
|-----------|--------|------|
| `make nsys` | 全部 | 所有变体顺序执行 |
| `make nsys-i32` | `-DHISTOGRAM_I32` | Int32 标量 |
| `make nsys-i32x4` | `-DHISTOGRAM_I32X4` | Int32 int4 向量化 |
