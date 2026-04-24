<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# flash-attn/tools

## Purpose
Build, install, and debugging utilities for the flash attention library.

## Key Files

| File | Description |
|------|-------------|
| `install.sh` | Builds and installs the pybind11 Python extension |
| `clear.sh` | Removes compiled artifacts and build cache |
| `print_swizzle_layout.py` | Prints XOR swizzle address mapping for QKV tiles |
| `utils.py` | Python-side timing helpers used by `flash_attn_mma.py` |

## For AI Agents

### Working In This Directory
- Run `bash tools/install.sh` from the `flash-attn/` parent directory.
- Use `print_swizzle_layout.py` to visualize which swizzle configuration eliminates bank conflicts for a given tile size.

<!-- MANUAL: -->
