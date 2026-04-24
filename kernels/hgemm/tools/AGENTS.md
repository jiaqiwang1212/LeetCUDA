<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# hgemm/tools

## Purpose
Build, install, and debugging utilities for the HGEMM library.

## Key Files

| File | Description |
|------|-------------|
| `install.sh` | Builds and installs the pybind11 Python extension (`pip install -e .`) |
| `clear.sh` | Removes compiled artifacts and build cache |
| `print_swizzle_layout.py` | Prints the XOR swizzle address mapping for visual inspection |
| `utils.py` | Python-side timing and correctness check helpers used by `hgemm.py` |

## For AI Agents

### Working In This Directory
- Run `bash tools/install.sh` from the `hgemm/` parent directory, not from inside `tools/`.
- `print_swizzle_layout.py` is identical to the one in `kernels/swizzle/` — use either for layout debugging.

<!-- MANUAL: -->
