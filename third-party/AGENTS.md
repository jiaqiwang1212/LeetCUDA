<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# third-party

## Purpose
Git submodule dependencies used by advanced kernels in this repository. Do not modify these directories directly — update via git submodule commands.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `cutlass/` | NVIDIA CUTLASS library including CuTe layout abstractions — required by `kernels/cutlass/` and `kernels/hgemm/cutlass/` |

## For AI Agents

### Working In This Directory
- **Do not edit files here.** These are external submodules.
- To initialize: `git submodule update --init --recursive --force`
- To update to latest: use `.dev/update_submodules.sh`
- CuTe headers from `cutlass/include/cute/` are used by kernels with `_cute.cu` suffix.

<!-- MANUAL: -->
