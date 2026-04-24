<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# nms

## Purpose
CUDA kernel for Non-Maximum Suppression (NMS) — object detection post-processing that filters overlapping bounding boxes based on IoU threshold. One of the more complex irregular-access kernels in the repo.

## Key Files

| File | Description |
|------|-------------|
| `nms.cu` | CUDA NMS kernel with bitfield-based suppression |
| `nms.cc` | C++ host wrapper / pybind glue |
| `nms.py` | PyTorch test vs `torchvision.ops.nms` |
| `README.md` | Algorithm explanation and IoU computation details |

## For AI Agents

### Working In This Directory
- Test with `uv run python nms.py` (requires torchvision for the baseline).
- Has both a `.cu` kernel file and a `.cc` host wrapper — both must be compiled together.

<!-- MANUAL: -->
