<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-24 | Updated: 2026-04-24 -->

# pytorch/distributed

## Purpose
Individual test scripts for every major PyTorch distributed collective operation using the NCCL backend. Each script demonstrates one collective in isolation — useful for learning the `torch.distributed` API and verifying multi-GPU communication.

## Key Files

| File | Description |
|------|-------------|
| `test_all_reduce.py` | `dist.all_reduce` — sum/mean tensors across all ranks |
| `test_all_gather.py` | `dist.all_gather` — each rank collects tensors from all others |
| `test_all_gather_objects.py` | `dist.all_gather_object` — gather arbitrary Python objects |
| `test_all_to_all.py` | `dist.all_to_all` — each rank sends distinct data to each other rank |
| `test_all_to_all_single.py` | `dist.all_to_all_single` — all-to-all with a single flat tensor |
| `test_all_to_all_single_ray.py` | Same as above but using Ray for process management |
| `test_broadcast.py` | `dist.broadcast` — root rank sends to all others |
| `test_reduce.py` | `dist.reduce` — reduce to a single root rank |
| `test_reduce_scatter.py` | `dist.reduce_scatter` — reduce then scatter shards to each rank |
| `test_scatter.py` | `dist.scatter` — root sends distinct chunks to each rank |
| `test_gather.py` | `dist.gather` — root collects from all ranks |
| `test_p2p.py` | Point-to-point `dist.send`/`dist.recv` |
| `test_dist_all.py` | Runs all collectives in sequence as a smoke test |
| `README.md` | Setup instructions and collective operation summary |

## For AI Agents

### Working In This Directory
- Launch with: `uv run torchrun --nproc_per_node=<N> test_<op>.py` (requires N GPUs).
- Single-GPU simulation: most scripts accept `--world_size 1` for local testing.
- `test_dist_all.py` is the quickest integration check.

<!-- MANUAL: -->
