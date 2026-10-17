# GPU memory versus reader batching

All 24 measured reads matched the expected sum and row count. All ended with zero tracked live bytes and no device-query failures.

| Pass / output hint (MiB) | Input passes / subpasses | Baseline read (ms) | Peak live RMM (MiB) | Sampled device increase (MiB) |
|---|---:|---:|---:|---:|
| 0 / 0 | 1 / 1 | 153.10 | 1861.60 | 1866.00 |
| 16 / 0 | 8 / 248 | 6423.41 | 44.05 | 48.00 |
| 64 / 0 | 8 / 32 | 998.97 | 94.36 | 98.00 |
| 64 / 8 | 8 / 32 | 1237.66 | 94.36 | 98.00 |
| 64 / 32 | 8 / 32 | 1024.86 | 94.36 | 98.00 |
| 64 / 128 | 8 / 32 | 1035.57 | 94.36 | 98.00 |
| 256 / 0 | 4 / 12 | 462.27 | 259.16 | 266.00 |
| 512 / 0 | 2 / 6 | 295.13 | 510.77 | 514.00 |

Zero disables a hint. Timings and trace counts are from the original separate runs.
Peak RMM bytes are live allocation requests routed through the current RMM resource, including reader temporaries, output columns and reduction scratch. The original default CUDA resource is wrapped, without adding a pool. Buffers are released between returned tables.
3 measured memory rounds per configuration, shuffled with seed 42, after one excluded warm-up per configuration.
Device usage is total minus free from cudaMemGetInfo, sampled on a separate thread with a requested 1 ms sleep. The observed peak is a lower bound: short-lived allocations may be missed. It includes the CUDA context, driver/library allocations and other GPU users; its increase is relative to each sample start.
Neither allocator accounting nor cudaMemGetInfo proves physical residency under WSL or measures process-exclusive VRAM. Async frees and future pool allocators can separate logical live bytes from physically reserved memory. These hints remain soft targets, not capacity guarantees.
Use peak live bytes to compare working sets, allow headroom for untracked allocations and concurrent workloads, and measure again if retaining output tables or changing allocator/workload.

![Peak live GPU allocations versus baseline reader time](memory.png)

[Raw memory samples](memory-samples.csv) · [Joined summary](memory-summary.csv) · [Capture setup](memory-capture.json)
