# Page-size sweep: CUDA trace

25 profiled sums verified; 875 kernel events and 1425 memcpy events recorded.
5 profiled rounds, following one excluded warm-up per layout. Compare performance using the separate unprofiled baseline.

| Page target (KiB) | Pages | Baseline read (ms) | Traced read (ms) | Kernel busy (ms) | Memcpy busy (ms) | Activity (%) | Gap (ms) | Decoder blocks | Decoder time (ms) | Header kernels (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4096 | 24.681 | 25.164 | 7.513 | 5.807 | 52.2 | 12.029 | 4096 | 0.189 | 7.119 |
| 64 | 1024 | 18.541 | 19.201 | 2.017 | 5.461 | 40.0 | 11.497 | 1024 | 0.178 | 1.721 |
| 256 | 256 | 17.231 | 19.396 | 0.716 | 5.504 | 33.2 | 12.951 | 256 | 0.202 | 0.419 |
| 1,024 | 64 | 17.412 | 17.903 | 0.800 | 5.495 | 35.2 | 11.617 | 64 | 0.611 | 0.113 |
| 16,384 | 4 | 26.378 | 27.462 | 9.862 | 5.519 | 57.9 | 11.649 | 4 | 9.775 | 0.012 |

All displayed times are per-layout medians. Columns need not add up: events can overlap and medians are taken independently.
Activity means union time of recorded kernels and memcpy events divided by the enclosing synchronized read range.
It is not SM utilization, occupancy, or a hardware counter. Gap means no recorded kernel or memcpy; it can include CPU work, file access, allocation, scheduling and profiling overhead.
CUDA memsets, if any, are not included in this activity definition. Decoder time sums kernels whose names contain decode_page_data.
Capture uses software CUDA tracing, NVTX labels and no CPU sampling or hardware counters.
The first measured iteration supplies timeline.csv; use the full .nsys-rep in Nsight Systems for interactive inspection.

| Page target (KiB) | Kernels/read | Data-decoder launches/read | Maximum concurrent kernels |
|---:|---:|---:|---:|
| 16 | 33 | 1 | 1 |
| 64 | 33 | 1 | 1 |
| 256 | 33 | 1 | 1 |
| 1,024 | 33 | 1 | 1 |
| 16,384 | 33 | 1 | 1 |

Concurrency is measured by overlap of kernel execution intervals, not by stream count or launch count.
Grid size describes scheduled blocks, not simultaneously resident blocks.
