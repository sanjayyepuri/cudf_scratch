# Native C++ CPU baseline on the same 15 GiB Parquet files

CPU: AMD Ryzen 9 3950X 16-Core Processor; 32 available logical CPUs. Arrow C++ 19.0.1; same hermetic LLVM toolchain as the GPU driver.
All 24 measured CPU reads passed row count, selected-column count and CPU-reference sum checks. Three rounds after one excluded warm-up per configuration.

| Workload | CPU workers | CPU read (s) | CPU sum (s) | CPU total (s) | GPU total (s) | GPU / CPU time |
|---|---:|---:|---:|---:|---:|---:|
| ZSTD · three integers | 1 | 24.611 | 0.337 | 25.084 | 3.850 | 0.15× |
| ZSTD · three integers | 8 | 3.809 | 0.170 | 4.454 | 3.850 | 0.86× |
| ZSTD · three integers | 32 | 2.970 | 0.157 | 3.678 | 3.850 | 1.05× |
| PLAIN · 1 MiB Pages | 8 | 7.595 | 0.397 | 8.629 | 7.047 | 0.82× |
| PLAIN · 16 KiB Pages | 8 | 7.754 | 0.381 | 8.871 | 9.934 | 1.12× |
| PLAIN · 16 MiB Pages | 8 | 7.516 | 0.429 | 8.667 | 6.843 | 0.79× |
| PLAIN · 256 KiB Pages | 8 | 6.981 | 0.395 | 8.031 | 6.892 | 0.86× |
| PLAIN · 64 KiB Pages | 8 | 7.107 | 0.387 | 8.231 | 7.401 | 0.90× |

**GPU / CPU time below 1 means the GPU driver was faster; above 1 means the CPU driver was faster.** GPU uses a fixed 2 GiB pass hint and disabled output hint, not the fastest hint selected after looking at results.

CPU reader: one independent Arrow Parquet reader per fixed worker. Each wave reads at most one row group per worker, reduces the amount columns in parallel, and releases all tables before the next wave. Nested Arrow threading and prebuffering are disabled. Both drivers decode the same selected columns and reduce only amount.
The ZSTD fixture tests 1, 8 and 32 workers. Uncompressed Page fixtures use 8 workers to keep decoded waves at most 2 GiB before scratch and input buffers; 32 workers could hold 8 GiB of values plus encoded inputs and exceed host RAM. Eight workers are not a claim of full-machine CPU saturation or best CPU performance.
Read and sum phases are timed separately, with total including metadata, worker setup and cleanup. GPU total includes CUDA allocation, synchronization, copies and per-table scalar transfers. CPU and GPU batching and library implementations differ; this is an explicit implementation baseline, not a GPU-only kernel speedup or universally optimized engine comparison.
CPU and GPU runs do not overlap. Cache flushing/residency is not controlled. Compressed input is 5.9 GiB; uncompressed files are each 15 GiB and exceed available host RAM. File-cache state, I/O implementations, scheduling and batch policy can affect the comparison.
Each timing column is an independently computed median; read and sum medians need not add exactly to total median. See raw samples and observed min–max ranges in summary.csv.
CPU RSS is a process-lifetime high-water mark including excluded warm-up, native allocator caches and libraries. It is not directly comparable to peak live GPU RMM requests.

```bash
bazelisk build //:cpu_benchmark
/tmp/cudf-layout-venv/bin/python experiments/cpu_baseline.py
/tmp/cudf-layout-venv/bin/python experiments/cpu_report.py
```

[Raw samples](samples.csv) · [CPU summary](summary.csv) · [Joined comparison](comparison.csv) · [Hardware and method](method.json)
