# Parquet layout experiment

8,388,608 identical non-null INT64 values; GPU sum verified as 41,933,446,337 in every sample.
GPU / driver: NVIDIA GeForce RTX 3090, 610.88. Kernel: 5.15.146.1-microsoft-standard-WSL2.
9 measured rounds; one unmeasured warm-up per file; shuffled file order each round.
Uncompressed PLAIN encoding, no dictionary or statistics. KVIKIO_COMPAT_MODE=ON (no GDS).

Times include metadata, file access, transfer, allocation and GPU decode, ending with GPU synchronization.
Sum is timed separately. CUDA initialization and table destruction are outside read timing.
Storage location: /tmp/cudf-layout-data. Files are warmed before measurement; no cache eviction.

| Rows/group | Page target (KiB) | Groups | Total pages | Pages/chunk | Median read (ms) | Min–max (ms) | Effective GiB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 131,072 | 64 | 64 | 1024 | 16–16 | 16.979 | 16.059–17.603 | 3.68 |
| 524,288 | 64 | 16 | 1024 | 64–64 | 17.040 | 16.575–17.515 | 3.67 |
| 2,097,152 | 16 | 4 | 4096 | 1024–1024 | 24.677 | 24.222–26.092 | 2.53 |
| 2,097,152 | 64 | 4 | 1024 | 256–256 | 19.014 | 17.322–20.921 | 3.29 |
| 2,097,152 | 256 | 4 | 256 | 64–64 | 16.830 | 15.671–18.279 | 3.71 |
| 2,097,152 | 1024 | 4 | 64 | 16–16 | 17.006 | 16.072–19.063 | 3.68 |
| 2,097,152 | 16384 | 4 | 4 | 1–1 | 25.838 | 25.038–26.487 | 2.42 |
| 8,388,608 | 64 | 1 | 1024 | 1024–1024 | 23.882 | 23.148–25.158 | 2.62 |

Page sweep: compare rows/group = 2,097,152. Row-group sweep: compare page target = 64 KiB.
Page targets are approximate; counts above come from actual Parquet encoding statistics.
Effective bandwidth counts decoded amount bytes; it is not physical storage throughput.
This is a small single-column layout study, not a reproduction of the paper’s full TPC-H/GDS benchmark.
Repeated rounds share one GPU process; medians and ranges are descriptive, not confidence intervals.
