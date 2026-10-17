# Parquet layout experiment

8,388,608 identical non-null INT64 values; GPU sum verified as 41,933,446,337 in every sample.
GPU / driver: NVIDIA GeForce RTX 3090, 610.88. Kernel: 5.15.146.1-microsoft-standard-WSL2.
15 measured rounds; one unmeasured warm-up per file; shuffled file order each round.
Uncompressed PLAIN encoding, no dictionary or statistics. KVIKIO_COMPAT_MODE=ON (no GDS).

Times include metadata, file access, transfer, allocation and GPU decode, ending with GPU synchronization.
Sum is timed separately. CUDA initialization and table destruction are outside read timing.
Storage location: /tmp/cudf-layout-pages. Files are warmed before measurement; no cache eviction.

| Rows/group | Page target (KiB) | Groups | Total pages | Pages/chunk | Median read (ms) | Min–max (ms) | Effective GiB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,097,152 | 16 | 4 | 4096 | 1024–1024 | 24.681 | 23.988–27.926 | 2.53 |
| 2,097,152 | 64 | 4 | 1024 | 256–256 | 18.541 | 17.627–20.346 | 3.37 |
| 2,097,152 | 256 | 4 | 256 | 64–64 | 17.231 | 15.995–19.096 | 3.63 |
| 2,097,152 | 1024 | 4 | 64 | 16–16 | 17.412 | 16.098–19.383 | 3.59 |
| 2,097,152 | 16384 | 4 | 4 | 1–1 | 26.378 | 25.264–27.823 | 2.37 |

Page sweep: rows/group = 2,097,152.
Page targets are approximate; counts above come from actual Parquet encoding statistics.
Effective bandwidth counts decoded amount bytes; it is not physical storage throughput.
This is a small single-column layout study, not a reproduction of the paper’s full TPC-H/GDS benchmark.
Repeated rounds share one GPU process; medians and ranges are descriptive, not confidence intervals.
