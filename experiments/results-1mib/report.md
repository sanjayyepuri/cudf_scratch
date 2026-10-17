# Parquet layout experiment

8,388,608 identical non-null INT64 values; GPU sum verified as 41,933,446,337 in every sample.
GPU / driver: NVIDIA GeForce RTX 3090, 610.88. Kernel: 5.15.146.1-microsoft-standard-WSL2.
15 measured rounds; one unmeasured warm-up per file; shuffled file order each round.
Uncompressed PLAIN encoding, no dictionary or statistics. KVIKIO_COMPAT_MODE=ON (no GDS).

Times include metadata, file access, transfer, allocation and GPU decode, ending with GPU synchronization.
Sum is timed separately. CUDA initialization and table destruction are outside read timing.
Storage location: /tmp/cudf-layout-1mib. Files are warmed before measurement; no cache eviction.

| Rows/group | Page target (KiB) | Groups | Total pages | Pages/chunk | Median read (ms) | Min–max (ms) | Effective GiB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 131,072 | 1024 | 64 | 64 | 1–1 | 16.840 | 15.692–17.679 | 3.71 |
| 262,144 | 1024 | 32 | 64 | 2–2 | 16.653 | 15.322–18.502 | 3.75 |
| 524,288 | 1024 | 16 | 64 | 4–4 | 16.602 | 15.132–18.116 | 3.76 |
| 1,048,576 | 1024 | 8 | 64 | 8–8 | 17.295 | 15.983–18.025 | 3.61 |
| 2,097,152 | 1024 | 4 | 64 | 16–16 | 16.832 | 15.607–19.399 | 3.71 |
| 4,194,304 | 1024 | 2 | 64 | 32–32 | 17.333 | 15.760–18.076 | 3.61 |
| 8,388,608 | 1024 | 1 | 64 | 64–64 | 17.703 | 16.118–19.488 | 3.53 |

Row-group sweep: page target = 1,024 KiB. Only row-group size varies.
Page targets are approximate; counts above come from actual Parquet encoding statistics.
Effective bandwidth counts decoded amount bytes; it is not physical storage throughput.
This is a small single-column layout study, not a reproduction of the paper’s full TPC-H/GDS benchmark.
Repeated rounds share one GPU process; medians and ranges are descriptive, not confidence intervals.
