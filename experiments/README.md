# Row groups and pages

The GPU program is C++; Python is only an optional fixture generator and results
summarizer. This keeps Arrow and benchmarking helpers out of the C++ SDK/build.
The C++ compiler and SDK continue to come from the locked Bazel toolchain.
The helper environment is version-pinned separately; it is not a hermetic Bazel target.

From the project root:

```bash
bazelisk build //:layout_benchmark
uv venv --python 3.12 /tmp/cudf-layout-venv
uv pip install --python /tmp/cudf-layout-venv/bin/python -r experiments/requirements.txt
/tmp/cudf-layout-venv/bin/python experiments/layout.py
```

Alternatively create a Python 3.12 venv with your usual tooling and install the
requirements with pip. `--help` lists options for repetitions, data and output
locations, row count, and the benchmark executable.

The default creates eight approximately 64 MiB Parquet files on the Linux
filesystem under `/tmp/cudf-layout-data`. Existing files with the same layout
names there are regenerated. Keep this on the Linux filesystem when using WSL,
since Windows-mounted storage introduces another performance variable.

Every file contains the same 8,388,608 non-null INT64 `amount` values, generated
with seed 42. Compression, dictionary encoding and statistics are disabled;
encoding is PLAIN. These are deliberate controls, not recommended production
settings. The sum must match the CPU reference for every GPU measurement.

- **Page sweep:** 2,097,152 rows/group with page targets of 16 KiB, 64 KiB,
  256 KiB, 1 MiB, and 16 MiB. The last case produces one page per chunk with
  this data, exposing the low-parallelism regime.
- **Row-group sweep:** 64 KiB page target with 131,072, 524,288, 2,097,152,
  and 8,388,608 rows/group. One layout is shared by both sweeps.

The writer's page target is approximate. Actual data-page counts come from
Parquet encoding statistics read by fastparquet. A page belongs to one column
chunk; it does not span multiple columns. With fixed data size and page size,
the row-group sweep can keep the total page count constant while changing how
pages are grouped into chunks.

The benchmark runs in one GPU process, warms every file once, then shuffles
file order with a fixed seed in each of nine measured rounds. Host wall-clock
read timing ends with `cudaDeviceSynchronize`, so asynchronous GPU work is
included. Reading includes metadata, allocation, host I/O, transfer and decode;
the GPU sum and scalar transfer are measured separately. Table destruction is
outside both intervals. The first warm-up may still include initialization work;
no warm-up samples enter the report.

`KVIKIO_COMPAT_MODE=ON` explicitly selects compatibility I/O, without GDS.
Caches are not evicted: this is a warm-cache end-to-end read experiment, not
a cold-storage or decoder-only measurement. The small single-column dataset
also differs from the paper's A100, PystachIO and TPC-H setup. Medians and ranges
are descriptive; small differences with overlapping ranges are inconclusive.

Outputs in `experiments/results/`:

- `report.md`: readable measurements and methodology.
- `samples.csv`: each measured GPU read and sum timing.
- `summary.csv`: per-layout medians/ranges, file sizes and actual page counts.
- `manifest.json`: seed, expected sum, versions, GPU/driver and layouts.
- `stderr.log`: runtime diagnostics.

Large generated Parquet files stay outside the repository. Rerunning replaces
these result files. No privileged cache flushing is required.

Sources: [Parquet concepts](https://parquet.apache.org/docs/concepts/),
[Arrow writer options](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html),
[paper, Section 3](https://arxiv.org/pdf/2602.17335v1).

## Fixed 1 MiB pages: row-group sweep

To rerun the follow-up while preserving the earlier results:

```bash
/tmp/cudf-layout-venv/bin/python experiments/layout.py \
  --sweep row-groups \
  --row-group-page-bytes 1048576 \
  --row-group-sizes 131072 262144 524288 1048576 2097152 4194304 8388608 \
  --repeats 15 \
  --data-dir /tmp/cudf-layout-1mib \
  --output experiments/results-1mib
```

Here “one megabyte” uses the binary unit: 1 MiB = 1,048,576 bytes. The row-group
sweep options also allow other page targets and lists of row-group sizes.

[Recorded results](results-1mib/report.md) cover seven layouts and 105 verified
measurements. Read medians were 16.60–17.70 ms, with overlapping observed ranges;
this run does not establish a clear best row-group size.

All seven files have **64 total data pages**. The smallest row groups contain
one page in each of 64 column chunks; the largest contains all 64 pages in one
chunk. A single page *per chunk* therefore does not necessarily mean too few
pages across the whole scan. This differs from the earlier 16 MiB-page case,
which had only four total pages. We have not profiled the kernels to attribute
timing differences to a specific cause.

## CUDA tracing with Nsight Systems

[Experiment write-up](REPORT.md) · [cuDF source-reading guide](SOURCE_GUIDE.md)

Install the profiler locally on Linux x86_64 / WSL, using Python 3.8+ and
`dpkg-deb` (provided by Ubuntu's dpkg package):

```bash
python3 experiments/install_nsys.py
```

The helper downloads NVIDIA Nsight Systems **2026.3.2.476**, checks its pinned
SHA-256, and extracts the official Ubuntu package without root or system package
changes. It was tested on this Ubuntu 20.04 WSL2 host. The download is about
448 MB and the extracted package includes GUI components, although we use the CLI.
This profiler is a host tool separate from the hermetic Bazel application build.

Default executable:
`/tmp/cudf-profiling/nsight/opt/nvidia/nsight-systems/2026.3.2/bin/nsys`.
Use `--prefix` and `--cache` for persistent locations; `/tmp` can be cleared.
A custom installation path must also be passed to `trace_pages.py --nsys`.

Reproduce the page-only baseline and trace:

```bash
bazelisk build //:layout_benchmark
/tmp/cudf-layout-venv/bin/python experiments/layout.py \
  --sweep pages --repeats 15 \
  --data-dir /tmp/cudf-layout-pages --output experiments/results-pages
/tmp/cudf-layout-venv/bin/python experiments/trace_pages.py
```

The tracer uses software CUDA/CUPTI tracing and NVTX ranges. `CUDF_PROFILE=1`
enables the enclosing `measured` range in the C++ driver; capture starts after
warm-up and stops at its end. `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` allows the
ASCII NVTX capture label. CPU sampling and context-switch tracing are disabled.
The wrapper sets these variables and `KVIKIO_COMPAT_MODE=ON` automatically.
Hardware counters are not requested. A counter probe reported insufficient
privilege here; that does not prevent the kernel/transfer tracing used below.

The capture command, profiler version and event counts are recorded in
`results-pages-traced/capture.json`. Five measured rounds cover all five page
sizes; profiled sums must still match the baseline CPU reference.

Files:

- `traces/pages.nsys-rep`: full interactive timeline; open it in a compatible
  Nsight Systems GUI. A Windows GUI can open the report produced in WSL.
- `traces/pages.sqlite`: exported events, useful for SQL inspection.
- `results-pages-traced/read-phases.csv`: per-read wall time, union of kernel
  intervals, transfer intervals, gaps and maximum concurrent kernel count.
- `results-pages-traced/kernels.csv`: kernel names, event timestamps, streams,
  launch grids, threads/block, register counts and shared memory.
- `results-pages-traced/cuda-apis.csv`: host CUDA runtime calls inside each read.
- `results-pages-traced/timeline.csv`: event intervals for one read per layout.
- `results-pages-traced/summary.csv` and `report.md`: summaries.

Analyze an existing capture without rerunning GPU work:

```bash
/tmp/cudf-layout-venv/bin/python experiments/trace_pages.py --analyze-only
```

Optional scientific figure (adds Matplotlib only to the helper environment):

```bash
uv pip install --python /tmp/cudf-layout-venv/bin/python -r experiments/profiling-requirements.txt
/tmp/cudf-layout-venv/bin/python experiments/plot_traces.py
```

The figure is saved as PNG and SVG in `results-pages-traced/`.

Interpret the measurements carefully: profiled timing includes instrumentation
cost. Use the **unprofiled baseline** for performance comparisons. Traced activity
is union time of kernel/memcpy intervals, not SM utilization or occupancy.
A gap has no recorded kernel or memcpy; it need not mean the entire GPU is idle.
CUDA memsets are not included in that definition. CPU work, file access,
allocation, synchronization and tool overhead can all contribute to gaps.
Different streams do not prove concurrent execution: inspect interval overlap.

Lossless `.gz` archives of the baseline page and subpass traces and SQLite exports are checked into Git,
alongside CSV measurements, figures and notes. New capture names are ignored by
default. Large generated Parquet fixtures remain reproducible from the scripts
and saved manifests rather than being copied into the repository.

Sources: [NVIDIA profiler packages](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/),
[Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[post-collection analysis](https://docs.nvidia.com/nsight-systems/AnalysisGuide/).

## Compressed multi-row-group follow-up

[The subpass experiment](SUBPASSES.md) independently varies the pass-memory and
output-table hints on eight compressed row groups and three selected integer columns.
It verifies actual subpass counts using CUDA/NVTX traces.

```bash
bazelisk build //:subpass_benchmark
/tmp/cudf-layout-venv/bin/python experiments/subpasses.py
```

The [saved subpass report](results-subpasses/report.md) and
[comparison chart](results-subpasses/subpasses.png) distinguish input passes,
page subpasses and returned tables. To redraw the chart without GPU work:

```bash
/tmp/cudf-layout-venv/bin/python experiments/plot_subpasses.py
```

## GPU memory follow-up

After running the subpass sweep, build the updated driver and reuse the saved fixture:

```bash
bazelisk build //:subpass_benchmark
/tmp/cudf-layout-venv/bin/python experiments/memory_subpasses.py
```

This runs three separate memory-instrumented rounds per configuration, after an
excluded warm-up. It preserves the original timing and trace results, checks sums
and row counts, and joins memory peaks to the baseline counts and timings.
`--repeats` changes the memory rounds; `--analyze-only` regenerates the summary and
chart from saved samples without GPU work. Use the same repeat count when analyzing.

`CUDF_MEMORY=1` enables an RMM statistics adaptor around the current device
resource and a separate device-memory sampler in the C++ driver. With the default
CUDA resource, this adds no pool. All default cuDF allocations, including internal
temporaries, route through the adaptor; the script requires zero live tracked
bytes after each reader is destroyed. Every returned table is consumed and released.
The device sampler calls `cudaMemGetInfo` with a requested 1 ms sleep between
queries; short peaks may be missed. Device usage includes context/library overhead
and other GPU users. Neither metric establishes physical residency under WSL.

[Memory report and chart](results-subpasses/memory-report.md) ·
[Raw samples](results-subpasses/memory-samples.csv) ·
[Joined memory/time/count summary](results-subpasses/memory-summary.csv)

To restore all checked-in profiler captures after cloning:

```bash
gzip -dk experiments/traces/*.gz
```

This keeps the archives and recreates the original Nsight reports and SQLite
exports. The experiment scripts also regenerate the uncompressed captures.
