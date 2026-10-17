# Reading a Parquet reader: from file format to cuDF kernels

Start with the format, then the host orchestration, then the CUDA launch code.
The links below pin an upstream cuDF snapshot so line numbers stay stable. This
snapshot is not a verified exact source revision of our prebuilt libcudf binary.
The observed kernel names and launch mappings agree with our trace.

## A short Parquet primer

A file contains row groups. Each row group contains one chunk per column, and
each chunk contains pages. Pages are encoding/compression units, unrelated to
operating-system memory pages. A reader typically:

1. Reads the footer to discover schema, row groups, chunk offsets and sizes.
2. Selects columns and row groups; optional statistics/indexes can prune work.
3. Reads selected column chunks, parses page headers, and prepares descriptors.
4. Decompresses page payloads when needed, then decodes their values.
5. Reconstructs validity/nested structure as required and builds output columns.

For our selected INT64 column, the reader can skip unrelated columns. The
encoding is PLAIN and there is no compression, dictionary, or nested data. This
makes the path easier to follow, though nullable-column bookkeeping can remain.

Read [Parquet concepts](https://parquet.apache.org/docs/concepts/),
[file layout](https://parquet.apache.org/docs/file-format/), and
[data pages](https://parquet.apache.org/docs/file-format/data-pages/) first.
For CPU C++ usage, see [Arrow's Parquet guide](https://arrow.apache.org/docs/cpp/parquet.html).

## Follow cuDF in this order

| Source | What to inspect |
|---|---|
| [Public read_parquet](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/functions.cpp#L638) | Creates data sources and the reader |
| [reader::read](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader.cpp#L24) | Thin forwarding layer |
| [reader_impl::read](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl.cpp#L1082) | prepare_data, then read_chunk_internal |
| [prepare_data](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl.cpp#L634) | Whole-file preprocessing and chunking |
| [Read selected chunks](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl_preprocess.cu#L592) | Encoded input bytes and chunk offsets |
| [Count pages and allocate descriptors](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl_preprocess.cu#L658) | count, allocate PageInfo, decode headers |
| [Allocate and decode output](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl.cpp#L710) | allocate_columns before decode_page_data |
| [Header-kernel launches](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/page_hdr.cu#L936) | One warp per chunk |
| [Fixed-width decoder launch](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/decode_fixed.cu#L1389) | One thread block per page |

For compressed files, inspect [subpass decompression](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl_chunking.cu#L347).
For footer parsing, inspect [metadata construction](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl_helpers.cpp#L525).
Other datatypes, encodings and nested schemas dispatch to additional decoder paths;
one block per page here is a property of this decoder, not a Parquet requirement.

## The two header kernels

`count_page_headers_kernel` walks chunk headers and counts pages.
`decode_page_headers_kernel` fills page descriptors with the information needed
for decoding. Both assign one warp per chunk. They do not decode the column's
actual values; `decode_page_data_generic` handles that in our fixed-width path.

Our trace has 33 kernel launches per read, so these three stages are only a
simplified view. Helper operations also launch kernels.

## Where memory is allocated

“Page memory” refers to several different things:

- **Encoded input bytes:** read_column_chunks delegates to read_column_chunks_async,
  using device buffers to hold chunk data. Chunk byte sizes are already known
  from metadata, before individual pages are counted.
- **Page descriptors:** after total_pages is known, read_compressed_data creates
  a zeroed device vector of PageInfo. This is per-page metadata, not a second
  allocation of every page payload.
- **Decoded output:** [allocate_columns](https://github.com/NVIDIA/cudf/blob/816f682244c080b438792eec6bc8eda6c3b0583a/cpp/src/io/parquet/reader_impl_preprocess.cu#L1046)
  prepares output-column buffers before the value-decoding kernel runs.
- **Compressed-file intermediates:** compressed inputs may also require
  decompression buffers, dictionaries and scratch space; that path is disabled
  by our uncompressed fixtures.

These allocations are orchestrated by host code through device memory resources.
Our uncompressed fixed-width page decoders write directly into output-column
buffers rather than allocate a separate decoded buffer for every page.

## Relating the code to our trace

[The experiment report](REPORT.md) contains measured grids and kernel times.
More pages increased the data decoder's grid size, not the number of launches.
Header kernels remained one block in our four-chunk case. Kernel execution did
not overlap in the captured reads. Grid size is available scheduled work;
resident blocks and SM occupancy are separate questions.
