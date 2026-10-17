# libcudf + hermetic-llvm + Bazel

One C++ program reads a Parquet file into a GPU table and prints its dimensions.

```bash
git clone https://github.com/sanjayyepuri/cudf_scratch.git
cd cudf_scratch
bazelisk run //:hello -- "$PWD/sample.parquet"
```

The sample contains 12 rows and four columns. Pass an absolute path to read your
own file. The first build downloads the compiler and native SDK and builds the
C++ runtime, so it takes longer than later builds.

The build targets Linux x86_64, including WSL2. Runtime requirements are glibc
2.28 or newer, a supported NVIDIA GPU, and a compatible host NVIDIA driver.

## What Bazel supplies

- `hermeticbuild/hermetic-llvm` 0.8.24 supplies the compiler, linker, C++ headers,
  and runtime. The target selects GNU `libstdc++` 14.3.0 to match libcudf's C++ ABI.
- `sdk-sources.json` pins libcudf 26.10.0 and its CUDA 12 native libraries by URL
  and SHA-256. `libcudf.bzl` extracts the native headers and shared libraries from
  NVIDIA's wheel archives. A pinned GCC support-library archive supplies
  `libgcc_s.so.1`; Conda is not needed to extract it. The application has no
  Python runtime dependency.
- zlib is built from a pinned Bazel dependency for libcudf's runtime use.

No Conda environment, Nix installation, system C++ compiler, or system CUDA
Toolkit is used. `.bazelrc` disables host compiler detection and uses sandboxed
build actions. Keep `MODULE.bazel.lock` under version control.

The GPU driver and the operating system's glibc remain runtime dependencies.
A hermetic compiler/build does not make GPU execution independent of the host.
The SDK contains prebuilt NVIDIA binaries; this project does not rebuild libcudf.

## Files to read

- `main.cc`: the complete Parquet reader.
- `BUILD.bazel`: the C++ target and ABI-compatible platform.
- `MODULE.bazel`: the pinned toolchain and library dependencies.
- `libcudf.bzl` and `sdk-sources.json`: native SDK packaging.

Validated on the local RTX 3090 with both the 12-row fixture and a million-row
Parquet file, with `CC`, `CXX`, `CONDA_PREFIX`, and `LD_LIBRARY_PATH` unset.

Build cache: `/tmp/cudf-hermetic-bazel`. No CUDA compiler is needed for this
example because the application calls precompiled libcudf GPU operations.

References: [hermetic-llvm](https://github.com/hermeticbuild/hermetic-llvm),
[libcudf Parquet reader](https://docs.nvidia.com/cudf/latest/libcudf/api_docs/io_readers/).
