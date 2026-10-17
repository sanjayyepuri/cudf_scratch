# libcudf + hermetic-llvm + Bazel

A small C++ example reads a Parquet file into GPU memory and prints its dimensions.

```bash
git clone https://github.com/sanjayyepuri/cudf_scratch.git
cd cudf_scratch
bazelisk run //:hello -- "$PWD/sample.parquet"
```

The sample contains 12 rows and four columns. Pass an absolute path to your own
Parquet file. The separate experiment drivers also run and verify GPU sums.

## Dependencies

- [hermetic-llvm](https://github.com/hermeticbuild/hermetic-llvm) 0.8.24 supplies
  the compiler, linker, and C++ headers. Host compiler detection is disabled.
- [rules_conda](https://github.com/calebzulawski/rules_conda) 0.3.2 installs the
  native SDK from the generated `conda.lock`, verifying package checksums.
  It supplies libcudf 26.10, CUDA 12.9, zlib, and their runtime dependencies.
- Standard `cc_library` rules expose the installed SDK to the application.
  `sdk.bzl` only exports its repository paths; it contains no download logic.

The SDK's GNU C++ runtime matches NVIDIA's prebuilt library ABI, including
its `_Float128` symbols. LLVM still performs compilation and linking.
There is no hand-maintained SDK archive manifest or copied zlib version script.

No system compiler, CUDA Toolkit, or Conda installation is needed. Bazel may
fetch Python as a build helper for the dependency rules; the application is C++.
The first build downloads the toolchain and SDK. Subsequent builds reuse them.

## Update dependencies

Edit the short `environment.yml`, then regenerate the lock:

```bash
bazelisk run //:deps.update
bazelisk test //:deps.test
```

Keep both `conda.lock` and `MODULE.bazel.lock` under version control. Ordinary
builds use the lock without resolving package versions again. Lock updates and
the lock-consistency test access the package channels.

This example targets Linux x86_64 / WSL2. Execution requires glibc 2.28 or newer,
a supported NVIDIA GPU, and a compatible host NVIDIA driver. Build actions are
sandboxed; GPU execution uses the host driver. Build cache: `/tmp/cudf-hermetic-bazel`.

## Layout experiment

See [experiments/README.md](experiments/README.md) for a repeatable C++ GPU
benchmark varying row-group and page sizes, with identical data and verified sums.
The hello-world reader stays unchanged. The recorded local measurements are in
[experiments/results/report.md](experiments/results/report.md).

The [experiment write-up](experiments/REPORT.md) includes CUDA tracing results.
See [profiler setup](experiments/README.md#cuda-tracing-with-nsight-systems) and
[the cuDF reader source guide](experiments/SOURCE_GUIDE.md) to explore further.
