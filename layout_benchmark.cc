#include <cudf/aggregation.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void synchronize()
{
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv)
{
  try {
    if (argc < 3) throw std::invalid_argument("Usage: layout_benchmark repetitions file.parquet [...]");
    auto repeats = std::stoi(argv[1]);
    if (repeats < 1) throw std::invalid_argument("Repetitions must be positive");
    std::vector<int> order(argc - 2);
    std::iota(order.begin(), order.end(), 2);
    std::mt19937 rng(42);
    using clock = std::chrono::steady_clock;
    std::cout << "file,iteration,rows,sum,read_ms,sum_ms\n";
    // Round zero warms CUDA, allocations and each file's OS cache; omit it from results.
    bool const profile = std::getenv("CUDF_PROFILE") != nullptr;
    for (int round = 0; round <= repeats; ++round) {
      if (profile && round == 1) {
        nvtxRangePushA("measured");
      }
      std::shuffle(order.begin(), order.end(), rng);
      for (auto index : order) {
        auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{argv[index]})
                         .column_names({"amount"}).ignore_missing_columns(false).build();
        auto label = std::string(round == 0 ? "warmup:" : "sample:") + argv[index] +
                     ":" + std::to_string(round);
        nvtxRangePushA(label.c_str());
        synchronize();
        nvtxRangePushA("benchmark/read");
        auto start = clock::now();
        auto result = cudf::io::read_parquet(options);
        synchronize();
        auto read_end = clock::now();
        nvtxRangePop();
        nvtxRangePushA("benchmark/sum");
        auto column = result.tbl->view().column(0);
        if (column.type() != cudf::data_type{cudf::type_id::INT64})
          throw std::invalid_argument("Expected INT64 amount column");
        auto sum = cudf::reduce(column, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::INT64});
        if (!sum->is_valid()) throw std::invalid_argument("Expected nonempty, non-null data");
        auto value = static_cast<cudf::numeric_scalar<int64_t> const&>(*sum).value();
        synchronize();
        auto end = clock::now();
        nvtxRangePop();
        nvtxRangePop();
        if (round > 0) {
          std::cout << '\"' << argv[index] << "\"," << round << ',' << result.tbl->num_rows()
                    << ',' << value << ','
                    << std::chrono::duration<double, std::milli>(read_end - start).count() << ','
                    << std::chrono::duration<double, std::milli>(end - read_end).count() << '\n';
        }
      }
    }
    if (profile) {
      nvtxRangePop();
    }
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
