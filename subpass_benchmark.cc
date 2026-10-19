#include <cudf/aggregation.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <memory>
#include <thread>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct range {
  explicit range(char const* name) { nvtxRangePushA(name); }
  ~range() { nvtxRangePop(); }
};
void sync_gpu()
{
  auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
using timer = std::chrono::steady_clock;
double milliseconds(timer::time_point begin) {
  return std::chrono::duration<double, std::milli>(timer::now() - begin).count();
}
struct config { size_t pass, chunk; };

// Optional separate memory runs: count every allocation routed through RMM,
// including cuDF temporaries using the current resource and default output MR.
struct memory_probe {
  cuda::mr::any_resource<cuda::mr::device_accessible> previous;
  rmm::mr::statistics_resource_adaptor stats;
  std::jthread sampler;
  size_t device_start = 0, device_peak = 0, samples = 0;
  int failures = 0;
  memory_probe() : previous(rmm::mr::reset_current_device_resource()), stats(previous) {
    rmm::mr::set_current_device_resource({stats});
    sample(); device_start = device_peak;
    int device = 0;
    cudaGetDevice(&device);
    sampler = std::jthread([this, device](std::stop_token stop) {
      if (cudaSetDevice(device) != cudaSuccess) { ++failures; return; }
      while (!stop.stop_requested()) {
        sample();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }
  void sample() {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) { ++failures; return; }
    device_peak = std::max(device_peak, total - free); ++samples;
  }
  void stop() {
    sampler.request_stop();
    if (sampler.joinable()) sampler.join();
  }
  ~memory_probe() {
    stop();
    rmm::mr::set_current_device_resource(std::move(previous));
  }
};

int main(int argc, char** argv)
{
  try {
    if (argc < 4) throw std::invalid_argument("Usage: subpass_benchmark repetitions file.parquet pass_bytes:chunk_bytes [...]");
    auto const repeats = std::stoi(argv[1]);
    if (repeats < 1) throw std::invalid_argument("Repetitions must be positive");
    std::vector<config> configs;
    for (int i = 3; i < argc; ++i) {
      std::string arg{argv[i]}; auto colon = arg.find(':');
      if (colon == std::string::npos) throw std::invalid_argument("Expected pass_bytes:chunk_bytes");
      configs.push_back({std::stoull(arg.substr(0, colon)), std::stoull(arg.substr(colon + 1))});
    }
    bool const single_column = std::getenv("CUDF_BENCH_SINGLE_COLUMN") != nullptr;
    std::vector<std::string> const selected_columns = single_column
      ? std::vector<std::string>{"amount"}
      : std::vector<std::string>{"amount", "customer_id", "quantity"};
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{argv[2]})
                     .column_names(selected_columns)
                     .ignore_missing_columns(false).build();
    std::mt19937 rng{42};
    bool const memory = std::getenv("CUDF_MEMORY") != nullptr;
    bool const profile = std::getenv("CUDF_PROFILE") != nullptr;
    std::cout << "pass_limit,chunk_limit,iteration,rows,sum,columns,output_chunks,max_chunk_rows,read_ms,sum_ms,total_ms";
    if (memory) std::cout << ",rmm_peak_bytes,rmm_final_bytes,rmm_total_bytes,device_start_bytes,device_sampled_peak_bytes,device_samples,device_sample_failures";
    std::cout << '\n';
    for (int round = 0; round <= repeats; ++round) {
      if (profile && round == 1) nvtxRangePushA("measured");
      std::shuffle(configs.begin(), configs.end(), rng);
      for (auto cfg : configs) {
        auto label = std::string(round == 0 ? "warmup:" : "sample:") +
                     std::to_string(cfg.pass) + ":" + std::to_string(cfg.chunk) + ":" + std::to_string(round);
        range sample{label.c_str()};
        std::cerr << "begin " << label << std::endl;
        sync_gpu();
        auto probe = memory ? std::make_unique<memory_probe>() : nullptr;
        auto total_start = timer::now();
        double read_ms = 0, sum_ms = 0;
        int64_t rows = 0, sum_value = 0;
        int chunks = 0, columns = 0;
        cudf::size_type max_rows = 0;
        {
          auto begin = timer::now();
          cudf::io::chunked_parquet_reader reader(cfg.chunk, cfg.pass, options);
          sync_gpu(); read_ms += milliseconds(begin);
          while (true) {
            begin = timer::now();
            bool more;
            { range phase{"benchmark/has_next"}; more = reader.has_next(); sync_gpu(); }
            read_ms += milliseconds(begin);
            if (!more) break;
            begin = timer::now();
            auto result = [&] { range phase{"benchmark/read_chunk"};
              auto out = reader.read_chunk(); sync_gpu(); return out; }();
            read_ms += milliseconds(begin);
            auto count = result.tbl->num_rows();
            rows += count; ++chunks;
            columns = result.tbl->num_columns();
            max_rows = std::max(max_rows, count);
            if (columns != static_cast<int>(selected_columns.size()))
              throw std::runtime_error("Unexpected output column count");
            if (count == 0) throw std::runtime_error("Unexpected empty chunk in nonempty benchmark fixture");
            begin = timer::now();
            { range phase{"benchmark/sum"};
              auto column = result.tbl->view().column(0);
              if (column.type() != cudf::data_type{cudf::type_id::INT64})
                throw std::runtime_error("Expected INT64 amount");
              auto value = cudf::reduce(column, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                                        cudf::data_type{cudf::type_id::INT64});
              if (!value->is_valid()) throw std::runtime_error("Invalid sum");
              sum_value += static_cast<cudf::numeric_scalar<int64_t> const&>(*value).value();
              sync_gpu();
            }
            sum_ms += milliseconds(begin);
          }
        }
        sync_gpu();
        auto const total_ms = milliseconds(total_start);
        if (probe) probe->stop();
        std::cerr << "done " << label << " rows=" << rows << " chunks=" << chunks << " ms=" << total_ms << std::endl;
        if (round > 0) {
          std::cout << cfg.pass << ',' << cfg.chunk << ',' << round << ',' << rows << ',' << sum_value
                    << ',' << columns << ',' << chunks << ',' << max_rows << ','
                    << read_ms << ',' << sum_ms << ',' << total_ms;
          if (probe) {
            auto bytes = probe->stats.get_bytes_counter();
            std::cout << ',' << bytes.peak << ',' << bytes.value << ',' << bytes.total
                      << ',' << probe->device_start << ',' << probe->device_peak
                      << ',' << probe->samples << ',' << probe->failures;
          }
          std::cout << '\n' << std::flush;
        }
      }
    }
    if (profile) nvtxRangePop();
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
