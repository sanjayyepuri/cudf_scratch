#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;
double elapsed(clock_type::time_point begin) {
  return std::chrono::duration<double,std::milli>(clock_type::now()-begin).count();
}
void check(arrow::Status const& status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}
template<class T> T checked(arrow::Result<T> result) {
  check(result.status()); return std::move(result).ValueOrDie();
}
std::unique_ptr<parquet::arrow::FileReader> open_reader(std::string const& path) {
  parquet::arrow::FileReaderBuilder builder;
  check(builder.OpenFile(path,false));
  parquet::ArrowReaderProperties properties;
  properties.set_use_threads(false);
  properties.set_pre_buffer(false);
  builder.properties(properties);
  return checked(builder.Build());
}
struct sample {
  int64_t rows=0,sum=0,max_wave_value_bytes=0;
  int groups=0;
  double read_ms=0,sum_ms=0,total_ms=0;
};
sample run(std::string const& path,int workers,bool single) {
  sample out;
  auto total_start=clock_type::now();
  auto master=open_reader(path);
  out.groups=master->parquet_reader()->metadata()->num_row_groups();
  std::shared_ptr<arrow::Schema> schema;
  check(master->GetSchema(&schema));
  std::vector<int> columns;
  for (auto const& name : single ? std::vector<std::string>{"amount"}
                                 : std::vector<std::string>{"amount","customer_id","quantity"}) {
    auto index=schema->GetFieldIndex(name);
    if (index<0) throw std::runtime_error("Missing selected column");
    columns.push_back(index); // Both fixtures contain only flat primitive leaves.
  }
  master.reset();
  std::vector<std::shared_ptr<arrow::Table>> tables(workers);
  std::vector<int64_t> sums(workers);
  std::vector<std::exception_ptr> errors(workers);
  std::atomic<bool> aborted{false};
  std::barrier<> phase(workers+1);
  std::vector<std::jthread> pool;
  for (int worker=0;worker<workers;++worker) {
    pool.emplace_back([&,worker] {
      std::unique_ptr<parquet::arrow::FileReader> reader;
      try { reader=open_reader(path); }
      catch (...) { errors[worker]=std::current_exception();aborted=true; }
      phase.arrive_and_wait();
      for (int first=0;first<out.groups;first+=workers) {
        phase.arrive_and_wait();
        try {
          if (!aborted && first+worker<out.groups) {
            check(reader->ReadRowGroup(first+worker,columns,&tables[worker]));
            if (tables[worker]->num_columns()!=static_cast<int>(columns.size()))
              throw std::runtime_error("Incorrect CPU projection");
          }
        } catch (...) { errors[worker]=std::current_exception();aborted=true; }
        phase.arrive_and_wait();
        phase.arrive_and_wait();
        sums[worker]=0;
        try {
          if (!aborted && tables[worker]) {
            auto result=checked(arrow::compute::Sum(tables[worker]->column(0)));
            auto scalar=result.scalar();
            if (!scalar->is_valid || scalar->type->id()!=arrow::Type::INT64)
              throw std::runtime_error("Invalid CPU sum");
            sums[worker]=static_cast<arrow::Int64Scalar const&>(*scalar).value;
          }
        } catch (...) { errors[worker]=std::current_exception();aborted=true; }
        phase.arrive_and_wait();
      }
    });
  }
  phase.arrive_and_wait();
  out.read_ms=elapsed(total_start);
  for (int first=0;first<out.groups;first+=workers) {
    auto begin=clock_type::now();
    phase.arrive_and_wait();phase.arrive_and_wait();
    out.read_ms+=elapsed(begin);
    int64_t wave_rows=0;
    for (auto const& table : tables) if (table) wave_rows+=table->num_rows();
    out.rows+=wave_rows;
    out.max_wave_value_bytes=std::max(out.max_wave_value_bytes,wave_rows*(single?8:20));
    begin=clock_type::now();
    phase.arrive_and_wait();phase.arrive_and_wait();
    out.sum_ms+=elapsed(begin);
    for (auto value : sums) out.sum+=value;
    for (auto& table : tables) table.reset();
  }
  for (auto& worker : pool) worker.join();
  for (auto const& error : errors) if (error) std::rethrow_exception(error);
  out.total_ms=elapsed(total_start);
  return out;
}
int main(int argc,char** argv) {
  try {
    if (argc!=5) throw std::runtime_error("Usage: cpu_benchmark repeats file threads single|three");
    int repeats=std::stoi(argv[1]),threads=std::stoi(argv[3]);
    std::string mode=argv[4];
    if (repeats<1 || threads<1 || (mode!="single" && mode!="three"))
      throw std::runtime_error("Invalid CPU benchmark configuration");
    std::cout<<"iteration,threads,rows,sum,columns,row_groups,read_ms,sum_ms,total_ms,max_wave_value_bytes,process_peak_rss_bytes\n";
    for (int iteration=0;iteration<=repeats;++iteration) {
      std::cerr<<"begin iteration="<<iteration<<" threads="<<threads<<std::endl;
      auto result=run(argv[2],threads,mode=="single");
      rusage usage{};
      if (getrusage(RUSAGE_SELF,&usage)!=0) throw std::runtime_error("getrusage failed");
      if (iteration) std::cout<<iteration<<','<<threads<<','<<result.rows<<','<<result.sum
        <<','<<(mode=="single"?1:3)<<','<<result.groups<<','<<result.read_ms<<','<<result.sum_ms
        <<','<<result.total_ms<<','<<result.max_wave_value_bytes<<','
        <<int64_t(usage.ru_maxrss)*1024<<'\n'<<std::flush;
      std::cerr<<"done iteration="<<iteration<<" ms="<<result.total_ms<<std::endl;
    }
  } catch (std::exception const& error) { std::cerr<<error.what()<<'\n';return 1; }
}
