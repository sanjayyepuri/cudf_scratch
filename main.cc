#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: hello /absolute/path/to/file.parquet\n";
    return 1;
  }

  try {
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info{argv[1]})
                     .build();
    auto result = cudf::io::read_parquet(options);
    std::cout << "Hello, libcudf! Read " << result.tbl->num_rows() << " rows and "
              << result.tbl->num_columns() << " columns into GPU memory.\n";
  } catch (std::exception const& error) {
    std::cerr << "Parquet read failed: " << error.what() << '\n';
    return 1;
  }
}
