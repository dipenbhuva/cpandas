cmake_minimum_required(VERSION 3.16)

project(cpandas C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(cpandas
  src/cpandas.c
)

target_include_directories(cpandas
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(NOT WIN32)
  target_link_libraries(cpandas PUBLIC m)
endif()
find_package(ZLIB)
if(ZLIB_FOUND)
  target_link_libraries(cpandas PUBLIC ZLIB::ZLIB)
  target_compile_definitions(cpandas PRIVATE CPANDAS_HAVE_ZLIB=1)
endif()

enable_testing()

add_executable(cpandas_tests
  tests/test_cpandas.c
)

target_link_libraries(cpandas_tests
  PRIVATE
    cpandas
)

add_test(NAME cpandas_tests COMMAND cpandas_tests)

add_executable(cpandas_fuzz_tests
  tests/test_csv_fuzz.c
)

target_link_libraries(cpandas_fuzz_tests
  PRIVATE
    cpandas
)

add_test(NAME cpandas_fuzz_tests COMMAND cpandas_fuzz_tests)

add_executable(cpandas_bench
  benchmarks/benchmark_cpandas.c
)

target_link_libraries(cpandas_bench
  PRIVATE
    cpandas
)
