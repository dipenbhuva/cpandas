cmake_minimum_required(VERSION 3.16)

project(cpandas C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(CPANDAS_WITH_ARROW "Enable Parquet support via Apache Arrow" OFF)
if(CPANDAS_WITH_ARROW)
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

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

if(CPANDAS_WITH_ARROW)
  find_package(Arrow CONFIG REQUIRED)
  find_package(Parquet CONFIG REQUIRED)

  if(TARGET Arrow::arrow_shared)
    set(CPANDAS_ARROW_TARGET Arrow::arrow_shared)
  elseif(TARGET Arrow::arrow_static)
    set(CPANDAS_ARROW_TARGET Arrow::arrow_static)
  elseif(TARGET Arrow::arrow)
    set(CPANDAS_ARROW_TARGET Arrow::arrow)
  else()
    message(FATAL_ERROR "Arrow CMake target not found")
  endif()

  if(TARGET Parquet::parquet_shared)
    set(CPANDAS_PARQUET_TARGET Parquet::parquet_shared)
  elseif(TARGET Parquet::parquet_static)
    set(CPANDAS_PARQUET_TARGET Parquet::parquet_static)
  elseif(TARGET Parquet::parquet)
    set(CPANDAS_PARQUET_TARGET Parquet::parquet)
  else()
    message(FATAL_ERROR "Parquet CMake target not found")
  endif()

  target_sources(cpandas PRIVATE src/cpandas_parquet.cpp)
  target_compile_definitions(cpandas PRIVATE CPANDAS_WITH_ARROW=1)
  target_link_libraries(cpandas PRIVATE ${CPANDAS_ARROW_TARGET}
                                      ${CPANDAS_PARQUET_TARGET})
endif()

enable_testing()

add_executable(cpandas_tests
  tests/test_cpandas.c
)

target_link_libraries(cpandas_tests
  PRIVATE
    cpandas
)
if(CPANDAS_WITH_ARROW)
  target_compile_definitions(cpandas_tests PRIVATE CPANDAS_WITH_ARROW=1)
endif()

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
