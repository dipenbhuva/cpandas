# C Pandas (cpandas)

C Pandas is a C11 library that provides pandas-like DataFrame and Series
functionality with a low-level, performance-oriented API.

## Features

- DataFrame and Series core types with columnar storage.
- Typed columns: int64, float64, string.
- CSV read/write with header support.
- Aggregations: count, sum, mean, min, max (Series and DataFrame helpers).
- Joins (inner/left/right/outer) with multi-key support, strategy selection (auto/hash/sorted/nested), and pivot tables.
- Tests and a simple benchmark target.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Test

```sh
ctest --test-dir build
```

## Benchmark

```sh
./build/cpandas_bench 500000
./build/cpandas_bench 50000 --join --strategy hash
./build/cpandas_bench 50000 --join --strategy all
```

## Status

See `CONVERSION_STATUS.md` for a checklist of implemented and remaining
pandas features.
