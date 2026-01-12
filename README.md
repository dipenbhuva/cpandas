# C Pandas (cpandas)

C Pandas is a C11 library that provides pandas-like DataFrame and Series
functionality with a low-level, performance-oriented API.

## Features

- DataFrame and Series core types with columnar storage.
- Typed columns: int64, float64, string.
- CSV read/write with header support.
- Aggregations: count, sum, mean, min, max (Series and DataFrame helpers).
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
```

## Status

See `CONVERSION_STATUS.md` for a checklist of implemented and remaining
pandas features.
