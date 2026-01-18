cpandas: A pandas-like DataFrame library written in C

cpandas is an experimental C library that implements pandas-style DataFrame and Series operations (CSV IO, joins, aggregations) for performance-critical and systems-level workloads where Python is not suitable.

If you are searching for pandas C, DataFrame in C, columnar data C, or a pandas alternative C, cpandas is a compact C11 DataFrame C library focused on predictable performance and low-level control. It is also referred to as C Pandas or "pandas in C" for discoverability.

## Features

- Columnar data C storage with typed columns (int64, float64, string).
- DataFrame and Series API with selection, sorting, joins, groupby, and pivot tables.
- Aggregations (count, sum, mean, min, max, median, std, corr, cov, rank, diff).
- CSV read/write, TSV export (`to_excel`), SQL script export (`to_sql`).
- Query filtering with AND/OR and parentheses.
- Vectorized arithmetic helpers and column-to-column comparisons.
- Tests and benchmarks for correctness and performance.

## Apache Arrow comparison

Apache Arrow is a columnar in-memory data format and cross-language standard for interchange. cpandas is a pandas-like DataFrame library written in C that focuses on operations inside C programs. If you need zero-copy IPC or broad language interoperability, Arrow is the better fit; if you need a lightweight pandas alternative C for in-process analytics, cpandas is a good choice. Arrow integration is not built in yet.

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
./build/cpandas_bench 50000 --join --strategy sorted --match-rate 0.5
./build/cpandas_bench 50000 --join --strategy hash --match-rate 0.8 --key-dup-rate 0.3
```

## Status

See `CONVERSION_STATUS.md` for a checklist of implemented and remaining pandas features.

## Repo metadata (SEO)

Suggested GitHub description:
- "pandas in C - DataFrame C library for columnar data"

Suggested GitHub topics:
- pandas, dataframe, c, columnar-data, analytics
