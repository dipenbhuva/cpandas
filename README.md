cpandas: A pandas-like DataFrame library written in C

cpandas is an experimental C library that implements pandas-style DataFrame and Series operations (CSV IO, joins, aggregations) for performance-critical and systems-level workloads where Python is not suitable.

If you are searching for pandas C, DataFrame in C, columnar data C, or a pandas alternative C, cpandas is a compact C11 DataFrame C library focused on predictable performance and low-level control. It is also referred to as C Pandas or "pandas in C" for discoverability.

## Features

- Columnar data C storage with typed columns (int64, float64, string).
- DataFrame and Series API with selection, sorting, joins, groupby, and pivot tables.
- Aggregations (count, sum, mean, min, max, median, std, corr, cov, rank, diff).
- CSV/TSV/JSON/NDJSON/Parquet/CPD read/write, TSV export (`to_excel`), SQL script export (`to_sql`).
- Pure C11 core (no C++ dependencies required).
- Query filtering with AND/OR and parentheses.
- Vectorized arithmetic helpers and column-to-column comparisons.
- Tests and benchmarks for correctness and performance.

## Apache Arrow comparison

Apache Arrow is a columnar in-memory data format and cross-language standard for interchange. cpandas is a pandas-like DataFrame library written in C that focuses on operations inside C programs. If you need zero-copy IPC or broad language interoperability, Arrow is the better fit; if you need a lightweight pandas alternative C for in-process analytics, cpandas is a good choice. Parquet read/write is available with a minimal C-only implementation.

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

## CPD format

CPD is a compact little-endian binary format used by cpandas for fast load/save of
typed, columnar data with null masks. It is intended for C-to-C workflows.

## Parquet support

cpandas supports a C-only Parquet subset: multiple row groups, uncompressed or
Snappy-compressed data pages, PLAIN, dictionary, and delta-binary-packed
(int64) encodings, primitive columns (int64, float64, string/UTF8) with
optional nulls, and read-only nested structs (flattened as `parent.child` with
`def_level < max_def_level` treated as null). Column statistics
(min/max/null_count) are written for each column chunk. Dictionary encoding is
applied when it reduces size; set `CPANDAS_PARQUET_ENCODING=delta` to force
delta encoding for int64 columns. GZIP compression is available when built with
zlib and can be selected with `CPANDAS_PARQUET_CODEC=gzip`. Repeated fields and
complex nested types are not supported yet. The default row-group size is
65,536 rows.

## Repo metadata (SEO)

Suggested GitHub description:
- "pandas in C - DataFrame C library for columnar data"

Suggested GitHub topics:
- pandas, dataframe, c, columnar-data, analytics
