# Pandas-to-C Conversion Status

This document tracks what has been converted from pandas concepts into the C library so far, and what remains based on the PRD scope.

## Converted / Implemented

Core structures and API
- DataFrame and Series core types with columnar storage.
- Typed columns: int64, float64, string.
- Null handling via per-column null flags.
- Error handling via `CpError`.

I/O
- CSV read/write with delimiter selection and optional header.
- Basic CSV quoting/escaping for commas, quotes, and newlines.
- TSV convenience wrappers for read/write.
- JSON read/write (array-of-objects).
- NDJSON read/write (line-delimited objects).
- CPD binary read/write (cpandas columnar format).
- Parquet read/write stubs (not implemented).

Data exploration
- `head` / `tail` row slicing helpers.
- DataFrame dtypes listing.
- `info` summary output.
- `describe` summary statistics (numeric columns).

Data operations
- Column lookup by name.
- Column selection by name.
- Column selection by dtype.
- Column drop and rename.
- Vectorized arithmetic helpers and column-to-column comparisons.
- Row append with type parsing (int64/float64/string).
- Row filtering by boolean mask.
- Row/column selection by position and name (`iloc`, `loc`).
- Predicate filtering helpers (numeric/string comparisons).
- Sorting by a single column (ascending/descending, nulls last).
- Sorting by multiple columns.
- Groupby aggregation by key column (int64/string) with count/sum/mean/min/max.
- Joins (inner/left/right/outer) on single or multiple key columns with configurable suffixes.
- Hash-indexed and sorted-index join paths for larger joins.
- Join strategy override for benchmarking (nested/hash/sorted/auto).
- Pivot tables (single index/column/value with count/sum/mean/min/max).
- Aggregations (Series-level and DataFrame-level by name/index):
  - count
  - sum (int64/float64)
  - mean (int64/float64)
  - min/max (int64/float64)

Data cleaning
- Null mask extraction.
- dropna (remove rows with any nulls).
- fillna (per-column fill values).

Validation and tooling
- Unit tests for CSV parsing/writing, error cases, and aggregations.
- Deterministic CSV roundtrip fuzz test.
- Simple benchmark target for append and sum throughput.

## GeeksforGeeks Pandas Functions Checklist

Source: https://www.geeksforgeeks.org/pandas/pandas-functions-in-python/

Implemented from the list
- DataFrame creation (`pd.DataFrame()` -> `cp_df_create`).
- Data exploration (`head`, `tail`, `dtypes`, `info`, `describe`).
- Column selection (`df["col"]`, `df[["col1", "col2"]]`).
- Sorting by column (`df.sort_values`).
- Column drop/rename.
- Missing value helpers (`isnull`, `dropna`, `fillna`).
- `isna` alias for `isnull`.
- Uniques/counts (`unique`, `nunique`, `value_counts`, `duplicated`, `drop_duplicates`).
- Sampling/selection (`sample`, `nlargest`, `nsmallest`).
- Conditional/replace (`where`, `mask`, `clip`, `replace`).
- `concat`.
- Query filtering (`query`).
- Type/index helpers (`astype`, `set_index`, `reset_index`, `at`).
- Apply/transform helpers (`apply`, `transform`) and iteration (`iterrows`, `iteritems`).
- Stats (`median`, `std`, `corr`, `cov`, `rank`, `diff`).
- Conversion/format (`to_numeric`, `to_datetime`, `to_string`, `to_excel`, `to_sql`).
- Plotting (`plot`).
- Groupby and `agg`.
- Merging (`pd.merge`).
- Pivot tables.
- Shape/metadata helpers (`shape`, `size`, `ndim`, `columns`).
- DataFrame copy (`copy`).

Remaining from the list
- None.

## Remaining / Not Yet Implemented

Core DataFrame operations
- None.

Joins
- Join performance tuning (parallelism, memory).

Indexing and missing values
- Missing value handling beyond blank/whitespace parsing and null flags
  (e.g., column-wise strategies).

I/O and formats
- Parquet (C-only) or additional formats.

Performance and scalability
- SIMD or parallel ops.
- Memory pooling and zero-copy views.

Interop and parity
- Python bindings for parity validation.
- Parity test harness against pandas.

Advanced pandas features
- Time series APIs.
- Multi-index pivot tables and margins.
- Advanced indexing (multi-index, label-based slices).

## Notes
- CSV parsing is intentionally minimal for MVP and does not support multiline
  quoted fields.
- dtype inference is not implemented; dtypes must be provided or default to string.
- `query` supports `AND`/`OR` chaining with parentheses and basic comparison
  operators; it does not support functions or complex expressions.
- `set_index` stores a single column as index metadata (column remains in data and
  must be int64 or string); `reset_index` clears it.
- `at` uses index metadata when present, otherwise treats the row label as a
  positional index.
- `apply` returns a single-column DataFrame from a row callback; `transform`
  replaces a single column via callback and keeps the same row count.
- `median`/`std` skip nulls and NaNs; `corr`/`cov` use pairwise complete rows with
  sample variance (ddof=1); `rank` uses average ties and `diff` compares to the
  previous row.
- `to_numeric` converts a single column to float64 using `astype` rules.
- `to_datetime` parses `YYYY-MM-DD` or `YYYY/MM/DD` with optional
  `HH:MM[:SS][.fff]` time and optional `Z` or `±HH:MM` offsets into int64 epoch
  seconds (UTC); blanks are nulls.
- `to_string` renders a space-aligned table with `null` for nulls.
- `to_sql` writes a SQL script (CREATE TABLE + INSERT statements) with identifiers
  quoted and string values escaped via doubled single quotes.
- `to_excel` writes a tab-separated text file with a header for Excel import.
- `plot` renders a simple SVG polyline chart of numeric columns vs row index.
- `read_json`/`write_json` handle arrays of JSON objects with primitive values;
  nested objects/arrays and non-ASCII unicode escapes are rejected.
- `read_ndjson`/`write_ndjson` handle line-delimited JSON objects with the same
  primitive-only constraints as `read_json`.
- `read_cpd`/`write_cpd` support a little-endian binary columnar format with
  schema, null masks, and per-column data blocks.
- `read_parquet`/`write_parquet` are stubbed pending an all-C implementation.
- Vectorized arithmetic outputs float64 and treats nulls/NaNs as null; division by
  zero yields nulls.
- `loc_labels`/`loc_slice` use the index metadata when present and positional
  indices otherwise; duplicate index labels return the first match.
- `read_csv_with_na`/`read_tsv_with_na` accept custom NA tokens for parsing.
