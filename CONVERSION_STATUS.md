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

Data exploration
- `head` / `tail` row slicing helpers.
- DataFrame dtypes listing.
- `info` summary output.
- `describe` summary statistics (numeric columns).

Data operations
- Column lookup by name.
- Column selection by name.
- Column drop and rename.
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
- Groupby and `agg`.
- Merging (`pd.merge`).
- Pivot tables.

Remaining from the list
None from the GeeksforGeeks list.

## Remaining / Not Yet Implemented

Core DataFrame operations
- Column selection by dtype and projection API.
- Multi-column predicate composition (AND/OR) and expression evaluation.
- Vectorized arithmetic and comparison ops beyond aggregations.

Joins
- Hash-join or indexed join performance work.

Indexing and missing values
- Index and index-based operations.
- Missing value handling beyond blank/whitespace parsing and null flags
  (e.g., custom NA tokens, column-wise strategies).

I/O and formats
- TSV convenience wrappers.
- Parquet/JSON or other formats.

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
