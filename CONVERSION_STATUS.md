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

Data operations
- Column lookup by name.
- Column selection by name.
- Row append with type parsing (int64/float64/string).
- Row filtering by boolean mask.
- Aggregations (Series-level and DataFrame-level by name/index):
  - count
  - sum (int64/float64)
  - mean (int64/float64)
  - min/max (int64/float64)

Validation and tooling
- Unit tests for CSV parsing/writing, error cases, and aggregations.
- Deterministic CSV roundtrip fuzz test.
- Simple benchmark target for append and sum throughput.

## Remaining / Not Yet Implemented

Core DataFrame operations
- Column selection by dtype and projection API.
- Row filtering via predicates (beyond mask-based filtering).
- Sorting by column(s).
- Vectorized arithmetic and comparison ops beyond aggregations.

Groupby and joins
- Groupby (split/apply/aggregate).
- Joins (inner, left) with key handling.

Indexing and missing values
- Index and index-based operations.
- Missing value handling beyond blank/whitespace parsing and null flags
  (e.g., custom NA tokens, fill/replace, dropna).

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
- Pivot tables.
- Advanced indexing (multi-index, label-based slices).

## Notes
- CSV parsing is intentionally minimal for MVP and does not support multiline
  quoted fields.
- dtype inference is not implemented; dtypes must be provided or default to string.
