# Pandas Functions Coverage (GeeksforGeeks List)

Source: https://www.geeksforgeeks.org/pandas/pandas-functions-in-python/

This checklist maps the GeeksforGeeks pandas function list to the current
cpandas C API.

## Implemented from the GfG list

DataFrame creation
- `pd.DataFrame()` -> `cp_df_create` (requires explicit schema and dtypes)

Data selection and filtering
- `df["columnname"]` -> `cp_df_get_col`
- `df[["column1", "column2"]]` -> `cp_df_select_cols`

Data manipulation
- `df.sort_values(by=column)` -> `cp_df_sort_values`
- Multi-column sort -> `cp_df_sort_values_multi` (extra vs. GfG list)

## Remaining from the GfG list

Data exploration
- `df.head()`
- `df.tail()`
- `df.info()`
- `df.describe()`
- `df.dtypes` (DataFrame-level dtype listing)

Data selection and filtering
- `df.loc` (label-based indexing)
- `df.iloc` (positional indexing)
- `df[df["column"] > value]` (predicate-based filtering; mask-only today)

Data manipulation
- `df["newcolumn"] = value` (add/assign column)
- `df.drop(columns=column)`
- `df.rename(columns={"oldname": "newname"})`

Data aggregation
- `df.groupby(column).mean()`
- `df.agg({"column": ["mean", "max"]})`

Data cleaning
- `df.isnull()`
- `df.dropna()`
- `df.fillna(value)`

Data merging
- `pd.merge(df1, df2, on=column)`

Pivot tables
- `df.pivot_table(values=column, index=index, columns=columns, aggfunc=mean)`

## Notes
- Mask-based filtering is available via `cp_df_filter_mask`.
- Series-level dtypes are available via `cp_series_dtype`, but a DataFrame-level
  dtype listing helper is not yet exposed.
- Aggregations (sum/mean/min/max/count) exist, but groupby and agg wrappers are
  not yet implemented.
