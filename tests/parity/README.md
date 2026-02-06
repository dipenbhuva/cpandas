# Parity Fixtures

This directory holds a small parity fixture set used by `cpandas_parity_tests`.
The tests load `inputs/basic.csv`, run a fixed set of operations, and compare
the outputs to the JSON files in `expected/`.

## Running the parity test

From the build directory:

```sh
ctest --test-dir build -R cpandas_parity_tests --output-on-failure
```

## Updating fixtures

The expected JSON files are meant to reflect pandas-style results for the same
operations (head/tail/sort/groupby/describe/dropna). If you update behavior or
add more cases, regenerate the JSON outputs from pandas and replace the files in
`expected/`.

Notes:
- JSON is in `records` form (array of objects).
- `null` represents missing values.
- Column order must match the cpandas output order.
