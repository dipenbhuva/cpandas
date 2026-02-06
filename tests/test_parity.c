#include "cpandas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "parity check failed: %s\n", msg); \
      return 0; \
    } \
  } while (0)

static int build_path(char *out,
                      size_t out_len,
                      const char *base,
                      const char *rel) {
  if (!out || out_len == 0 || !base || !rel) {
    return 0;
  }
  int n = snprintf(out, out_len, "%s/%s", base, rel);
  return n > 0 && (size_t)n < out_len;
}

static int series_equal(const CpSeries *a,
                        const CpSeries *b,
                        double tol) {
  if (!a || !b) {
    return 0;
  }
  if (cp_series_dtype(a) != cp_series_dtype(b)) {
    return 0;
  }
  if (cp_series_len(a) != cp_series_len(b)) {
    return 0;
  }
  const char *name_a = cp_series_name(a);
  const char *name_b = cp_series_name(b);
  if (!name_a) {
    name_a = "";
  }
  if (!name_b) {
    name_b = "";
  }
  if (strcmp(name_a, name_b) != 0) {
    return 0;
  }

  size_t len = cp_series_len(a);
  CpDType dtype = cp_series_dtype(a);
  for (size_t i = 0; i < len; ++i) {
    int a_null = 0;
    int b_null = 0;
    if (dtype == CP_DTYPE_INT64) {
      int64_t av = 0;
      int64_t bv = 0;
      if (!cp_series_get_int64(a, i, &av, &a_null)) {
        return 0;
      }
      if (!cp_series_get_int64(b, i, &bv, &b_null)) {
        return 0;
      }
      if (a_null != b_null) {
        return 0;
      }
      if (!a_null && av != bv) {
        return 0;
      }
    } else if (dtype == CP_DTYPE_FLOAT64) {
      double av = 0.0;
      double bv = 0.0;
      if (!cp_series_get_float64(a, i, &av, &a_null)) {
        return 0;
      }
      if (!cp_series_get_float64(b, i, &bv, &b_null)) {
        return 0;
      }
      if (a_null != b_null) {
        return 0;
      }
      if (!a_null) {
        if (isnan(av) && isnan(bv)) {
          continue;
        }
        if (fabs(av - bv) > tol) {
          return 0;
        }
      }
    } else if (dtype == CP_DTYPE_STRING) {
      const char *av = NULL;
      const char *bv = NULL;
      if (!cp_series_get_string(a, i, &av, &a_null)) {
        return 0;
      }
      if (!cp_series_get_string(b, i, &bv, &b_null)) {
        return 0;
      }
      if (a_null != b_null) {
        return 0;
      }
      if (!a_null) {
        if (!av) {
          av = "";
        }
        if (!bv) {
          bv = "";
        }
        if (strcmp(av, bv) != 0) {
          return 0;
        }
      }
    } else {
      return 0;
    }
  }
  return 1;
}

static int df_equal(const CpDataFrame *a,
                    const CpDataFrame *b,
                    double tol,
                    CpError *err) {
  if (!a || !b) {
    return 0;
  }
  size_t nrows = cp_df_nrows(a);
  size_t ncols = cp_df_ncols(a);
  if (cp_df_nrows(b) != nrows || cp_df_ncols(b) != ncols) {
    return 0;
  }

  const char **a_names = (const char **)malloc(ncols * sizeof(const char *));
  const char **b_names = (const char **)malloc(ncols * sizeof(const char *));
  if (!a_names || !b_names) {
    free(a_names);
    free(b_names);
    return 0;
  }
  if (!cp_df_columns(a, a_names, ncols, err) ||
      !cp_df_columns(b, b_names, ncols, err)) {
    free(a_names);
    free(b_names);
    return 0;
  }

  for (size_t col = 0; col < ncols; ++col) {
    const char *aname = a_names[col] ? a_names[col] : "";
    const char *bname = b_names[col] ? b_names[col] : "";
    if (strcmp(aname, bname) != 0) {
      free(a_names);
      free(b_names);
      return 0;
    }
    const CpSeries *as = cp_df_get_col(a, aname);
    const CpSeries *bs = cp_df_get_col(b, bname);
    if (!series_equal(as, bs, tol)) {
      free(a_names);
      free(b_names);
      return 0;
    }
  }

  free(a_names);
  free(b_names);
  return 1;
}

static int run_case(const char *label,
                    CpDataFrame *actual,
                    const char *expected_path,
                    const CpDType *expected_dtypes,
                    size_t expected_count,
                    CpError *err) {
  if (!actual) {
    fprintf(stderr, "parity case %s: missing actual\n", label);
    return 0;
  }
  CpDataFrame *expected =
      cp_df_read_json(expected_path, expected_dtypes, expected_count, err);
  if (!expected) {
    fprintf(stderr, "parity case %s: failed to read expected\n", label);
    cp_df_free(actual);
    return 0;
  }
  int ok = df_equal(actual, expected, 1e-9, err);
  if (!ok) {
    fprintf(stderr, "parity case %s: mismatch\n", label);
  }
  cp_df_free(actual);
  cp_df_free(expected);
  return ok;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <parity_dir>\n", argv[0]);
    return 1;
  }
  const char *base_dir = argv[1];
  char path[1024];
  CpError err;
  cp_error_clear(&err);

  if (!build_path(path, sizeof(path), base_dir, "inputs/basic.csv")) {
    fprintf(stderr, "invalid parity input path\n");
    return 1;
  }
  CpDType input_dtypes[] = {
      CP_DTYPE_STRING,
      CP_DTYPE_INT64,
      CP_DTYPE_FLOAT64
  };
  CpDataFrame *df = cp_df_read_csv(path, ',', 1, input_dtypes, 3, &err);
  if (!df) {
    fprintf(stderr, "failed to load parity input\n");
    return 1;
  }

  int ok = 1;
  CpDType base_dtypes[] = {
      CP_DTYPE_STRING,
      CP_DTYPE_INT64,
      CP_DTYPE_FLOAT64
  };
  CpDType group_dtypes[] = {CP_DTYPE_STRING, CP_DTYPE_INT64};
  CpDType describe_dtypes[] = {
      CP_DTYPE_STRING,
      CP_DTYPE_FLOAT64,
      CP_DTYPE_FLOAT64
  };

  if (build_path(path, sizeof(path), base_dir, "expected/head.json")) {
    ok &= run_case("head",
                   cp_df_head(df, 2, &err),
                   path,
                   base_dtypes,
                   3,
                   &err);
  } else {
    ok = 0;
  }
  if (build_path(path, sizeof(path), base_dir, "expected/tail.json")) {
    ok &= run_case("tail",
                   cp_df_tail(df, 2, &err),
                   path,
                   base_dtypes,
                   3,
                   &err);
  } else {
    ok = 0;
  }
  if (build_path(path, sizeof(path), base_dir, "expected/sort_sales.json")) {
    ok &= run_case("sort_sales",
                   cp_df_sort_values(df, "sales", 1, &err),
                   path,
                   base_dtypes,
                   3,
                   &err);
  } else {
    ok = 0;
  }
  if (build_path(path,
                 sizeof(path),
                 base_dir,
                 "expected/groupby_sales_sum.json")) {
    const char *value_cols[] = {"sales"};
    CpAggOp ops[] = {CP_AGG_SUM};
    ok &= run_case("groupby_sales_sum",
                   cp_df_groupby_agg(df, "city", value_cols, ops, 1, &err),
                   path,
                   group_dtypes,
                   2,
                   &err);
  } else {
    ok = 0;
  }
  if (build_path(path, sizeof(path), base_dir, "expected/describe.json")) {
    ok &= run_case("describe",
                   cp_df_describe(df, &err),
                   path,
                   describe_dtypes,
                   3,
                   &err);
  } else {
    ok = 0;
  }
  if (build_path(path, sizeof(path), base_dir, "expected/dropna.json")) {
    ok &= run_case("dropna",
                   cp_df_dropna(df, &err),
                   path,
                   base_dtypes,
                   3,
                   &err);
  } else {
    ok = 0;
  }

  cp_df_free(df);
  return ok ? 0 : 1;
}
