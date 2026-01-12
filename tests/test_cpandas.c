#include "cpandas.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <unistd.h>
#endif

static int tests_failed = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      tests_failed += 1;                                                  \
    }                                                                     \
  } while (0)

static char *dup_string(const char *s) {
  size_t len = strlen(s);
  char *out = (char *)malloc(len + 1);
  if (!out) {
    return NULL;
  }
  memcpy(out, s, len + 1);
  return out;
}

static char *make_temp_path(void) {
#ifdef _WIN32
  char buf[L_tmpnam];
  if (!tmpnam(buf)) {
    return NULL;
  }
  return dup_string(buf);
#else
  char *tmpl = dup_string("cpandas_test_XXXXXX");
  if (!tmpl) {
    return NULL;
  }
  int fd = mkstemp(tmpl);
  if (fd == -1) {
    free(tmpl);
    return NULL;
  }
  close(fd);
  return tmpl;
#endif
}

static int write_file(const char *path, const char *content) {
  FILE *fp = fopen(path, "wb");
  if (!fp) {
    return 0;
  }
  size_t len = strlen(content);
  size_t written = fwrite(content, 1, len, fp);
  fclose(fp);
  return written == len;
}

static char *read_file(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    return NULL;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  long size = ftell(fp);
  if (size < 0) {
    fclose(fp);
    return NULL;
  }
  rewind(fp);
  char *buf = (char *)malloc((size_t)size + 1);
  if (!buf) {
    fclose(fp);
    return NULL;
  }
  size_t read = fread(buf, 1, (size_t)size, fp);
  fclose(fp);
  if (read != (size_t)size) {
    free(buf);
    return NULL;
  }
  buf[size] = '\0';
  return buf;
}

static void test_read_csv_header(void) {
  CpError err;
  cp_error_clear(&err);

  char *path = make_temp_path();
  CHECK(path != NULL);
  if (!path) {
    return;
  }

  const char *csv =
      "id,score,name\n"
      "1,98.5,Alice\n"
      "2,,Bob\n"
      ",73.25,\"Charlie, Jr.\"\n";
  CHECK(write_file(path, csv));

  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_read_csv(path, ',', 1, dtypes, 3, &err);
  CHECK(df != NULL);

  if (df) {
    CHECK(cp_df_ncols(df) == 3);
    CHECK(cp_df_nrows(df) == 3);

    const CpSeries *id = cp_df_get_col(df, "id");
    const CpSeries *score = cp_df_get_col(df, "score");
    const CpSeries *name = cp_df_get_col(df, "name");
    CHECK(id && score && name);

    int is_null = 0;
    int64_t id_val = 0;
    double score_val = 0.0;
    const char *name_val = NULL;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);

    CHECK(cp_series_get_float64(score, 0, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 98.5) < 1e-9);

    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);

    CHECK(cp_series_get_float64(score, 1, &score_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);

    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_float64(score, 2, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 73.25) < 1e-9);

    CHECK(cp_series_get_string(name, 2, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Charlie, Jr.") == 0);
  } else {
    CHECK(err.code != CP_OK);
  }

  if (df) {
    cp_df_free(df);
  }
  remove(path);
  free(path);
}

static void test_read_csv_no_header(void) {
  CpError err;
  cp_error_clear(&err);

  char *path = make_temp_path();
  CHECK(path != NULL);
  if (!path) {
    return;
  }

  const char *csv = "10,20\n30,40\n";
  CHECK(write_file(path, csv));

  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *df = cp_df_read_csv(path, ',', 0, dtypes, 2, &err);
  CHECK(df != NULL);

  if (df) {
    CHECK(cp_df_ncols(df) == 2);
    CHECK(cp_df_nrows(df) == 2);

    const CpSeries *col0 = cp_df_get_col(df, "col0");
    const CpSeries *col1 = cp_df_get_col(df, "col1");
    CHECK(col0 && col1);

    int is_null = 0;
    int64_t v0 = 0;
    int64_t v1 = 0;

    CHECK(cp_series_get_int64(col0, 0, &v0, &is_null));
    CHECK(!is_null && v0 == 10);
    CHECK(cp_series_get_int64(col1, 0, &v1, &is_null));
    CHECK(!is_null && v1 == 20);

    CHECK(cp_series_get_int64(col0, 1, &v0, &is_null));
    CHECK(!is_null && v0 == 30);
    CHECK(cp_series_get_int64(col1, 1, &v1, &is_null));
    CHECK(!is_null && v1 == 40);
  } else {
    CHECK(err.code != CP_OK);
  }

  if (df) {
    cp_df_free(df);
  }
  remove(path);
  free(path);
}

static void test_write_csv_header(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "2.5", "Alice, Jr."};
  const char *row2[] = {"2", "", ""};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));

  char *path = make_temp_path();
  CHECK(path != NULL);
  if (!path) {
    cp_df_free(df);
    return;
  }

  CHECK(cp_df_write_csv(df, path, ',', 1, &err));
  char *contents = read_file(path);
  CHECK(contents != NULL);

  const char *expected =
      "id,score,name\n"
      "1,2.5,\"Alice, Jr.\"\n"
      "2,,\n";

  if (contents) {
    CHECK(strcmp(contents, expected) == 0);
  }

  free(contents);
  remove(path);
  free(path);
  cp_df_free(df);
}

static void test_append_row_errors(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"a", "b"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *short_row[] = {"1"};
  CHECK(!cp_df_append_row(df, short_row, 1, &err));
  CHECK(err.code == CP_ERR_INVALID);
  CHECK(cp_df_nrows(df) == 0);

  cp_error_clear(&err);
  const char *bad_row[] = {"abc", "2"};
  CHECK(!cp_df_append_row(df, bad_row, 2, &err));
  CHECK(err.code == CP_ERR_PARSE);
  CHECK(cp_df_nrows(df) == 0);

  const CpSeries *col_a = cp_df_get_col(df, "a");
  const CpSeries *col_b = cp_df_get_col(df, "b");
  CHECK(col_a && col_b);
  if (col_a && col_b) {
    CHECK(cp_series_len(col_a) == 0);
    CHECK(cp_series_len(col_b) == 0);
  }

  cp_df_free(df);
}

static void test_read_csv_mismatch(void) {
  CpError err;
  cp_error_clear(&err);

  char *path = make_temp_path();
  CHECK(path != NULL);
  if (!path) {
    return;
  }

  const char *csv = "a,b\n1,2,3\n";
  CHECK(write_file(path, csv));

  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *df = cp_df_read_csv(path, ',', 1, dtypes, 2, &err);
  CHECK(df == NULL);
  CHECK(err.code == CP_ERR_PARSE);

  if (df) {
    cp_df_free(df);
  }
  remove(path);
  free(path);
}

static void test_aggregations(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"ival", "fval"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "1.5"};
  const char *row2[] = {"2", ""};
  const char *row3[] = {"", "-2.25"};
  const char *row4[] = {"-5", "0"};

  CHECK(cp_df_append_row(df, row1, 2, &err));
  CHECK(cp_df_append_row(df, row2, 2, &err));
  CHECK(cp_df_append_row(df, row3, 2, &err));
  CHECK(cp_df_append_row(df, row4, 2, &err));

  const CpSeries *ival = cp_df_get_col(df, "ival");
  const CpSeries *fval = cp_df_get_col(df, "fval");
  CHECK(ival && fval);

  size_t count = 0;
  size_t nulls = 0;
  int64_t isum = 0;
  double fsum = 0.0;
  double mean = 0.0;
  int64_t imin = 0;
  int64_t imax = 0;
  double fmin = 0.0;
  double fmax = 0.0;

  CHECK(cp_series_count(ival, &count, &nulls, &err));
  CHECK(count == 3 && nulls == 1);
  CHECK(cp_series_sum_int64(ival, &isum, &count, &nulls, &err));
  CHECK(isum == -2 && count == 3 && nulls == 1);
  CHECK(cp_series_mean(ival, &mean, &count, &nulls, &err));
  CHECK(fabs(mean - (-2.0 / 3.0)) < 1e-9);
  CHECK(cp_series_min_int64(ival, &imin, &nulls, &err));
  CHECK(imin == -5 && nulls == 1);
  CHECK(cp_series_max_int64(ival, &imax, &nulls, &err));
  CHECK(imax == 2 && nulls == 1);

  CHECK(cp_series_count(fval, &count, &nulls, &err));
  CHECK(count == 3 && nulls == 1);
  CHECK(cp_series_sum_float64(fval, &fsum, &count, &nulls, &err));
  CHECK(fabs(fsum - (-0.75)) < 1e-9 && count == 3 && nulls == 1);
  CHECK(cp_series_mean(fval, &mean, &count, &nulls, &err));
  CHECK(fabs(mean - (-0.25)) < 1e-9);
  CHECK(cp_series_min_float64(fval, &fmin, &nulls, &err));
  CHECK(fabs(fmin - (-2.25)) < 1e-9 && nulls == 1);
  CHECK(cp_series_max_float64(fval, &fmax, &nulls, &err));
  CHECK(fabs(fmax - 1.5) < 1e-9 && nulls == 1);

  cp_df_free(df);

  cp_error_clear(&err);
  const char *null_names[] = {"empty"};
  CpDType null_dtypes[] = {CP_DTYPE_INT64};
  CpDataFrame *null_df = cp_df_create(1, null_names, null_dtypes, 0, &err);
  CHECK(null_df != NULL);
  if (null_df) {
    const char *null_row1[] = {""};
    const char *null_row2[] = {"   "};
    CHECK(cp_df_append_row(null_df, null_row1, 1, &err));
    CHECK(cp_df_append_row(null_df, null_row2, 1, &err));

    const CpSeries *empty = cp_df_get_col(null_df, "empty");
    CHECK(empty != NULL);

    CHECK(!cp_series_mean(empty, NULL, NULL, NULL, &err));
    CHECK(err.code == CP_ERR_INVALID);

    cp_df_free(null_df);
  }
}

static void test_df_aggregation_helpers(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"ival", "fval", "sval"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "1.5", "a"};
  const char *row2[] = {"2", "", ""};
  const char *row3[] = {"-5", "-2.25", "b"};
  const char *row4[] = {"", "0", ""};

  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));
  CHECK(cp_df_append_row(df, row4, 3, &err));

  size_t count = 0;
  size_t nulls = 0;
  int64_t isum = 0;
  int64_t imin = 0;
  int64_t imax = 0;
  double fsum = 0.0;
  double mean = 0.0;
  double fmin = 0.0;
  double fmax = 0.0;
  CpAggInt64 agg_i64;
  CpAggFloat64 agg_f64;

  CHECK(cp_df_count(df, "sval", &count, &nulls, &err));
  CHECK(count == 2 && nulls == 2);

  CHECK(cp_df_sum_int64(df, "ival", &isum, &count, &nulls, &err));
  CHECK(isum == -2 && count == 3 && nulls == 1);
  CHECK(cp_df_mean(df, "ival", &mean, &count, &nulls, &err));
  CHECK(fabs(mean - (-2.0 / 3.0)) < 1e-9);
  CHECK(cp_df_min_int64(df, "ival", &imin, &nulls, &err));
  CHECK(imin == -5 && nulls == 1);
  CHECK(cp_df_max_int64(df, "ival", &imax, &nulls, &err));
  CHECK(imax == 2 && nulls == 1);

  CHECK(cp_df_sum_float64(df, "fval", &fsum, &count, &nulls, &err));
  CHECK(fabs(fsum - (-0.75)) < 1e-9 && count == 3 && nulls == 1);
  CHECK(cp_df_mean(df, "fval", &mean, &count, &nulls, &err));
  CHECK(fabs(mean - (-0.25)) < 1e-9);
  CHECK(cp_df_min_float64(df, "fval", &fmin, &nulls, &err));
  CHECK(fabs(fmin - (-2.25)) < 1e-9 && nulls == 1);
  CHECK(cp_df_max_float64(df, "fval", &fmax, &nulls, &err));
  CHECK(fabs(fmax - 1.5) < 1e-9 && nulls == 1);

  CHECK(cp_df_sum_int64_result(df, "ival", &agg_i64, &err));
  CHECK(agg_i64.value == -2 && agg_i64.count == 3 && agg_i64.nulls == 1);
  CHECK(cp_df_mean_result(df, "ival", &agg_f64, &err));
  CHECK(fabs(agg_f64.value - (-2.0 / 3.0)) < 1e-9);
  CHECK(agg_f64.count == 3 && agg_f64.nulls == 1);
  CHECK(cp_df_min_int64_result(df, "ival", &agg_i64, &err));
  CHECK(agg_i64.value == -5 && agg_i64.count == 3 && agg_i64.nulls == 1);

  CHECK(cp_df_sum_float64_result(df, "fval", &agg_f64, &err));
  CHECK(fabs(agg_f64.value - (-0.75)) < 1e-9);
  CHECK(agg_f64.count == 3 && agg_f64.nulls == 1);
  CHECK(cp_df_max_float64_result(df, "fval", &agg_f64, &err));
  CHECK(fabs(agg_f64.value - 1.5) < 1e-9);
  CHECK(agg_f64.count == 3 && agg_f64.nulls == 1);

  CHECK(cp_df_sum_int64_at(df, 0, &isum, &count, &nulls, &err));
  CHECK(isum == -2 && count == 3 && nulls == 1);
  CHECK(cp_df_mean_at(df, 1, &mean, &count, &nulls, &err));
  CHECK(fabs(mean - (-0.25)) < 1e-9 && count == 3 && nulls == 1);
  CHECK(cp_df_count_at(df, 2, &count, &nulls, &err));
  CHECK(count == 2 && nulls == 2);
  CHECK(cp_df_min_float64_result_at(df, 1, &agg_f64, &err));
  CHECK(fabs(agg_f64.value - (-2.25)) < 1e-9);
  CHECK(agg_f64.count == 3 && agg_f64.nulls == 1);

  cp_error_clear(&err);
  CHECK(!cp_df_sum_int64(df, "fval", &isum, &count, &nulls, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_mean(df, "missing", &mean, &count, &nulls, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_sum_int64_at(df, 1, &isum, &count, &nulls, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_mean_at(df, 9, &mean, &count, &nulls, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_sum_int64_result(df, "ival", NULL, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_df_free(df);
}

static void test_select_and_filter(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "10.5", "Alice"};
  const char *row2[] = {"2", "", "Bob"};
  const char *row3[] = {"3", "8.0", ""};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));

  const char *sel[] = {"name", "id"};
  CpDataFrame *selected = cp_df_select_cols(df, sel, 2, &err);
  CHECK(selected != NULL);
  if (selected) {
    CHECK(cp_df_ncols(selected) == 2);
    CHECK(cp_df_nrows(selected) == 3);

    const CpSeries *name = cp_df_get_col(selected, "name");
    const CpSeries *id = cp_df_get_col(selected, "id");
    CHECK(name && id);

    const char *name_val = NULL;
    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);

    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);

    CHECK(cp_series_get_string(name, 2, &name_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
  }

  cp_error_clear(&err);
  const char *bad_sel[] = {"missing"};
  CpDataFrame *bad = cp_df_select_cols(df, bad_sel, 1, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  uint8_t mask[] = {1, 0, 1};
  CpDataFrame *filtered = cp_df_filter_mask(df, mask, 3, &err);
  CHECK(filtered != NULL);
  if (filtered) {
    CHECK(cp_df_nrows(filtered) == 2);
    const CpSeries *fid = cp_df_get_col(filtered, "id");
    const CpSeries *fname = cp_df_get_col(filtered, "name");
    CHECK(fid && fname);

    int64_t id_val = 0;
    const char *name_val = NULL;
    int is_null = 0;

    CHECK(cp_series_get_int64(fid, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_string(fname, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);

    CHECK(cp_series_get_int64(fid, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
    CHECK(cp_series_get_string(fname, 1, &name_val, &is_null));
    CHECK(is_null);
  }

  cp_error_clear(&err);
  uint8_t bad_mask[] = {1, 0};
  CpDataFrame *bad_filter = cp_df_filter_mask(df, bad_mask, 2, &err);
  CHECK(bad_filter == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (selected) {
    cp_df_free(selected);
  }
  if (filtered) {
    cp_df_free(filtered);
  }
  cp_df_free(df);
}

static void test_sort_values(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"3", "2.0", "Bob"};
  const char *row2[] = {"1", "5.0", "Alice"};
  const char *row3[] = {"2", "1.5", "Charlie"};
  const char *row4[] = {"", "4.5", ""};
  const char *row5[] = {"2", "0.5", "Bob"};

  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));
  CHECK(cp_df_append_row(df, row4, 3, &err));
  CHECK(cp_df_append_row(df, row5, 3, &err));

  CpDataFrame *asc = cp_df_sort_values(df, "id", 1, &err);
  CHECK(asc != NULL);
  if (asc) {
    const CpSeries *id = cp_df_get_col(asc, "id");
    const CpSeries *name = cp_df_get_col(asc, "name");
    CHECK(id && name);

    int64_t id_val = 0;
    const char *name_val = NULL;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Charlie") == 0);

    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_string(name, 2, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);

    CHECK(cp_series_get_int64(id, 3, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);

    CHECK(cp_series_get_int64(id, 4, &id_val, &is_null));
    CHECK(is_null);
  }

  CpDataFrame *desc = cp_df_sort_values(df, "id", 0, &err);
  CHECK(desc != NULL);
  if (desc) {
    const CpSeries *id = cp_df_get_col(desc, "id");
    CHECK(id != NULL);

    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_int64(id, 3, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_int64(id, 4, &id_val, &is_null));
    CHECK(is_null);
  }

  CpDataFrame *by_name = cp_df_sort_values(df, "name", 1, &err);
  CHECK(by_name != NULL);
  if (by_name) {
    const CpSeries *id = cp_df_get_col(by_name, "id");
    const CpSeries *name = cp_df_get_col(by_name, "name");
    CHECK(id && name);

    const char *name_val = NULL;
    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);

    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);

    CHECK(cp_series_get_string(name, 2, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);

    CHECK(cp_series_get_string(name, 3, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Charlie") == 0);

    CHECK(cp_series_get_string(name, 4, &name_val, &is_null));
    CHECK(is_null);
  }

  cp_error_clear(&err);
  CpDataFrame *bad = cp_df_sort_values(df, "missing", 1, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (asc) {
    cp_df_free(asc);
  }
  if (desc) {
    cp_df_free(desc);
  }
  if (by_name) {
    cp_df_free(by_name);
  }
  cp_df_free(df);
}

static void test_sort_values_multi(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"3", "2.0", "Bob"};
  const char *row2[] = {"1", "5.0", "Alice"};
  const char *row3[] = {"2", "1.5", "Charlie"};
  const char *row4[] = {"", "4.5", ""};
  const char *row5[] = {"2", "0.5", "Bob"};

  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));
  CHECK(cp_df_append_row(df, row4, 3, &err));
  CHECK(cp_df_append_row(df, row5, 3, &err));

  const char *keys1[] = {"name", "id"};
  int asc1[] = {1, 0};
  CpDataFrame *sorted = cp_df_sort_values_multi(df, keys1, 2, asc1, &err);
  CHECK(sorted != NULL);
  if (sorted) {
    const CpSeries *id = cp_df_get_col(sorted, "id");
    const CpSeries *name = cp_df_get_col(sorted, "name");
    CHECK(id && name);

    const char *name_val = NULL;
    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);

    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);

    CHECK(cp_series_get_string(name, 2, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);

    CHECK(cp_series_get_string(name, 3, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Charlie") == 0);

    CHECK(cp_series_get_string(name, 4, &name_val, &is_null));
    CHECK(is_null);
  }

  const char *keys2[] = {"id", "score"};
  int asc2[] = {1, 0};
  CpDataFrame *sorted2 = cp_df_sort_values_multi(df, keys2, 2, asc2, &err);
  CHECK(sorted2 != NULL);
  if (sorted2) {
    const CpSeries *id = cp_df_get_col(sorted2, "id");
    const CpSeries *score = cp_df_get_col(sorted2, "score");
    CHECK(id && score);

    int64_t id_val = 0;
    double score_val = 0.0;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_float64(score, 0, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 5.0) < 1e-9);

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_float64(score, 1, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 1.5) < 1e-9);

    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_float64(score, 2, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 0.5) < 1e-9);

    CHECK(cp_series_get_int64(id, 3, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);

    CHECK(cp_series_get_int64(id, 4, &id_val, &is_null));
    CHECK(is_null);
  }

  cp_error_clear(&err);
  CpDataFrame *bad = cp_df_sort_values_multi(df, NULL, 0, NULL, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (sorted) {
    cp_df_free(sorted);
  }
  if (sorted2) {
    cp_df_free(sorted2);
  }
  cp_df_free(df);
}

static void test_head_tail(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "Alice"};
  const char *row2[] = {"2", "Bob"};
  const char *row3[] = {"3", "Charlie"};
  CHECK(cp_df_append_row(df, row1, 2, &err));
  CHECK(cp_df_append_row(df, row2, 2, &err));
  CHECK(cp_df_append_row(df, row3, 2, &err));

  CpDataFrame *head = cp_df_head(df, 2, &err);
  CHECK(head != NULL);
  if (head) {
    CHECK(cp_df_nrows(head) == 2);
    const CpSeries *id = cp_df_get_col(head, "id");
    const CpSeries *name = cp_df_get_col(head, "name");
    CHECK(id && name);

    int64_t id_val = 0;
    const char *name_val = NULL;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Alice") == 0);
  }

  CpDataFrame *tail = cp_df_tail(df, 2, &err);
  CHECK(tail != NULL);
  if (tail) {
    CHECK(cp_df_nrows(tail) == 2);
    const CpSeries *id = cp_df_get_col(tail, "id");
    CHECK(id != NULL);

    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
  }

  CpDataFrame *head0 = cp_df_head(df, 0, &err);
  CHECK(head0 != NULL);
  if (head0) {
    CHECK(cp_df_nrows(head0) == 0);
  }

  CpDataFrame *tail_big = cp_df_tail(df, 10, &err);
  CHECK(tail_big != NULL);
  if (tail_big) {
    CHECK(cp_df_nrows(tail_big) == 3);
  }

  if (head) {
    cp_df_free(head);
  }
  if (tail) {
    cp_df_free(tail);
  }
  if (head0) {
    cp_df_free(head0);
  }
  if (tail_big) {
    cp_df_free(tail_big);
  }
  cp_df_free(df);
}

static void test_dtypes_and_rename_drop_fill(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"a", "b", "c"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  CpDType out[3] = {0};
  CHECK(cp_df_dtypes(df, out, 3, &err));
  CHECK(out[0] == CP_DTYPE_INT64);
  CHECK(out[1] == CP_DTYPE_FLOAT64);
  CHECK(out[2] == CP_DTYPE_STRING);

  cp_error_clear(&err);
  CpDType short_out[1] = {0};
  CHECK(!cp_df_dtypes(df, short_out, 1, &err));
  CHECK(err.code == CP_ERR_INVALID);

  const char *row1[] = {"1", "1.5", "x"};
  const char *row2[] = {"2", "", ""};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));

  const char *drop_cols[] = {"b"};
  CpDataFrame *dropped = cp_df_drop_cols(df, drop_cols, 1, &err);
  CHECK(dropped != NULL);
  if (dropped) {
    CHECK(cp_df_ncols(dropped) == 2);
    CHECK(cp_df_get_col(dropped, "a") != NULL);
    CHECK(cp_df_get_col(dropped, "c") != NULL);
  }

  const char *old_names[] = {"a", "c"};
  const char *new_names[] = {"alpha", "gamma"};
  CpDataFrame *renamed = cp_df_rename_cols(df, old_names, new_names, 2, &err);
  CHECK(renamed != NULL);
  if (renamed) {
    CHECK(cp_df_get_col(renamed, "alpha") != NULL);
    CHECK(cp_df_get_col(renamed, "gamma") != NULL);
    CHECK(cp_df_get_col(renamed, "a") == NULL);
  }

  cp_error_clear(&err);
  const char *drop_all[] = {"a", "b", "c"};
  CpDataFrame *none = cp_df_drop_cols(df, drop_all, 3, &err);
  CHECK(none == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  const char *fill_values[] = {"0", "2.5", "unknown"};
  CpDataFrame *filled = cp_df_fillna(df, fill_values, 3, &err);
  CHECK(filled != NULL);
  if (filled) {
    const CpSeries *b = cp_df_get_col(filled, "b");
    const CpSeries *c = cp_df_get_col(filled, "c");
    CHECK(b && c);

    double b_val = 0.0;
    const char *c_val = NULL;
    int is_null = 0;

    CHECK(cp_series_get_float64(b, 1, &b_val, &is_null));
    CHECK(!is_null && fabs(b_val - 2.5) < 1e-9);
    CHECK(cp_series_get_string(c, 1, &c_val, &is_null));
    CHECK(!is_null && strcmp(c_val, "unknown") == 0);
  }

  cp_error_clear(&err);
  const char *bad_fill[] = {"", "1.0", "x"};
  CpDataFrame *bad = cp_df_fillna(df, bad_fill, 3, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (dropped) {
    cp_df_free(dropped);
  }
  if (renamed) {
    cp_df_free(renamed);
  }
  if (filled) {
    cp_df_free(filled);
  }
  cp_df_free(df);
}

static void test_isnull_dropna(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "1.0", "Alice"};
  const char *row2[] = {"", "", "Bob"};
  const char *row3[] = {"3", "", ""};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));

  uint8_t mask[9] = {0};
  CHECK(cp_df_isnull_mask(df, mask, 9, &err));
  CHECK(mask[0] == 0);
  CHECK(mask[1] == 0);
  CHECK(mask[2] == 0);
  CHECK(mask[3] == 1);
  CHECK(mask[4] == 1);
  CHECK(mask[5] == 0);
  CHECK(mask[6] == 0);
  CHECK(mask[7] == 1);
  CHECK(mask[8] == 1);

  CpDataFrame *dropna = cp_df_dropna(df, &err);
  CHECK(dropna != NULL);
  if (dropna) {
    CHECK(cp_df_nrows(dropna) == 1);
    const CpSeries *id = cp_df_get_col(dropna, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
  }

  if (dropna) {
    cp_df_free(dropna);
  }
  cp_df_free(df);
}

static void test_info_describe(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "2.0", "Alice"};
  const char *row2[] = {"2", "-1.0", ""};
  const char *row3[] = {"", "3.0", "Bob"};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));

  char *path = make_temp_path();
  CHECK(path != NULL);
  if (path) {
    FILE *fp = fopen(path, "w");
    CHECK(fp != NULL);
    if (fp) {
      CHECK(cp_df_info(df, fp, &err));
      fclose(fp);
      char *contents = read_file(path);
      CHECK(contents != NULL);
      if (contents) {
        CHECK(strstr(contents, "Rows: 3") != NULL);
        CHECK(strstr(contents, "Columns: 3") != NULL);
        CHECK(strstr(contents, "[0] id (int64) non-null: 2") != NULL);
        CHECK(strstr(contents, "[1] score (float64) non-null: 3") != NULL);
        CHECK(strstr(contents, "[2] name (string) non-null: 2") != NULL);
      }
      free(contents);
    }
    remove(path);
    free(path);
  }

  cp_error_clear(&err);
  CHECK(!cp_df_info(df, NULL, &err));
  CHECK(err.code == CP_ERR_INVALID);

  CpDataFrame *desc = cp_df_describe(df, &err);
  CHECK(desc != NULL);
  if (desc) {
    CHECK(cp_df_nrows(desc) == 4);
    CHECK(cp_df_ncols(desc) == 3);
    const CpSeries *stat = cp_df_get_col(desc, "stat");
    const CpSeries *id = cp_df_get_col(desc, "id");
    const CpSeries *score = cp_df_get_col(desc, "score");
    CHECK(stat && id && score);

    const char *stat_val = NULL;
    double val = 0.0;
    int is_null = 0;

    CHECK(cp_series_get_string(stat, 0, &stat_val, &is_null));
    CHECK(!is_null && strcmp(stat_val, "count") == 0);
    CHECK(cp_series_get_float64(id, 0, &val, &is_null));
    CHECK(!is_null && fabs(val - 2.0) < 1e-9);

    CHECK(cp_series_get_string(stat, 1, &stat_val, &is_null));
    CHECK(!is_null && strcmp(stat_val, "mean") == 0);
    CHECK(cp_series_get_float64(id, 1, &val, &is_null));
    CHECK(!is_null && fabs(val - 1.5) < 1e-9);

    CHECK(cp_series_get_string(stat, 2, &stat_val, &is_null));
    CHECK(!is_null && strcmp(stat_val, "min") == 0);
    CHECK(cp_series_get_float64(score, 2, &val, &is_null));
    CHECK(!is_null && fabs(val - (-1.0)) < 1e-9);

    CHECK(cp_series_get_string(stat, 3, &stat_val, &is_null));
    CHECK(!is_null && strcmp(stat_val, "max") == 0);
    CHECK(cp_series_get_float64(score, 3, &val, &is_null));
    CHECK(!is_null && fabs(val - 3.0) < 1e-9);
  }

  cp_df_free(desc);
  cp_df_free(df);
}

int main(void) {
  test_read_csv_header();
  test_read_csv_no_header();
  test_write_csv_header();
  test_append_row_errors();
  test_read_csv_mismatch();
  test_aggregations();
  test_df_aggregation_helpers();
  test_select_and_filter();
  test_sort_values();
  test_sort_values_multi();
  test_head_tail();
  test_dtypes_and_rename_drop_fill();
  test_isnull_dropna();
  test_info_describe();

  if (tests_failed != 0) {
    fprintf(stderr, "%d test(s) failed\n", tests_failed);
  }
  return tests_failed == 0 ? 0 : 1;
}
