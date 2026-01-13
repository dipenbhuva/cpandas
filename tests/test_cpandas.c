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

static void test_loc_iloc(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"10", "1.0", "A"};
  const char *row2[] = {"20", "2.0", "B"};
  const char *row3[] = {"30", "3.0", "C"};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));

  size_t rows[] = {2, 0};
  size_t cols[] = {1, 0};
  CpDataFrame *iloc = cp_df_iloc(df, rows, 2, cols, 2, &err);
  CHECK(iloc != NULL);
  if (iloc) {
    CHECK(cp_df_nrows(iloc) == 2);
    const CpSeries *score = cp_df_get_col(iloc, "score");
    const CpSeries *id = cp_df_get_col(iloc, "id");
    CHECK(score && id);

    double score_val = 0.0;
    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_float64(score, 0, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 3.0) < 1e-9);
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 30);

    CHECK(cp_series_get_float64(score, 1, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 1.0) < 1e-9);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 10);
  }

  size_t rows2[] = {1};
  const char *loc_cols[] = {"name", "id"};
  CpDataFrame *loc = cp_df_loc(df, rows2, 1, loc_cols, 2, &err);
  CHECK(loc != NULL);
  if (loc) {
    CHECK(cp_df_ncols(loc) == 2);
    CHECK(cp_df_nrows(loc) == 1);
    const CpSeries *name = cp_df_get_col(loc, "name");
    const CpSeries *id = cp_df_get_col(loc, "id");
    CHECK(name && id);

    const char *name_val = NULL;
    int64_t id_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "B") == 0);
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 20);
  }

  cp_error_clear(&err);
  size_t bad_row[] = {5};
  CpDataFrame *bad = cp_df_iloc(df, bad_row, 1, NULL, 0, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  const char *bad_cols[] = {"missing"};
  CpDataFrame *bad_loc = cp_df_loc(df, NULL, 0, bad_cols, 1, &err);
  CHECK(bad_loc == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (iloc) {
    cp_df_free(iloc);
  }
  if (loc) {
    cp_df_free(loc);
  }
  cp_df_free(df);
}

static void test_groupby_agg(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"city", "sales", "score"};
  CpDType dtypes[] = {CP_DTYPE_STRING, CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"NY", "10", "1.5"};
  const char *row2[] = {"SF", "5", "2.0"};
  const char *row3[] = {"NY", "7", "2.5"};
  const char *row4[] = {"LA", "8", "2.5"};
  const char *row5[] = {"SF", "", "2.0"};
  const char *row6[] = {"", "4", "9.0"};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));
  CHECK(cp_df_append_row(df, row4, 3, &err));
  CHECK(cp_df_append_row(df, row5, 3, &err));
  CHECK(cp_df_append_row(df, row6, 3, &err));

  const char *value_cols[] = {"sales", "score", "score", "sales"};
  CpAggOp ops[] = {CP_AGG_SUM, CP_AGG_MEAN, CP_AGG_MAX, CP_AGG_COUNT};
  CpDataFrame *grouped = cp_df_groupby_agg(df, "city", value_cols, ops, 4, &err);
  CHECK(grouped != NULL);
  if (grouped) {
    CHECK(cp_df_nrows(grouped) == 3);
    CHECK(cp_df_ncols(grouped) == 5);
    const CpSeries *city = cp_df_get_col(grouped, "city");
    const CpSeries *sales_sum = cp_df_get_col(grouped, "sales_sum");
    const CpSeries *score_mean = cp_df_get_col(grouped, "score_mean");
    const CpSeries *score_max = cp_df_get_col(grouped, "score_max");
    const CpSeries *sales_count = cp_df_get_col(grouped, "sales_count");
    CHECK(city && sales_sum && score_mean && score_max && sales_count);

    const char *city_val = NULL;
    int64_t i64_val = 0;
    double f64_val = 0.0;
    int is_null = 0;

    CHECK(cp_series_get_string(city, 0, &city_val, &is_null));
    CHECK(!is_null && strcmp(city_val, "NY") == 0);
    CHECK(cp_series_get_int64(sales_sum, 0, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 17);
    CHECK(cp_series_get_float64(score_mean, 0, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 2.0) < 1e-9);
    CHECK(cp_series_get_float64(score_max, 0, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 2.5) < 1e-9);
    CHECK(cp_series_get_int64(sales_count, 0, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 2);

    CHECK(cp_series_get_string(city, 1, &city_val, &is_null));
    CHECK(!is_null && strcmp(city_val, "SF") == 0);
    CHECK(cp_series_get_int64(sales_sum, 1, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 5);
    CHECK(cp_series_get_float64(score_mean, 1, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 2.0) < 1e-9);
    CHECK(cp_series_get_float64(score_max, 1, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 2.0) < 1e-9);
    CHECK(cp_series_get_int64(sales_count, 1, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 1);

    CHECK(cp_series_get_string(city, 2, &city_val, &is_null));
    CHECK(!is_null && strcmp(city_val, "LA") == 0);
    CHECK(cp_series_get_int64(sales_sum, 2, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 8);
    CHECK(cp_series_get_float64(score_mean, 2, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 2.5) < 1e-9);
    CHECK(cp_series_get_float64(score_max, 2, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 2.5) < 1e-9);
    CHECK(cp_series_get_int64(sales_count, 2, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 1);
  }

  if (grouped) {
    cp_df_free(grouped);
  }

  cp_error_clear(&err);
  const char *bad_cols[] = {"city"};
  CpAggOp bad_ops[] = {CP_AGG_SUM};
  CpDataFrame *bad = cp_df_groupby_agg(df, "city", bad_cols, bad_ops, 1, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);
  if (bad) {
    cp_df_free(bad);
  }

  cp_df_free(df);

  cp_error_clear(&err);
  const char *names2[] = {"group", "value"};
  CpDType dtypes2[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df2 = cp_df_create(2, names2, dtypes2, 0, &err);
  CHECK(df2 != NULL);
  if (!df2) {
    return;
  }

  const char *g1[] = {"1", "2.0"};
  const char *g2[] = {"1", "4.0"};
  const char *g3[] = {"2", "6.0"};
  CHECK(cp_df_append_row(df2, g1, 2, &err));
  CHECK(cp_df_append_row(df2, g2, 2, &err));
  CHECK(cp_df_append_row(df2, g3, 2, &err));

  const char *value_cols2[] = {"value"};
  CpAggOp ops2[] = {CP_AGG_MEAN};
  CpDataFrame *grouped2 = cp_df_groupby_agg(df2, "group", value_cols2, ops2, 1, &err);
  CHECK(grouped2 != NULL);
  if (grouped2) {
    CHECK(cp_df_nrows(grouped2) == 2);
    const CpSeries *group = cp_df_get_col(grouped2, "group");
    const CpSeries *value_mean = cp_df_get_col(grouped2, "value_mean");
    CHECK(group && value_mean);

    int is_null = 0;
    int64_t group_val = 0;
    double mean_val = 0.0;

    CHECK(cp_series_get_int64(group, 0, &group_val, &is_null));
    CHECK(!is_null && group_val == 1);
    CHECK(cp_series_get_float64(value_mean, 0, &mean_val, &is_null));
    CHECK(!is_null && fabs(mean_val - 3.0) < 1e-9);

    CHECK(cp_series_get_int64(group, 1, &group_val, &is_null));
    CHECK(!is_null && group_val == 2);
    CHECK(cp_series_get_float64(value_mean, 1, &mean_val, &is_null));
    CHECK(!is_null && fabs(mean_val - 6.0) < 1e-9);
  }

  if (grouped2) {
    cp_df_free(grouped2);
  }
  cp_df_free(df2);
}

static void test_join_inner_left(void) {
  CpError err;
  cp_error_clear(&err);

  const char *left_names[] = {"id", "name", "score"};
  CpDType left_types[] = {CP_DTYPE_INT64, CP_DTYPE_STRING, CP_DTYPE_FLOAT64};
  CpDataFrame *left = cp_df_create(3, left_names, left_types, 0, &err);
  CHECK(left != NULL);
  if (!left) {
    return;
  }

  const char *l0[] = {"1", "Ann", "1.0"};
  const char *l1[] = {"2", "Bob", "2.0"};
  const char *l2[] = {"2", "Beth", "2.5"};
  const char *l3[] = {"3", "Cid", "3.0"};
  const char *l4[] = {"", "NullKey", "4.0"};
  const char *l5[] = {"4", "Dan", "4.5"};
  CHECK(cp_df_append_row(left, l0, 3, &err));
  CHECK(cp_df_append_row(left, l1, 3, &err));
  CHECK(cp_df_append_row(left, l2, 3, &err));
  CHECK(cp_df_append_row(left, l3, 3, &err));
  CHECK(cp_df_append_row(left, l4, 3, &err));
  CHECK(cp_df_append_row(left, l5, 3, &err));

  const char *right_names[] = {"id", "city", "score"};
  CpDType right_types[] = {CP_DTYPE_INT64, CP_DTYPE_STRING, CP_DTYPE_FLOAT64};
  CpDataFrame *right = cp_df_create(3, right_names, right_types, 0, &err);
  CHECK(right != NULL);
  if (!right) {
    cp_df_free(left);
    return;
  }

  const char *r0[] = {"1", "NY", "10.0"};
  const char *r1[] = {"2", "SF", "20.0"};
  const char *r2[] = {"2", "LA", "21.0"};
  const char *r3[] = {"5", "TX", "50.0"};
  const char *r4[] = {"", "NA", "99.0"};
  const char *r5[] = {"3", "SEA", ""};
  CHECK(cp_df_append_row(right, r0, 3, &err));
  CHECK(cp_df_append_row(right, r1, 3, &err));
  CHECK(cp_df_append_row(right, r2, 3, &err));
  CHECK(cp_df_append_row(right, r3, 3, &err));
  CHECK(cp_df_append_row(right, r4, 3, &err));
  CHECK(cp_df_append_row(right, r5, 3, &err));

  CpDataFrame *inner = cp_df_join(left, right, "id", "id", CP_JOIN_INNER, &err);
  CHECK(inner != NULL);
  if (inner) {
    CHECK(cp_df_nrows(inner) == 6);
    CHECK(cp_df_ncols(inner) == 5);

    const CpSeries *id = cp_df_get_col(inner, "id");
    const CpSeries *name = cp_df_get_col(inner, "name");
    const CpSeries *score = cp_df_get_col(inner, "score");
    const CpSeries *city = cp_df_get_col(inner, "city");
    const CpSeries *score_right = cp_df_get_col(inner, "score_right");
    CHECK(id && name && score && city && score_right);

    int is_null = 0;
    int64_t id_val = 0;
    double f64_val = 0.0;
    const char *str_val = NULL;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_string(city, 0, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "NY") == 0);
    CHECK(cp_series_get_float64(score_right, 0, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 10.0) < 1e-9);

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_string(city, 1, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "SF") == 0);
    CHECK(cp_series_get_float64(score_right, 1, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 20.0) < 1e-9);

    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_string(city, 2, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "LA") == 0);
    CHECK(cp_series_get_float64(score_right, 2, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 21.0) < 1e-9);

    CHECK(cp_series_get_int64(id, 5, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
    CHECK(cp_series_get_string(city, 5, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "SEA") == 0);
    CHECK(cp_series_get_float64(score_right, 5, &f64_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_string(name, 0, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "Ann") == 0);
    CHECK(cp_series_get_float64(score, 0, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 1.0) < 1e-9);
  }

  CpDataFrame *left_join = cp_df_join(left, right, "id", "id", CP_JOIN_LEFT, &err);
  CHECK(left_join != NULL);
  if (left_join) {
    CHECK(cp_df_nrows(left_join) == 8);

    const CpSeries *id = cp_df_get_col(left_join, "id");
    const CpSeries *name = cp_df_get_col(left_join, "name");
    const CpSeries *city = cp_df_get_col(left_join, "city");
    const CpSeries *score_right = cp_df_get_col(left_join, "score_right");
    CHECK(id && name && city && score_right);

    int is_null = 0;
    int64_t id_val = 0;
    const char *str_val = NULL;
    double f64_val = 0.0;

    CHECK(cp_series_get_int64(id, 6, &id_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(name, 6, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "NullKey") == 0);
    CHECK(cp_series_get_string(city, 6, &str_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_int64(id, 7, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
    CHECK(cp_series_get_string(name, 7, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "Dan") == 0);
    CHECK(cp_series_get_string(city, 7, &str_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_float64(score_right, 7, &f64_val, &is_null));
    CHECK(is_null);
  }

  CpDataFrame *right_join = cp_df_join(left, right, "id", "id", CP_JOIN_RIGHT, &err);
  CHECK(right_join != NULL);
  if (right_join) {
    CHECK(cp_df_nrows(right_join) == 8);

    const CpSeries *id = cp_df_get_col(right_join, "id");
    const CpSeries *name = cp_df_get_col(right_join, "name");
    const CpSeries *city = cp_df_get_col(right_join, "city");
    const CpSeries *score_right = cp_df_get_col(right_join, "score_right");
    CHECK(id && name && city && score_right);

    int is_null = 0;
    int64_t id_val = 0;
    const char *str_val = NULL;
    double f64_val = 0.0;

    CHECK(cp_series_get_int64(id, 6, &id_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(name, 6, &str_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(city, 6, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "TX") == 0);
    CHECK(cp_series_get_float64(score_right, 6, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 50.0) < 1e-9);

    CHECK(cp_series_get_int64(id, 7, &id_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(name, 7, &str_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(city, 7, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "NA") == 0);
    CHECK(cp_series_get_float64(score_right, 7, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 99.0) < 1e-9);
  }

  CpDataFrame *outer_join = cp_df_join(left, right, "id", "id", CP_JOIN_OUTER, &err);
  CHECK(outer_join != NULL);
  if (outer_join) {
    CHECK(cp_df_nrows(outer_join) == 10);

    const CpSeries *id = cp_df_get_col(outer_join, "id");
    const CpSeries *name = cp_df_get_col(outer_join, "name");
    const CpSeries *city = cp_df_get_col(outer_join, "city");
    const CpSeries *score_right = cp_df_get_col(outer_join, "score_right");
    CHECK(id && name && city && score_right);

    int is_null = 0;
    int64_t id_val = 0;
    const char *str_val = NULL;
    double f64_val = 0.0;

    CHECK(cp_series_get_int64(id, 6, &id_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(name, 6, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "NullKey") == 0);
    CHECK(cp_series_get_string(city, 6, &str_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_int64(id, 7, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
    CHECK(cp_series_get_string(name, 7, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "Dan") == 0);
    CHECK(cp_series_get_string(city, 7, &str_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_string(city, 8, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "TX") == 0);
    CHECK(cp_series_get_float64(score_right, 8, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 50.0) < 1e-9);

    CHECK(cp_series_get_string(city, 9, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "NA") == 0);
    CHECK(cp_series_get_float64(score_right, 9, &f64_val, &is_null));
    CHECK(!is_null && fabs(f64_val - 99.0) < 1e-9);
  }

  if (inner) {
    cp_df_free(inner);
  }
  if (left_join) {
    cp_df_free(left_join);
  }
  if (right_join) {
    cp_df_free(right_join);
  }
  if (outer_join) {
    cp_df_free(outer_join);
  }
  cp_df_free(right);
  cp_df_free(left);
}

static void test_join_multi_key(void) {
  CpError err;
  cp_error_clear(&err);

  const char *left_names[] = {"id", "day", "value"};
  CpDType left_types[] = {CP_DTYPE_INT64, CP_DTYPE_STRING, CP_DTYPE_INT64};
  CpDataFrame *left = cp_df_create(3, left_names, left_types, 0, &err);
  CHECK(left != NULL);
  if (!left) {
    return;
  }

  const char *l0[] = {"1", "Mon", "10"};
  const char *l1[] = {"1", "Tue", "11"};
  const char *l2[] = {"2", "Mon", "20"};
  const char *l3[] = {"2", "Wed", "22"};
  CHECK(cp_df_append_row(left, l0, 3, &err));
  CHECK(cp_df_append_row(left, l1, 3, &err));
  CHECK(cp_df_append_row(left, l2, 3, &err));
  CHECK(cp_df_append_row(left, l3, 3, &err));

  const char *right_names[] = {"id", "day", "value", "note"};
  CpDType right_types[] = {CP_DTYPE_INT64, CP_DTYPE_STRING, CP_DTYPE_INT64, CP_DTYPE_STRING};
  CpDataFrame *right = cp_df_create(4, right_names, right_types, 0, &err);
  CHECK(right != NULL);
  if (!right) {
    cp_df_free(left);
    return;
  }

  const char *r0[] = {"1", "Mon", "100", "A"};
  const char *r1[] = {"1", "Wed", "101", "B"};
  const char *r2[] = {"2", "Mon", "200", "C"};
  const char *r3[] = {"2", "Mon", "201", "D"};
  CHECK(cp_df_append_row(right, r0, 4, &err));
  CHECK(cp_df_append_row(right, r1, 4, &err));
  CHECK(cp_df_append_row(right, r2, 4, &err));
  CHECK(cp_df_append_row(right, r3, 4, &err));

  const char *left_keys[] = {"id", "day"};
  const char *right_keys[] = {"id", "day"};
  CpDataFrame *inner = cp_df_join_multi(left,
                                        right,
                                        left_keys,
                                        right_keys,
                                        2,
                                        CP_JOIN_INNER,
                                        "_x",
                                        "_y",
                                        &err);
  CHECK(inner != NULL);
  if (inner) {
    CHECK(cp_df_nrows(inner) == 3);
    CHECK(cp_df_ncols(inner) == 5);

    const CpSeries *id = cp_df_get_col(inner, "id");
    const CpSeries *day = cp_df_get_col(inner, "day");
    const CpSeries *value_x = cp_df_get_col(inner, "value_x");
    const CpSeries *value_y = cp_df_get_col(inner, "value_y");
    const CpSeries *note = cp_df_get_col(inner, "note");
    CHECK(id && day && value_x && value_y && note);

    int is_null = 0;
    int64_t id_val = 0;
    int64_t value_val = 0;
    const char *str_val = NULL;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_string(day, 0, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "Mon") == 0);
    CHECK(cp_series_get_int64(value_x, 0, &value_val, &is_null));
    CHECK(!is_null && value_val == 10);
    CHECK(cp_series_get_int64(value_y, 0, &value_val, &is_null));
    CHECK(!is_null && value_val == 100);
    CHECK(cp_series_get_string(note, 0, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "A") == 0);

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_string(day, 1, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "Mon") == 0);
    CHECK(cp_series_get_int64(value_x, 1, &value_val, &is_null));
    CHECK(!is_null && value_val == 20);
    CHECK(cp_series_get_int64(value_y, 1, &value_val, &is_null));
    CHECK(!is_null && value_val == 200);
    CHECK(cp_series_get_string(note, 1, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "C") == 0);

    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_string(day, 2, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "Mon") == 0);
    CHECK(cp_series_get_int64(value_x, 2, &value_val, &is_null));
    CHECK(!is_null && value_val == 20);
    CHECK(cp_series_get_int64(value_y, 2, &value_val, &is_null));
    CHECK(!is_null && value_val == 201);
    CHECK(cp_series_get_string(note, 2, &str_val, &is_null));
    CHECK(!is_null && strcmp(str_val, "D") == 0);
  }

  if (inner) {
    cp_df_free(inner);
  }
  cp_df_free(right);
  cp_df_free(left);
}

static void test_pivot_table(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"region", "quarter", "sales"};
  CpDType dtypes[] = {CP_DTYPE_STRING, CP_DTYPE_STRING, CP_DTYPE_INT64};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"East", "Q1", "10"};
  const char *r2[] = {"East", "Q2", "20"};
  const char *r3[] = {"West", "Q1", "5"};
  const char *r4[] = {"West", "Q2", "15"};
  const char *r5[] = {"West", "Q2", "5"};
  const char *r6[] = {"East", "Q3", ""};
  const char *r7[] = {"", "Q1", "7"};
  const char *r8[] = {"East", "", "9"};
  CHECK(cp_df_append_row(df, r1, 3, &err));
  CHECK(cp_df_append_row(df, r2, 3, &err));
  CHECK(cp_df_append_row(df, r3, 3, &err));
  CHECK(cp_df_append_row(df, r4, 3, &err));
  CHECK(cp_df_append_row(df, r5, 3, &err));
  CHECK(cp_df_append_row(df, r6, 3, &err));
  CHECK(cp_df_append_row(df, r7, 3, &err));
  CHECK(cp_df_append_row(df, r8, 3, &err));

  CpDataFrame *pivot = cp_df_pivot_table(df, "region", "quarter", "sales", CP_AGG_SUM, &err);
  CHECK(pivot != NULL);
  if (pivot) {
    CHECK(cp_df_nrows(pivot) == 2);
    CHECK(cp_df_ncols(pivot) == 4);

    const CpSeries *region = cp_df_get_col(pivot, "region");
    const CpSeries *q1 = cp_df_get_col(pivot, "Q1");
    const CpSeries *q2 = cp_df_get_col(pivot, "Q2");
    const CpSeries *q3 = cp_df_get_col(pivot, "Q3");
    CHECK(region && q1 && q2 && q3);

    const char *region_val = NULL;
    int64_t i64_val = 0;
    int is_null = 0;

    CHECK(cp_series_get_string(region, 0, &region_val, &is_null));
    CHECK(!is_null && strcmp(region_val, "East") == 0);
    CHECK(cp_series_get_int64(q1, 0, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 10);
    CHECK(cp_series_get_int64(q2, 0, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 20);
    CHECK(cp_series_get_int64(q3, 0, &i64_val, &is_null));
    CHECK(is_null);

    CHECK(cp_series_get_string(region, 1, &region_val, &is_null));
    CHECK(!is_null && strcmp(region_val, "West") == 0);
    CHECK(cp_series_get_int64(q1, 1, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 5);
    CHECK(cp_series_get_int64(q2, 1, &i64_val, &is_null));
    CHECK(!is_null && i64_val == 20);
    CHECK(cp_series_get_int64(q3, 1, &i64_val, &is_null));
    CHECK(is_null);
  }

  if (pivot) {
    cp_df_free(pivot);
  }
  cp_df_free(df);

  cp_error_clear(&err);
  const char *names2[] = {"id", "metric", "value"};
  CpDType dtypes2[] = {CP_DTYPE_INT64, CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df2 = cp_df_create(3, names2, dtypes2, 0, &err);
  CHECK(df2 != NULL);
  if (!df2) {
    return;
  }

  const char *v1[] = {"1", "100", "1.0"};
  const char *v2[] = {"1", "200", "3.0"};
  const char *v3[] = {"1", "100", "5.0"};
  const char *v4[] = {"2", "100", "6.0"};
  const char *v5[] = {"2", "200", ""};
  CHECK(cp_df_append_row(df2, v1, 3, &err));
  CHECK(cp_df_append_row(df2, v2, 3, &err));
  CHECK(cp_df_append_row(df2, v3, 3, &err));
  CHECK(cp_df_append_row(df2, v4, 3, &err));
  CHECK(cp_df_append_row(df2, v5, 3, &err));

  CpDataFrame *pivot2 = cp_df_pivot_table(df2, "id", "metric", "value", CP_AGG_MEAN, &err);
  CHECK(pivot2 != NULL);
  if (pivot2) {
    CHECK(cp_df_nrows(pivot2) == 2);
    CHECK(cp_df_ncols(pivot2) == 3);

    const CpSeries *id = cp_df_get_col(pivot2, "id");
    const CpSeries *c100 = cp_df_get_col(pivot2, "100");
    const CpSeries *c200 = cp_df_get_col(pivot2, "200");
    CHECK(id && c100 && c200);

    int is_null = 0;
    int64_t id_val = 0;
    double mean_val = 0.0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_float64(c100, 0, &mean_val, &is_null));
    CHECK(!is_null && fabs(mean_val - 3.0) < 1e-9);
    CHECK(cp_series_get_float64(c200, 0, &mean_val, &is_null));
    CHECK(!is_null && fabs(mean_val - 3.0) < 1e-9);

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_float64(c100, 1, &mean_val, &is_null));
    CHECK(!is_null && fabs(mean_val - 6.0) < 1e-9);
    CHECK(cp_series_get_float64(c200, 1, &mean_val, &is_null));
    CHECK(is_null);
  }

  if (pivot2) {
    cp_df_free(pivot2);
  }
  cp_df_free(df2);

  cp_error_clear(&err);
  const char *names3[] = {"group", "metric", "label"};
  CpDType dtypes3[] = {CP_DTYPE_INT64, CP_DTYPE_INT64, CP_DTYPE_STRING};
  CpDataFrame *df3 = cp_df_create(3, names3, dtypes3, 0, &err);
  CHECK(df3 != NULL);
  if (!df3) {
    return;
  }

  const char *c1[] = {"1", "10", "A"};
  const char *c2[] = {"1", "10", ""};
  const char *c3[] = {"1", "20", "B"};
  const char *c4[] = {"2", "10", "C"};
  CHECK(cp_df_append_row(df3, c1, 3, &err));
  CHECK(cp_df_append_row(df3, c2, 3, &err));
  CHECK(cp_df_append_row(df3, c3, 3, &err));
  CHECK(cp_df_append_row(df3, c4, 3, &err));

  CpDataFrame *pivot3 = cp_df_pivot_table(df3, "group", "metric", "label", CP_AGG_COUNT, &err);
  CHECK(pivot3 != NULL);
  if (pivot3) {
    CHECK(cp_df_nrows(pivot3) == 2);
    CHECK(cp_df_ncols(pivot3) == 3);

    const CpSeries *group = cp_df_get_col(pivot3, "group");
    const CpSeries *c10 = cp_df_get_col(pivot3, "10");
    const CpSeries *c20 = cp_df_get_col(pivot3, "20");
    CHECK(group && c10 && c20);

    int is_null = 0;
    int64_t group_val = 0;
    int64_t count_val = 0;

    CHECK(cp_series_get_int64(group, 0, &group_val, &is_null));
    CHECK(!is_null && group_val == 1);
    CHECK(cp_series_get_int64(c10, 0, &count_val, &is_null));
    CHECK(!is_null && count_val == 1);
    CHECK(cp_series_get_int64(c20, 0, &count_val, &is_null));
    CHECK(!is_null && count_val == 1);

    CHECK(cp_series_get_int64(group, 1, &group_val, &is_null));
    CHECK(!is_null && group_val == 2);
    CHECK(cp_series_get_int64(c10, 1, &count_val, &is_null));
    CHECK(!is_null && count_val == 1);
    CHECK(cp_series_get_int64(c20, 1, &count_val, &is_null));
    CHECK(!is_null && count_val == 0);
  }

  if (pivot3) {
    cp_df_free(pivot3);
  }

  cp_error_clear(&err);
  CpDataFrame *bad = cp_df_pivot_table(df3, "group", "metric", "label", CP_AGG_SUM, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);
  if (bad) {
    cp_df_free(bad);
  }
  cp_df_free(df3);
}

static void test_predicate_filters(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "1.5", "Alice"};
  const char *row2[] = {"2", "-1.0", "Bob"};
  const char *row3[] = {"3", "0.0", "Charlie"};
  const char *row4[] = {"", "2.0", "Bob"};
  const char *row5[] = {"4", "", ""};
  CHECK(cp_df_append_row(df, row1, 3, &err));
  CHECK(cp_df_append_row(df, row2, 3, &err));
  CHECK(cp_df_append_row(df, row3, 3, &err));
  CHECK(cp_df_append_row(df, row4, 3, &err));
  CHECK(cp_df_append_row(df, row5, 3, &err));

  uint8_t mask[5] = {0};
  CHECK(cp_df_mask_int64(df, "id", CP_OP_GT, 2, mask, 5, &err));
  CHECK(mask[0] == 0);
  CHECK(mask[1] == 0);
  CHECK(mask[2] == 1);
  CHECK(mask[3] == 0);
  CHECK(mask[4] == 1);

  CpDataFrame *filtered = cp_df_filter_int64(df, "id", CP_OP_GT, 2, &err);
  CHECK(filtered != NULL);
  if (filtered) {
    CHECK(cp_df_nrows(filtered) == 2);
    const CpSeries *id = cp_df_get_col(filtered, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
  }

  uint8_t mask2[5] = {0};
  CHECK(cp_df_mask_float64(df, "score", CP_OP_LE, 0.0, mask2, 5, &err));
  CHECK(mask2[0] == 0);
  CHECK(mask2[1] == 1);
  CHECK(mask2[2] == 1);
  CHECK(mask2[3] == 0);
  CHECK(mask2[4] == 0);

  CpDataFrame *filtered2 = cp_df_filter_float64(df, "score", CP_OP_LE, 0.0, &err);
  CHECK(filtered2 != NULL);
  if (filtered2) {
    CHECK(cp_df_nrows(filtered2) == 2);
    const CpSeries *name = cp_df_get_col(filtered2, "name");
    CHECK(name != NULL);
    const char *name_val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(name, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Charlie") == 0);
  }

  CpDataFrame *filtered3 = cp_df_filter_string(df, "name", CP_OP_EQ, "Bob", &err);
  CHECK(filtered3 != NULL);
  if (filtered3) {
    CHECK(cp_df_nrows(filtered3) == 2);
    const CpSeries *id = cp_df_get_col(filtered3, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(is_null);
  }

  cp_error_clear(&err);
  CHECK(!cp_df_mask_int64(df, "score", CP_OP_GT, 1, mask, 5, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_mask_string(df, "name", CP_OP_EQ, NULL, mask, 5, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_mask_float64(df, "score", CP_OP_GT, 1.0, mask, 2, &err));
  CHECK(err.code == CP_ERR_INVALID);

  if (filtered) {
    cp_df_free(filtered);
  }
  if (filtered2) {
    cp_df_free(filtered2);
  }
  if (filtered3) {
    cp_df_free(filtered3);
  }
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
  test_loc_iloc();
  test_groupby_agg();
  test_join_inner_left();
  test_join_multi_key();
  test_pivot_table();
  test_predicate_filters();

  if (tests_failed != 0) {
    fprintf(stderr, "%d test(s) failed\n", tests_failed);
  }
  return tests_failed == 0 ? 0 : 1;
}
