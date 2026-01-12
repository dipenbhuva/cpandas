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

int main(void) {
  test_read_csv_header();
  test_read_csv_no_header();
  test_write_csv_header();
  test_append_row_errors();
  test_read_csv_mismatch();
  test_aggregations();
  test_df_aggregation_helpers();
  test_select_and_filter();

  if (tests_failed != 0) {
    fprintf(stderr, "%d test(s) failed\n", tests_failed);
  }
  return tests_failed == 0 ? 0 : 1;
}
