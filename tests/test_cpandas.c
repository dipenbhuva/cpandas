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

typedef struct {
  const CpSeries *id;
  const CpSeries *score;
} ApplySumCtx;

static int apply_sum_row(const CpDataFrame *df,
                         size_t row,
                         void *user_data,
                         CpValue *out,
                         CpError *err) {
  (void)df;
  (void)err;
  if (!user_data || !out) {
    return 0;
  }
  ApplySumCtx *ctx = (ApplySumCtx *)user_data;
  int64_t id_val = 0;
  double score_val = 0.0;
  int is_null = 0;
  if (!cp_series_get_int64(ctx->id, row, &id_val, &is_null)) {
    return 0;
  }
  if (is_null) {
    out->is_null = 1;
    return 1;
  }
  if (!cp_series_get_float64(ctx->score, row, &score_val, &is_null)) {
    return 0;
  }
  if (is_null) {
    out->is_null = 1;
    return 1;
  }
  out->is_null = 0;
  out->value.i64 = id_val + (int64_t)score_val;
  return 1;
}

static int transform_double(const CpSeries *series,
                            size_t row,
                            void *user_data,
                            CpValue *out,
                            CpError *err) {
  (void)user_data;
  (void)err;
  if (!series || !out) {
    return 0;
  }
  double value = 0.0;
  int is_null = 0;
  if (!cp_series_get_float64(series, row, &value, &is_null)) {
    return 0;
  }
  if (is_null) {
    out->is_null = 1;
    return 1;
  }
  out->is_null = 0;
  out->value.f64 = value * 2.0;
  return 1;
}

typedef struct {
  const CpSeries *id;
  int64_t sum;
} IterRowsCtx;

static int iterrows_sum(const CpDataFrame *df,
                        size_t row,
                        void *user_data,
                        CpError *err) {
  (void)df;
  (void)err;
  if (!user_data) {
    return 0;
  }
  IterRowsCtx *ctx = (IterRowsCtx *)user_data;
  int64_t id_val = 0;
  int is_null = 0;
  if (!cp_series_get_int64(ctx->id, row, &id_val, &is_null)) {
    return 0;
  }
  if (!is_null) {
    ctx->sum += id_val;
  }
  return 1;
}

typedef struct {
  size_t count;
} IterItemsCtx;

static int iteritems_count(const CpSeries *series,
                           size_t col,
                           void *user_data,
                           CpError *err) {
  (void)series;
  (void)col;
  (void)err;
  if (!user_data) {
    return 0;
  }
  IterItemsCtx *ctx = (IterItemsCtx *)user_data;
  ctx->count += 1;
  return 1;
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

static void test_metadata_helpers(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 2, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1", "Alice"};
  const char *row2[] = {"2", "Bob"};
  CHECK(cp_df_append_row(df, row1, 2, &err));
  CHECK(cp_df_append_row(df, row2, 2, &err));

  size_t rows = 0;
  size_t cols = 0;
  CHECK(cp_df_shape(df, &rows, &cols, &err));
  CHECK(rows == 2);
  CHECK(cols == 2);
  CHECK(cp_df_size(df) == 4);
  CHECK(cp_df_ndim(df) == 2);

  const char *col_names[2] = {NULL, NULL};
  CHECK(cp_df_columns(df, col_names, 2, &err));
  CHECK(col_names[0] && strcmp(col_names[0], "id") == 0);
  CHECK(col_names[1] && strcmp(col_names[1], "name") == 0);

  CpDataFrame *copy = cp_df_copy(df, &err);
  CHECK(copy != NULL);
  if (copy) {
    CHECK(cp_df_nrows(copy) == 2);
    CHECK(cp_df_ncols(copy) == 2);

    const CpSeries *id = cp_df_get_col(copy, "id");
    const CpSeries *name = cp_df_get_col(copy, "name");
    CHECK(id && name);

    int64_t id_val = 0;
    const char *name_val = NULL;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bob") == 0);
  }

  const char *row3[] = {"3", "Cara"};
  CHECK(cp_df_append_row(df, row3, 2, &err));
  CHECK(cp_df_nrows(df) == 3);
  if (copy) {
    CHECK(cp_df_nrows(copy) == 2);
  }

  cp_error_clear(&err);
  CHECK(!cp_df_shape(NULL, &rows, &cols, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_shape(df, NULL, NULL, &err));
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  CHECK(!cp_df_columns(df, col_names, 1, &err));
  CHECK(err.code == CP_ERR_INVALID);

  if (copy) {
    cp_df_free(copy);
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

static void test_unique_counts_int64(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id"};
  CpDType dtypes[] = {CP_DTYPE_INT64};
  CpDataFrame *df = cp_df_create(1, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1"};
  const char *row2[] = {"2"};
  const char *row3[] = {"2"};
  const char *row4[] = {""};
  const char *row5[] = {"3"};
  const char *row6[] = {"3"};
  CHECK(cp_df_append_row(df, row1, 1, &err));
  CHECK(cp_df_append_row(df, row2, 1, &err));
  CHECK(cp_df_append_row(df, row3, 1, &err));
  CHECK(cp_df_append_row(df, row4, 1, &err));
  CHECK(cp_df_append_row(df, row5, 1, &err));
  CHECK(cp_df_append_row(df, row6, 1, &err));

  CpDataFrame *unique = cp_df_unique(df, "id", &err);
  CHECK(unique != NULL);
  if (unique) {
    CHECK(cp_df_nrows(unique) == 4);
    const CpSeries *id = cp_df_get_col(unique, "id");
    CHECK(id != NULL);

    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &v, &is_null));
    CHECK(!is_null && v == 1);
    CHECK(cp_series_get_int64(id, 1, &v, &is_null));
    CHECK(!is_null && v == 2);
    CHECK(cp_series_get_int64(id, 2, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(id, 3, &v, &is_null));
    CHECK(!is_null && v == 3);
  }

  size_t nunique = 0;
  CHECK(cp_df_nunique(df, "id", &nunique, &err));
  CHECK(nunique == 3);

  CpDataFrame *counts = cp_df_value_counts(df, "id", &err);
  CHECK(counts != NULL);
  if (counts) {
    CHECK(cp_df_nrows(counts) == 3);
    const CpSeries *values = cp_df_get_col(counts, "id");
    const CpSeries *cnt = cp_df_get_col(counts, "count");
    CHECK(values && cnt);

    int64_t count_1 = 0;
    int64_t count_2 = 0;
    int64_t count_3 = 0;
    for (size_t i = 0; i < cp_series_len(values); ++i) {
      int64_t val = 0;
      int64_t c = 0;
      int is_null = 0;
      CHECK(cp_series_get_int64(values, i, &val, &is_null));
      CHECK(!is_null);
      CHECK(cp_series_get_int64(cnt, i, &c, &is_null));
      CHECK(!is_null);
      if (val == 1) {
        count_1 = c;
      } else if (val == 2) {
        count_2 = c;
      } else if (val == 3) {
        count_3 = c;
      }
    }
    CHECK(count_1 == 1);
    CHECK(count_2 == 2);
    CHECK(count_3 == 2);
  }

  uint8_t dup[6] = {0};
  uint8_t expected_first[6] = {0, 0, 1, 0, 0, 1};
  uint8_t expected_last[6] = {0, 1, 0, 0, 1, 0};
  uint8_t expected_none[6] = {0, 1, 1, 0, 1, 1};
  CHECK(cp_df_duplicated(df, "id", CP_DUP_KEEP_FIRST, dup, 6, &err));
  for (size_t i = 0; i < 6; ++i) {
    CHECK(dup[i] == expected_first[i]);
  }
  CHECK(cp_df_duplicated(df, "id", CP_DUP_KEEP_LAST, dup, 6, &err));
  for (size_t i = 0; i < 6; ++i) {
    CHECK(dup[i] == expected_last[i]);
  }
  CHECK(cp_df_duplicated(df, "id", CP_DUP_KEEP_NONE, dup, 6, &err));
  for (size_t i = 0; i < 6; ++i) {
    CHECK(dup[i] == expected_none[i]);
  }

  CpDataFrame *drop_first = cp_df_drop_duplicates(df, "id", CP_DUP_KEEP_FIRST,
                                                  &err);
  CHECK(drop_first != NULL);
  if (drop_first) {
    CHECK(cp_df_nrows(drop_first) == 4);
    const CpSeries *id = cp_df_get_col(drop_first, "id");
    CHECK(id != NULL);
    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &v, &is_null));
    CHECK(!is_null && v == 1);
    CHECK(cp_series_get_int64(id, 1, &v, &is_null));
    CHECK(!is_null && v == 2);
    CHECK(cp_series_get_int64(id, 2, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(id, 3, &v, &is_null));
    CHECK(!is_null && v == 3);
  }

  CpDataFrame *drop_none = cp_df_drop_duplicates(df, "id", CP_DUP_KEEP_NONE,
                                                 &err);
  CHECK(drop_none != NULL);
  if (drop_none) {
    CHECK(cp_df_nrows(drop_none) == 2);
    const CpSeries *id = cp_df_get_col(drop_none, "id");
    CHECK(id != NULL);
    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &v, &is_null));
    CHECK(!is_null && v == 1);
    CHECK(cp_series_get_int64(id, 1, &v, &is_null));
    CHECK(is_null);
  }

  if (unique) {
    cp_df_free(unique);
  }
  if (counts) {
    cp_df_free(counts);
  }
  if (drop_first) {
    cp_df_free(drop_first);
  }
  if (drop_none) {
    cp_df_free(drop_none);
  }
  cp_df_free(df);
}

static void test_unique_counts_string(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"name"};
  CpDType dtypes[] = {CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(1, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"Alice"};
  const char *row2[] = {"Bob"};
  const char *row3[] = {"Alice"};
  const char *row4[] = {""};
  const char *row5[] = {"Bob"};
  CHECK(cp_df_append_row(df, row1, 1, &err));
  CHECK(cp_df_append_row(df, row2, 1, &err));
  CHECK(cp_df_append_row(df, row3, 1, &err));
  CHECK(cp_df_append_row(df, row4, 1, &err));
  CHECK(cp_df_append_row(df, row5, 1, &err));

  CpDataFrame *unique = cp_df_unique(df, "name", &err);
  CHECK(unique != NULL);
  if (unique) {
    CHECK(cp_df_nrows(unique) == 3);
    const CpSeries *name = cp_df_get_col(unique, "name");
    CHECK(name != NULL);

    const char *val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(name, 0, &val, &is_null));
    CHECK(!is_null && strcmp(val, "Alice") == 0);
    CHECK(cp_series_get_string(name, 1, &val, &is_null));
    CHECK(!is_null && strcmp(val, "Bob") == 0);
    CHECK(cp_series_get_string(name, 2, &val, &is_null));
    CHECK(is_null);
  }

  size_t nunique = 0;
  CHECK(cp_df_nunique(df, "name", &nunique, &err));
  CHECK(nunique == 2);

  CpDataFrame *counts = cp_df_value_counts(df, "name", &err);
  CHECK(counts != NULL);
  if (counts) {
    CHECK(cp_df_nrows(counts) == 2);
    const CpSeries *values = cp_df_get_col(counts, "name");
    const CpSeries *cnt = cp_df_get_col(counts, "count");
    CHECK(values && cnt);

    int64_t count_alice = 0;
    int64_t count_bob = 0;
    for (size_t i = 0; i < cp_series_len(values); ++i) {
      const char *val = NULL;
      int64_t c = 0;
      int is_null = 0;
      CHECK(cp_series_get_string(values, i, &val, &is_null));
      CHECK(!is_null);
      CHECK(cp_series_get_int64(cnt, i, &c, &is_null));
      CHECK(!is_null);
      if (strcmp(val, "Alice") == 0) {
        count_alice = c;
      } else if (strcmp(val, "Bob") == 0) {
        count_bob = c;
      }
    }
    CHECK(count_alice == 2);
    CHECK(count_bob == 2);
  }

  uint8_t dup[5] = {0};
  uint8_t expected[5] = {0, 0, 1, 0, 1};
  CHECK(cp_df_duplicated(df, "name", CP_DUP_KEEP_FIRST, dup, 5, &err));
  for (size_t i = 0; i < 5; ++i) {
    CHECK(dup[i] == expected[i]);
  }

  if (unique) {
    cp_df_free(unique);
  }
  if (counts) {
    cp_df_free(counts);
  }
  cp_df_free(df);
}

static void test_sample(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id"};
  CpDType dtypes[] = {CP_DTYPE_INT64};
  CpDataFrame *df = cp_df_create(1, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *row1[] = {"1"};
  const char *row2[] = {"2"};
  const char *row3[] = {"3"};
  const char *row4[] = {"4"};
  const char *row5[] = {"5"};
  CHECK(cp_df_append_row(df, row1, 1, &err));
  CHECK(cp_df_append_row(df, row2, 1, &err));
  CHECK(cp_df_append_row(df, row3, 1, &err));
  CHECK(cp_df_append_row(df, row4, 1, &err));
  CHECK(cp_df_append_row(df, row5, 1, &err));

  CpDataFrame *empty = cp_df_sample(df, 0, 0, 42, &err);
  CHECK(empty != NULL);
  if (empty) {
    CHECK(cp_df_nrows(empty) == 0);
  }

  CpDataFrame *sample = cp_df_sample(df, 3, 0, 123, &err);
  CHECK(sample != NULL);
  if (sample) {
    CHECK(cp_df_nrows(sample) == 3);
    const CpSeries *id = cp_df_get_col(sample, "id");
    CHECK(id != NULL);
    int64_t vals[3] = {0, 0, 0};
    for (size_t i = 0; i < 3; ++i) {
      int is_null = 0;
      CHECK(cp_series_get_int64(id, i, &vals[i], &is_null));
      CHECK(!is_null);
      CHECK(vals[i] >= 1 && vals[i] <= 5);
    }
    CHECK(vals[0] != vals[1]);
    CHECK(vals[0] != vals[2]);
    CHECK(vals[1] != vals[2]);
  }

  CpDataFrame *sample_rep = cp_df_sample(df, 8, 1, 7, &err);
  CHECK(sample_rep != NULL);
  if (sample_rep) {
    CHECK(cp_df_nrows(sample_rep) == 8);
    const CpSeries *id = cp_df_get_col(sample_rep, "id");
    CHECK(id != NULL);
    for (size_t i = 0; i < 8; ++i) {
      int64_t v = 0;
      int is_null = 0;
      CHECK(cp_series_get_int64(id, i, &v, &is_null));
      CHECK(!is_null);
      CHECK(v >= 1 && v <= 5);
    }
  }

  cp_error_clear(&err);
  CpDataFrame *bad = cp_df_sample(df, 6, 0, 1, &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (empty) {
    cp_df_free(empty);
  }
  if (sample) {
    cp_df_free(sample);
  }
  if (sample_rep) {
    cp_df_free(sample_rep);
  }
  cp_df_free(df);
}

static void test_nlargest_nsmallest(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "3.0"};
  const char *r2[] = {"2", ""};
  const char *r3[] = {"3", "nan"};
  const char *r4[] = {"4", "1.5"};
  const char *r5[] = {"5", "2.5"};
  CHECK(cp_df_append_row(df, r1, 2, &err));
  CHECK(cp_df_append_row(df, r2, 2, &err));
  CHECK(cp_df_append_row(df, r3, 2, &err));
  CHECK(cp_df_append_row(df, r4, 2, &err));
  CHECK(cp_df_append_row(df, r5, 2, &err));

  CpDataFrame *largest = cp_df_nlargest(df, "score", 2, &err);
  CHECK(largest != NULL);
  if (largest) {
    CHECK(cp_df_nrows(largest) == 2);
    const CpSeries *id = cp_df_get_col(largest, "id");
    const CpSeries *score = cp_df_get_col(largest, "score");
    CHECK(id && score);
    int64_t id_val = 0;
    double score_val = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_float64(score, 0, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 3.0) < 1e-9);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 5);
  }

  CpDataFrame *smallest = cp_df_nsmallest(df, "score", 2, &err);
  CHECK(smallest != NULL);
  if (smallest) {
    CHECK(cp_df_nrows(smallest) == 2);
    const CpSeries *id = cp_df_get_col(smallest, "id");
    const CpSeries *score = cp_df_get_col(smallest, "score");
    CHECK(id && score);
    int64_t id_val = 0;
    double score_val = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
    CHECK(cp_series_get_float64(score, 0, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 1.5) < 1e-9);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 5);
  }

  CpDataFrame *largest_all = cp_df_nlargest(df, "score", 10, &err);
  CHECK(largest_all != NULL);
  if (largest_all) {
    CHECK(cp_df_nrows(largest_all) == 3);
  }

  CpDataFrame *df_str = NULL;
  const char *s_names[] = {"label"};
  CpDType s_dtypes[] = {CP_DTYPE_STRING};
  df_str = cp_df_create(1, s_names, s_dtypes, 0, &err);
  CHECK(df_str != NULL);
  if (df_str) {
    const char *srow[] = {"a"};
    CHECK(cp_df_append_row(df_str, srow, 1, &err));
    cp_error_clear(&err);
    CpDataFrame *bad = cp_df_nlargest(df_str, "label", 1, &err);
    CHECK(bad == NULL);
    CHECK(err.code == CP_ERR_INVALID);
  }

  if (largest) {
    cp_df_free(largest);
  }
  if (smallest) {
    cp_df_free(smallest);
  }
  if (largest_all) {
    cp_df_free(largest_all);
  }
  if (df_str) {
    cp_df_free(df_str);
  }
  cp_df_free(df);
}

static void test_where_mask_clip_replace(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "3.0", "Alice"};
  const char *r2[] = {"2", "", "Bob"};
  const char *r3[] = {"3", "1.0", ""};
  const char *r4[] = {"4", "nan", "Dana"};
  const char *r5[] = {"5", "5.0", "Bob"};
  const char *r6[] = {"", "2.0", "Eve"};
  CHECK(cp_df_append_row(df, r1, 3, &err));
  CHECK(cp_df_append_row(df, r2, 3, &err));
  CHECK(cp_df_append_row(df, r3, 3, &err));
  CHECK(cp_df_append_row(df, r4, 3, &err));
  CHECK(cp_df_append_row(df, r5, 3, &err));
  CHECK(cp_df_append_row(df, r6, 3, &err));

  uint8_t mask[6] = {1, 0, 1, 0, 0, 1};
  const char *repl[] = {"0", "2.5", "missing"};

  CpDataFrame *where_df = cp_df_where(df, mask, 6, repl, 3, &err);
  CHECK(where_df != NULL);
  if (where_df) {
    const CpSeries *id = cp_df_get_col(where_df, "id");
    const CpSeries *score = cp_df_get_col(where_df, "score");
    const CpSeries *name = cp_df_get_col(where_df, "name");
    CHECK(id && score && name);

    int64_t id_val = 0;
    double score_val = 0.0;
    const char *name_val = NULL;
    int is_null = 0;

    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 0);
    CHECK(cp_series_get_float64(score, 1, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 2.5) < 1e-9);
    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "missing") == 0);

    CHECK(cp_series_get_int64(id, 5, &id_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(name, 5, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Eve") == 0);
  }

  CpDataFrame *mask_df = cp_df_mask(df, mask, 6, repl, 3, &err);
  CHECK(mask_df != NULL);
  if (mask_df) {
    const CpSeries *id = cp_df_get_col(mask_df, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 0);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
  }

  CpDataFrame *where_null = cp_df_where(df, mask, 6, NULL, 0, &err);
  CHECK(where_null != NULL);
  if (where_null) {
    const CpSeries *id = cp_df_get_col(where_null, "id");
    const CpSeries *score = cp_df_get_col(where_null, "score");
    const CpSeries *name = cp_df_get_col(where_null, "name");
    CHECK(id && score && name);
    int is_null = 0;
    int64_t id_val = 0;
    double score_val = 0.0;
    const char *name_val = NULL;
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_float64(score, 1, &score_val, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(is_null);
  }

  CpDataFrame *clipped = cp_df_clip(df, "score", 2.0, 4.0, &err);
  CHECK(clipped != NULL);
  if (clipped) {
    const CpSeries *score = cp_df_get_col(clipped, "score");
    CHECK(score != NULL);
    double score_val = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_float64(score, 2, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 2.0) < 1e-9);
    CHECK(cp_series_get_float64(score, 4, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val - 4.0) < 1e-9);
    CHECK(cp_series_get_float64(score, 3, &score_val, &is_null));
    CHECK(!is_null && isnan(score_val));
  }

  CpDataFrame *repl_name = cp_df_replace(df, "name", "Bob", "Bobby", &err);
  CHECK(repl_name != NULL);
  if (repl_name) {
    const CpSeries *name = cp_df_get_col(repl_name, "name");
    CHECK(name != NULL);
    const char *name_val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Bobby") == 0);
  }

  CpDataFrame *repl_nan = cp_df_replace(df, "score", "nan", "0.0", &err);
  CHECK(repl_nan != NULL);
  if (repl_nan) {
    const CpSeries *score = cp_df_get_col(repl_nan, "score");
    CHECK(score != NULL);
    double score_val = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_float64(score, 3, &score_val, &is_null));
    CHECK(!is_null && fabs(score_val) < 1e-9);
  }

  CpDataFrame *repl_null = cp_df_replace(df, "id", "", "7", &err);
  CHECK(repl_null != NULL);
  if (repl_null) {
    const CpSeries *id = cp_df_get_col(repl_null, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 5, &id_val, &is_null));
    CHECK(!is_null && id_val == 7);
  }

  cp_error_clear(&err);
  CpDataFrame *bad_clip = cp_df_clip(df, "name", 0.0, 1.0, &err);
  CHECK(bad_clip == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  cp_error_clear(&err);
  const char *bad_vals[] = {"1"};
  CpDataFrame *bad_where = cp_df_where(df, mask, 6, bad_vals, 1, &err);
  CHECK(bad_where == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (where_df) {
    cp_df_free(where_df);
  }
  if (mask_df) {
    cp_df_free(mask_df);
  }
  if (where_null) {
    cp_df_free(where_null);
  }
  if (clipped) {
    cp_df_free(clipped);
  }
  if (repl_name) {
    cp_df_free(repl_name);
  }
  if (repl_nan) {
    cp_df_free(repl_nan);
  }
  if (repl_null) {
    cp_df_free(repl_null);
  }
  cp_df_free(df);
}

static void test_concat(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_STRING};
  CpDataFrame *df1 = cp_df_create(2, names, dtypes, 0, &err);
  CpDataFrame *df2 = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df1 != NULL);
  CHECK(df2 != NULL);
  if (!df1 || !df2) {
    if (df1) {
      cp_df_free(df1);
    }
    if (df2) {
      cp_df_free(df2);
    }
    return;
  }

  const char *r1[] = {"1", "Alice"};
  const char *r2[] = {"2", "Bob"};
  const char *r3[] = {"3", "Cara"};
  CHECK(cp_df_append_row(df1, r1, 2, &err));
  CHECK(cp_df_append_row(df1, r2, 2, &err));
  CHECK(cp_df_append_row(df2, r3, 2, &err));

  const CpDataFrame *row_dfs[] = {df1, df2};
  CpDataFrame *rows = cp_df_concat(row_dfs, 2, CP_CONCAT_ROWS, &err);
  CHECK(rows != NULL);
  if (rows) {
    CHECK(cp_df_nrows(rows) == 3);
    const CpSeries *id = cp_df_get_col(rows, "id");
    const CpSeries *name = cp_df_get_col(rows, "name");
    CHECK(id && name);
    int64_t id_val = 0;
    const char *name_val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
    CHECK(cp_series_get_string(name, 2, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "Cara") == 0);
  }

  const char *bad_names[] = {"id", "label"};
  CpDataFrame *df_bad = cp_df_create(2, bad_names, dtypes, 0, &err);
  CHECK(df_bad != NULL);
  if (df_bad) {
    const CpDataFrame *bad_dfs[] = {df1, df_bad};
    cp_error_clear(&err);
    CpDataFrame *bad = cp_df_concat(bad_dfs, 2, CP_CONCAT_ROWS, &err);
    CHECK(bad == NULL);
    CHECK(err.code == CP_ERR_INVALID);
  }

  const char *a_names[] = {"id", "score"};
  CpDType a_dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  const char *b_names[] = {"name"};
  CpDType b_dtypes[] = {CP_DTYPE_STRING};
  CpDataFrame *dfa = cp_df_create(2, a_names, a_dtypes, 0, &err);
  CpDataFrame *dfb = cp_df_create(1, b_names, b_dtypes, 0, &err);
  CHECK(dfa != NULL);
  CHECK(dfb != NULL);
  if (dfa && dfb) {
    const char *ra1[] = {"1", "1.5"};
    const char *ra2[] = {"2", "2.5"};
    const char *rb1[] = {"Alice"};
    const char *rb2[] = {"Bob"};
    CHECK(cp_df_append_row(dfa, ra1, 2, &err));
    CHECK(cp_df_append_row(dfa, ra2, 2, &err));
    CHECK(cp_df_append_row(dfb, rb1, 1, &err));
    CHECK(cp_df_append_row(dfb, rb2, 1, &err));

    const CpDataFrame *col_dfs[] = {dfa, dfb};
    CpDataFrame *cols = cp_df_concat(col_dfs, 2, CP_CONCAT_COLS, &err);
    CHECK(cols != NULL);
    if (cols) {
      CHECK(cp_df_ncols(cols) == 3);
      CHECK(cp_df_nrows(cols) == 2);
      const CpSeries *name = cp_df_get_col(cols, "name");
      const CpSeries *score = cp_df_get_col(cols, "score");
      CHECK(name && score);
      const char *name_val = NULL;
      double score_val = 0.0;
      int is_null = 0;
      CHECK(cp_series_get_string(name, 1, &name_val, &is_null));
      CHECK(!is_null && strcmp(name_val, "Bob") == 0);
      CHECK(cp_series_get_float64(score, 0, &score_val, &is_null));
      CHECK(!is_null && fabs(score_val - 1.5) < 1e-9);
      cp_df_free(cols);
    }

    CpDataFrame *df_short = cp_df_create(1, b_names, b_dtypes, 0, &err);
    CHECK(df_short != NULL);
    if (df_short) {
      const char *rb_short[] = {"Solo"};
      CHECK(cp_df_append_row(df_short, rb_short, 1, &err));
      const CpDataFrame *bad_cols[] = {dfa, df_short};
      cp_error_clear(&err);
      CpDataFrame *bad = cp_df_concat(bad_cols, 2, CP_CONCAT_COLS, &err);
      CHECK(bad == NULL);
      CHECK(err.code == CP_ERR_INVALID);
      cp_df_free(df_short);
    }

    const char *dup_names[] = {"id"};
    CpDType dup_dtypes[] = {CP_DTYPE_INT64};
    CpDataFrame *df_dup = cp_df_create(1, dup_names, dup_dtypes, 0, &err);
    CHECK(df_dup != NULL);
    if (df_dup) {
      const char *dup1[] = {"10"};
      const char *dup2[] = {"11"};
      CHECK(cp_df_append_row(df_dup, dup1, 1, &err));
      CHECK(cp_df_append_row(df_dup, dup2, 1, &err));
      const CpDataFrame *dup_cols[] = {dfa, df_dup};
      cp_error_clear(&err);
      CpDataFrame *bad = cp_df_concat(dup_cols, 2, CP_CONCAT_COLS, &err);
      CHECK(bad == NULL);
      CHECK(err.code == CP_ERR_INVALID);
      cp_df_free(df_dup);
    }
  }

  if (rows) {
    cp_df_free(rows);
  }
  if (df_bad) {
    cp_df_free(df_bad);
  }
  if (dfa) {
    cp_df_free(dfa);
  }
  if (dfb) {
    cp_df_free(dfb);
  }
  cp_df_free(df1);
  cp_df_free(df2);
}

static void test_apply_transform_iter(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "2.0"};
  const char *r2[] = {"2", ""};
  const char *r3[] = {"3", "4.0"};
  CHECK(cp_df_append_row(df, r1, 2, &err));
  CHECK(cp_df_append_row(df, r2, 2, &err));
  CHECK(cp_df_append_row(df, r3, 2, &err));

  ApplySumCtx apply_ctx = {0};
  apply_ctx.id = cp_df_get_col(df, "id");
  apply_ctx.score = cp_df_get_col(df, "score");
  CHECK(apply_ctx.id != NULL);
  CHECK(apply_ctx.score != NULL);

  CpDataFrame *applied =
      cp_df_apply(df, CP_DTYPE_INT64, "sum", apply_sum_row, &apply_ctx, &err);
  CHECK(applied != NULL);
  if (applied) {
    CHECK(cp_df_ncols(applied) == 1);
    CHECK(cp_df_nrows(applied) == 3);
    const CpSeries *sum = cp_df_get_col(applied, "sum");
    CHECK(sum != NULL);
    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(sum, 0, &v, &is_null));
    CHECK(!is_null && v == 3);
    CHECK(cp_series_get_int64(sum, 1, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(sum, 2, &v, &is_null));
    CHECK(!is_null && v == 7);
  }

  CpDataFrame *transformed =
      cp_df_transform(df, "score", CP_DTYPE_FLOAT64, transform_double, NULL,
                      &err);
  CHECK(transformed != NULL);
  if (transformed) {
    const CpSeries *score = cp_df_get_col(transformed, "score");
    CHECK(score != NULL);
    double v = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_float64(score, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 4.0) < 1e-9);
    CHECK(cp_series_get_float64(score, 1, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_float64(score, 2, &v, &is_null));
    CHECK(!is_null && fabs(v - 8.0) < 1e-9);
  }

  IterRowsCtx iter_ctx = {0};
  iter_ctx.id = apply_ctx.id;
  iter_ctx.sum = 0;
  CHECK(cp_df_iterrows(df, iterrows_sum, &iter_ctx, &err));
  CHECK(iter_ctx.sum == 6);

  IterItemsCtx items_ctx = {0};
  items_ctx.count = 0;
  CHECK(cp_df_iteritems(df, iteritems_count, &items_ctx, &err));
  CHECK(items_ctx.count == 2);

  if (applied) {
    cp_df_free(applied);
  }
  if (transformed) {
    cp_df_free(transformed);
  }
  cp_df_free(df);
}

static void test_astype_index_at(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "tag"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "2.0", "100"};
  const char *r2[] = {"2", "3.0", ""};
  const char *r3[] = {"3", "nan", "300"};
  CHECK(cp_df_append_row(df, r1, 3, &err));
  CHECK(cp_df_append_row(df, r2, 3, &err));
  CHECK(cp_df_append_row(df, r3, 3, &err));

  CpDataFrame *cast_tag = cp_df_astype(df, "tag", CP_DTYPE_INT64, &err);
  CHECK(cast_tag != NULL);
  if (cast_tag) {
    const CpSeries *tag = cp_df_get_col(cast_tag, "tag");
    CHECK(tag != NULL);
    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(tag, 0, &v, &is_null));
    CHECK(!is_null && v == 100);
    CHECK(cp_series_get_int64(tag, 1, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(tag, 2, &v, &is_null));
    CHECK(!is_null && v == 300);
  }

  CpDataFrame *cast_id = cp_df_astype(df, "id", CP_DTYPE_STRING, &err);
  CHECK(cast_id != NULL);
  if (cast_id) {
    const CpSeries *id = cp_df_get_col(cast_id, "id");
    CHECK(id != NULL);
    const char *val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(id, 2, &val, &is_null));
    CHECK(!is_null && strcmp(val, "3") == 0);
  }

  CpDataFrame *cast_score = cp_df_astype(df, "score", CP_DTYPE_STRING, &err);
  CHECK(cast_score != NULL);
  if (cast_score) {
    const CpSeries *score = cp_df_get_col(cast_score, "score");
    CHECK(score != NULL);
    const char *val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(score, 0, &val, &is_null));
    CHECK(!is_null && strcmp(val, "2") == 0);
    CHECK(cp_series_get_string(score, 2, &val, &is_null));
    CHECK(!is_null && strcmp(val, "nan") == 0);
  }

  CpDataFrame *indexed = cp_df_set_index(df, "id", &err);
  CHECK(indexed != NULL);
  if (indexed) {
    double score_val = 0.0;
    int is_null = 0;
    CHECK(cp_df_at_float64(indexed, "1", "score", &score_val, &is_null,
                           &err));
    CHECK(!is_null && fabs(score_val - 2.0) < 1e-9);

    const char *tag_val = NULL;
    CHECK(cp_df_at_string(indexed, "2", "tag", &tag_val, &is_null, &err));
    CHECK(is_null);

    cp_error_clear(&err);
    int64_t id_val = 0;
    CHECK(!cp_df_at_int64(indexed, "9", "id", &id_val, &is_null, &err));
    CHECK(err.code == CP_ERR_INVALID);
  }

  CpDataFrame *indexed_str = cp_df_set_index(df, "tag", &err);
  CHECK(indexed_str != NULL);
  if (indexed_str) {
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_df_at_int64(indexed_str, "300", "id", &id_val, &is_null, &err));
    CHECK(!is_null && id_val == 3);
  }

  CpDataFrame *reset = NULL;
  if (indexed) {
    reset = cp_df_reset_index(indexed, &err);
    CHECK(reset != NULL);
    if (reset) {
      int64_t id_val = 0;
      int is_null = 0;
      CHECK(cp_df_at_int64(reset, "0", "id", &id_val, &is_null, &err));
      CHECK(!is_null && id_val == 1);
    }
  }

  cp_error_clear(&err);
  CpDataFrame *bad_index = cp_df_set_index(df, "score", &err);
  CHECK(bad_index == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  CpDataFrame *df_bad = NULL;
  CpDataFrame *bad_cast = NULL;
  const char *f_names[] = {"score"};
  CpDType f_dtypes[] = {CP_DTYPE_FLOAT64};
  df_bad = cp_df_create(1, f_names, f_dtypes, 0, &err);
  CHECK(df_bad != NULL);
  if (df_bad) {
    const char *rb[] = {"1.5"};
    CHECK(cp_df_append_row(df_bad, rb, 1, &err));
    cp_error_clear(&err);
    bad_cast = cp_df_astype(df_bad, "score", CP_DTYPE_INT64, &err);
    CHECK(bad_cast == NULL);
    CHECK(err.code == CP_ERR_INVALID);
  }

  if (cast_tag) {
    cp_df_free(cast_tag);
  }
  if (cast_id) {
    cp_df_free(cast_id);
  }
  if (cast_score) {
    cp_df_free(cast_score);
  }
  if (indexed) {
    cp_df_free(indexed);
  }
  if (indexed_str) {
    cp_df_free(indexed_str);
  }
  if (reset) {
    cp_df_free(reset);
  }
  if (df_bad) {
    cp_df_free(df_bad);
  }
  if (bad_cast) {
    cp_df_free(bad_cast);
  }
  cp_df_free(df);
}

static void test_conversion_helpers(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"num", "dt"};
  CpDType dtypes[] = {CP_DTYPE_STRING, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "1970-01-01"};
  const char *r2[] = {"2.5", "1970-01-02 00:00:01+02:00"};
  const char *r3[] = {"", "1970/01/01T01:02:03Z"};
  const char *r4[] = {"nan", ""};
  CHECK(cp_df_append_row(df, r1, 2, &err));
  CHECK(cp_df_append_row(df, r2, 2, &err));
  CHECK(cp_df_append_row(df, r3, 2, &err));
  CHECK(cp_df_append_row(df, r4, 2, &err));

  CpDataFrame *numeric = cp_df_to_numeric(df, "num", &err);
  CHECK(numeric != NULL);
  if (numeric) {
    const CpSeries *num = cp_df_get_col(numeric, "num");
    CHECK(num != NULL);
    double v = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_float64(num, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
    CHECK(cp_series_get_float64(num, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 2.5) < 1e-9);
    CHECK(cp_series_get_float64(num, 2, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_float64(num, 3, &v, &is_null));
    CHECK(!is_null && isnan(v));
  }

  CpDataFrame *datetimes = cp_df_to_datetime(df, "dt", &err);
  CHECK(datetimes != NULL);
  if (datetimes) {
    const CpSeries *dt = cp_df_get_col(datetimes, "dt");
    CHECK(dt != NULL);
    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(dt, 0, &v, &is_null));
    CHECK(!is_null && v == 0);
    CHECK(cp_series_get_int64(dt, 1, &v, &is_null));
    CHECK(!is_null && v == 79201);
    CHECK(cp_series_get_int64(dt, 2, &v, &is_null));
    CHECK(!is_null && v == 3723);
    CHECK(cp_series_get_int64(dt, 3, &v, &is_null));
    CHECK(is_null);
  }

  uint8_t mask_a[8] = {0};
  uint8_t mask_b[8] = {0};
  CHECK(cp_df_isnull_mask(df, mask_a, 8, &err));
  CHECK(cp_df_isna_mask(df, mask_b, 8, &err));
  CHECK(memcmp(mask_a, mask_b, 8) == 0);

  char *repr = cp_df_to_string(df, &err);
  CHECK(repr != NULL);
  if (repr) {
    CHECK(strstr(repr, "num") != NULL);
    CHECK(strstr(repr, "dt") != NULL);
    CHECK(strstr(repr, "1970-01-01") != NULL);
    CHECK(strstr(repr, "null") != NULL);
  }
  free(repr);

  if (numeric) {
    cp_df_free(numeric);
  }
  if (datetimes) {
    cp_df_free(datetimes);
  }
  cp_df_free(df);

  cp_error_clear(&err);
  const char *bad_names[] = {"dt"};
  CpDType bad_dtypes[] = {CP_DTYPE_STRING};
  CpDataFrame *bad_df = cp_df_create(1, bad_names, bad_dtypes, 0, &err);
  CHECK(bad_df != NULL);
  if (bad_df) {
    const char *row[] = {"1970-02-30"};
    CHECK(cp_df_append_row(bad_df, row, 1, &err));
    CpDataFrame *bad = cp_df_to_datetime(bad_df, "dt", &err);
    CHECK(bad == NULL);
    CHECK(err.code == CP_ERR_PARSE);
    if (bad) {
      cp_df_free(bad);
    }
    cp_df_free(bad_df);
  }
}

static void test_stats_helpers(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64};
  CpDataFrame *df = cp_df_create(2, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"10", "1.0"};
  const char *r2[] = {"20", "2.0"};
  const char *r3[] = {"10", ""};
  const char *r4[] = {"30", "4.0"};
  const char *r5[] = {"", "5.0"};
  const char *r6[] = {"40", "nan"};
  CHECK(cp_df_append_row(df, r1, 2, &err));
  CHECK(cp_df_append_row(df, r2, 2, &err));
  CHECK(cp_df_append_row(df, r3, 2, &err));
  CHECK(cp_df_append_row(df, r4, 2, &err));
  CHECK(cp_df_append_row(df, r5, 2, &err));
  CHECK(cp_df_append_row(df, r6, 2, &err));

  double median = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  CHECK(cp_df_median(df, "id", &median, &count, &nulls, &err));
  CHECK(fabs(median - 20.0) < 1e-9);
  CHECK(count == 5);
  CHECK(nulls == 1);

  CHECK(cp_df_median_at(df, 1, &median, &count, &nulls, &err));
  CHECK(fabs(median - 3.0) < 1e-9);
  CHECK(count == 4);
  CHECK(nulls == 2);

  double std = 0.0;
  CHECK(cp_df_std(df, "score", &std, &count, &nulls, &err));
  CHECK(fabs(std - 1.825741858) < 1e-6);
  CHECK(count == 4);
  CHECK(nulls == 2);

  CHECK(cp_df_std_at(df, 1, &std, &count, &nulls, &err));
  CHECK(fabs(std - 1.825741858) < 1e-6);

  CpDataFrame *diff_id = cp_df_diff(df, "id", &err);
  CHECK(diff_id != NULL);
  if (diff_id) {
    const CpSeries *diff = cp_df_get_col(diff_id, "id");
    CHECK(diff != NULL);
    int64_t v = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(diff, 0, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(diff, 1, &v, &is_null));
    CHECK(!is_null && v == 10);
    CHECK(cp_series_get_int64(diff, 2, &v, &is_null));
    CHECK(!is_null && v == -10);
    CHECK(cp_series_get_int64(diff, 4, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_int64(diff, 5, &v, &is_null));
    CHECK(is_null);
  }

  CpDataFrame *diff_score = cp_df_diff(df, "score", &err);
  CHECK(diff_score != NULL);
  if (diff_score) {
    const CpSeries *diff = cp_df_get_col(diff_score, "score");
    CHECK(diff != NULL);
    double v = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_float64(diff, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
    CHECK(cp_series_get_float64(diff, 2, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_float64(diff, 4, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
    CHECK(cp_series_get_float64(diff, 5, &v, &is_null));
    CHECK(is_null);
  }

  CpDataFrame *rank = cp_df_rank(df, "id", &err);
  CHECK(rank != NULL);
  if (rank) {
    const CpSeries *r = cp_df_get_col(rank, "id");
    CHECK(r != NULL);
    double v = 0.0;
    int is_null = 0;
    CHECK(cp_series_get_float64(r, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.5) < 1e-9);
    CHECK(cp_series_get_float64(r, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 3.0) < 1e-9);
    CHECK(cp_series_get_float64(r, 2, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.5) < 1e-9);
    CHECK(cp_series_get_float64(r, 3, &v, &is_null));
    CHECK(!is_null && fabs(v - 4.0) < 1e-9);
    CHECK(cp_series_get_float64(r, 4, &v, &is_null));
    CHECK(is_null);
    CHECK(cp_series_get_float64(r, 5, &v, &is_null));
    CHECK(!is_null && fabs(v - 5.0) < 1e-9);
  }

  if (diff_id) {
    cp_df_free(diff_id);
  }
  if (diff_score) {
    cp_df_free(diff_score);
  }
  if (rank) {
    cp_df_free(rank);
  }
  cp_df_free(df);
}

static void test_corr_cov(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"a", "b", "label"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "2.0", "x"};
  const char *r2[] = {"2", "4.0", "y"};
  const char *r3[] = {"3", "6.0", "z"};
  const char *r4[] = {"4", "8.0", "w"};
  CHECK(cp_df_append_row(df, r1, 3, &err));
  CHECK(cp_df_append_row(df, r2, 3, &err));
  CHECK(cp_df_append_row(df, r3, 3, &err));
  CHECK(cp_df_append_row(df, r4, 3, &err));

  CpDataFrame *corr = cp_df_corr(df, &err);
  CHECK(corr != NULL);
  if (corr) {
    CHECK(cp_df_ncols(corr) == 3);
    CHECK(cp_df_nrows(corr) == 2);
    const CpSeries *header = cp_df_get_col(corr, "column");
    const CpSeries *col_a = cp_df_get_col(corr, "a");
    const CpSeries *col_b = cp_df_get_col(corr, "b");
    CHECK(header && col_a && col_b);

    const char *name_val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(header, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "a") == 0);
    CHECK(cp_series_get_string(header, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "b") == 0);

    double v = 0.0;
    CHECK(cp_series_get_float64(col_a, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
    CHECK(cp_series_get_float64(col_b, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
    CHECK(cp_series_get_float64(col_a, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
    CHECK(cp_series_get_float64(col_b, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.0) < 1e-9);
  }

  CpDataFrame *cov = cp_df_cov(df, &err);
  CHECK(cov != NULL);
  if (cov) {
    CHECK(cp_df_ncols(cov) == 3);
    CHECK(cp_df_nrows(cov) == 2);
    const CpSeries *header = cp_df_get_col(cov, "column");
    const CpSeries *col_a = cp_df_get_col(cov, "a");
    const CpSeries *col_b = cp_df_get_col(cov, "b");
    CHECK(header && col_a && col_b);

    const char *name_val = NULL;
    int is_null = 0;
    CHECK(cp_series_get_string(header, 0, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "a") == 0);
    CHECK(cp_series_get_string(header, 1, &name_val, &is_null));
    CHECK(!is_null && strcmp(name_val, "b") == 0);

    double v = 0.0;
    CHECK(cp_series_get_float64(col_a, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 1.6666666667) < 1e-6);
    CHECK(cp_series_get_float64(col_b, 0, &v, &is_null));
    CHECK(!is_null && fabs(v - 3.3333333333) < 1e-6);
    CHECK(cp_series_get_float64(col_a, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 3.3333333333) < 1e-6);
    CHECK(cp_series_get_float64(col_b, 1, &v, &is_null));
    CHECK(!is_null && fabs(v - 6.6666666667) < 1e-6);
  }

  if (corr) {
    cp_df_free(corr);
  }
  if (cov) {
    cp_df_free(cov);
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

  CpDataFrame *inner =
      cp_df_join_with_strategy(left,
                               right,
                               "id",
                               "id",
                               CP_JOIN_INNER,
                               CP_JOIN_STRATEGY_HASH,
                               &err);
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

static void test_join_hash_path(void) {
  CpError err;
  cp_error_clear(&err);

  const char *left_names[] = {"id", "left_val"};
  CpDType left_types[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *left = cp_df_create(2, left_names, left_types, 0, &err);
  CHECK(left != NULL);
  if (!left) {
    return;
  }

  const char *right_names[] = {"id", "right_val"};
  CpDType right_types[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *right = cp_df_create(2, right_names, right_types, 0, &err);
  CHECK(right != NULL);
  if (!right) {
    cp_df_free(left);
    return;
  }

  for (int i = 0; i < 50; ++i) {
    char id_buf[32];
    char val_buf[32];
    snprintf(id_buf, sizeof(id_buf), "%d", i);
    snprintf(val_buf, sizeof(val_buf), "%d", i * 10);
    const char *row[] = {id_buf, val_buf};
    CHECK(cp_df_append_row(left, row, 2, &err));
  }

  for (int i = 25; i < 75; ++i) {
    char id_buf[32];
    char val_buf[32];
    snprintf(id_buf, sizeof(id_buf), "%d", i);
    snprintf(val_buf, sizeof(val_buf), "%d", i * 100);
    const char *row[] = {id_buf, val_buf};
    CHECK(cp_df_append_row(right, row, 2, &err));
  }

  CpDataFrame *inner = cp_df_join(left, right, "id", "id", CP_JOIN_INNER, &err);
  CHECK(inner != NULL);
  if (inner) {
    CHECK(cp_df_nrows(inner) == 25);
    CHECK(cp_df_ncols(inner) == 3);

    const CpSeries *id = cp_df_get_col(inner, "id");
    const CpSeries *left_val = cp_df_get_col(inner, "left_val");
    const CpSeries *right_val = cp_df_get_col(inner, "right_val");
    CHECK(id && left_val && right_val);

    int is_null = 0;
    int64_t id_val = 0;
    int64_t left_v = 0;
    int64_t right_v = 0;

    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 25);
    CHECK(cp_series_get_int64(left_val, 0, &left_v, &is_null));
    CHECK(!is_null && left_v == 250);
    CHECK(cp_series_get_int64(right_val, 0, &right_v, &is_null));
    CHECK(!is_null && right_v == 2500);

    CHECK(cp_series_get_int64(id, 24, &id_val, &is_null));
    CHECK(!is_null && id_val == 49);
    CHECK(cp_series_get_int64(left_val, 24, &left_v, &is_null));
    CHECK(!is_null && left_v == 490);
    CHECK(cp_series_get_int64(right_val, 24, &right_v, &is_null));
    CHECK(!is_null && right_v == 4900);
  }

  if (inner) {
    cp_df_free(inner);
  }
  cp_df_free(right);
  cp_df_free(left);
}

static void assert_join_strategy_result(CpDataFrame *df) {
  CHECK(df != NULL);
  if (!df) {
    return;
  }
  CHECK(cp_df_nrows(df) == 4);
  CHECK(cp_df_ncols(df) == 3);

  const CpSeries *id = cp_df_get_col(df, "id");
  const CpSeries *left_val = cp_df_get_col(df, "left_val");
  const CpSeries *right_val = cp_df_get_col(df, "right_val");
  CHECK(id && left_val && right_val);

  int64_t exp_id[] = {2, 2, 2, 2};
  int64_t exp_left[] = {20, 20, 21, 21};
  int64_t exp_right[] = {200, 201, 200, 201};

  for (size_t i = 0; i < 4; ++i) {
    int is_null = 0;
    int64_t value = 0;
    CHECK(cp_series_get_int64(id, i, &value, &is_null));
    CHECK(!is_null && value == exp_id[i]);
    CHECK(cp_series_get_int64(left_val, i, &value, &is_null));
    CHECK(!is_null && value == exp_left[i]);
    CHECK(cp_series_get_int64(right_val, i, &value, &is_null));
    CHECK(!is_null && value == exp_right[i]);
  }
}

static void test_join_strategy_forced(void) {
  CpError err;
  cp_error_clear(&err);

  const char *left_names[] = {"id", "left_val"};
  CpDType left_types[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *left = cp_df_create(2, left_names, left_types, 0, &err);
  CHECK(left != NULL);
  if (!left) {
    return;
  }

  const char *right_names[] = {"id", "right_val"};
  CpDType right_types[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};
  CpDataFrame *right = cp_df_create(2, right_names, right_types, 0, &err);
  CHECK(right != NULL);
  if (!right) {
    cp_df_free(left);
    return;
  }

  const char *l0[] = {"1", "10"};
  const char *l1[] = {"2", "20"};
  const char *l2[] = {"2", "21"};
  CHECK(cp_df_append_row(left, l0, 2, &err));
  CHECK(cp_df_append_row(left, l1, 2, &err));
  CHECK(cp_df_append_row(left, l2, 2, &err));

  const char *r0[] = {"2", "200"};
  const char *r1[] = {"2", "201"};
  const char *r2[] = {"3", "300"};
  CHECK(cp_df_append_row(right, r0, 2, &err));
  CHECK(cp_df_append_row(right, r1, 2, &err));
  CHECK(cp_df_append_row(right, r2, 2, &err));

  CpDataFrame *nested =
      cp_df_join_with_strategy(left,
                               right,
                               "id",
                               "id",
                               CP_JOIN_INNER,
                               CP_JOIN_STRATEGY_NESTED,
                               &err);
  assert_join_strategy_result(nested);

  CpDataFrame *sorted =
      cp_df_join_with_strategy(left,
                               right,
                               "id",
                               "id",
                               CP_JOIN_INNER,
                               CP_JOIN_STRATEGY_SORTED,
                               &err);
  assert_join_strategy_result(sorted);

  CpDataFrame *hash =
      cp_df_join_with_strategy(left,
                               right,
                               "id",
                               "id",
                               CP_JOIN_INNER,
                               CP_JOIN_STRATEGY_HASH,
                               &err);
  assert_join_strategy_result(hash);

  if (nested) {
    cp_df_free(nested);
  }
  if (sorted) {
    cp_df_free(sorted);
  }
  if (hash) {
    cp_df_free(hash);
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

static void test_query(void) {
  CpError err;
  cp_error_clear(&err);

  const char *names[] = {"id", "score", "name"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpDataFrame *df = cp_df_create(3, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  const char *r1[] = {"1", "2.5", "Alice"};
  const char *r2[] = {"2", "4.0", "Bob"};
  const char *r3[] = {"3", "", "Bob"};
  const char *r4[] = {"4", "1.0", ""};
  const char *r5[] = {"5", "nan", "Cara"};
  CHECK(cp_df_append_row(df, r1, 3, &err));
  CHECK(cp_df_append_row(df, r2, 3, &err));
  CHECK(cp_df_append_row(df, r3, 3, &err));
  CHECK(cp_df_append_row(df, r4, 3, &err));
  CHECK(cp_df_append_row(df, r5, 3, &err));

  CpDataFrame *q1 = cp_df_query(df, "id >= 3", &err);
  CHECK(q1 != NULL);
  if (q1) {
    CHECK(cp_df_nrows(q1) == 3);
    const CpSeries *id = cp_df_get_col(q1, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 3);
    CHECK(cp_series_get_int64(id, 2, &id_val, &is_null));
    CHECK(!is_null && id_val == 5);
  }

  CpDataFrame *q2 = cp_df_query(df, "name == \"Bob\"", &err);
  CHECK(q2 != NULL);
  if (q2) {
    CHECK(cp_df_nrows(q2) == 2);
    const CpSeries *id = cp_df_get_col(q2, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
  }

  CpDataFrame *q3 = cp_df_query(df, "score < 3.0", &err);
  CHECK(q3 != NULL);
  if (q3) {
    CHECK(cp_df_nrows(q3) == 2);
    const CpSeries *id = cp_df_get_col(q3, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
  }

  CpDataFrame *q4 = cp_df_query(df, "name == null", &err);
  CHECK(q4 != NULL);
  if (q4) {
    CHECK(cp_df_nrows(q4) == 1);
    const CpSeries *id = cp_df_get_col(q4, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
  }

  CpDataFrame *q5 = cp_df_query(df, "score == nan", &err);
  CHECK(q5 != NULL);
  if (q5) {
    CHECK(cp_df_nrows(q5) == 1);
    const CpSeries *id = cp_df_get_col(q5, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 5);
  }

  CpDataFrame *q6 = cp_df_query(df, "name == \"Bob\" AND score >= 4.0", &err);
  CHECK(q6 != NULL);
  if (q6) {
    CHECK(cp_df_nrows(q6) == 1);
    const CpSeries *id = cp_df_get_col(q6, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
  }

  CpDataFrame *q7 = cp_df_query(df, "id == 1 or id == 4", &err);
  CHECK(q7 != NULL);
  if (q7) {
    CHECK(cp_df_nrows(q7) == 2);
    const CpSeries *id = cp_df_get_col(q7, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 4);
  }

  CpDataFrame *q8 =
      cp_df_query(df, "id == 1 or id == 2 and score >= 3.0", &err);
  CHECK(q8 != NULL);
  if (q8) {
    CHECK(cp_df_nrows(q8) == 2);
    const CpSeries *id = cp_df_get_col(q8, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 1);
    CHECK(cp_series_get_int64(id, 1, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
  }

  CpDataFrame *q9 =
      cp_df_query(df, "(id == 1 or id == 2) and score >= 3.0", &err);
  CHECK(q9 != NULL);
  if (q9) {
    CHECK(cp_df_nrows(q9) == 1);
    const CpSeries *id = cp_df_get_col(q9, "id");
    CHECK(id != NULL);
    int64_t id_val = 0;
    int is_null = 0;
    CHECK(cp_series_get_int64(id, 0, &id_val, &is_null));
    CHECK(!is_null && id_val == 2);
  }

  cp_error_clear(&err);
  CpDataFrame *bad = cp_df_query(df, "id ?? 3", &err);
  CHECK(bad == NULL);
  CHECK(err.code == CP_ERR_INVALID);

  if (q1) {
    cp_df_free(q1);
  }
  if (q2) {
    cp_df_free(q2);
  }
  if (q3) {
    cp_df_free(q3);
  }
  if (q4) {
    cp_df_free(q4);
  }
  if (q5) {
    cp_df_free(q5);
  }
  if (q6) {
    cp_df_free(q6);
  }
  if (q7) {
    cp_df_free(q7);
  }
  if (q8) {
    cp_df_free(q8);
  }
  if (q9) {
    cp_df_free(q9);
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
  test_metadata_helpers();
  test_dtypes_and_rename_drop_fill();
  test_unique_counts_int64();
  test_unique_counts_string();
  test_sample();
  test_nlargest_nsmallest();
  test_where_mask_clip_replace();
  test_concat();
  test_apply_transform_iter();
  test_astype_index_at();
  test_conversion_helpers();
  test_stats_helpers();
  test_corr_cov();
  test_isnull_dropna();
  test_info_describe();
  test_loc_iloc();
  test_groupby_agg();
  test_join_inner_left();
  test_join_multi_key();
  test_join_hash_path();
  test_join_strategy_forced();
  test_pivot_table();
  test_predicate_filters();
  test_query();

  if (tests_failed != 0) {
    fprintf(stderr, "%d test(s) failed\n", tests_failed);
  }
  return tests_failed == 0 ? 0 : 1;
}
