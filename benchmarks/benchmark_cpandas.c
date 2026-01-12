#include "cpandas.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double elapsed_seconds(clock_t start, clock_t end) {
  return (double)(end - start) / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
  size_t rows = 200000;
  if (argc > 1) {
    char *end = NULL;
    unsigned long long parsed = strtoull(argv[1], &end, 10);
    if (end && *end == '\0' && parsed > 0) {
      rows = (size_t)parsed;
    }
  }

  const char *names[] = {"id", "value", "label"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpError err;
  cp_error_clear(&err);

  CpDataFrame *df = cp_df_create(3, names, dtypes, rows, &err);
  if (!df) {
    fprintf(stderr, "failed to create dataframe: %s\n", err.message);
    return 1;
  }

  clock_t start = clock();
  for (size_t i = 0; i < rows; ++i) {
    char id_buf[32];
    char val_buf[64];
    snprintf(id_buf, sizeof(id_buf), "%zu", i + 1);
    snprintf(val_buf, sizeof(val_buf), "%.3f", (double)i * 0.5);
    const char *row[] = {id_buf, val_buf, "alpha"};
    if (!cp_df_append_row(df, row, 3, &err)) {
      fprintf(stderr, "append failed at row %zu: %s\n", i, err.message);
      cp_df_free(df);
      return 1;
    }
  }
  clock_t end = clock();
  double append_s = elapsed_seconds(start, end);

  const CpSeries *id_series = cp_df_get_col(df, "id");
  const CpSeries *val_series = cp_df_get_col(df, "value");

  int64_t isum = 0;
  double fsum = 0.0;
  size_t count = 0;
  size_t nulls = 0;

  start = clock();
  if (!cp_series_sum_int64(id_series, &isum, &count, &nulls, &err)) {
    fprintf(stderr, "sum int failed: %s\n", err.message);
  }
  if (!cp_series_sum_float64(val_series, &fsum, &count, &nulls, &err)) {
    fprintf(stderr, "sum float failed: %s\n", err.message);
  }
  end = clock();
  double sum_s = elapsed_seconds(start, end);

  printf("rows: %zu\n", rows);
  printf("append: %.4fs (%.0f rows/s)\n", append_s,
         append_s > 0.0 ? (double)rows / append_s : 0.0);
  printf("sum: %.6fs\n", sum_s);
  printf("checksum int: %" PRId64 ", float: %.3f\n", isum, fsum);

  cp_df_free(df);
  return 0;
}
