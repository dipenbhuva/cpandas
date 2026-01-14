#include "cpandas.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_seconds(clock_t start, clock_t end) {
  return (double)(end - start) / (double)CLOCKS_PER_SEC;
}

static const char *strategy_name(CpJoinStrategy strategy) {
  switch (strategy) {
    case CP_JOIN_STRATEGY_AUTO:
      return "auto";
    case CP_JOIN_STRATEGY_NESTED:
      return "nested";
    case CP_JOIN_STRATEGY_HASH:
      return "hash";
    case CP_JOIN_STRATEGY_SORTED:
      return "sorted";
    default:
      return "unknown";
  }
}

static int parse_join_strategy(const char *value,
                               CpJoinStrategy *out_strategy,
                               int *out_all) {
  if (!value || !out_strategy || !out_all) {
    return 0;
  }
  *out_all = 0;
  if (strcmp(value, "auto") == 0) {
    *out_strategy = CP_JOIN_STRATEGY_AUTO;
    return 1;
  }
  if (strcmp(value, "nested") == 0) {
    *out_strategy = CP_JOIN_STRATEGY_NESTED;
    return 1;
  }
  if (strcmp(value, "hash") == 0) {
    *out_strategy = CP_JOIN_STRATEGY_HASH;
    return 1;
  }
  if (strcmp(value, "sorted") == 0) {
    *out_strategy = CP_JOIN_STRATEGY_SORTED;
    return 1;
  }
  if (strcmp(value, "all") == 0) {
    *out_strategy = CP_JOIN_STRATEGY_AUTO;
    *out_all = 1;
    return 1;
  }
  return 0;
}

static void print_usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [rows] [--join] [--strategy auto|nested|hash|sorted|all] "
          "[--match-rate 0-1]\n",
          prog);
}

int main(int argc, char **argv) {
  size_t rows = 200000;
  int run_join = 0;
  CpJoinStrategy join_strategy = CP_JOIN_STRATEGY_AUTO;
  int join_all = 0;
  double match_rate = 1.0;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--join") == 0) {
      run_join = 1;
      continue;
    }
    if (strcmp(arg, "--strategy") == 0 || strcmp(arg, "--join-strategy") == 0) {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return 1;
      }
      int parsed_all = 0;
      if (!parse_join_strategy(argv[i + 1], &join_strategy, &parsed_all)) {
        print_usage(argv[0]);
        return 1;
      }
      join_all = parsed_all;
      run_join = 1;
      i += 1;
      continue;
    }
    if (strcmp(arg, "--match-rate") == 0) {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return 1;
      }
      char *end = NULL;
      double parsed = strtod(argv[i + 1], &end);
      if (!end || *end != '\0' || parsed < 0.0 || parsed > 1.0) {
        print_usage(argv[0]);
        return 1;
      }
      match_rate = parsed;
      run_join = 1;
      i += 1;
      continue;
    }
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
    if (arg[0] == '-') {
      print_usage(argv[0]);
      return 1;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(arg, &end, 10);
    if (!end || *end != '\0' || parsed == 0) {
      print_usage(argv[0]);
      return 1;
    }
    rows = (size_t)parsed;
  }

  const char *names[] = {"id", "value", "label"};
  CpDType dtypes[] = {CP_DTYPE_INT64, CP_DTYPE_FLOAT64, CP_DTYPE_STRING};
  CpError err;
  cp_error_clear(&err);

  if (!run_join) {
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

  size_t join_rows = rows;
  if ((join_all || join_strategy == CP_JOIN_STRATEGY_NESTED) &&
      join_rows > 20000) {
    join_rows = 20000;
    printf("join rows capped at %zu for nested strategy\n", join_rows);
  }
  size_t match_count = (size_t)((double)join_rows * match_rate);
  if (match_count > join_rows) {
    match_count = join_rows;
  }

  const char *join_left_names[] = {"id", "left_val"};
  const char *join_right_names[] = {"id", "right_val"};
  CpDType join_types[] = {CP_DTYPE_INT64, CP_DTYPE_INT64};

  CpDataFrame *left = cp_df_create(2, join_left_names, join_types, join_rows, &err);
  if (!left) {
    fprintf(stderr, "failed to create left join dataframe: %s\n", err.message);
    return 1;
  }
  CpDataFrame *right = cp_df_create(2, join_right_names, join_types, join_rows, &err);
  if (!right) {
    fprintf(stderr, "failed to create right join dataframe: %s\n", err.message);
    cp_df_free(left);
    return 1;
  }

  for (size_t i = 0; i < join_rows; ++i) {
    char id_buf[32];
    char val_buf[32];
    snprintf(id_buf, sizeof(id_buf), "%zu", i);
    snprintf(val_buf, sizeof(val_buf), "%zu", i * 2);
    const char *row[] = {id_buf, val_buf};
    if (!cp_df_append_row(left, row, 2, &err)) {
      fprintf(stderr, "left append failed at row %zu: %s\n", i, err.message);
      cp_df_free(left);
      cp_df_free(right);
      return 1;
    }
  }

  for (size_t i = 0; i < join_rows; ++i) {
    char id_buf[32];
    char val_buf[32];
    size_t key = i < match_count ? i : join_rows + (i - match_count);
    snprintf(id_buf, sizeof(id_buf), "%zu", key);
    snprintf(val_buf, sizeof(val_buf), "%zu", i * 3);
    const char *row[] = {id_buf, val_buf};
    if (!cp_df_append_row(right, row, 2, &err)) {
      fprintf(stderr, "right append failed at row %zu: %s\n", i, err.message);
      cp_df_free(left);
      cp_df_free(right);
      return 1;
    }
  }

  printf("join rows: %zu\n", join_rows);
  printf("match rate: %.2f (matches: %zu)\n", match_rate, match_count);

  CpJoinStrategy strategies[] = {CP_JOIN_STRATEGY_NESTED,
                                 CP_JOIN_STRATEGY_SORTED,
                                 CP_JOIN_STRATEGY_HASH,
                                 CP_JOIN_STRATEGY_AUTO};
  size_t strategy_count = join_all ? 4 : 1;

  for (size_t i = 0; i < strategy_count; ++i) {
    CpJoinStrategy strategy = join_all ? strategies[i] : join_strategy;
    clock_t start = clock();
    CpDataFrame *joined =
        cp_df_join_with_strategy(left,
                                 right,
                                 "id",
                                 "id",
                                 CP_JOIN_INNER,
                                 strategy,
                                 &err);
    clock_t end = clock();
    double join_s = elapsed_seconds(start, end);
    if (!joined) {
      fprintf(stderr, "join failed (%s): %s\n", strategy_name(strategy), err.message);
      continue;
    }
    size_t out_rows = cp_df_nrows(joined);
    printf("join %s: %.4fs (%.0f rows/s, out %zu)\n",
           strategy_name(strategy),
           join_s,
           join_s > 0.0 ? (double)join_rows / join_s : 0.0,
           out_rows);
    cp_df_free(joined);
  }

  cp_df_free(left);
  cp_df_free(right);
  return 0;
}
