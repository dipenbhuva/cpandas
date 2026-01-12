#include "cpandas.h"

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

static uint32_t rng_state = 0x12345678u;

static uint32_t rng_next(void) {
  uint32_t x = rng_state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rng_state = x;
  return x;
}

static int rng_range(int min_val, int max_val) {
  uint32_t span = (uint32_t)(max_val - min_val + 1);
  return (int)(min_val + (rng_next() % span));
}

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
  char *tmpl = dup_string("cpandas_fuzz_XXXXXX");
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

static char *make_random_field(int *is_null) {
  if (rng_range(0, 9) == 0) {
    if (is_null) {
      *is_null = 1;
    }
    return NULL;
  }

  int len = rng_range(1, 24);
  char *buf = (char *)malloc((size_t)len + 1);
  if (!buf) {
    return NULL;
  }

  const char *alphabet =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,\"";
  size_t alpha_len = strlen(alphabet);

  int has_non_space = 0;
  for (int i = 0; i < len; ++i) {
    char ch = alphabet[rng_range(0, (int)alpha_len - 1)];
    buf[i] = ch;
    if (ch != ' ') {
      has_non_space = 1;
    }
  }
  if (!has_non_space) {
    buf[0] = 'A';
  }
  buf[len] = '\0';

  if (is_null) {
    *is_null = 0;
  }
  return buf;
}

static void free_expected(char ***values, int **nulls, size_t rows, size_t cols) {
  if (!values || !nulls) {
    return;
  }
  for (size_t r = 0; r < rows; ++r) {
    if (values[r]) {
      for (size_t c = 0; c < cols; ++c) {
        free(values[r][c]);
      }
      free(values[r]);
    }
    free(nulls[r]);
  }
  free(values);
  free(nulls);
}

static void test_csv_roundtrip_fuzz(void) {
  const size_t rows = 120;
  const size_t cols = 6;

  const char *names[] = {"c0", "c1", "c2", "c3", "c4", "c5"};
  CpDType dtypes[] = {CP_DTYPE_STRING, CP_DTYPE_STRING, CP_DTYPE_STRING,
                      CP_DTYPE_STRING, CP_DTYPE_STRING, CP_DTYPE_STRING};

  CpError err;
  cp_error_clear(&err);

  CpDataFrame *df = cp_df_create(cols, names, dtypes, 0, &err);
  CHECK(df != NULL);
  if (!df) {
    return;
  }

  char ***expected = (char ***)calloc(rows, sizeof(char **));
  int **expected_nulls = (int **)calloc(rows, sizeof(int *));
  CHECK(expected != NULL && expected_nulls != NULL);
  if (!expected || !expected_nulls) {
    cp_df_free(df);
    free(expected);
    free(expected_nulls);
    return;
  }

  for (size_t r = 0; r < rows; ++r) {
    expected[r] = (char **)calloc(cols, sizeof(char *));
    expected_nulls[r] = (int *)calloc(cols, sizeof(int));
    CHECK(expected[r] != NULL && expected_nulls[r] != NULL);
    if (!expected[r] || !expected_nulls[r]) {
      free_expected(expected, expected_nulls, rows, cols);
      cp_df_free(df);
      return;
    }

    const char *row_values[6] = {0};
    for (size_t c = 0; c < cols; ++c) {
      int is_null = 0;
      char *value = make_random_field(&is_null);
      if (is_null) {
        expected_nulls[r][c] = 1;
        row_values[c] = NULL;
      } else {
        expected_nulls[r][c] = 0;
        expected[r][c] = value;
        row_values[c] = value;
      }
    }

    CHECK(cp_df_append_row(df, row_values, cols, &err));
    if (err.code != CP_OK) {
      fprintf(stderr, "append error: %s\n", err.message);
      break;
    }
  }

  char *path = make_temp_path();
  CHECK(path != NULL);
  if (!path) {
    free_expected(expected, expected_nulls, rows, cols);
    cp_df_free(df);
    return;
  }

  CHECK(cp_df_write_csv(df, path, ',', 1, &err));

  CpDataFrame *df2 = cp_df_read_csv(path, ',', 1, dtypes, cols, &err);
  CHECK(df2 != NULL);
  if (df2) {
    CHECK(cp_df_nrows(df2) == rows);
    for (size_t c = 0; c < cols; ++c) {
      const CpSeries *series = cp_df_get_col(df2, names[c]);
      CHECK(series != NULL);
      if (!series) {
        continue;
      }
      for (size_t r = 0; r < rows; ++r) {
        const char *value = NULL;
        int is_null = 0;
        CHECK(cp_series_get_string(series, r, &value, &is_null));
        CHECK(is_null == expected_nulls[r][c]);
        if (!is_null) {
          CHECK(value != NULL);
          if (value) {
            CHECK(strcmp(value, expected[r][c]) == 0);
          }
        }
      }
    }
  }

  if (df2) {
    cp_df_free(df2);
  }
  cp_df_free(df);
  free_expected(expected, expected_nulls, rows, cols);
  remove(path);
  free(path);
}

int main(void) {
  test_csv_roundtrip_fuzz();
  if (tests_failed != 0) {
    fprintf(stderr, "%d test(s) failed\n", tests_failed);
  }
  return tests_failed == 0 ? 0 : 1;
}
