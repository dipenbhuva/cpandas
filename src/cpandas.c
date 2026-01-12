#include "cpandas.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct CpSeries {
  char *name;
  CpDType dtype;
  size_t length;
  size_t capacity;
  unsigned char *is_null;
  union {
    int64_t *i64;
    double *f64;
    char **str;
  } data;
};

struct CpDataFrame {
  size_t ncols;
  size_t nrows;
  CpSeries **cols;
};

CpDataFrame *cp_df_filter_mask(const CpDataFrame *df,
                               const uint8_t *mask,
                               size_t mask_len,
                               CpError *err);

static const char *cp_dtype_name(CpDType dtype) {
  switch (dtype) {
    case CP_DTYPE_INT64:
      return "int64";
    case CP_DTYPE_FLOAT64:
      return "float64";
    case CP_DTYPE_STRING:
      return "string";
    default:
      return "unknown";
  }
}

static const char *cp_agg_op_name(CpAggOp op) {
  switch (op) {
    case CP_AGG_COUNT:
      return "count";
    case CP_AGG_SUM:
      return "sum";
    case CP_AGG_MEAN:
      return "mean";
    case CP_AGG_MIN:
      return "min";
    case CP_AGG_MAX:
      return "max";
    default:
      return "unknown";
  }
}

static void cp_error_set(CpError *err,
                         CpErrCode code,
                         size_t row,
                         size_t col,
                         const char *fmt,
                         ...);

static int cp_eval_compare_int64(int64_t lhs,
                                 CpCompareOp op,
                                 int64_t rhs,
                                 int *out,
                                 CpError *err) {
  if (!out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid compare output");
    return 0;
  }
  switch (op) {
    case CP_OP_EQ:
      *out = (lhs == rhs);
      return 1;
    case CP_OP_NE:
      *out = (lhs != rhs);
      return 1;
    case CP_OP_LT:
      *out = (lhs < rhs);
      return 1;
    case CP_OP_LE:
      *out = (lhs <= rhs);
      return 1;
    case CP_OP_GT:
      *out = (lhs > rhs);
      return 1;
    case CP_OP_GE:
      *out = (lhs >= rhs);
      return 1;
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid compare op");
      return 0;
  }
}

static int cp_eval_compare_float64(double lhs,
                                   CpCompareOp op,
                                   double rhs,
                                   int *out,
                                   CpError *err) {
  if (!out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid compare output");
    return 0;
  }
  if (isnan(lhs) || isnan(rhs)) {
    *out = 0;
    return 1;
  }
  switch (op) {
    case CP_OP_EQ:
      *out = (lhs == rhs);
      return 1;
    case CP_OP_NE:
      *out = (lhs != rhs);
      return 1;
    case CP_OP_LT:
      *out = (lhs < rhs);
      return 1;
    case CP_OP_LE:
      *out = (lhs <= rhs);
      return 1;
    case CP_OP_GT:
      *out = (lhs > rhs);
      return 1;
    case CP_OP_GE:
      *out = (lhs >= rhs);
      return 1;
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid compare op");
      return 0;
  }
}

static int cp_eval_compare_string(const char *lhs,
                                  CpCompareOp op,
                                  const char *rhs,
                                  int *out,
                                  CpError *err) {
  if (!out || !lhs || !rhs) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid compare input");
    return 0;
  }
  int cmp = strcmp(lhs, rhs);
  switch (op) {
    case CP_OP_EQ:
      *out = (cmp == 0);
      return 1;
    case CP_OP_NE:
      *out = (cmp != 0);
      return 1;
    case CP_OP_LT:
      *out = (cmp < 0);
      return 1;
    case CP_OP_LE:
      *out = (cmp <= 0);
      return 1;
    case CP_OP_GT:
      *out = (cmp > 0);
      return 1;
    case CP_OP_GE:
      *out = (cmp >= 0);
      return 1;
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid compare op");
      return 0;
  }
}

typedef struct {
  size_t count;
  int has_value;
  int64_t sum_i64;
  int64_t min_i64;
  int64_t max_i64;
  double sum_f64;
  double min_f64;
  double max_f64;
} CpAggState;

typedef struct {
  const CpSeries *series;
  CpAggOp op;
  CpDType out_dtype;
  char *name;
} CpAggSpec;

static void cp_error_set(CpError *err,
                         CpErrCode code,
                         size_t row,
                         size_t col,
                         const char *fmt,
                         ...) {
  if (!err) {
    return;
  }
  err->code = code;
  err->row = row;
  err->col = col;
  if (fmt && fmt[0] != '\0') {
    va_list args;
    va_start(args, fmt);
    vsnprintf(err->message, sizeof(err->message), fmt, args);
    va_end(args);
  } else {
    err->message[0] = '\0';
  }
}

void cp_error_clear(CpError *err) {
  if (!err) {
    return;
  }
  err->code = CP_OK;
  err->row = 0;
  err->col = 0;
  err->message[0] = '\0';
}

static char *cp_strndup(const char *s, size_t len) {
  char *out = (char *)malloc(len + 1);
  if (!out) {
    return NULL;
  }
  if (len > 0) {
    memcpy(out, s, len);
  }
  out[len] = '\0';
  return out;
}

static char *cp_strdup(const char *s) {
  if (!s) {
    return NULL;
  }
  return cp_strndup(s, strlen(s));
}

static int cp_series_reserve(CpSeries *s, size_t needed, CpError *err) {
  if (needed <= s->capacity) {
    return 1;
  }
  size_t new_cap = s->capacity == 0 ? 16 : s->capacity;
  while (new_cap < needed) {
    new_cap *= 2;
  }

  unsigned char *new_nulls =
      (unsigned char *)realloc(s->is_null, new_cap * sizeof(unsigned char));
  if (!new_nulls) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return 0;
  }
  if (new_cap > s->capacity) {
    memset(new_nulls + s->capacity, 0, new_cap - s->capacity);
  }
  s->is_null = new_nulls;

  switch (s->dtype) {
    case CP_DTYPE_INT64: {
      int64_t *new_data =
          (int64_t *)realloc(s->data.i64, new_cap * sizeof(int64_t));
      if (!new_data) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        return 0;
      }
      s->data.i64 = new_data;
      break;
    }
    case CP_DTYPE_FLOAT64: {
      double *new_data =
          (double *)realloc(s->data.f64, new_cap * sizeof(double));
      if (!new_data) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        return 0;
      }
      s->data.f64 = new_data;
      break;
    }
    case CP_DTYPE_STRING: {
      char **new_data = (char **)realloc(s->data.str, new_cap * sizeof(char *));
      if (!new_data) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        return 0;
      }
      if (new_cap > s->capacity) {
        memset(new_data + s->capacity, 0,
               (new_cap - s->capacity) * sizeof(char *));
      }
      s->data.str = new_data;
      break;
    }
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unknown dtype");
      return 0;
  }

  s->capacity = new_cap;
  return 1;
}

static void cp_series_free(CpSeries *s) {
  if (!s) {
    return;
  }
  free(s->name);
  if (s->dtype == CP_DTYPE_STRING && s->data.str) {
    for (size_t i = 0; i < s->length; ++i) {
      free(s->data.str[i]);
    }
  }
  free(s->data.str);
  free(s->is_null);
  free(s);
}

static CpSeries *cp_series_create(const char *name,
                                  CpDType dtype,
                                  size_t capacity,
                                  CpError *err) {
  CpSeries *s = (CpSeries *)calloc(1, sizeof(CpSeries));
  if (!s) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  s->name = cp_strdup(name ? name : "");
  if (!s->name) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    cp_series_free(s);
    return NULL;
  }
  s->dtype = dtype;
  s->length = 0;
  s->capacity = 0;
  s->is_null = NULL;
  s->data.str = NULL;

  if (capacity > 0) {
    if (!cp_series_reserve(s, capacity, err)) {
      cp_series_free(s);
      return NULL;
    }
  }

  return s;
}

static int cp_series_append_int64(CpSeries *s,
                                  int64_t value,
                                  int is_null,
                                  CpError *err) {
  if (!s || s->dtype != CP_DTYPE_INT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  if (!cp_series_reserve(s, s->length + 1, err)) {
    return 0;
  }
  s->data.i64[s->length] = value;
  s->is_null[s->length] = is_null ? 1 : 0;
  s->length += 1;
  return 1;
}

static int cp_series_append_float64(CpSeries *s,
                                    double value,
                                    int is_null,
                                    CpError *err) {
  if (!s || s->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  if (!cp_series_reserve(s, s->length + 1, err)) {
    return 0;
  }
  s->data.f64[s->length] = value;
  s->is_null[s->length] = is_null ? 1 : 0;
  s->length += 1;
  return 1;
}

static int cp_series_append_string(CpSeries *s,
                                   const char *value,
                                   int is_null,
                                   CpError *err) {
  if (!s || s->dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  if (!cp_series_reserve(s, s->length + 1, err)) {
    return 0;
  }
  s->data.str[s->length] = NULL;
  if (!is_null) {
    s->data.str[s->length] = cp_strdup(value ? value : "");
    if (!s->data.str[s->length]) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return 0;
    }
  }
  s->is_null[s->length] = is_null ? 1 : 0;
  s->length += 1;
  return 1;
}

static int cp_series_append_null(CpSeries *s, CpError *err) {
  if (!s) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid series");
    return 0;
  }
  switch (s->dtype) {
    case CP_DTYPE_INT64:
      return cp_series_append_int64(s, 0, 1, err);
    case CP_DTYPE_FLOAT64:
      return cp_series_append_float64(s, 0.0, 1, err);
    case CP_DTYPE_STRING:
      return cp_series_append_string(s, NULL, 1, err);
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unknown dtype");
      return 0;
  }
}

static int cp_series_append_from(CpSeries *dest,
                                 const CpSeries *src,
                                 size_t idx,
                                 CpError *err) {
  if (!dest || !src || dest->dtype != src->dtype) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  if (idx >= src->length) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "row index out of range");
    return 0;
  }
  int is_null = src->is_null[idx] ? 1 : 0;
  switch (dest->dtype) {
    case CP_DTYPE_INT64:
      return cp_series_append_int64(dest, src->data.i64[idx], is_null, err);
    case CP_DTYPE_FLOAT64:
      return cp_series_append_float64(dest, src->data.f64[idx], is_null, err);
    case CP_DTYPE_STRING:
      return cp_series_append_string(dest, src->data.str[idx], is_null, err);
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unknown dtype");
      return 0;
  }
}

static void cp_series_pop(CpSeries *s) {
  if (!s || s->length == 0) {
    return;
  }
  size_t idx = s->length - 1;
  if (s->dtype == CP_DTYPE_STRING) {
    free(s->data.str[idx]);
    s->data.str[idx] = NULL;
  }
  s->length -= 1;
}

static int cp_is_blank(const char *s) {
  if (!s) {
    return 1;
  }
  for (const char *p = s; *p; ++p) {
    if (!isspace((unsigned char)*p)) {
      return 0;
    }
  }
  return 1;
}

static int cp_parse_int64(const char *s,
                          int64_t *out,
                          int *is_null,
                          CpError *err,
                          size_t row,
                          size_t col) {
  if (cp_is_blank(s)) {
    if (out) {
      *out = 0;
    }
    if (is_null) {
      *is_null = 1;
    }
    return 1;
  }
  const char *start = s;
  while (isspace((unsigned char)*start)) {
    start++;
  }
  errno = 0;
  char *end = NULL;
  long long value = strtoll(start, &end, 10);
  if (errno == ERANGE) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "int64 overflow");
    return 0;
  }
  while (end && isspace((unsigned char)*end)) {
    end++;
  }
  if (!end || *end != '\0') {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid int64 value");
    return 0;
  }
  if (out) {
    *out = (int64_t)value;
  }
  if (is_null) {
    *is_null = 0;
  }
  return 1;
}

static int cp_parse_float64(const char *s,
                            double *out,
                            int *is_null,
                            CpError *err,
                            size_t row,
                            size_t col) {
  if (cp_is_blank(s)) {
    if (out) {
      *out = 0.0;
    }
    if (is_null) {
      *is_null = 1;
    }
    return 1;
  }
  const char *start = s;
  while (isspace((unsigned char)*start)) {
    start++;
  }
  errno = 0;
  char *end = NULL;
  double value = strtod(start, &end);
  if (errno == ERANGE) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "float64 overflow");
    return 0;
  }
  while (end && isspace((unsigned char)*end)) {
    end++;
  }
  if (!end || *end != '\0') {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid float64 value");
    return 0;
  }
  if (out) {
    *out = value;
  }
  if (is_null) {
    *is_null = 0;
  }
  return 1;
}

static int cp_parse_string(const char *s,
                           const char **out,
                           int *is_null) {
  if (cp_is_blank(s)) {
    if (out) {
      *out = NULL;
    }
    if (is_null) {
      *is_null = 1;
    }
    return 1;
  }
  if (out) {
    *out = s;
  }
  if (is_null) {
    *is_null = 0;
  }
  return 1;
}

static void cp_free_fields(char **fields, size_t count) {
  if (!fields) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    free(fields[i]);
  }
  free(fields);
}

static int cp_fields_push(char ***fields,
                          size_t *count,
                          size_t *capacity,
                          const char *buf,
                          size_t len,
                          CpError *err) {
  if (*count + 1 > *capacity) {
    size_t new_cap = *capacity == 0 ? 8 : (*capacity * 2);
    char **new_fields = (char **)realloc(*fields, new_cap * sizeof(char *));
    if (!new_fields) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return 0;
    }
    *fields = new_fields;
    *capacity = new_cap;
  }
  char *field = cp_strndup(buf, len);
  if (!field) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return 0;
  }
  (*fields)[*count] = field;
  *count += 1;
  return 1;
}

static int cp_field_append(char **buf, size_t *len, size_t *cap, char ch) {
  if (*len + 1 >= *cap) {
    size_t new_cap = *cap == 0 ? 64 : (*cap * 2);
    char *new_buf = (char *)realloc(*buf, new_cap);
    if (!new_buf) {
      return 0;
    }
    *buf = new_buf;
    *cap = new_cap;
  }
  (*buf)[*len] = ch;
  *len += 1;
  return 1;
}

static int cp_parse_csv_line(const char *line,
                             char delimiter,
                             char ***out_fields,
                             size_t *out_count,
                             CpError *err) {
  char **fields = NULL;
  size_t field_count = 0;
  size_t field_cap = 0;

  size_t i = 0;
  while (1) {
    char *field_buf = NULL;
    size_t field_len = 0;
    size_t buf_cap = 0;

    if (line[i] == '"') {
      int closed = 0;
      i += 1;
      while (line[i]) {
        if (line[i] == '"') {
          if (line[i + 1] == '"') {
            if (!cp_field_append(&field_buf, &field_len, &buf_cap, '"')) {
              cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
              free(field_buf);
              cp_free_fields(fields, field_count);
              return 0;
            }
            i += 2;
            continue;
          }
          i += 1;
          closed = 1;
          break;
        }
        if (!cp_field_append(&field_buf, &field_len, &buf_cap, line[i])) {
          cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
          free(field_buf);
          cp_free_fields(fields, field_count);
          return 0;
        }
        i += 1;
      }
      if (!closed) {
        cp_error_set(err, CP_ERR_PARSE, 0, 0, "unterminated quoted field");
        free(field_buf);
        cp_free_fields(fields, field_count);
        return 0;
      }
      if (line[i] && line[i] != delimiter) {
        while (line[i] && line[i] != delimiter) {
          if (!isspace((unsigned char)line[i])) {
            cp_error_set(err, CP_ERR_PARSE, 0, 0, "invalid quoted field");
            free(field_buf);
            cp_free_fields(fields, field_count);
            return 0;
          }
          i += 1;
        }
      }
    } else {
      while (line[i] && line[i] != delimiter) {
        if (!cp_field_append(&field_buf, &field_len, &buf_cap, line[i])) {
          cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
          free(field_buf);
          cp_free_fields(fields, field_count);
          return 0;
        }
        i += 1;
      }
    }

    if (!field_buf) {
      field_buf = (char *)malloc(1);
      if (!field_buf) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        cp_free_fields(fields, field_count);
        return 0;
      }
    }
    if (field_len == 0) {
      field_buf[0] = '\0';
    } else {
      field_buf[field_len] = '\0';
    }

    if (!cp_fields_push(&fields, &field_count, &field_cap, field_buf,
                        field_len, err)) {
      free(field_buf);
      cp_free_fields(fields, field_count);
      return 0;
    }
    free(field_buf);

    if (line[i] == delimiter) {
      i += 1;
      continue;
    }
    if (line[i] == '\0') {
      break;
    }
  }

  *out_fields = fields;
  *out_count = field_count;
  return 1;
}

static char *cp_read_line(FILE *fp, CpError *err) {
  size_t cap = 256;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  int ch;
  while ((ch = fgetc(fp)) != EOF) {
    if (len + 1 >= cap) {
      size_t new_cap = cap * 2;
      char *new_buf = (char *)realloc(buf, new_cap);
      if (!new_buf) {
        free(buf);
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        return NULL;
      }
      buf = new_buf;
      cap = new_cap;
    }
    if (ch == '\n') {
      break;
    }
    buf[len++] = (char)ch;
  }
  if (ch == EOF && len == 0) {
    free(buf);
    return NULL;
  }
  if (len > 0 && buf[len - 1] == '\r') {
    len -= 1;
  }
  buf[len] = '\0';
  return buf;
}

CpDataFrame *cp_df_create(size_t ncols,
                          const char **names,
                          const CpDType *dtypes,
                          size_t capacity,
                          CpError *err) {
  if (ncols == 0 || !names || !dtypes) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe schema");
    return NULL;
  }
  CpDataFrame *df = (CpDataFrame *)calloc(1, sizeof(CpDataFrame));
  if (!df) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  df->ncols = ncols;
  df->nrows = 0;
  df->cols = (CpSeries **)calloc(ncols, sizeof(CpSeries *));
  if (!df->cols) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    free(df);
    return NULL;
  }

  for (size_t i = 0; i < ncols; ++i) {
    df->cols[i] = cp_series_create(names[i], dtypes[i], capacity, err);
    if (!df->cols[i]) {
      cp_df_free(df);
      return NULL;
    }
  }
  return df;
}

void cp_df_free(CpDataFrame *df) {
  if (!df) {
    return;
  }
  for (size_t i = 0; i < df->ncols; ++i) {
    cp_series_free(df->cols[i]);
  }
  free(df->cols);
  free(df);
}

size_t cp_df_nrows(const CpDataFrame *df) {
  return df ? df->nrows : 0;
}

size_t cp_df_ncols(const CpDataFrame *df) {
  return df ? df->ncols : 0;
}

const CpSeries *cp_df_get_col(const CpDataFrame *df, const char *name) {
  if (!df || !name) {
    return NULL;
  }
  for (size_t i = 0; i < df->ncols; ++i) {
    if (df->cols[i] && df->cols[i]->name && strcmp(df->cols[i]->name, name) == 0) {
      return df->cols[i];
    }
  }
  return NULL;
}

static const CpSeries *cp_df_require_col(const CpDataFrame *df,
                                         const char *name,
                                         CpError *err) {
  if (!df || !name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid column");
    return NULL;
  }
  const CpSeries *series = cp_df_get_col(df, name);
  if (!series) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }
  return series;
}

static const CpSeries *cp_df_require_col_index(const CpDataFrame *df,
                                               size_t index,
                                               CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid column index");
    return NULL;
  }
  if (index >= df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column index out of range");
    return NULL;
  }
  if (!df->cols[index]) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }
  return df->cols[index];
}

static int cp_df_append_row_from_sources(CpDataFrame *df,
                                         const CpSeries **src_cols,
                                         size_t ncols,
                                         size_t row,
                                         CpError *err) {
  if (!df || !src_cols || ncols != df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, row, 0, "invalid row source");
    return 0;
  }
  for (size_t i = 0; i < ncols; ++i) {
    if (!cp_series_append_from(df->cols[i], src_cols[i], row, err)) {
      for (size_t j = 0; j < i; ++j) {
        cp_series_pop(df->cols[j]);
      }
      return 0;
    }
  }
  df->nrows += 1;
  return 1;
}

static int cp_name_in_list(const char *name,
                           const char **names,
                           size_t count) {
  if (!name || !names) {
    return 0;
  }
  for (size_t i = 0; i < count; ++i) {
    if (names[i] && strcmp(name, names[i]) == 0) {
      return 1;
    }
  }
  return 0;
}

static int cp_names_have_duplicates(const char **names, size_t count) {
  if (!names) {
    return 0;
  }
  for (size_t i = 0; i < count; ++i) {
    if (!names[i]) {
      continue;
    }
    for (size_t j = i + 1; j < count; ++j) {
      if (names[j] && strcmp(names[i], names[j]) == 0) {
        return 1;
      }
    }
  }
  return 0;
}

static const char *cp_unique_name_with_suffix(const char *base,
                                              const char **existing,
                                              size_t existing_count,
                                              const char *suffix,
                                              int *owned,
                                              CpError *err) {
  if (!owned || !suffix) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid name ownership");
    return NULL;
  }
  *owned = 0;
  const char *safe = base ? base : "";
  if (!cp_name_in_list(safe, existing, existing_count)) {
    return safe;
  }

  size_t attempt = 1;
  for (;;) {
    char num_buf[32];
    if (attempt == 1) {
      num_buf[0] = '\0';
    } else {
      snprintf(num_buf, sizeof(num_buf), "%zu", attempt);
    }
    size_t len = strlen(safe) + strlen(suffix) + strlen(num_buf) + 1;
    char *name = (char *)malloc(len);
    if (!name) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    snprintf(name, len, "%s%s%s", safe, suffix, num_buf);
    if (!cp_name_in_list(name, existing, existing_count)) {
      *owned = 1;
      return name;
    }
    free(name);
    attempt += 1;
  }
}

static int cp_join_key_is_null(const CpSeries *s, size_t row) {
  if (!s) {
    return 1;
  }
  if (row >= s->length) {
    return 1;
  }
  if (s->is_null[row]) {
    return 1;
  }
  if (s->dtype == CP_DTYPE_STRING && !s->data.str[row]) {
    return 1;
  }
  return 0;
}

static int cp_join_key_equal(const CpSeries *left_key,
                             size_t left_row,
                             const CpSeries *right_key,
                             size_t right_row) {
  if (!left_key || !right_key) {
    return 0;
  }
  if (left_key->dtype == CP_DTYPE_INT64) {
    return left_key->data.i64[left_row] == right_key->data.i64[right_row];
  }
  const char *lhs = left_key->data.str[left_row];
  const char *rhs = right_key->data.str[right_row];
  if (!lhs || !rhs) {
    return 0;
  }
  return strcmp(lhs, rhs) == 0;
}

static int cp_agg_output_dtype(const CpSeries *series, CpAggOp op, CpDType *out) {
  if (!series || !out) {
    return 0;
  }
  if (op == CP_AGG_COUNT) {
    *out = CP_DTYPE_INT64;
    return 1;
  }
  if (op == CP_AGG_MEAN) {
    *out = CP_DTYPE_FLOAT64;
    return 1;
  }
  if (series->dtype == CP_DTYPE_INT64) {
    *out = CP_DTYPE_INT64;
    return 1;
  }
  if (series->dtype == CP_DTYPE_FLOAT64) {
    *out = CP_DTYPE_FLOAT64;
    return 1;
  }
  return 0;
}

static CpDataFrame *cp_df_empty_like(const CpDataFrame *df, CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  size_t ncols = df->ncols;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t i = 0; i < ncols; ++i) {
    dtypes[i] = df->cols[i]->dtype;
    names[i] = df->cols[i]->name;
  }
  CpDataFrame *out = cp_df_create(ncols, names, dtypes, 0, err);
  free(dtypes);
  free(names);
  return out;
}

static int cp_indices_have_duplicates(const size_t *indices, size_t count) {
  if (!indices) {
    return 0;
  }
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = i + 1; j < count; ++j) {
      if (indices[i] == indices[j]) {
        return 1;
      }
    }
  }
  return 0;
}

CpDataFrame *cp_df_iloc(const CpDataFrame *df,
                        const size_t *row_indices,
                        size_t row_count,
                        const size_t *col_indices,
                        size_t col_count,
                        CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  size_t sel_cols = col_indices ? col_count : ncols;
  if (sel_cols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns selected");
    return NULL;
  }

  if (col_indices && cp_indices_have_duplicates(col_indices, col_count)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "duplicate column indices");
    return NULL;
  }

  CpDType *dtypes = (CpDType *)malloc(sel_cols * sizeof(CpDType));
  const char **names = (const char **)malloc(sel_cols * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(sel_cols * sizeof(const CpSeries *));
  if (!dtypes || !names || !src_cols) {
    free(dtypes);
    free(names);
    free(src_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  if (col_indices) {
    for (size_t i = 0; i < sel_cols; ++i) {
      if (col_indices[i] >= ncols) {
        free(dtypes);
        free(names);
        free(src_cols);
        cp_error_set(err, CP_ERR_INVALID, 0, col_indices[i],
                     "column index out of range");
        return NULL;
      }
      const CpSeries *series = df->cols[col_indices[i]];
      dtypes[i] = series->dtype;
      names[i] = series->name;
      src_cols[i] = series;
    }
  } else {
    for (size_t i = 0; i < sel_cols; ++i) {
      dtypes[i] = df->cols[i]->dtype;
      names[i] = df->cols[i]->name;
      src_cols[i] = df->cols[i];
    }
  }

  size_t out_rows = row_indices ? row_count : nrows;
  CpDataFrame *out = cp_df_create(sel_cols, names, dtypes, out_rows, err);
  if (!out) {
    free(dtypes);
    free(names);
    free(src_cols);
    return NULL;
  }

  if (row_indices) {
    for (size_t i = 0; i < row_count; ++i) {
      size_t row = row_indices[i];
      if (row >= nrows) {
        cp_error_set(err, CP_ERR_INVALID, row, 0, "row index out of range");
        cp_df_free(out);
        out = NULL;
        break;
      }
      if (!cp_df_append_row_from_sources(out, src_cols, sel_cols, row, err)) {
        cp_df_free(out);
        out = NULL;
        break;
      }
    }
  } else {
    for (size_t row = 0; row < nrows; ++row) {
      if (!cp_df_append_row_from_sources(out, src_cols, sel_cols, row, err)) {
        cp_df_free(out);
        out = NULL;
        break;
      }
    }
  }

  free(dtypes);
  free(names);
  free(src_cols);
  return out;
}

CpDataFrame *cp_df_loc(const CpDataFrame *df,
                       const size_t *row_indices,
                       size_t row_count,
                       const char **names,
                       size_t name_count,
                       CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (names && name_count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns selected");
    return NULL;
  }

  size_t *col_indices = NULL;
  size_t col_count = 0;
  if (names) {
    col_count = name_count;
    col_indices = (size_t *)malloc(name_count * sizeof(size_t));
    if (!col_indices) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    for (size_t i = 0; i < name_count; ++i) {
      const CpSeries *series = cp_df_get_col(df, names[i]);
      if (!series) {
        free(col_indices);
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
        return NULL;
      }
      size_t idx = 0;
      for (; idx < df->ncols; ++idx) {
        if (df->cols[idx] == series) {
          break;
        }
      }
      if (idx >= df->ncols) {
        free(col_indices);
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
        return NULL;
      }
      col_indices[i] = idx;
    }
  }

  CpDataFrame *out =
      cp_df_iloc(df, row_indices, row_count, col_indices, col_count, err);
  free(col_indices);
  return out;
}

CpDataFrame *cp_df_select_cols(const CpDataFrame *df,
                               const char **names,
                               size_t count,
                               CpError *err) {
  if (!df || !names || count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid selection");
    return NULL;
  }

  CpDType *dtypes = (CpDType *)malloc(count * sizeof(CpDType));
  const char **sel_names = (const char **)malloc(count * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(count * sizeof(const CpSeries *));
  if (!dtypes || !sel_names || !src_cols) {
    free(dtypes);
    free(sel_names);
    free(src_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < count; ++i) {
    const CpSeries *series = cp_df_get_col(df, names[i]);
    if (!series) {
      free(dtypes);
      free(sel_names);
      free(src_cols);
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
      return NULL;
    }
    dtypes[i] = series->dtype;
    sel_names[i] = series->name;
    src_cols[i] = series;
  }

  CpDataFrame *out = cp_df_create(count, sel_names, dtypes, df->nrows, err);
  if (!out) {
    free(dtypes);
    free(sel_names);
    free(src_cols);
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (!cp_df_append_row_from_sources(out, src_cols, count, row, err)) {
      cp_df_free(out);
      out = NULL;
      break;
    }
  }

  free(dtypes);
  free(sel_names);
  free(src_cols);
  return out;
}

CpDataFrame *cp_df_head(const CpDataFrame *df, size_t n, CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  size_t nrows = df->nrows;
  size_t take = n < nrows ? n : nrows;
  if (take == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask && nrows > 0) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t i = 0; i < take; ++i) {
    mask[i] = 1;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  return out;
}

CpDataFrame *cp_df_tail(const CpDataFrame *df, size_t n, CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  size_t nrows = df->nrows;
  size_t take = n < nrows ? n : nrows;
  if (take == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask && nrows > 0) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  size_t start = nrows - take;
  for (size_t i = start; i < nrows; ++i) {
    mask[i] = 1;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  return out;
}

int cp_df_dtypes(const CpDataFrame *df,
                 CpDType *out,
                 size_t out_len,
                 CpError *err) {
  if (!df || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  if (out_len < df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output buffer too small");
    return 0;
  }
  for (size_t i = 0; i < df->ncols; ++i) {
    out[i] = df->cols[i]->dtype;
  }
  return 1;
}

CpDataFrame *cp_df_drop_cols(const CpDataFrame *df,
                             const char **names,
                             size_t count,
                             CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (!names && count > 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid column list");
    return NULL;
  }

  for (size_t i = 0; i < count; ++i) {
    if (names[i] && !cp_df_get_col(df, names[i])) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
      return NULL;
    }
  }

  size_t keep_count = 0;
  for (size_t i = 0; i < df->ncols; ++i) {
    if (!cp_name_in_list(df->cols[i]->name, names, count)) {
      keep_count += 1;
    }
  }
  if (keep_count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns remaining");
    return NULL;
  }

  const char **keep_names =
      (const char **)malloc(keep_count * sizeof(const char *));
  if (!keep_names) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  size_t idx = 0;
  for (size_t i = 0; i < df->ncols; ++i) {
    if (!cp_name_in_list(df->cols[i]->name, names, count)) {
      keep_names[idx++] = df->cols[i]->name;
    }
  }

  CpDataFrame *out = cp_df_select_cols(df, keep_names, keep_count, err);
  free(keep_names);
  return out;
}

CpDataFrame *cp_df_rename_cols(const CpDataFrame *df,
                               const char **old_names,
                               const char **new_names,
                               size_t count,
                               CpError *err) {
  if (!df || !old_names || !new_names || count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid rename mapping");
    return NULL;
  }

  size_t ncols = df->ncols;
  const char **out_names =
      (const char **)malloc(ncols * sizeof(const char *));
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(ncols * sizeof(const CpSeries *));
  if (!out_names || !dtypes || !src_cols) {
    free(out_names);
    free(dtypes);
    free(src_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < ncols; ++i) {
    const char *name = df->cols[i]->name;
    const char *new_name = name;
    for (size_t j = 0; j < count; ++j) {
      if (old_names[j] && name && strcmp(old_names[j], name) == 0) {
        if (!new_names[j]) {
          free(out_names);
          free(dtypes);
          free(src_cols);
          cp_error_set(err, CP_ERR_INVALID, 0, 0, "new name is required");
          return NULL;
        }
        new_name = new_names[j];
        break;
      }
    }
    out_names[i] = new_name;
    dtypes[i] = df->cols[i]->dtype;
    src_cols[i] = df->cols[i];
  }

  if (cp_names_have_duplicates(out_names, ncols)) {
    free(out_names);
    free(dtypes);
    free(src_cols);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "duplicate column names");
    return NULL;
  }

  CpDataFrame *out = cp_df_create(ncols, out_names, dtypes, df->nrows, err);
  if (!out) {
    free(out_names);
    free(dtypes);
    free(src_cols);
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (!cp_df_append_row_from_sources(out, src_cols, ncols, row, err)) {
      cp_df_free(out);
      out = NULL;
      break;
    }
  }

  free(out_names);
  free(dtypes);
  free(src_cols);
  return out;
}

int cp_df_isnull_mask(const CpDataFrame *df,
                      uint8_t *out,
                      size_t out_len,
                      CpError *err) {
  if (!df || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  size_t needed = df->nrows * df->ncols;
  if (out_len < needed) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output buffer too small");
    return 0;
  }
  size_t idx = 0;
  for (size_t row = 0; row < df->nrows; ++row) {
    for (size_t col = 0; col < df->ncols; ++col) {
      out[idx++] = df->cols[col]->is_null[row] ? 1 : 0;
    }
  }
  return 1;
}

CpDataFrame *cp_df_dropna(const CpDataFrame *df, CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  size_t nrows = df->nrows;
  if (nrows == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask && nrows > 0) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t row = 0; row < nrows; ++row) {
    int keep = 1;
    for (size_t col = 0; col < df->ncols; ++col) {
      if (df->cols[col]->is_null[row]) {
        keep = 0;
        break;
      }
    }
    mask[row] = keep ? 1 : 0;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  return out;
}

CpDataFrame *cp_df_fillna(const CpDataFrame *df,
                          const char **values,
                          size_t count,
                          CpError *err) {
  if (!df || !values) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return NULL;
  }
  if (count != df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "fill values count mismatch");
    return NULL;
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  int *fill_enabled = (int *)calloc(ncols, sizeof(int));
  int64_t *fill_i64 = (int64_t *)calloc(ncols, sizeof(int64_t));
  double *fill_f64 = (double *)calloc(ncols, sizeof(double));
  const char **fill_str =
      (const char **)calloc(ncols, sizeof(const char *));
  if (!fill_enabled || !fill_i64 || !fill_f64 || !fill_str) {
    free(fill_enabled);
    free(fill_i64);
    free(fill_f64);
    free(fill_str);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t col = 0; col < ncols; ++col) {
    if (!values[col]) {
      fill_enabled[col] = 0;
      continue;
    }
    fill_enabled[col] = 1;
    int is_null = 0;
    switch (df->cols[col]->dtype) {
      case CP_DTYPE_INT64: {
        int64_t v = 0;
        if (!cp_parse_int64(values[col], &v, &is_null, err, 0, col)) {
          free(fill_enabled);
          free(fill_i64);
          free(fill_f64);
          free(fill_str);
          return NULL;
        }
        if (is_null) {
          cp_error_set(err, CP_ERR_INVALID, 0, col,
                       "fill value is null");
          free(fill_enabled);
          free(fill_i64);
          free(fill_f64);
          free(fill_str);
          return NULL;
        }
        fill_i64[col] = v;
        break;
      }
      case CP_DTYPE_FLOAT64: {
        double v = 0.0;
        if (!cp_parse_float64(values[col], &v, &is_null, err, 0, col)) {
          free(fill_enabled);
          free(fill_i64);
          free(fill_f64);
          free(fill_str);
          return NULL;
        }
        if (is_null) {
          cp_error_set(err, CP_ERR_INVALID, 0, col,
                       "fill value is null");
          free(fill_enabled);
          free(fill_i64);
          free(fill_f64);
          free(fill_str);
          return NULL;
        }
        fill_f64[col] = v;
        break;
      }
      case CP_DTYPE_STRING:
        fill_str[col] = values[col];
        break;
      default:
        cp_error_set(err, CP_ERR_INVALID, 0, col, "unknown dtype");
        free(fill_enabled);
        free(fill_i64);
        free(fill_f64);
        free(fill_str);
        return NULL;
    }
  }

  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(ncols * sizeof(const CpSeries *));
  if (!dtypes || !names || !src_cols) {
    free(dtypes);
    free(names);
    free(src_cols);
    free(fill_enabled);
    free(fill_i64);
    free(fill_f64);
    free(fill_str);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t col = 0; col < ncols; ++col) {
    src_cols[col] = df->cols[col];
    names[col] = df->cols[col]->name;
    dtypes[col] = df->cols[col]->dtype;
  }

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, nrows, err);
  if (!out) {
    free(dtypes);
    free(names);
    free(src_cols);
    free(fill_enabled);
    free(fill_i64);
    free(fill_f64);
    free(fill_str);
    return NULL;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *src = src_cols[col];
      if (src->is_null[row] && fill_enabled[col]) {
        int ok = 0;
        switch (src->dtype) {
          case CP_DTYPE_INT64:
            ok = cp_series_append_int64(dest, fill_i64[col], 0, err);
            break;
          case CP_DTYPE_FLOAT64:
            ok = cp_series_append_float64(dest, fill_f64[col], 0, err);
            break;
          case CP_DTYPE_STRING:
            ok = cp_series_append_string(dest, fill_str[col], 0, err);
            break;
          default:
            ok = 0;
            cp_error_set(err, CP_ERR_INVALID, row, col, "unknown dtype");
            break;
        }
        if (!ok) {
          for (size_t j = 0; j < col; ++j) {
            cp_series_pop(out->cols[j]);
          }
          cp_df_free(out);
          out = NULL;
          break;
        }
      } else {
        if (!cp_series_append_from(dest, src, row, err)) {
          for (size_t j = 0; j < col; ++j) {
            cp_series_pop(out->cols[j]);
          }
          cp_df_free(out);
          out = NULL;
          break;
        }
      }
    }
    if (!out) {
      break;
    }
    out->nrows += 1;
  }

  free(dtypes);
  free(names);
  free(src_cols);
  free(fill_enabled);
  free(fill_i64);
  free(fill_f64);
  free(fill_str);
  return out;
}

int cp_df_info(const CpDataFrame *df, FILE *out, CpError *err) {
  if (!df || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  if (fprintf(out, "DataFrame\n") < 0 ||
      fprintf(out, "Rows: %zu\n", df->nrows) < 0 ||
      fprintf(out, "Columns: %zu\n", df->ncols) < 0 ||
      fprintf(out, "Columns detail:\n") < 0) {
    cp_error_set(err, CP_ERR_IO, 0, 0, "failed to write info");
    return 0;
  }
  for (size_t i = 0; i < df->ncols; ++i) {
    size_t count = 0;
    size_t nulls = 0;
    if (!cp_series_count(df->cols[i], &count, &nulls, err)) {
      return 0;
    }
    if (fprintf(out, "  [%zu] %s (%s) non-null: %zu\n", i,
                df->cols[i]->name ? df->cols[i]->name : "",
                cp_dtype_name(df->cols[i]->dtype),
                count) < 0) {
      cp_error_set(err, CP_ERR_IO, 0, 0, "failed to write info");
      return 0;
    }
  }
  return 1;
}

CpDataFrame *cp_df_describe(const CpDataFrame *df, CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }

  size_t numeric_count = 0;
  for (size_t i = 0; i < df->ncols; ++i) {
    CpDType dtype = df->cols[i]->dtype;
    if (dtype == CP_DTYPE_INT64 || dtype == CP_DTYPE_FLOAT64) {
      numeric_count += 1;
    }
  }
  if (numeric_count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no numeric columns");
    return NULL;
  }

  const CpSeries **numeric_cols =
      (const CpSeries **)malloc(numeric_count * sizeof(const CpSeries *));
  if (!numeric_cols) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  size_t idx = 0;
  for (size_t i = 0; i < df->ncols; ++i) {
    CpDType dtype = df->cols[i]->dtype;
    if (dtype == CP_DTYPE_INT64 || dtype == CP_DTYPE_FLOAT64) {
      numeric_cols[idx++] = df->cols[i];
    }
  }

  size_t out_cols = numeric_count + 1;
  const char **names = (const char **)malloc(out_cols * sizeof(const char *));
  CpDType *dtypes = (CpDType *)malloc(out_cols * sizeof(CpDType));
  if (!names || !dtypes) {
    free(names);
    free(dtypes);
    free(numeric_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  names[0] = "stat";
  dtypes[0] = CP_DTYPE_STRING;
  for (size_t i = 0; i < numeric_count; ++i) {
    names[i + 1] = numeric_cols[i]->name;
    dtypes[i + 1] = CP_DTYPE_FLOAT64;
  }

  CpDataFrame *out = cp_df_create(out_cols, names, dtypes, 4, err);
  free(names);
  free(dtypes);
  if (!out) {
    free(numeric_cols);
    return NULL;
  }

  double *counts = (double *)calloc(numeric_count, sizeof(double));
  double *means = (double *)calloc(numeric_count, sizeof(double));
  double *mins = (double *)calloc(numeric_count, sizeof(double));
  double *maxs = (double *)calloc(numeric_count, sizeof(double));
  if (!counts || !means || !mins || !maxs) {
    free(counts);
    free(means);
    free(mins);
    free(maxs);
    free(numeric_cols);
    cp_df_free(out);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < numeric_count; ++i) {
    const CpSeries *series = numeric_cols[i];
    size_t count = 0;
    double sum = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    int found = 0;
    for (size_t row = 0; row < series->length; ++row) {
      if (series->is_null[row]) {
        continue;
      }
      double value = 0.0;
      if (series->dtype == CP_DTYPE_INT64) {
        value = (double)series->data.i64[row];
      } else {
        value = series->data.f64[row];
      }
      if (!found) {
        min_val = value;
        max_val = value;
        found = 1;
      } else {
        if (value < min_val) {
          min_val = value;
        }
        if (value > max_val) {
          max_val = value;
        }
      }
      sum += value;
      count += 1;
    }
    counts[i] = (double)count;
    if (count == 0) {
      means[i] = NAN;
      mins[i] = NAN;
      maxs[i] = NAN;
    } else {
      means[i] = sum / (double)count;
      mins[i] = min_val;
      maxs[i] = max_val;
    }
  }

  const char *stat_names[] = {"count", "mean", "min", "max"};
  for (size_t row = 0; row < 4; ++row) {
    char **values = (char **)malloc(out_cols * sizeof(char *));
    char **allocated = (char **)calloc(out_cols, sizeof(char *));
    if (!values || !allocated) {
      free(values);
      free(allocated);
      free(counts);
      free(means);
      free(mins);
      free(maxs);
      free(numeric_cols);
      cp_df_free(out);
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }

    values[0] = (char *)stat_names[row];
    for (size_t col = 0; col < numeric_count; ++col) {
      double value = 0.0;
      switch (row) {
        case 0:
          value = counts[col];
          break;
        case 1:
          value = means[col];
          break;
        case 2:
          value = mins[col];
          break;
        case 3:
          value = maxs[col];
          break;
        default:
          value = NAN;
          break;
      }
      char buf[64];
      if (isnan(value)) {
        snprintf(buf, sizeof(buf), "nan");
      } else {
        snprintf(buf, sizeof(buf), "%.17g", value);
      }
      allocated[col + 1] = cp_strdup(buf);
      if (!allocated[col + 1]) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        for (size_t j = 0; j < out_cols; ++j) {
          free(allocated[j]);
        }
        free(values);
        free(allocated);
        free(counts);
        free(means);
        free(mins);
        free(maxs);
        free(numeric_cols);
        cp_df_free(out);
        return NULL;
      }
      values[col + 1] = allocated[col + 1];
    }

    if (!cp_df_append_row(out, (const char **)values, out_cols, err)) {
      for (size_t j = 0; j < out_cols; ++j) {
        free(allocated[j]);
      }
      free(values);
      free(allocated);
      free(counts);
      free(means);
      free(mins);
      free(maxs);
      free(numeric_cols);
      cp_df_free(out);
      return NULL;
    }

    for (size_t j = 0; j < out_cols; ++j) {
      free(allocated[j]);
    }
    free(values);
    free(allocated);
  }

  free(counts);
  free(means);
  free(mins);
  free(maxs);
  free(numeric_cols);
  return out;
}

CpDataFrame *cp_df_groupby_agg(const CpDataFrame *df,
                               const char *key,
                               const char **value_cols,
                               const CpAggOp *ops,
                               size_t count,
                               CpError *err) {
  if (!df || !key || !value_cols || !ops || count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid groupby arguments");
    return NULL;
  }

  const CpSeries *key_series = cp_df_require_col(df, key, err);
  if (!key_series) {
    return NULL;
  }
  if (key_series->dtype != CP_DTYPE_INT64 && key_series->dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported key dtype");
    return NULL;
  }

  CpAggSpec *specs = (CpAggSpec *)calloc(count, sizeof(CpAggSpec));
  if (!specs) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < count; ++i) {
    const CpSeries *series = cp_df_require_col(df, value_cols[i], err);
    if (!series) {
      free(specs);
      return NULL;
    }
    if (ops[i] != CP_AGG_COUNT &&
        series->dtype != CP_DTYPE_INT64 &&
        series->dtype != CP_DTYPE_FLOAT64) {
      free(specs);
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "aggregation requires numeric dtype");
      return NULL;
    }
    CpDType out_dtype = CP_DTYPE_FLOAT64;
    if (!cp_agg_output_dtype(series, ops[i], &out_dtype)) {
      free(specs);
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid aggregation dtype");
      return NULL;
    }
    const char *op_name = cp_agg_op_name(ops[i]);
    size_t name_len = strlen(series->name ? series->name : "") + 1 + strlen(op_name) + 1;
    char *col_name = (char *)malloc(name_len);
    if (!col_name) {
      free(specs);
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    snprintf(col_name, name_len, "%s_%s", series->name ? series->name : "", op_name);
    specs[i].series = series;
    specs[i].op = ops[i];
    specs[i].out_dtype = out_dtype;
    specs[i].name = col_name;
  }

  size_t group_cap = 8;
  size_t group_count = 0;
  typedef struct {
    int64_t key_i64;
    const char *key_str;
    CpAggState *states;
  } CpGroup;
  CpGroup *groups = (CpGroup *)calloc(group_cap, sizeof(CpGroup));
  if (!groups) {
    for (size_t i = 0; i < count; ++i) {
      free(specs[i].name);
    }
    free(specs);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (key_series->is_null[row]) {
      continue;
    }
    int64_t key_i64 = 0;
    const char *key_str = NULL;
    if (key_series->dtype == CP_DTYPE_INT64) {
      key_i64 = key_series->data.i64[row];
    } else {
      key_str = key_series->data.str[row];
      if (!key_str) {
        continue;
      }
    }

    size_t group_idx = group_count;
    for (size_t g = 0; g < group_count; ++g) {
      if (key_series->dtype == CP_DTYPE_INT64) {
        if (groups[g].key_i64 == key_i64) {
          group_idx = g;
          break;
        }
      } else {
        if (groups[g].key_str && strcmp(groups[g].key_str, key_str) == 0) {
          group_idx = g;
          break;
        }
      }
    }

    if (group_idx == group_count) {
      if (group_count == group_cap) {
        size_t new_cap = group_cap * 2;
        CpGroup *new_groups =
            (CpGroup *)realloc(groups, new_cap * sizeof(CpGroup));
        if (!new_groups) {
          cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
          for (size_t i = 0; i < group_count; ++i) {
            free(groups[i].states);
          }
          free(groups);
          for (size_t i = 0; i < count; ++i) {
            free(specs[i].name);
          }
          free(specs);
          return NULL;
        }
        memset(new_groups + group_cap, 0, (new_cap - group_cap) * sizeof(CpGroup));
        groups = new_groups;
        group_cap = new_cap;
      }
      groups[group_idx].states = (CpAggState *)calloc(count, sizeof(CpAggState));
      if (!groups[group_idx].states) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        for (size_t i = 0; i < group_count; ++i) {
          free(groups[i].states);
        }
        free(groups);
        for (size_t i = 0; i < count; ++i) {
          free(specs[i].name);
        }
        free(specs);
        return NULL;
      }
      if (key_series->dtype == CP_DTYPE_INT64) {
        groups[group_idx].key_i64 = key_i64;
      } else {
        groups[group_idx].key_str = key_str;
      }
      group_count += 1;
    }

    CpAggState *states = groups[group_idx].states;
    for (size_t i = 0; i < count; ++i) {
      const CpSeries *series = specs[i].series;
      if (series->is_null[row]) {
        continue;
      }
      if (series->dtype == CP_DTYPE_INT64) {
        int64_t value = series->data.i64[row];
        if (ops[i] == CP_AGG_COUNT) {
          states[i].count += 1;
          continue;
        }
        if (ops[i] == CP_AGG_SUM || ops[i] == CP_AGG_MEAN) {
          if ((value > 0 && states[i].sum_i64 > INT64_MAX - value) ||
              (value < 0 && states[i].sum_i64 < INT64_MIN - value)) {
            cp_error_set(err, CP_ERR_INVALID, row, i, "int64 sum overflow");
            for (size_t g = 0; g < group_count; ++g) {
              free(groups[g].states);
            }
            free(groups);
            for (size_t j = 0; j < count; ++j) {
              free(specs[j].name);
            }
            free(specs);
            return NULL;
          }
          states[i].sum_i64 += value;
          states[i].count += 1;
          states[i].has_value = 1;
        } else if (ops[i] == CP_AGG_MIN) {
          if (!states[i].has_value || value < states[i].min_i64) {
            states[i].min_i64 = value;
          }
          states[i].has_value = 1;
        } else if (ops[i] == CP_AGG_MAX) {
          if (!states[i].has_value || value > states[i].max_i64) {
            states[i].max_i64 = value;
          }
          states[i].has_value = 1;
        }
      } else if (series->dtype == CP_DTYPE_FLOAT64) {
        double value = series->data.f64[row];
        if (ops[i] == CP_AGG_COUNT) {
          states[i].count += 1;
          continue;
        }
        if (ops[i] == CP_AGG_SUM || ops[i] == CP_AGG_MEAN) {
          states[i].sum_f64 += value;
          states[i].count += 1;
          states[i].has_value = 1;
        } else if (ops[i] == CP_AGG_MIN) {
          if (!states[i].has_value || value < states[i].min_f64) {
            states[i].min_f64 = value;
          }
          states[i].has_value = 1;
        } else if (ops[i] == CP_AGG_MAX) {
          if (!states[i].has_value || value > states[i].max_f64) {
            states[i].max_f64 = value;
          }
          states[i].has_value = 1;
        }
      } else if (ops[i] == CP_AGG_COUNT) {
        states[i].count += 1;
      }
    }
  }

  size_t out_cols = count + 1;
  const char **names = (const char **)malloc(out_cols * sizeof(const char *));
  CpDType *dtypes = (CpDType *)malloc(out_cols * sizeof(CpDType));
  if (!names || !dtypes) {
    free(names);
    free(dtypes);
    for (size_t g = 0; g < group_count; ++g) {
      free(groups[g].states);
    }
    free(groups);
    for (size_t i = 0; i < count; ++i) {
      free(specs[i].name);
    }
    free(specs);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  names[0] = key_series->name ? key_series->name : "key";
  dtypes[0] = key_series->dtype;
  for (size_t i = 0; i < count; ++i) {
    names[i + 1] = specs[i].name;
    dtypes[i + 1] = specs[i].out_dtype;
  }

  CpDataFrame *out = cp_df_create(out_cols, names, dtypes, group_count, err);
  free(names);
  free(dtypes);
  if (!out) {
    for (size_t g = 0; g < group_count; ++g) {
      free(groups[g].states);
    }
    free(groups);
    for (size_t i = 0; i < count; ++i) {
      free(specs[i].name);
    }
    free(specs);
    return NULL;
  }

  for (size_t g = 0; g < group_count; ++g) {
    int ok = 1;
    for (size_t col = 0; col < out_cols; ++col) {
      CpSeries *dest = out->cols[col];
      if (col == 0) {
        if (key_series->dtype == CP_DTYPE_INT64) {
          ok = cp_series_append_int64(dest, groups[g].key_i64, 0, err);
        } else {
          ok = cp_series_append_string(dest, groups[g].key_str, 0, err);
        }
      } else {
        size_t spec_idx = col - 1;
        CpAggState *state = &groups[g].states[spec_idx];
        CpAggOp op = specs[spec_idx].op;
        CpDType out_dtype = specs[spec_idx].out_dtype;
        if (op == CP_AGG_COUNT) {
          ok = cp_series_append_int64(dest, (int64_t)state->count, 0, err);
        } else if (op == CP_AGG_MEAN) {
          if (state->count == 0) {
            ok = cp_series_append_float64(dest, 0.0, 1, err);
          } else {
            double mean = 0.0;
            if (specs[spec_idx].series->dtype == CP_DTYPE_INT64) {
              mean = (double)state->sum_i64 / (double)state->count;
            } else {
              mean = state->sum_f64 / (double)state->count;
            }
            ok = cp_series_append_float64(dest, mean, 0, err);
          }
        } else if (out_dtype == CP_DTYPE_INT64) {
          if (op == CP_AGG_SUM) {
            if (state->count == 0) {
              ok = cp_series_append_int64(dest, 0, 1, err);
            } else {
              ok = cp_series_append_int64(dest, state->sum_i64, 0, err);
            }
          } else if (op == CP_AGG_MIN) {
            if (!state->has_value) {
              ok = cp_series_append_int64(dest, 0, 1, err);
            } else {
              ok = cp_series_append_int64(dest, state->min_i64, 0, err);
            }
          } else if (op == CP_AGG_MAX) {
            if (!state->has_value) {
              ok = cp_series_append_int64(dest, 0, 1, err);
            } else {
              ok = cp_series_append_int64(dest, state->max_i64, 0, err);
            }
          } else {
            ok = 0;
          }
        } else {
          if (op == CP_AGG_SUM) {
            if (state->count == 0) {
              ok = cp_series_append_float64(dest, 0.0, 1, err);
            } else {
              ok = cp_series_append_float64(dest, state->sum_f64, 0, err);
            }
          } else if (op == CP_AGG_MIN) {
            if (!state->has_value) {
              ok = cp_series_append_float64(dest, 0.0, 1, err);
            } else {
              ok = cp_series_append_float64(dest, state->min_f64, 0, err);
            }
          } else if (op == CP_AGG_MAX) {
            if (!state->has_value) {
              ok = cp_series_append_float64(dest, 0.0, 1, err);
            } else {
              ok = cp_series_append_float64(dest, state->max_f64, 0, err);
            }
          } else {
            ok = 0;
          }
        }
      }

      if (!ok) {
        for (size_t j = 0; j < col; ++j) {
          cp_series_pop(out->cols[j]);
        }
        cp_df_free(out);
        out = NULL;
        break;
      }
    }
    if (!out) {
      break;
    }
    out->nrows += 1;
  }

  for (size_t g = 0; g < group_count; ++g) {
    free(groups[g].states);
  }
  free(groups);
  for (size_t i = 0; i < count; ++i) {
    free(specs[i].name);
  }
  free(specs);
  return out;
}

static int cp_join_append_row(CpDataFrame *out,
                              const CpSeries **sources,
                              const unsigned char *from_right,
                              size_t ncols,
                              size_t left_row,
                              size_t right_row,
                              int has_right,
                              CpError *err) {
  if (!out || !sources || !from_right) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join row");
    return 0;
  }
  for (size_t col = 0; col < ncols; ++col) {
    int ok = 0;
    if (from_right[col]) {
      if (has_right) {
        ok = cp_series_append_from(out->cols[col], sources[col], right_row, err);
      } else {
        ok = cp_series_append_null(out->cols[col], err);
      }
    } else {
      ok = cp_series_append_from(out->cols[col], sources[col], left_row, err);
    }
    if (!ok) {
      for (size_t j = 0; j < col; ++j) {
        cp_series_pop(out->cols[j]);
      }
      return 0;
    }
  }
  out->nrows += 1;
  return 1;
}

CpDataFrame *cp_df_join(const CpDataFrame *left,
                        const CpDataFrame *right,
                        const char *left_key,
                        const char *right_key,
                        CpJoinType how,
                        CpError *err) {
  if (!left || !right || !left_key || !right_key) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join arguments");
    return NULL;
  }
  if (left->ncols == 0 || right->ncols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "empty schema");
    return NULL;
  }
  if (how != CP_JOIN_INNER && how != CP_JOIN_LEFT) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported join type");
    return NULL;
  }

  const CpSeries *left_key_series = cp_df_require_col(left, left_key, err);
  if (!left_key_series) {
    return NULL;
  }
  const CpSeries *right_key_series = cp_df_require_col(right, right_key, err);
  if (!right_key_series) {
    return NULL;
  }

  if (left_key_series->dtype != right_key_series->dtype) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "join key dtype mismatch");
    return NULL;
  }
  if (left_key_series->dtype != CP_DTYPE_INT64 &&
      left_key_series->dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported join key dtype");
    return NULL;
  }

  int same_key_name = strcmp(left_key, right_key) == 0;
  size_t out_cols = left->ncols + right->ncols - (same_key_name ? 1 : 0);
  if (out_cols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join schema");
    return NULL;
  }

  const char **out_names = (const char **)malloc(out_cols * sizeof(const char *));
  CpDType *out_dtypes = (CpDType *)malloc(out_cols * sizeof(CpDType));
  const CpSeries **out_sources =
      (const CpSeries **)malloc(out_cols * sizeof(const CpSeries *));
  unsigned char *out_from_right =
      (unsigned char *)calloc(out_cols, sizeof(unsigned char));
  unsigned char *name_owned =
      (unsigned char *)calloc(out_cols, sizeof(unsigned char));
  if (!out_names || !out_dtypes || !out_sources || !out_from_right || !name_owned) {
    free(out_names);
    free(out_dtypes);
    free(out_sources);
    free(out_from_right);
    free(name_owned);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  size_t out_idx = 0;
  for (size_t i = 0; i < left->ncols; ++i) {
    out_names[out_idx] = left->cols[i]->name;
    out_dtypes[out_idx] = left->cols[i]->dtype;
    out_sources[out_idx] = left->cols[i];
    out_from_right[out_idx] = 0;
    out_idx += 1;
  }

  for (size_t i = 0; i < right->ncols; ++i) {
    const CpSeries *series = right->cols[i];
    if (same_key_name && strcmp(series->name, right_key) == 0) {
      continue;
    }
    int owned = 0;
    const char *name = cp_unique_name_with_suffix(series->name,
                                                  out_names,
                                                  out_idx,
                                                  "_right",
                                                  &owned,
                                                  err);
    if (!name) {
      for (size_t j = 0; j < out_idx; ++j) {
        if (name_owned[j]) {
          free((char *)out_names[j]);
        }
      }
      free(out_names);
      free(out_dtypes);
      free(out_sources);
      free(out_from_right);
      free(name_owned);
      return NULL;
    }
    out_names[out_idx] = name;
    out_dtypes[out_idx] = series->dtype;
    out_sources[out_idx] = series;
    out_from_right[out_idx] = 1;
    if (owned) {
      name_owned[out_idx] = 1;
    }
    out_idx += 1;
  }

  if (out_idx != out_cols) {
    for (size_t j = 0; j < out_idx; ++j) {
      if (name_owned[j]) {
        free((char *)out_names[j]);
      }
    }
    free(out_names);
    free(out_dtypes);
    free(out_sources);
    free(out_from_right);
    free(name_owned);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "join schema mismatch");
    return NULL;
  }

  size_t total_rows = 0;
  for (size_t lrow = 0; lrow < left->nrows; ++lrow) {
    if (cp_join_key_is_null(left_key_series, lrow)) {
      if (how == CP_JOIN_LEFT) {
        if (total_rows == SIZE_MAX) {
          cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
          total_rows = 0;
          break;
        }
        total_rows += 1;
      }
      continue;
    }
    size_t matches = 0;
    for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
      if (cp_join_key_is_null(right_key_series, rrow)) {
        continue;
      }
      if (cp_join_key_equal(left_key_series, lrow, right_key_series, rrow)) {
        matches += 1;
      }
    }
    if (matches == 0) {
      if (how == CP_JOIN_LEFT) {
        if (total_rows == SIZE_MAX) {
          cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
          total_rows = 0;
          break;
        }
        total_rows += 1;
      }
    } else {
      if (total_rows > SIZE_MAX - matches) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
        total_rows = 0;
        break;
      }
      total_rows += matches;
    }
  }

  CpDataFrame *out = cp_df_create(out_cols, out_names, out_dtypes, total_rows, err);
  for (size_t j = 0; j < out_cols; ++j) {
    if (name_owned[j]) {
      free((char *)out_names[j]);
    }
  }
  free(out_names);
  free(out_dtypes);
  free(name_owned);

  if (!out) {
    free(out_sources);
    free(out_from_right);
    return NULL;
  }

  for (size_t lrow = 0; lrow < left->nrows; ++lrow) {
    if (cp_join_key_is_null(left_key_series, lrow)) {
      if (how == CP_JOIN_LEFT) {
        if (!cp_join_append_row(out,
                                out_sources,
                                out_from_right,
                                out_cols,
                                lrow,
                                0,
                                0,
                                err)) {
          cp_df_free(out);
          out = NULL;
          break;
        }
      }
      continue;
    }

    int matched = 0;
    for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
      if (cp_join_key_is_null(right_key_series, rrow)) {
        continue;
      }
      if (cp_join_key_equal(left_key_series, lrow, right_key_series, rrow)) {
        matched = 1;
        if (!cp_join_append_row(out,
                                out_sources,
                                out_from_right,
                                out_cols,
                                lrow,
                                rrow,
                                1,
                                err)) {
          cp_df_free(out);
          out = NULL;
          break;
        }
      }
    }

    if (!out) {
      break;
    }

    if (!matched && how == CP_JOIN_LEFT) {
      if (!cp_join_append_row(out,
                              out_sources,
                              out_from_right,
                              out_cols,
                              lrow,
                              0,
                              0,
                              err)) {
        cp_df_free(out);
        out = NULL;
        break;
      }
    }
  }

  free(out_sources);
  free(out_from_right);
  return out;
}

CpDataFrame *cp_df_pivot_table(const CpDataFrame *df,
                               const char *index,
                               const char *columns,
                               const char *values,
                               CpAggOp op,
                               CpError *err) {
  if (!df || !index || !columns || !values) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid pivot arguments");
    return NULL;
  }

  const CpSeries *index_series = cp_df_require_col(df, index, err);
  if (!index_series) {
    return NULL;
  }
  const CpSeries *columns_series = cp_df_require_col(df, columns, err);
  if (!columns_series) {
    return NULL;
  }
  const CpSeries *values_series = cp_df_require_col(df, values, err);
  if (!values_series) {
    return NULL;
  }

  if ((index_series->dtype != CP_DTYPE_INT64 &&
       index_series->dtype != CP_DTYPE_STRING) ||
      (columns_series->dtype != CP_DTYPE_INT64 &&
       columns_series->dtype != CP_DTYPE_STRING)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported pivot key dtype");
    return NULL;
  }
  if (op != CP_AGG_COUNT &&
      values_series->dtype != CP_DTYPE_INT64 &&
      values_series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "pivot aggregation requires numeric dtype");
    return NULL;
  }

  CpDType out_dtype = CP_DTYPE_FLOAT64;
  if (!cp_agg_output_dtype(values_series, op, &out_dtype)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid pivot aggregation");
    return NULL;
  }

  size_t index_cap = 8;
  size_t index_count = 0;
  int64_t *index_i64 = NULL;
  const char **index_str = NULL;
  if (index_series->dtype == CP_DTYPE_INT64) {
    index_i64 = (int64_t *)malloc(index_cap * sizeof(int64_t));
  } else {
    index_str = (const char **)malloc(index_cap * sizeof(const char *));
  }

  size_t col_cap = 8;
  size_t col_count = 0;
  int64_t *col_i64 = NULL;
  const char **col_str = NULL;
  if (columns_series->dtype == CP_DTYPE_INT64) {
    col_i64 = (int64_t *)malloc(col_cap * sizeof(int64_t));
  } else {
    col_str = (const char **)malloc(col_cap * sizeof(const char *));
  }

  if ((index_series->dtype == CP_DTYPE_INT64 && !index_i64) ||
      (index_series->dtype == CP_DTYPE_STRING && !index_str) ||
      (columns_series->dtype == CP_DTYPE_INT64 && !col_i64) ||
      (columns_series->dtype == CP_DTYPE_STRING && !col_str)) {
    free(index_i64);
    free(index_str);
    free(col_i64);
    free(col_str);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (cp_join_key_is_null(index_series, row) ||
        cp_join_key_is_null(columns_series, row)) {
      continue;
    }

    if (index_series->dtype == CP_DTYPE_INT64) {
      int64_t key = index_series->data.i64[row];
      size_t idx = 0;
      for (; idx < index_count; ++idx) {
        if (index_i64[idx] == key) {
          break;
        }
      }
      if (idx == index_count) {
        if (index_count == index_cap) {
          size_t new_cap = index_cap * 2;
          int64_t *next = (int64_t *)realloc(index_i64, new_cap * sizeof(int64_t));
          if (!next) {
            cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
            free(index_i64);
            free(index_str);
            free(col_i64);
            free(col_str);
            return NULL;
          }
          index_i64 = next;
          index_cap = new_cap;
        }
        index_i64[index_count++] = key;
      }
    } else {
      const char *key = index_series->data.str[row];
      if (!key) {
        continue;
      }
      size_t idx = 0;
      for (; idx < index_count; ++idx) {
        if (index_str[idx] && strcmp(index_str[idx], key) == 0) {
          break;
        }
      }
      if (idx == index_count) {
        if (index_count == index_cap) {
          size_t new_cap = index_cap * 2;
          const char **next =
              (const char **)realloc(index_str, new_cap * sizeof(const char *));
          if (!next) {
            cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
            free(index_i64);
            free(index_str);
            free(col_i64);
            free(col_str);
            return NULL;
          }
          index_str = next;
          index_cap = new_cap;
        }
        index_str[index_count++] = key;
      }
    }

    if (columns_series->dtype == CP_DTYPE_INT64) {
      int64_t key = columns_series->data.i64[row];
      size_t idx = 0;
      for (; idx < col_count; ++idx) {
        if (col_i64[idx] == key) {
          break;
        }
      }
      if (idx == col_count) {
        if (col_count == col_cap) {
          size_t new_cap = col_cap * 2;
          int64_t *next = (int64_t *)realloc(col_i64, new_cap * sizeof(int64_t));
          if (!next) {
            cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
            free(index_i64);
            free(index_str);
            free(col_i64);
            free(col_str);
            return NULL;
          }
          col_i64 = next;
          col_cap = new_cap;
        }
        col_i64[col_count++] = key;
      }
    } else {
      const char *key = columns_series->data.str[row];
      if (!key) {
        continue;
      }
      size_t idx = 0;
      for (; idx < col_count; ++idx) {
        if (col_str[idx] && strcmp(col_str[idx], key) == 0) {
          break;
        }
      }
      if (idx == col_count) {
        if (col_count == col_cap) {
          size_t new_cap = col_cap * 2;
          const char **next =
              (const char **)realloc(col_str, new_cap * sizeof(const char *));
          if (!next) {
            cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
            free(index_i64);
            free(index_str);
            free(col_i64);
            free(col_str);
            return NULL;
          }
          col_str = next;
          col_cap = new_cap;
        }
        col_str[col_count++] = key;
      }
    }
  }

  if (index_count > 0 && col_count > 0 &&
      index_count > SIZE_MAX / col_count) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "pivot size overflow");
    free(index_i64);
    free(index_str);
    free(col_i64);
    free(col_str);
    return NULL;
  }

  size_t cell_count = index_count * col_count;
  CpAggState *states = NULL;
  if (cell_count > 0) {
    states = (CpAggState *)calloc(cell_count, sizeof(CpAggState));
    if (!states) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      free(index_i64);
      free(index_str);
      free(col_i64);
      free(col_str);
      return NULL;
    }
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (cp_join_key_is_null(index_series, row) ||
        cp_join_key_is_null(columns_series, row)) {
      continue;
    }

    size_t index_idx = index_count;
    if (index_series->dtype == CP_DTYPE_INT64) {
      int64_t key = index_series->data.i64[row];
      for (size_t i = 0; i < index_count; ++i) {
        if (index_i64[i] == key) {
          index_idx = i;
          break;
        }
      }
    } else {
      const char *key = index_series->data.str[row];
      for (size_t i = 0; i < index_count; ++i) {
        if (index_str[i] && key && strcmp(index_str[i], key) == 0) {
          index_idx = i;
          break;
        }
      }
    }
    if (index_idx == index_count) {
      continue;
    }

    size_t col_idx = col_count;
    if (columns_series->dtype == CP_DTYPE_INT64) {
      int64_t key = columns_series->data.i64[row];
      for (size_t i = 0; i < col_count; ++i) {
        if (col_i64[i] == key) {
          col_idx = i;
          break;
        }
      }
    } else {
      const char *key = columns_series->data.str[row];
      for (size_t i = 0; i < col_count; ++i) {
        if (col_str[i] && key && strcmp(col_str[i], key) == 0) {
          col_idx = i;
          break;
        }
      }
    }
    if (col_idx == col_count) {
      continue;
    }

    CpAggState *state = &states[index_idx * col_count + col_idx];
    if (op == CP_AGG_COUNT) {
      if (!values_series->is_null[row]) {
        state->count += 1;
      }
      continue;
    }
    if (values_series->is_null[row]) {
      continue;
    }

    if (values_series->dtype == CP_DTYPE_INT64) {
      int64_t value = values_series->data.i64[row];
      if (op == CP_AGG_SUM || op == CP_AGG_MEAN) {
        if ((value > 0 && state->sum_i64 > INT64_MAX - value) ||
            (value < 0 && state->sum_i64 < INT64_MIN - value)) {
          cp_error_set(err, CP_ERR_INVALID, row, 0, "int64 sum overflow");
          free(states);
          free(index_i64);
          free(index_str);
          free(col_i64);
          free(col_str);
          return NULL;
        }
        state->sum_i64 += value;
        state->count += 1;
        state->has_value = 1;
      } else if (op == CP_AGG_MIN) {
        if (!state->has_value || value < state->min_i64) {
          state->min_i64 = value;
        }
        state->has_value = 1;
      } else if (op == CP_AGG_MAX) {
        if (!state->has_value || value > state->max_i64) {
          state->max_i64 = value;
        }
        state->has_value = 1;
      }
    } else if (values_series->dtype == CP_DTYPE_FLOAT64) {
      double value = values_series->data.f64[row];
      if (op == CP_AGG_SUM || op == CP_AGG_MEAN) {
        state->sum_f64 += value;
        state->count += 1;
        state->has_value = 1;
      } else if (op == CP_AGG_MIN) {
        if (!state->has_value || value < state->min_f64) {
          state->min_f64 = value;
        }
        state->has_value = 1;
      } else if (op == CP_AGG_MAX) {
        if (!state->has_value || value > state->max_f64) {
          state->max_f64 = value;
        }
        state->has_value = 1;
      }
    }
  }

  size_t out_cols = col_count + 1;
  const char **names = (const char **)malloc(out_cols * sizeof(const char *));
  CpDType *dtypes = (CpDType *)malloc(out_cols * sizeof(CpDType));
  char **owned_names = (char **)calloc(col_count, sizeof(char *));
  if (!names || !dtypes || !owned_names) {
    free(names);
    free(dtypes);
    free(owned_names);
    free(states);
    free(index_i64);
    free(index_str);
    free(col_i64);
    free(col_str);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  names[0] = index_series->name ? index_series->name : "index";
  dtypes[0] = index_series->dtype;
  for (size_t col = 0; col < col_count; ++col) {
    char *base = NULL;
    if (columns_series->dtype == CP_DTYPE_INT64) {
      char buf[64];
      snprintf(buf, sizeof(buf), "%" PRId64, col_i64[col]);
      base = cp_strdup(buf);
    } else {
      base = cp_strdup(col_str[col] ? col_str[col] : "");
    }
    if (!base) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      for (size_t i = 0; i < col; ++i) {
        free(owned_names[i]);
      }
      free(names);
      free(dtypes);
      free(owned_names);
      free(states);
      free(index_i64);
      free(index_str);
      free(col_i64);
      free(col_str);
      return NULL;
    }
    int owned = 0;
    const char *name = cp_unique_name_with_suffix(base,
                                                  names,
                                                  col + 1,
                                                  "_col",
                                                  &owned,
                                                  err);
    if (!name) {
      free(base);
      for (size_t i = 0; i < col; ++i) {
        free(owned_names[i]);
      }
      free(names);
      free(dtypes);
      free(owned_names);
      free(states);
      free(index_i64);
      free(index_str);
      free(col_i64);
      free(col_str);
      return NULL;
    }
    if (name != base) {
      free(base);
    }
    names[col + 1] = name;
    owned_names[col] = (char *)name;
    dtypes[col + 1] = out_dtype;
  }

  CpDataFrame *out = cp_df_create(out_cols, names, dtypes, index_count, err);
  for (size_t col = 0; col < col_count; ++col) {
    free(owned_names[col]);
  }
  free(names);
  free(dtypes);
  free(owned_names);

  if (!out) {
    free(states);
    free(index_i64);
    free(index_str);
    free(col_i64);
    free(col_str);
    return NULL;
  }

  int values_is_int = values_series->dtype == CP_DTYPE_INT64;
  for (size_t row = 0; row < index_count; ++row) {
    int ok = 1;
    if (index_series->dtype == CP_DTYPE_INT64) {
      ok = cp_series_append_int64(out->cols[0], index_i64[row], 0, err);
    } else {
      ok = cp_series_append_string(out->cols[0], index_str[row], 0, err);
    }
    if (ok) {
      for (size_t col = 0; col < col_count; ++col) {
        CpAggState *state = cell_count > 0 ? &states[row * col_count + col] : NULL;
        CpSeries *dest = out->cols[col + 1];
        if (op == CP_AGG_COUNT) {
          ok = cp_series_append_int64(dest, (int64_t)state->count, 0, err);
        } else if (op == CP_AGG_MEAN) {
          if (state->count == 0) {
            ok = cp_series_append_float64(dest, 0.0, 1, err);
          } else {
            double mean = values_is_int
                              ? (double)state->sum_i64 / (double)state->count
                              : state->sum_f64 / (double)state->count;
            ok = cp_series_append_float64(dest, mean, 0, err);
          }
        } else if (values_is_int) {
          if (op == CP_AGG_SUM) {
            if (state->count == 0) {
              ok = cp_series_append_int64(dest, 0, 1, err);
            } else {
              ok = cp_series_append_int64(dest, state->sum_i64, 0, err);
            }
          } else if (op == CP_AGG_MIN) {
            if (!state->has_value) {
              ok = cp_series_append_int64(dest, 0, 1, err);
            } else {
              ok = cp_series_append_int64(dest, state->min_i64, 0, err);
            }
          } else if (op == CP_AGG_MAX) {
            if (!state->has_value) {
              ok = cp_series_append_int64(dest, 0, 1, err);
            } else {
              ok = cp_series_append_int64(dest, state->max_i64, 0, err);
            }
          } else {
            ok = 0;
          }
        } else {
          if (op == CP_AGG_SUM) {
            if (state->count == 0) {
              ok = cp_series_append_float64(dest, 0.0, 1, err);
            } else {
              ok = cp_series_append_float64(dest, state->sum_f64, 0, err);
            }
          } else if (op == CP_AGG_MIN) {
            if (!state->has_value) {
              ok = cp_series_append_float64(dest, 0.0, 1, err);
            } else {
              ok = cp_series_append_float64(dest, state->min_f64, 0, err);
            }
          } else if (op == CP_AGG_MAX) {
            if (!state->has_value) {
              ok = cp_series_append_float64(dest, 0.0, 1, err);
            } else {
              ok = cp_series_append_float64(dest, state->max_f64, 0, err);
            }
          } else {
            ok = 0;
          }
        }
        if (!ok) {
          for (size_t j = 0; j < col + 1; ++j) {
            cp_series_pop(out->cols[j]);
          }
          cp_df_free(out);
          out = NULL;
          break;
        }
      }
    }
    if (!out) {
      break;
    }
    out->nrows += 1;
  }

  free(states);
  free(index_i64);
  free(index_str);
  free(col_i64);
  free(col_str);
  return out;
}

int cp_df_mask_int64(const CpDataFrame *df,
                     const char *name,
                     CpCompareOp op,
                     int64_t value,
                     uint8_t *out,
                     size_t out_len,
                     CpError *err) {
  if (!df || !name || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  if (out_len < df->nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output buffer too small");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  if (series->dtype != CP_DTYPE_INT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (series->is_null[row]) {
      out[row] = 0;
      continue;
    }
    int match = 0;
    if (!cp_eval_compare_int64(series->data.i64[row], op, value, &match, err)) {
      return 0;
    }
    out[row] = match ? 1 : 0;
  }
  return 1;
}

int cp_df_mask_float64(const CpDataFrame *df,
                       const char *name,
                       CpCompareOp op,
                       double value,
                       uint8_t *out,
                       size_t out_len,
                       CpError *err) {
  if (!df || !name || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  if (out_len < df->nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output buffer too small");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  if (series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (series->is_null[row]) {
      out[row] = 0;
      continue;
    }
    int match = 0;
    if (!cp_eval_compare_float64(series->data.f64[row], op, value, &match, err)) {
      return 0;
    }
    out[row] = match ? 1 : 0;
  }
  return 1;
}

int cp_df_mask_string(const CpDataFrame *df,
                      const char *name,
                      CpCompareOp op,
                      const char *value,
                      uint8_t *out,
                      size_t out_len,
                      CpError *err) {
  if (!df || !name || !out || !value) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  if (out_len < df->nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output buffer too small");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  if (series->dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (series->is_null[row]) {
      out[row] = 0;
      continue;
    }
    const char *lhs = series->data.str[row] ? series->data.str[row] : "";
    int match = 0;
    if (!cp_eval_compare_string(lhs, op, value, &match, err)) {
      return 0;
    }
    out[row] = match ? 1 : 0;
  }
  return 1;
}

CpDataFrame *cp_df_filter_int64(const CpDataFrame *df,
                                const char *name,
                                CpCompareOp op,
                                int64_t value,
                                CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (df->nrows == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(df->nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  if (!cp_df_mask_int64(df, name, op, value, mask, df->nrows, err)) {
    free(mask);
    return NULL;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, df->nrows, err);
  free(mask);
  return out;
}

CpDataFrame *cp_df_filter_float64(const CpDataFrame *df,
                                  const char *name,
                                  CpCompareOp op,
                                  double value,
                                  CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (df->nrows == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(df->nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  if (!cp_df_mask_float64(df, name, op, value, mask, df->nrows, err)) {
    free(mask);
    return NULL;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, df->nrows, err);
  free(mask);
  return out;
}

CpDataFrame *cp_df_filter_string(const CpDataFrame *df,
                                 const char *name,
                                 CpCompareOp op,
                                 const char *value,
                                 CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (df->nrows == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(df->nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  if (!cp_df_mask_string(df, name, op, value, mask, df->nrows, err)) {
    free(mask);
    return NULL;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, df->nrows, err);
  free(mask);
  return out;
}

CpDataFrame *cp_df_filter_mask(const CpDataFrame *df,
                               const uint8_t *mask,
                               size_t mask_len,
                               CpError *err) {
  if (!df || !mask) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid filter");
    return NULL;
  }
  if (mask_len != df->nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "mask length mismatch");
    return NULL;
  }

  size_t keep = 0;
  for (size_t i = 0; i < mask_len; ++i) {
    if (mask[i]) {
      keep += 1;
    }
  }

  size_t ncols = df->ncols;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(ncols * sizeof(const CpSeries *));
  if (!dtypes || !names || !src_cols) {
    free(dtypes);
    free(names);
    free(src_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < ncols; ++i) {
    src_cols[i] = df->cols[i];
    names[i] = df->cols[i]->name;
    dtypes[i] = df->cols[i]->dtype;
  }

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, keep, err);
  if (!out) {
    free(dtypes);
    free(names);
    free(src_cols);
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    if (!mask[row]) {
      continue;
    }
    if (!cp_df_append_row_from_sources(out, src_cols, ncols, row, err)) {
      cp_df_free(out);
      out = NULL;
      break;
    }
  }

  free(dtypes);
  free(names);
  free(src_cols);
  return out;
}

static int cp_compare_int64(int64_t a, int64_t b) {
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

static int cp_compare_float64(double a, double b) {
  int a_nan = isnan(a);
  int b_nan = isnan(b);
  if (a_nan && b_nan) {
    return 0;
  }
  if (a_nan) {
    return 1;
  }
  if (b_nan) {
    return -1;
  }
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

static int cp_series_compare_values(const CpSeries *s, size_t a, size_t b) {
  switch (s->dtype) {
    case CP_DTYPE_INT64:
      return cp_compare_int64(s->data.i64[a], s->data.i64[b]);
    case CP_DTYPE_FLOAT64:
      return cp_compare_float64(s->data.f64[a], s->data.f64[b]);
    case CP_DTYPE_STRING: {
      const char *av = s->data.str[a] ? s->data.str[a] : "";
      const char *bv = s->data.str[b] ? s->data.str[b] : "";
      int cmp = strcmp(av, bv);
      if (cmp < 0) {
        return -1;
      }
      if (cmp > 0) {
        return 1;
      }
      return 0;
    }
    default:
      return 0;
  }
}

static int cp_series_compare_dir(const CpSeries *s,
                                 size_t a,
                                 size_t b,
                                 int ascending) {
  int a_null = s->is_null[a] ? 1 : 0;
  int b_null = s->is_null[b] ? 1 : 0;
  if (a_null && b_null) {
    return 0;
  }
  if (a_null) {
    return 1;
  }
  if (b_null) {
    return -1;
  }
  int cmp = cp_series_compare_values(s, a, b);
  if (cmp == 0) {
    return 0;
  }
  if (ascending) {
    return cmp;
  }
  return -cmp;
}

static void cp_sort_indices_merge(size_t *indices,
                                  size_t *tmp,
                                  size_t left,
                                  size_t right,
                                  const CpSeries *series,
                                  int ascending) {
  if (right - left <= 1) {
    return;
  }
  size_t mid = left + (right - left) / 2;
  cp_sort_indices_merge(indices, tmp, left, mid, series, ascending);
  cp_sort_indices_merge(indices, tmp, mid, right, series, ascending);

  size_t i = left;
  size_t j = mid;
  size_t k = left;
  while (i < mid && j < right) {
    int cmp = cp_series_compare_dir(series, indices[i], indices[j], ascending);
    if (cmp <= 0) {
      tmp[k++] = indices[i++];
    } else {
      tmp[k++] = indices[j++];
    }
  }
  while (i < mid) {
    tmp[k++] = indices[i++];
  }
  while (j < right) {
    tmp[k++] = indices[j++];
  }
  for (size_t idx = left; idx < right; ++idx) {
    indices[idx] = tmp[idx];
  }
}

static int cp_compare_rows_multi(const CpSeries **keys,
                                 const int *ascending,
                                 size_t key_count,
                                 size_t a,
                                 size_t b) {
  for (size_t i = 0; i < key_count; ++i) {
    int asc = 1;
    if (ascending) {
      asc = ascending[i] != 0;
    }
    int cmp = cp_series_compare_dir(keys[i], a, b, asc);
    if (cmp != 0) {
      return cmp;
    }
  }
  return 0;
}

static void cp_sort_indices_merge_multi(size_t *indices,
                                        size_t *tmp,
                                        size_t left,
                                        size_t right,
                                        const CpSeries **keys,
                                        const int *ascending,
                                        size_t key_count) {
  if (right - left <= 1) {
    return;
  }
  size_t mid = left + (right - left) / 2;
  cp_sort_indices_merge_multi(indices, tmp, left, mid, keys, ascending,
                              key_count);
  cp_sort_indices_merge_multi(indices, tmp, mid, right, keys, ascending,
                              key_count);

  size_t i = left;
  size_t j = mid;
  size_t k = left;
  while (i < mid && j < right) {
    int cmp =
        cp_compare_rows_multi(keys, ascending, key_count, indices[i], indices[j]);
    if (cmp <= 0) {
      tmp[k++] = indices[i++];
    } else {
      tmp[k++] = indices[j++];
    }
  }
  while (i < mid) {
    tmp[k++] = indices[i++];
  }
  while (j < right) {
    tmp[k++] = indices[j++];
  }
  for (size_t idx = left; idx < right; ++idx) {
    indices[idx] = tmp[idx];
  }
}

CpDataFrame *cp_df_sort_values_multi(const CpDataFrame *df,
                                     const char **names,
                                     size_t count,
                                     const int *ascending,
                                     CpError *err) {
  if (!df || !names || count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid sort keys");
    return NULL;
  }

  const CpSeries **keys = (const CpSeries **)malloc(count * sizeof(const CpSeries *));
  if (!keys) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < count; ++i) {
    const CpSeries *series = cp_df_require_col(df, names[i], err);
    if (!series) {
      free(keys);
      return NULL;
    }
    if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64 &&
        series->dtype != CP_DTYPE_STRING) {
      free(keys);
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported sort dtype");
      return NULL;
    }
    keys[i] = series;
  }

  size_t nrows = df->nrows;
  size_t ncols = df->ncols;
  size_t *indices = NULL;
  size_t *tmp = NULL;
  if (nrows > 0) {
    indices = (size_t *)malloc(nrows * sizeof(size_t));
    tmp = (size_t *)malloc(nrows * sizeof(size_t));
    if (!indices || !tmp) {
      free(indices);
      free(tmp);
      free(keys);
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    for (size_t i = 0; i < nrows; ++i) {
      indices[i] = i;
    }
    if (nrows > 1) {
      cp_sort_indices_merge_multi(indices, tmp, 0, nrows, keys, ascending,
                                  count);
    }
  }

  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **out_names = (const char **)malloc(ncols * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(ncols * sizeof(const CpSeries *));
  if (!dtypes || !out_names || !src_cols) {
    free(dtypes);
    free(out_names);
    free(src_cols);
    free(indices);
    free(tmp);
    free(keys);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t i = 0; i < ncols; ++i) {
    src_cols[i] = df->cols[i];
    out_names[i] = df->cols[i]->name;
    dtypes[i] = df->cols[i]->dtype;
  }

  CpDataFrame *out = cp_df_create(ncols, out_names, dtypes, nrows, err);
  if (!out) {
    free(dtypes);
    free(out_names);
    free(src_cols);
    free(indices);
    free(tmp);
    free(keys);
    return NULL;
  }

  for (size_t pos = 0; pos < nrows; ++pos) {
    size_t row = indices ? indices[pos] : pos;
    if (!cp_df_append_row_from_sources(out, src_cols, ncols, row, err)) {
      cp_df_free(out);
      out = NULL;
      break;
    }
  }

  free(dtypes);
  free(out_names);
  free(src_cols);
  free(indices);
  free(tmp);
  free(keys);
  return out;
}

CpDataFrame *cp_df_sort_values(const CpDataFrame *df,
                               const char *name,
                               int ascending,
                               CpError *err) {
  const char *names[1] = {name};
  int asc[1] = {ascending};
  return cp_df_sort_values_multi(df, names, 1, asc, err);
}

int cp_df_append_row(CpDataFrame *df,
                     const char **values,
                     size_t nvalues,
                     CpError *err) {
  if (!df || !values || nvalues != df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid row data");
    return 0;
  }
  size_t row = df->nrows;
  for (size_t i = 0; i < df->ncols; ++i) {
    CpSeries *col = df->cols[i];
    int ok = 0;
    switch (col->dtype) {
      case CP_DTYPE_INT64: {
        int64_t v = 0;
        int is_null = 0;
        ok = cp_parse_int64(values[i], &v, &is_null, err, row, i);
        if (ok) {
          ok = cp_series_append_int64(col, v, is_null, err);
        }
        break;
      }
      case CP_DTYPE_FLOAT64: {
        double v = 0.0;
        int is_null = 0;
        ok = cp_parse_float64(values[i], &v, &is_null, err, row, i);
        if (ok) {
          ok = cp_series_append_float64(col, v, is_null, err);
        }
        break;
      }
      case CP_DTYPE_STRING: {
        const char *v = NULL;
        int is_null = 0;
        ok = cp_parse_string(values[i], &v, &is_null);
        if (ok) {
          ok = cp_series_append_string(col, v, is_null, err);
        }
        break;
      }
      default:
        cp_error_set(err, CP_ERR_INVALID, row, i, "unknown dtype");
        ok = 0;
        break;
    }
    if (!ok) {
      for (size_t j = 0; j < i; ++j) {
        cp_series_pop(df->cols[j]);
      }
      return 0;
    }
  }
  df->nrows += 1;
  return 1;
}

static int cp_is_line_blank(const char *line) {
  if (!line) {
    return 1;
  }
  return cp_is_blank(line);
}

static char **cp_make_default_names(size_t ncols, CpError *err) {
  char **names = (char **)calloc(ncols, sizeof(char *));
  if (!names) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t i = 0; i < ncols; ++i) {
    char buf[32];
    snprintf(buf, sizeof(buf), "col%zu", i);
    names[i] = cp_strdup(buf);
    if (!names[i]) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      for (size_t j = 0; j < i; ++j) {
        free(names[j]);
      }
      free(names);
      return NULL;
    }
  }
  return names;
}

CpDataFrame *cp_df_read_csv(const char *path,
                            char delimiter,
                            int has_header,
                            const CpDType *dtypes,
                            size_t dtype_count,
                            CpError *err) {
  if (!path) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "path is required");
    return NULL;
  }
  FILE *fp = fopen(path, "r");
  if (!fp) {
    cp_error_set(err, CP_ERR_IO, 0, 0, "failed to open file");
    return NULL;
  }

  char *line = NULL;
  while ((line = cp_read_line(fp, err)) != NULL) {
    if (!cp_is_line_blank(line)) {
      break;
    }
    free(line);
  }

  if (!line) {
    fclose(fp);
    cp_error_set(err, CP_ERR_PARSE, 0, 0, "empty csv");
    return NULL;
  }

  char **fields = NULL;
  size_t ncols = 0;
  if (!cp_parse_csv_line(line, delimiter, &fields, &ncols, err)) {
    free(line);
    fclose(fp);
    return NULL;
  }
  free(line);

  if (ncols == 0) {
    cp_free_fields(fields, ncols);
    fclose(fp);
    cp_error_set(err, CP_ERR_PARSE, 0, 0, "no columns found");
    return NULL;
  }

  char **col_names = NULL;
  if (has_header) {
    col_names = fields;
  } else {
    col_names = cp_make_default_names(ncols, err);
    if (!col_names) {
      cp_free_fields(fields, ncols);
      fclose(fp);
      return NULL;
    }
  }

  CpDType *local_dtypes = NULL;
  if (dtypes) {
    if (dtype_count != ncols) {
      if (!has_header) {
        cp_free_fields(col_names, ncols);
        cp_free_fields(fields, ncols);
      } else {
        cp_free_fields(fields, ncols);
      }
      fclose(fp);
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype count mismatch");
      return NULL;
    }
  } else {
    local_dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
    if (!local_dtypes) {
      if (!has_header) {
        cp_free_fields(col_names, ncols);
        cp_free_fields(fields, ncols);
      } else {
        cp_free_fields(fields, ncols);
      }
      fclose(fp);
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    for (size_t i = 0; i < ncols; ++i) {
      local_dtypes[i] = CP_DTYPE_STRING;
    }
    dtypes = local_dtypes;
    dtype_count = ncols;
  }

  const char **name_ptrs = (const char **)col_names;
  CpDataFrame *df = cp_df_create(ncols, name_ptrs, dtypes, 0, err);
  if (!df) {
    if (!has_header) {
      cp_free_fields(col_names, ncols);
      cp_free_fields(fields, ncols);
    } else {
      cp_free_fields(fields, ncols);
    }
    free(local_dtypes);
    fclose(fp);
    return NULL;
  }

  if (has_header) {
    cp_free_fields(fields, ncols);
  } else {
    cp_free_fields(col_names, ncols);
    const char **row_values = (const char **)fields;
    if (!cp_df_append_row(df, row_values, ncols, err)) {
      cp_free_fields(fields, ncols);
      cp_df_free(df);
      free(local_dtypes);
      fclose(fp);
      return NULL;
    }
    cp_free_fields(fields, ncols);
  }

  size_t line_no = 1;
  while ((line = cp_read_line(fp, err)) != NULL) {
    line_no += 1;
    if (cp_is_line_blank(line)) {
      free(line);
      continue;
    }
    char **row_fields = NULL;
    size_t field_count = 0;
    if (!cp_parse_csv_line(line, delimiter, &row_fields, &field_count, err)) {
      free(line);
      cp_df_free(df);
      free(local_dtypes);
      fclose(fp);
      return NULL;
    }
    free(line);

    if (field_count != ncols) {
      cp_error_set(err, CP_ERR_PARSE, df->nrows, 0,
                   "column count mismatch on line %zu", line_no);
      cp_free_fields(row_fields, field_count);
      cp_df_free(df);
      free(local_dtypes);
      fclose(fp);
      return NULL;
    }

    const char **row_values = (const char **)row_fields;
    if (!cp_df_append_row(df, row_values, field_count, err)) {
      cp_free_fields(row_fields, field_count);
      cp_df_free(df);
      free(local_dtypes);
      fclose(fp);
      return NULL;
    }
    cp_free_fields(row_fields, field_count);
  }

  free(local_dtypes);
  fclose(fp);
  return df;
}

static int cp_write_csv_field(FILE *fp, const char *s, char delimiter) {
  int needs_quotes = 0;
  for (const char *p = s; *p; ++p) {
    if (*p == delimiter || *p == '"' || *p == '\n' || *p == '\r') {
      needs_quotes = 1;
      break;
    }
  }
  if (!needs_quotes) {
    return fputs(s, fp) >= 0;
  }
  if (fputc('"', fp) == EOF) {
    return 0;
  }
  for (const char *p = s; *p; ++p) {
    if (*p == '"') {
      if (fputc('"', fp) == EOF) {
        return 0;
      }
    }
    if (fputc(*p, fp) == EOF) {
      return 0;
    }
  }
  return fputc('"', fp) != EOF;
}

int cp_df_write_csv(const CpDataFrame *df,
                    const char *path,
                    char delimiter,
                    int include_header,
                    CpError *err) {
  if (!df || !path) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  FILE *fp = fopen(path, "w");
  if (!fp) {
    cp_error_set(err, CP_ERR_IO, 0, 0, "failed to open file");
    return 0;
  }

  if (include_header) {
    for (size_t i = 0; i < df->ncols; ++i) {
      if (i > 0 && fputc(delimiter, fp) == EOF) {
        cp_error_set(err, CP_ERR_IO, 0, 0, "failed to write header");
        fclose(fp);
        return 0;
      }
      const char *name = df->cols[i]->name ? df->cols[i]->name : "";
      if (!cp_write_csv_field(fp, name, delimiter)) {
        cp_error_set(err, CP_ERR_IO, 0, 0, "failed to write header");
        fclose(fp);
        return 0;
      }
    }
    if (fputc('\n', fp) == EOF) {
      cp_error_set(err, CP_ERR_IO, 0, 0, "failed to write header");
      fclose(fp);
      return 0;
    }
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    for (size_t col = 0; col < df->ncols; ++col) {
      if (col > 0 && fputc(delimiter, fp) == EOF) {
        cp_error_set(err, CP_ERR_IO, row, col, "failed to write csv");
        fclose(fp);
        return 0;
      }
      CpSeries *series = df->cols[col];
      if (series->is_null[row]) {
        continue;
      }
      switch (series->dtype) {
        case CP_DTYPE_INT64: {
          if (fprintf(fp, "%" PRId64, series->data.i64[row]) < 0) {
            cp_error_set(err, CP_ERR_IO, row, col, "failed to write csv");
            fclose(fp);
            return 0;
          }
          break;
        }
        case CP_DTYPE_FLOAT64: {
          if (fprintf(fp, "%.17g", series->data.f64[row]) < 0) {
            cp_error_set(err, CP_ERR_IO, row, col, "failed to write csv");
            fclose(fp);
            return 0;
          }
          break;
        }
        case CP_DTYPE_STRING: {
          const char *value = series->data.str[row];
          if (!value) {
            break;
          }
          if (!cp_write_csv_field(fp, value, delimiter)) {
            cp_error_set(err, CP_ERR_IO, row, col, "failed to write csv");
            fclose(fp);
            return 0;
          }
          break;
        }
        default:
          cp_error_set(err, CP_ERR_INVALID, row, col, "unknown dtype");
          fclose(fp);
          return 0;
      }
    }
    if (fputc('\n', fp) == EOF) {
      cp_error_set(err, CP_ERR_IO, row, 0, "failed to write csv");
      fclose(fp);
      return 0;
    }
  }

  fclose(fp);
  return 1;
}

const char *cp_series_name(const CpSeries *s) {
  return s ? s->name : NULL;
}

CpDType cp_series_dtype(const CpSeries *s) {
  return s ? s->dtype : CP_DTYPE_STRING;
}

size_t cp_series_len(const CpSeries *s) {
  return s ? s->length : 0;
}

int cp_series_get_int64(const CpSeries *s,
                        size_t idx,
                        int64_t *out,
                        int *is_null) {
  if (!s || s->dtype != CP_DTYPE_INT64 || idx >= s->length) {
    return 0;
  }
  if (out) {
    *out = s->data.i64[idx];
  }
  if (is_null) {
    *is_null = s->is_null[idx] ? 1 : 0;
  }
  return 1;
}

int cp_series_get_float64(const CpSeries *s,
                          size_t idx,
                          double *out,
                          int *is_null) {
  if (!s || s->dtype != CP_DTYPE_FLOAT64 || idx >= s->length) {
    return 0;
  }
  if (out) {
    *out = s->data.f64[idx];
  }
  if (is_null) {
    *is_null = s->is_null[idx] ? 1 : 0;
  }
  return 1;
}

int cp_series_get_string(const CpSeries *s,
                         size_t idx,
                         const char **out,
                         int *is_null) {
  if (!s || s->dtype != CP_DTYPE_STRING || idx >= s->length) {
    return 0;
  }
  if (out) {
    *out = s->data.str[idx];
  }
  if (is_null) {
    *is_null = s->is_null[idx] ? 1 : 0;
  }
  return 1;
}

int cp_series_count(const CpSeries *s, size_t *out, size_t *out_nulls, CpError *err) {
  if (!s) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid series");
    return 0;
  }
  size_t count = 0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
    } else {
      count += 1;
    }
  }
  if (out) {
    *out = count;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_sum_int64(const CpSeries *s,
                        int64_t *out,
                        size_t *out_count,
                        size_t *out_nulls,
                        CpError *err) {
  if (!s || s->dtype != CP_DTYPE_INT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  int64_t sum = 0;
  size_t count = 0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
      continue;
    }
    int64_t value = s->data.i64[i];
    if ((value > 0 && sum > INT64_MAX - value) ||
        (value < 0 && sum < INT64_MIN - value)) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "int64 sum overflow");
      return 0;
    }
    sum += value;
    count += 1;
  }
  if (out) {
    *out = sum;
  }
  if (out_count) {
    *out_count = count;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_sum_float64(const CpSeries *s,
                          double *out,
                          size_t *out_count,
                          size_t *out_nulls,
                          CpError *err) {
  if (!s || s->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  double sum = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
      continue;
    }
    sum += s->data.f64[i];
    count += 1;
  }
  if (out) {
    *out = sum;
  }
  if (out_count) {
    *out_count = count;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_mean(const CpSeries *s,
                   double *out,
                   size_t *out_count,
                   size_t *out_nulls,
                   CpError *err) {
  if (!s || (s->dtype != CP_DTYPE_INT64 && s->dtype != CP_DTYPE_FLOAT64)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  double sum = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  if (s->dtype == CP_DTYPE_INT64) {
    for (size_t i = 0; i < s->length; ++i) {
      if (s->is_null[i]) {
        nulls += 1;
        continue;
      }
      sum += (double)s->data.i64[i];
      count += 1;
    }
  } else {
    for (size_t i = 0; i < s->length; ++i) {
      if (s->is_null[i]) {
        nulls += 1;
        continue;
      }
      sum += s->data.f64[i];
      count += 1;
    }
  }
  if (count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "mean of empty series");
    return 0;
  }
  if (out) {
    *out = sum / (double)count;
  }
  if (out_count) {
    *out_count = count;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_min_int64(const CpSeries *s,
                        int64_t *out,
                        size_t *out_nulls,
                        CpError *err) {
  if (!s || s->dtype != CP_DTYPE_INT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  int found = 0;
  int64_t min_val = 0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
      continue;
    }
    if (!found) {
      min_val = s->data.i64[i];
      found = 1;
    } else if (s->data.i64[i] < min_val) {
      min_val = s->data.i64[i];
    }
  }
  if (!found) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "min of empty series");
    return 0;
  }
  if (out) {
    *out = min_val;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_max_int64(const CpSeries *s,
                        int64_t *out,
                        size_t *out_nulls,
                        CpError *err) {
  if (!s || s->dtype != CP_DTYPE_INT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  int found = 0;
  int64_t max_val = 0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
      continue;
    }
    if (!found) {
      max_val = s->data.i64[i];
      found = 1;
    } else if (s->data.i64[i] > max_val) {
      max_val = s->data.i64[i];
    }
  }
  if (!found) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "max of empty series");
    return 0;
  }
  if (out) {
    *out = max_val;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_min_float64(const CpSeries *s,
                          double *out,
                          size_t *out_nulls,
                          CpError *err) {
  if (!s || s->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  int found = 0;
  double min_val = 0.0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
      continue;
    }
    if (!found) {
      min_val = s->data.f64[i];
      found = 1;
    } else if (s->data.f64[i] < min_val) {
      min_val = s->data.f64[i];
    }
  }
  if (!found) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "min of empty series");
    return 0;
  }
  if (out) {
    *out = min_val;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_series_max_float64(const CpSeries *s,
                          double *out,
                          size_t *out_nulls,
                          CpError *err) {
  if (!s || s->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  int found = 0;
  double max_val = 0.0;
  size_t nulls = 0;
  for (size_t i = 0; i < s->length; ++i) {
    if (s->is_null[i]) {
      nulls += 1;
      continue;
    }
    if (!found) {
      max_val = s->data.f64[i];
      found = 1;
    } else if (s->data.f64[i] > max_val) {
      max_val = s->data.f64[i];
    }
  }
  if (!found) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "max of empty series");
    return 0;
  }
  if (out) {
    *out = max_val;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

int cp_df_count(const CpDataFrame *df,
                const char *name,
                size_t *out,
                size_t *out_nulls,
                CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_count(series, out, out_nulls, err);
}

int cp_df_sum_int64(const CpDataFrame *df,
                    const char *name,
                    int64_t *out,
                    size_t *out_count,
                    size_t *out_nulls,
                    CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_sum_int64(series, out, out_count, out_nulls, err);
}

int cp_df_sum_float64(const CpDataFrame *df,
                      const char *name,
                      double *out,
                      size_t *out_count,
                      size_t *out_nulls,
                      CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_sum_float64(series, out, out_count, out_nulls, err);
}

int cp_df_mean(const CpDataFrame *df,
               const char *name,
               double *out,
               size_t *out_count,
               size_t *out_nulls,
               CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_mean(series, out, out_count, out_nulls, err);
}

int cp_df_min_int64(const CpDataFrame *df,
                    const char *name,
                    int64_t *out,
                    size_t *out_nulls,
                    CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_min_int64(series, out, out_nulls, err);
}

int cp_df_max_int64(const CpDataFrame *df,
                    const char *name,
                    int64_t *out,
                    size_t *out_nulls,
                    CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_max_int64(series, out, out_nulls, err);
}

int cp_df_min_float64(const CpDataFrame *df,
                      const char *name,
                      double *out,
                      size_t *out_nulls,
                      CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_min_float64(series, out, out_nulls, err);
}

int cp_df_max_float64(const CpDataFrame *df,
                      const char *name,
                      double *out,
                      size_t *out_nulls,
                      CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_max_float64(series, out, out_nulls, err);
}

int cp_df_count_at(const CpDataFrame *df,
                   size_t col_idx,
                   size_t *out,
                   size_t *out_nulls,
                   CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_count(series, out, out_nulls, err);
}

int cp_df_sum_int64_at(const CpDataFrame *df,
                       size_t col_idx,
                       int64_t *out,
                       size_t *out_count,
                       size_t *out_nulls,
                       CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_sum_int64(series, out, out_count, out_nulls, err);
}

int cp_df_sum_float64_at(const CpDataFrame *df,
                         size_t col_idx,
                         double *out,
                         size_t *out_count,
                         size_t *out_nulls,
                         CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_sum_float64(series, out, out_count, out_nulls, err);
}

int cp_df_mean_at(const CpDataFrame *df,
                  size_t col_idx,
                  double *out,
                  size_t *out_count,
                  size_t *out_nulls,
                  CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_mean(series, out, out_count, out_nulls, err);
}

int cp_df_min_int64_at(const CpDataFrame *df,
                       size_t col_idx,
                       int64_t *out,
                       size_t *out_nulls,
                       CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_min_int64(series, out, out_nulls, err);
}

int cp_df_max_int64_at(const CpDataFrame *df,
                       size_t col_idx,
                       int64_t *out,
                       size_t *out_nulls,
                       CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_max_int64(series, out, out_nulls, err);
}

int cp_df_min_float64_at(const CpDataFrame *df,
                         size_t col_idx,
                         double *out,
                         size_t *out_nulls,
                         CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_min_float64(series, out, out_nulls, err);
}

int cp_df_max_float64_at(const CpDataFrame *df,
                         size_t col_idx,
                         double *out,
                         size_t *out_nulls,
                         CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_max_float64(series, out, out_nulls, err);
}

static int cp_agg_require_out_int64(CpAggInt64 *out, CpError *err) {
  if (!out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output is required");
    return 0;
  }
  return 1;
}

static int cp_agg_require_out_float64(CpAggFloat64 *out, CpError *err) {
  if (!out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output is required");
    return 0;
  }
  return 1;
}

int cp_df_sum_int64_result(const CpDataFrame *df,
                           const char *name,
                           CpAggInt64 *out,
                           CpError *err) {
  if (!cp_agg_require_out_int64(out, err)) {
    return 0;
  }
  int64_t value = 0;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_df_sum_int64(df, name, &value, &count, &nulls, err)) {
    return 0;
  }
  out->value = value;
  out->count = count;
  out->nulls = nulls;
  return 1;
}

int cp_df_sum_float64_result(const CpDataFrame *df,
                             const char *name,
                             CpAggFloat64 *out,
                             CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  double value = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_df_sum_float64(df, name, &value, &count, &nulls, err)) {
    return 0;
  }
  out->value = value;
  out->count = count;
  out->nulls = nulls;
  return 1;
}

int cp_df_mean_result(const CpDataFrame *df,
                      const char *name,
                      CpAggFloat64 *out,
                      CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  double value = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_df_mean(df, name, &value, &count, &nulls, err)) {
    return 0;
  }
  out->value = value;
  out->count = count;
  out->nulls = nulls;
  return 1;
}

int cp_df_min_int64_result(const CpDataFrame *df,
                           const char *name,
                           CpAggInt64 *out,
                           CpError *err) {
  if (!cp_agg_require_out_int64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  int64_t value = 0;
  size_t nulls = 0;
  if (!cp_series_min_int64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_max_int64_result(const CpDataFrame *df,
                           const char *name,
                           CpAggInt64 *out,
                           CpError *err) {
  if (!cp_agg_require_out_int64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  int64_t value = 0;
  size_t nulls = 0;
  if (!cp_series_max_int64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_min_float64_result(const CpDataFrame *df,
                             const char *name,
                             CpAggFloat64 *out,
                             CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  double value = 0.0;
  size_t nulls = 0;
  if (!cp_series_min_float64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_max_float64_result(const CpDataFrame *df,
                             const char *name,
                             CpAggFloat64 *out,
                             CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  double value = 0.0;
  size_t nulls = 0;
  if (!cp_series_max_float64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_sum_int64_result_at(const CpDataFrame *df,
                              size_t col_idx,
                              CpAggInt64 *out,
                              CpError *err) {
  if (!cp_agg_require_out_int64(out, err)) {
    return 0;
  }
  int64_t value = 0;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_df_sum_int64_at(df, col_idx, &value, &count, &nulls, err)) {
    return 0;
  }
  out->value = value;
  out->count = count;
  out->nulls = nulls;
  return 1;
}

int cp_df_sum_float64_result_at(const CpDataFrame *df,
                                size_t col_idx,
                                CpAggFloat64 *out,
                                CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  double value = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_df_sum_float64_at(df, col_idx, &value, &count, &nulls, err)) {
    return 0;
  }
  out->value = value;
  out->count = count;
  out->nulls = nulls;
  return 1;
}

int cp_df_mean_result_at(const CpDataFrame *df,
                         size_t col_idx,
                         CpAggFloat64 *out,
                         CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  double value = 0.0;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_df_mean_at(df, col_idx, &value, &count, &nulls, err)) {
    return 0;
  }
  out->value = value;
  out->count = count;
  out->nulls = nulls;
  return 1;
}

int cp_df_min_int64_result_at(const CpDataFrame *df,
                              size_t col_idx,
                              CpAggInt64 *out,
                              CpError *err) {
  if (!cp_agg_require_out_int64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  int64_t value = 0;
  size_t nulls = 0;
  if (!cp_series_min_int64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_max_int64_result_at(const CpDataFrame *df,
                              size_t col_idx,
                              CpAggInt64 *out,
                              CpError *err) {
  if (!cp_agg_require_out_int64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  int64_t value = 0;
  size_t nulls = 0;
  if (!cp_series_max_int64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_min_float64_result_at(const CpDataFrame *df,
                                size_t col_idx,
                                CpAggFloat64 *out,
                                CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  double value = 0.0;
  size_t nulls = 0;
  if (!cp_series_min_float64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}

int cp_df_max_float64_result_at(const CpDataFrame *df,
                                size_t col_idx,
                                CpAggFloat64 *out,
                                CpError *err) {
  if (!cp_agg_require_out_float64(out, err)) {
    return 0;
  }
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  double value = 0.0;
  size_t nulls = 0;
  if (!cp_series_max_float64(series, &value, &nulls, err)) {
    return 0;
  }
  size_t len = cp_series_len(series);
  out->value = value;
  out->nulls = nulls;
  out->count = len >= nulls ? len - nulls : 0;
  return 1;
}
