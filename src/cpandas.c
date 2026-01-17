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
  int has_index;
  size_t index_col;
};

CpDataFrame *cp_df_filter_mask(const CpDataFrame *df,
                               const uint8_t *mask,
                               size_t mask_len,
                               CpError *err);
static int cp_df_append_row_from_sources(CpDataFrame *df,
                                         const CpSeries **src_cols,
                                         size_t ncols,
                                         size_t row,
                                         CpError *err);
static const CpSeries *cp_df_require_col(const CpDataFrame *df,
                                         const char *name,
                                         CpError *err);
static void cp_sort_indices_merge(size_t *indices,
                                  size_t *tmp,
                                  size_t left,
                                  size_t right,
                                  const CpSeries *series,
                                  int ascending);
static void cp_sort_indices_merge_multi(size_t *indices,
                                        size_t *tmp,
                                        size_t left,
                                        size_t right,
                                        const CpSeries **keys,
                                        const int *ascending,
                                        size_t key_count);

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
static char *cp_strndup(const char *s, size_t len);
static int cp_parse_int64(const char *s,
                          int64_t *out,
                          int *is_null,
                          CpError *err,
                          size_t row,
                          size_t col);
static int cp_parse_float64(const char *s,
                            double *out,
                            int *is_null,
                            CpError *err,
                            size_t row,
                            size_t col);
static int cp_parse_string(const char *s,
                           const char **out,
                           int *is_null);

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

static uint32_t cp_rand_next(uint32_t *state) {
  uint32_t x = state ? *state : 0;
  if (x == 0) {
    x = 0x6d2b79f5u;
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  if (state) {
    *state = x;
  }
  return x;
}

static int cp_series_is_nan(const CpSeries *series, size_t idx) {
  if (!series || series->dtype != CP_DTYPE_FLOAT64) {
    return 0;
  }
  if (idx >= series->length || series->is_null[idx]) {
    return 0;
  }
  return isnan(series->data.f64[idx]) ? 1 : 0;
}

static int cp_series_is_valid_numeric(const CpSeries *series, size_t idx) {
  if (!series || idx >= series->length) {
    return 0;
  }
  if (series->is_null[idx]) {
    return 0;
  }
  if (series->dtype == CP_DTYPE_FLOAT64 && cp_series_is_nan(series, idx)) {
    return 0;
  }
  return 1;
}

static int cp_series_get_numeric(const CpSeries *series,
                                 size_t idx,
                                 double *out) {
  if (!series || !out || idx >= series->length) {
    return 0;
  }
  if (series->is_null[idx]) {
    return 0;
  }
  switch (series->dtype) {
    case CP_DTYPE_INT64:
      *out = (double)series->data.i64[idx];
      return 1;
    case CP_DTYPE_FLOAT64:
      if (cp_series_is_nan(series, idx)) {
        return 0;
      }
      *out = series->data.f64[idx];
      return 1;
    default:
      return 0;
  }
}

static int cp_series_collect_numeric(const CpSeries *series,
                                     double **out_vals,
                                     size_t *out_count,
                                     size_t *out_nulls,
                                     CpError *err) {
  if (!series || !out_vals || !out_count || !out_nulls) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid numeric series");
    return 0;
  }
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  size_t len = series->length;
  double *vals = NULL;
  if (len > 0) {
    vals = (double *)malloc(len * sizeof(double));
    if (!vals) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return 0;
    }
  }
  size_t count = 0;
  size_t nulls = 0;
  for (size_t i = 0; i < len; ++i) {
    double v = 0.0;
    if (!cp_series_get_numeric(series, i, &v)) {
      nulls += 1;
      continue;
    }
    vals[count++] = v;
  }
  *out_vals = vals;
  *out_count = count;
  *out_nulls = nulls;
  return 1;
}

static int cp_compare_double(const void *a, const void *b) {
  double av = *(const double *)a;
  double bv = *(const double *)b;
  if (av < bv) {
    return -1;
  }
  if (av > bv) {
    return 1;
  }
  return 0;
}

static int cp_series_median(const CpSeries *series,
                            double *out,
                            size_t *out_count,
                            size_t *out_nulls,
                            CpError *err) {
  double *vals = NULL;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_series_collect_numeric(series, &vals, &count, &nulls, err)) {
    return 0;
  }
  if (count == 0) {
    free(vals);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "median of empty series");
    return 0;
  }
  qsort(vals, count, sizeof(double), cp_compare_double);
  double median = 0.0;
  if (count % 2 == 1) {
    median = vals[count / 2];
  } else {
    median = (vals[count / 2 - 1] + vals[count / 2]) / 2.0;
  }
  free(vals);
  if (out) {
    *out = median;
  }
  if (out_count) {
    *out_count = count;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

static int cp_series_std(const CpSeries *series,
                         double *out,
                         size_t *out_count,
                         size_t *out_nulls,
                         CpError *err) {
  double *vals = NULL;
  size_t count = 0;
  size_t nulls = 0;
  if (!cp_series_collect_numeric(series, &vals, &count, &nulls, err)) {
    return 0;
  }
  if (count == 0) {
    free(vals);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "std of empty series");
    return 0;
  }
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += vals[i];
  }
  double mean = sum / (double)count;
  double sq_sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double diff = vals[i] - mean;
    sq_sum += diff * diff;
  }
  free(vals);
  double std = 0.0;
  if (count > 1) {
    std = sqrt(sq_sum / (double)(count - 1));
  }
  if (out) {
    *out = std;
  }
  if (out_count) {
    *out_count = count;
  }
  if (out_nulls) {
    *out_nulls = nulls;
  }
  return 1;
}

static const char *cp_skip_space(const char *s) {
  if (!s) {
    return NULL;
  }
  while (*s && isspace((unsigned char)*s)) {
    s++;
  }
  return s;
}

static int cp_is_ident_char(int ch) {
  return isalnum((unsigned char)ch) || ch == '_' || ch == '.';
}

static int cp_str_eq_ci(const char *a, const char *b) {
  if (!a || !b) {
    return 0;
  }
  while (*a && *b) {
    if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
      return 0;
    }
    a++;
    b++;
  }
  return *a == '\0' && *b == '\0';
}

typedef enum {
  CP_QUERY_NODE_PRED = 0,
  CP_QUERY_NODE_AND = 1,
  CP_QUERY_NODE_OR = 2
} CpQueryNodeType;

typedef struct CpQueryNode {
  CpQueryNodeType type;
  struct CpQueryNode *left;
  struct CpQueryNode *right;
  const CpSeries *series;
  char *col;
  char *value;
  int value_quoted;
  int is_null_literal;
  int is_nan_literal;
  CpCompareOp op;
  int64_t i64_value;
  double f64_value;
} CpQueryNode;

static void cp_query_node_free(CpQueryNode *node) {
  if (!node) {
    return;
  }
  cp_query_node_free(node->left);
  cp_query_node_free(node->right);
  free(node->col);
  free(node->value);
  free(node);
}

static int cp_query_match_keyword(const char **p, const char *kw) {
  if (!p || !*p || !kw) {
    return 0;
  }
  const char *s = cp_skip_space(*p);
  if (!s) {
    return 0;
  }
  size_t len = strlen(kw);
  for (size_t i = 0; i < len; ++i) {
    if (s[i] == '\0') {
      return 0;
    }
    if (tolower((unsigned char)s[i]) != tolower((unsigned char)kw[i])) {
      return 0;
    }
  }
  if (s[len] != '\0' && cp_is_ident_char(s[len])) {
    return 0;
  }
  *p = s + len;
  return 1;
}

static CpQueryNode *cp_query_parse_expr(const CpDataFrame *df,
                                        const char **p,
                                        CpError *err);
static CpQueryNode *cp_query_parse_or(const CpDataFrame *df,
                                      const char **p,
                                      CpError *err);
static CpQueryNode *cp_query_parse_and(const CpDataFrame *df,
                                       const char **p,
                                       CpError *err);
static CpQueryNode *cp_query_parse_term(const CpDataFrame *df,
                                        const char **p,
                                        CpError *err);
static CpQueryNode *cp_query_parse_pred(const CpDataFrame *df,
                                        const char **p,
                                        CpError *err);

static CpQueryNode *cp_query_parse_expr(const CpDataFrame *df,
                                        const char **p,
                                        CpError *err) {
  return cp_query_parse_or(df, p, err);
}

static CpQueryNode *cp_query_parse_or(const CpDataFrame *df,
                                      const char **p,
                                      CpError *err) {
  CpQueryNode *node = cp_query_parse_and(df, p, err);
  if (!node) {
    return NULL;
  }
  for (;;) {
    const char *cursor = *p;
    if (!cp_query_match_keyword(&cursor, "or")) {
      break;
    }
    *p = cursor;
    CpQueryNode *rhs = cp_query_parse_and(df, p, err);
    if (!rhs) {
      cp_query_node_free(node);
      return NULL;
    }
    CpQueryNode *parent = (CpQueryNode *)calloc(1, sizeof(CpQueryNode));
    if (!parent) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      cp_query_node_free(rhs);
      cp_query_node_free(node);
      return NULL;
    }
    parent->type = CP_QUERY_NODE_OR;
    parent->left = node;
    parent->right = rhs;
    node = parent;
  }
  return node;
}

static CpQueryNode *cp_query_parse_and(const CpDataFrame *df,
                                       const char **p,
                                       CpError *err) {
  CpQueryNode *node = cp_query_parse_term(df, p, err);
  if (!node) {
    return NULL;
  }
  for (;;) {
    const char *cursor = *p;
    if (!cp_query_match_keyword(&cursor, "and")) {
      break;
    }
    *p = cursor;
    CpQueryNode *rhs = cp_query_parse_term(df, p, err);
    if (!rhs) {
      cp_query_node_free(node);
      return NULL;
    }
    CpQueryNode *parent = (CpQueryNode *)calloc(1, sizeof(CpQueryNode));
    if (!parent) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      cp_query_node_free(rhs);
      cp_query_node_free(node);
      return NULL;
    }
    parent->type = CP_QUERY_NODE_AND;
    parent->left = node;
    parent->right = rhs;
    node = parent;
  }
  return node;
}

static CpQueryNode *cp_query_parse_term(const CpDataFrame *df,
                                        const char **p,
                                        CpError *err) {
  const char *cursor = cp_skip_space(*p);
  if (!cursor || *cursor == '\0') {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "missing query term");
    return NULL;
  }
  if (*cursor == '(') {
    cursor++;
    CpQueryNode *node = cp_query_parse_expr(df, &cursor, err);
    if (!node) {
      return NULL;
    }
    cursor = cp_skip_space(cursor);
    if (!cursor || *cursor != ')') {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unterminated group");
      cp_query_node_free(node);
      return NULL;
    }
    cursor++;
    *p = cursor;
    return node;
  }
  return cp_query_parse_pred(df, p, err);
}

static CpQueryNode *cp_query_parse_pred(const CpDataFrame *df,
                                        const char **p,
                                        CpError *err) {
  if (!df || !p || !*p) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid query");
    return NULL;
  }

  const char *cursor = cp_skip_space(*p);
  if (!cursor || *cursor == '\0') {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "empty query");
    return NULL;
  }

  const char *start = cursor;
  while (*cursor && cp_is_ident_char(*cursor)) {
    cursor++;
  }
  if (start == cursor) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "missing column");
    return NULL;
  }
  char *col = cp_strndup(start, (size_t)(cursor - start));
  if (!col) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  const CpSeries *series = cp_df_require_col(df, col, err);
  if (!series) {
    free(col);
    return NULL;
  }

  cursor = cp_skip_space(cursor);
  if (!cursor || *cursor == '\0') {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "missing operator");
    free(col);
    return NULL;
  }

  CpCompareOp op = CP_OP_EQ;
  if (cursor[0] == '=' && cursor[1] == '=') {
    op = CP_OP_EQ;
    cursor += 2;
  } else if (cursor[0] == '!' && cursor[1] == '=') {
    op = CP_OP_NE;
    cursor += 2;
  } else if (cursor[0] == '<' && cursor[1] == '=') {
    op = CP_OP_LE;
    cursor += 2;
  } else if (cursor[0] == '>' && cursor[1] == '=') {
    op = CP_OP_GE;
    cursor += 2;
  } else if (cursor[0] == '<') {
    op = CP_OP_LT;
    cursor += 1;
  } else if (cursor[0] == '>') {
    op = CP_OP_GT;
    cursor += 1;
  } else if (cursor[0] == '=') {
    op = CP_OP_EQ;
    cursor += 1;
  } else {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid operator");
    free(col);
    return NULL;
  }

  cursor = cp_skip_space(cursor);
  if (!cursor || *cursor == '\0') {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "missing value");
    free(col);
    return NULL;
  }

  char *value = NULL;
  int value_quoted = 0;
  if (*cursor == '"' || *cursor == '\'') {
    char quote = *cursor;
    cursor++;
    start = cursor;
    while (*cursor && *cursor != quote) {
      cursor++;
    }
    if (*cursor != quote) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unterminated string");
      free(col);
      return NULL;
    }
    value = cp_strndup(start, (size_t)(cursor - start));
    if (!value) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      free(col);
      return NULL;
    }
    value_quoted = 1;
    cursor++;
  } else {
    start = cursor;
    while (*cursor && !isspace((unsigned char)*cursor) && *cursor != ')') {
      cursor++;
    }
    if (start == cursor) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "missing value");
      free(col);
      return NULL;
    }
    value = cp_strndup(start, (size_t)(cursor - start));
    if (!value) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      free(col);
      return NULL;
    }
    value_quoted = 0;
  }

  CpQueryNode *node = (CpQueryNode *)calloc(1, sizeof(CpQueryNode));
  if (!node) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    free(col);
    free(value);
    return NULL;
  }
  node->type = CP_QUERY_NODE_PRED;
  node->series = series;
  node->col = col;
  node->value = value;
  node->value_quoted = value_quoted;
  node->op = op;

  if (!value_quoted && cp_str_eq_ci(value, "null")) {
    if (op != CP_OP_EQ && op != CP_OP_NE) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0,
                   "null comparison requires == or !=");
      cp_query_node_free(node);
      return NULL;
    }
    node->is_null_literal = 1;
    *p = cursor;
    return node;
  }

  if (!value_quoted && series->dtype == CP_DTYPE_FLOAT64 &&
      cp_str_eq_ci(value, "nan")) {
    if (op != CP_OP_EQ && op != CP_OP_NE) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0,
                   "nan comparison requires == or !=");
      cp_query_node_free(node);
      return NULL;
    }
    node->is_nan_literal = 1;
    *p = cursor;
    return node;
  }

  switch (series->dtype) {
    case CP_DTYPE_INT64: {
      int is_null = 0;
      if (!cp_parse_int64(value, &node->i64_value, &is_null, err, 0, 0)) {
        cp_query_node_free(node);
        return NULL;
      }
      if (is_null) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "query value is null");
        cp_query_node_free(node);
        return NULL;
      }
      break;
    }
    case CP_DTYPE_FLOAT64: {
      int is_null = 0;
      if (!cp_parse_float64(value, &node->f64_value, &is_null, err, 0, 0)) {
        cp_query_node_free(node);
        return NULL;
      }
      if (is_null) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "query value is null");
        cp_query_node_free(node);
        return NULL;
      }
      break;
    }
    case CP_DTYPE_STRING:
      break;
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
      cp_query_node_free(node);
      return NULL;
  }

  *p = cursor;
  return node;
}

static int cp_query_eval_node(const CpQueryNode *node,
                              size_t row,
                              int *out,
                              CpError *err) {
  if (!node || !out) {
    cp_error_set(err, CP_ERR_INVALID, row, 0, "invalid query node");
    return 0;
  }
  switch (node->type) {
    case CP_QUERY_NODE_PRED: {
      const CpSeries *series = node->series;
      if (!series) {
        cp_error_set(err, CP_ERR_INVALID, row, 0, "column not found");
        return 0;
      }
      if (node->is_null_literal) {
        int is_null = series->is_null[row] ? 1 : 0;
        *out = (node->op == CP_OP_EQ) ? is_null : !is_null;
        return 1;
      }
      if (node->is_nan_literal) {
        int is_nan =
            !series->is_null[row] && isnan(series->data.f64[row]) ? 1 : 0;
        *out = (node->op == CP_OP_EQ) ? is_nan : !is_nan;
        return 1;
      }
      switch (series->dtype) {
        case CP_DTYPE_INT64:
          if (series->is_null[row]) {
            *out = 0;
            return 1;
          }
          return cp_eval_compare_int64(series->data.i64[row],
                                       node->op,
                                       node->i64_value,
                                       out,
                                       err);
        case CP_DTYPE_FLOAT64:
          if (series->is_null[row]) {
            *out = 0;
            return 1;
          }
          return cp_eval_compare_float64(series->data.f64[row],
                                         node->op,
                                         node->f64_value,
                                         out,
                                         err);
        case CP_DTYPE_STRING: {
          if (series->is_null[row]) {
            *out = 0;
            return 1;
          }
          const char *lhs = series->data.str[row] ? series->data.str[row] : "";
          return cp_eval_compare_string(lhs, node->op, node->value, out, err);
        }
        default:
          cp_error_set(err, CP_ERR_INVALID, row, 0, "unsupported dtype");
          return 0;
      }
    }
    case CP_QUERY_NODE_AND: {
      int lhs = 0;
      if (!cp_query_eval_node(node->left, row, &lhs, err)) {
        return 0;
      }
      if (!lhs) {
        *out = 0;
        return 1;
      }
      int rhs = 0;
      if (!cp_query_eval_node(node->right, row, &rhs, err)) {
        return 0;
      }
      *out = rhs ? 1 : 0;
      return 1;
    }
    case CP_QUERY_NODE_OR: {
      int lhs = 0;
      if (!cp_query_eval_node(node->left, row, &lhs, err)) {
        return 0;
      }
      if (lhs) {
        *out = 1;
        return 1;
      }
      int rhs = 0;
      if (!cp_query_eval_node(node->right, row, &rhs, err)) {
        return 0;
      }
      *out = rhs ? 1 : 0;
      return 1;
    }
    default:
      cp_error_set(err, CP_ERR_INVALID, row, 0, "unsupported query node");
      return 0;
  }
}

static int cp_prepare_replacements(const CpDataFrame *df,
                                   const char **values,
                                   size_t count,
                                   int *is_null,
                                   int64_t *i64,
                                   double *f64,
                                   const char **str,
                                   CpError *err) {
  if (!df || !is_null || !i64 || !f64 || !str) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid replacements");
    return 0;
  }
  size_t ncols = df->ncols;
  if (!values) {
    for (size_t col = 0; col < ncols; ++col) {
      is_null[col] = 1;
      str[col] = NULL;
    }
    return 1;
  }
  if (count != ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "replacement count mismatch");
    return 0;
  }
  for (size_t col = 0; col < ncols; ++col) {
    is_null[col] = 0;
    str[col] = NULL;
    if (!values[col]) {
      is_null[col] = 1;
      continue;
    }
    int value_is_null = 0;
    switch (df->cols[col]->dtype) {
      case CP_DTYPE_INT64:
        if (!cp_parse_int64(values[col], &i64[col], &value_is_null, err, 0,
                            col)) {
          return 0;
        }
        is_null[col] = value_is_null;
        break;
      case CP_DTYPE_FLOAT64:
        if (!cp_parse_float64(values[col], &f64[col], &value_is_null, err, 0,
                              col)) {
          return 0;
        }
        is_null[col] = value_is_null;
        break;
      case CP_DTYPE_STRING:
        if (!cp_parse_string(values[col], &str[col], &value_is_null)) {
          cp_error_set(err, CP_ERR_INVALID, 0, col, "invalid string value");
          return 0;
        }
        is_null[col] = value_is_null;
        break;
      default:
        cp_error_set(err, CP_ERR_INVALID, 0, col, "unknown dtype");
        return 0;
    }
  }
  return 1;
}

static int cp_df_schema_matches(const CpDataFrame *base,
                                const CpDataFrame *other,
                                CpError *err) {
  if (!base || !other) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return 0;
  }
  if (base->ncols != other->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column count mismatch");
    return 0;
  }
  for (size_t col = 0; col < base->ncols; ++col) {
    const CpSeries *lhs = base->cols[col];
    const CpSeries *rhs = other->cols[col];
    if (!lhs || !rhs) {
      cp_error_set(err, CP_ERR_INVALID, 0, col, "column missing");
      return 0;
    }
    if (lhs->dtype != rhs->dtype) {
      cp_error_set(err, CP_ERR_INVALID, 0, col, "column dtype mismatch");
      return 0;
    }
    if (strcmp(lhs->name, rhs->name) != 0) {
      cp_error_set(err, CP_ERR_INVALID, 0, col, "column name mismatch");
      return 0;
    }
  }
  return 1;
}

static int cp_df_find_col_index(const CpDataFrame *df,
                                const char *name,
                                size_t *out,
                                CpError *err) {
  if (!df || !name || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid column lookup");
    return 0;
  }
  for (size_t i = 0; i < df->ncols; ++i) {
    if (df->cols[i] && df->cols[i]->name &&
        strcmp(df->cols[i]->name, name) == 0) {
      *out = i;
      return 1;
    }
  }
  cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
  return 0;
}

static int cp_df_find_row_label(const CpDataFrame *df,
                                const char *label,
                                size_t *out,
                                CpError *err) {
  if (!df || !label || !out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid row lookup");
    return 0;
  }
  if (df->has_index) {
    if (df->index_col >= df->ncols) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid index column");
      return 0;
    }
    const CpSeries *index = df->cols[df->index_col];
    if (!index) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid index column");
      return 0;
    }
    if (index->dtype == CP_DTYPE_INT64) {
      int64_t key = 0;
      int is_null = 0;
      if (!cp_parse_int64(label, &key, &is_null, err, 0, df->index_col)) {
        return 0;
      }
      if (is_null) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "row label is null");
        return 0;
      }
      for (size_t row = 0; row < df->nrows; ++row) {
        if (index->is_null[row]) {
          continue;
        }
        if (index->data.i64[row] == key) {
          *out = row;
          return 1;
        }
      }
    } else if (index->dtype == CP_DTYPE_STRING) {
      for (size_t row = 0; row < df->nrows; ++row) {
        if (index->is_null[row]) {
          continue;
        }
        const char *val = index->data.str[row];
        if (val && strcmp(val, label) == 0) {
          *out = row;
          return 1;
        }
      }
    } else {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported index dtype");
      return 0;
    }
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "row label not found");
    return 0;
  }

  int64_t idx = 0;
  int is_null = 0;
  if (!cp_parse_int64(label, &idx, &is_null, err, 0, 0)) {
    return 0;
  }
  if (is_null || idx < 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "row index invalid");
    return 0;
  }
  if ((size_t)idx >= df->nrows) {
    cp_error_set(err, CP_ERR_INVALID, (size_t)idx, 0, "row index out of range");
    return 0;
  }
  *out = (size_t)idx;
  return 1;
}

static int cp_series_value_equal(const CpSeries *series,
                                 size_t left,
                                 size_t right) {
  if (!series || left >= series->length || right >= series->length) {
    return 0;
  }
  int left_null = series->is_null[left] ? 1 : 0;
  int right_null = series->is_null[right] ? 1 : 0;
  if (left_null || right_null) {
    return left_null && right_null;
  }
  switch (series->dtype) {
    case CP_DTYPE_INT64:
      return series->data.i64[left] == series->data.i64[right];
    case CP_DTYPE_FLOAT64: {
      double a = series->data.f64[left];
      double b = series->data.f64[right];
      if (isnan(a) && isnan(b)) {
        return 1;
      }
      return a == b;
    }
    case CP_DTYPE_STRING: {
      const char *a = series->data.str[left];
      const char *b = series->data.str[right];
      if (!a || !b) {
        return a == b;
      }
      return strcmp(a, b) == 0;
    }
    default:
      return 0;
  }
}

static size_t cp_series_find_value(const CpSeries *series,
                                   const size_t *indices,
                                   size_t count,
                                   size_t row) {
  if (!series || !indices) {
    return SIZE_MAX;
  }
  for (size_t i = 0; i < count; ++i) {
    if (cp_series_value_equal(series, row, indices[i])) {
      return i;
    }
  }
  return SIZE_MAX;
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

static int cp_series_append_value(CpSeries *s,
                                  CpDType dtype,
                                  const CpValue *value,
                                  size_t row,
                                  size_t col,
                                  CpError *err) {
  if (!s || !value) {
    cp_error_set(err, CP_ERR_INVALID, row, col, "invalid value");
    return 0;
  }
  if (s->dtype != dtype) {
    cp_error_set(err, CP_ERR_INVALID, row, col, "dtype mismatch");
    return 0;
  }
  if (value->is_null) {
    return cp_series_append_null(s, err);
  }
  switch (dtype) {
    case CP_DTYPE_INT64:
      return cp_series_append_int64(s, value->value.i64, 0, err);
    case CP_DTYPE_FLOAT64:
      return cp_series_append_float64(s, value->value.f64, 0, err);
    case CP_DTYPE_STRING:
      if (!value->value.str) {
        cp_error_set(err, CP_ERR_INVALID, row, col,
                     "null string value");
        return 0;
      }
      return cp_series_append_string(s, value->value.str, 0, err);
    default:
      cp_error_set(err, CP_ERR_INVALID, row, col, "unknown dtype");
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

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} CpStrBuf;

static int cp_strbuf_init(CpStrBuf *buf, size_t cap, CpError *err) {
  if (!buf) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid string buffer");
    return 0;
  }
  if (cap == 0) {
    cap = 16;
  }
  buf->data = (char *)malloc(cap);
  if (!buf->data) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return 0;
  }
  buf->len = 0;
  buf->cap = cap;
  buf->data[0] = '\0';
  return 1;
}

static void cp_strbuf_free(CpStrBuf *buf) {
  if (!buf) {
    return;
  }
  free(buf->data);
  buf->data = NULL;
  buf->len = 0;
  buf->cap = 0;
}

static int cp_strbuf_ensure(CpStrBuf *buf, size_t extra, CpError *err) {
  if (!buf) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid string buffer");
    return 0;
  }
  size_t needed = buf->len + extra + 1;
  if (needed <= buf->cap) {
    return 1;
  }
  size_t new_cap = buf->cap ? buf->cap : 16;
  while (new_cap < needed) {
    new_cap *= 2;
  }
  char *next = (char *)realloc(buf->data, new_cap);
  if (!next) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return 0;
  }
  buf->data = next;
  buf->cap = new_cap;
  return 1;
}

static int cp_strbuf_append(CpStrBuf *buf,
                            const char *s,
                            size_t len,
                            CpError *err) {
  if (!s) {
    s = "";
    len = 0;
  }
  if (!cp_strbuf_ensure(buf, len, err)) {
    return 0;
  }
  if (len > 0) {
    memcpy(buf->data + buf->len, s, len);
  }
  buf->len += len;
  buf->data[buf->len] = '\0';
  return 1;
}

static int cp_strbuf_append_char(CpStrBuf *buf, char c, CpError *err) {
  if (!cp_strbuf_ensure(buf, 1, err)) {
    return 0;
  }
  buf->data[buf->len++] = c;
  buf->data[buf->len] = '\0';
  return 1;
}

static int cp_strbuf_append_padded(CpStrBuf *buf,
                                   const char *s,
                                   size_t len,
                                   size_t width,
                                   int right_align,
                                   CpError *err) {
  if (!s) {
    s = "";
    len = 0;
  }
  if (len > width) {
    width = len;
  }
  size_t pad = width - len;
  if (right_align && pad > 0) {
    for (size_t i = 0; i < pad; ++i) {
      if (!cp_strbuf_append_char(buf, ' ', err)) {
        return 0;
      }
    }
  }
  if (!cp_strbuf_append(buf, s, len, err)) {
    return 0;
  }
  if (!right_align && pad > 0) {
    for (size_t i = 0; i < pad; ++i) {
      if (!cp_strbuf_append_char(buf, ' ', err)) {
        return 0;
      }
    }
  }
  return 1;
}

static int cp_is_leap_year(int year) {
  if (year % 400 == 0) {
    return 1;
  }
  if (year % 100 == 0) {
    return 0;
  }
  return (year % 4 == 0);
}

static int cp_days_in_month(int year, int month) {
  static const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month < 1 || month > 12) {
    return 0;
  }
  if (month == 2 && cp_is_leap_year(year)) {
    return 29;
  }
  return days[month - 1];
}

static int64_t cp_days_from_civil(int year, int month, int day) {
  int y = year;
  int m = month;
  int d = day;
  y -= m <= 2;
  int era = (y >= 0 ? y : y - 399) / 400;
  unsigned int yoe = (unsigned int)(y - era * 400);
  unsigned int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + (unsigned int)d - 1;
  unsigned int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return (int64_t)era * 146097 + (int64_t)doe - 719468;
}

static int cp_parse_n_digits(const char **p, int count, int *out) {
  if (!p || !*p) {
    return 0;
  }
  int value = 0;
  const char *s = *p;
  for (int i = 0; i < count; ++i) {
    if (*s == '\0' || !isdigit((unsigned char)*s)) {
      return 0;
    }
    value = value * 10 + (*s - '0');
    s++;
  }
  *p = s;
  if (out) {
    *out = value;
  }
  return 1;
}

static int cp_parse_datetime(const char *s,
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
  const char *p = s;
  while (isspace((unsigned char)*p)) {
    p++;
  }

  int year = 0;
  int month = 0;
  int day = 0;
  if (!cp_parse_n_digits(&p, 4, &year)) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }
  char delim = *p;
  if (delim != '-' && delim != '/') {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }
  p++;
  if (!cp_parse_n_digits(&p, 2, &month) || *p != delim) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }
  p++;
  if (!cp_parse_n_digits(&p, 2, &day)) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }
  int max_day = cp_days_in_month(year, month);
  if (max_day == 0 || day < 1 || day > max_day) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }

  int hour = 0;
  int minute = 0;
  int second = 0;
  int has_time = 0;
  int has_tz = 0;
  int tz_sign = 1;
  int tz_hour = 0;
  int tz_min = 0;

  while (isspace((unsigned char)*p)) {
    p++;
  }

  if (*p == 'T' || *p == 't') {
    p++;
    has_time = 1;
  } else if (isdigit((unsigned char)*p)) {
    has_time = 1;
  }

  if (has_time) {
    if (!cp_parse_n_digits(&p, 2, &hour) || *p != ':') {
      cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
      return 0;
    }
    p++;
    if (!cp_parse_n_digits(&p, 2, &minute)) {
      cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
      return 0;
    }
    if (*p == ':') {
      p++;
      if (!cp_parse_n_digits(&p, 2, &second)) {
        cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
        return 0;
      }
    }
    if (*p == '.') {
      p++;
      while (isdigit((unsigned char)*p)) {
        p++;
      }
    }
  }

  while (isspace((unsigned char)*p)) {
    p++;
  }

  if (*p == 'Z' || *p == 'z') {
    has_tz = 1;
    p++;
  } else if (*p == '+' || *p == '-') {
    has_tz = 1;
    tz_sign = (*p == '-') ? -1 : 1;
    p++;
    if (!cp_parse_n_digits(&p, 2, &tz_hour)) {
      cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
      return 0;
    }
    if (*p == ':') {
      p++;
    }
    if (!cp_parse_n_digits(&p, 2, &tz_min)) {
      cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
      return 0;
    }
    if (tz_hour < 0 || tz_hour > 23 || tz_min < 0 || tz_min > 59) {
      cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
      return 0;
    }
  }

  while (isspace((unsigned char)*p)) {
    p++;
  }
  if (*p != '\0') {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }

  if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 ||
      second > 59) {
    cp_error_set(err, CP_ERR_PARSE, row, col, "invalid datetime");
    return 0;
  }

  int64_t days = cp_days_from_civil(year, month, day);
  int64_t total = days * 86400 + (int64_t)hour * 3600 +
                  (int64_t)minute * 60 + (int64_t)second;
  if (has_tz) {
    int64_t offset = (int64_t)tz_sign * ((int64_t)tz_hour * 3600 +
                                        (int64_t)tz_min * 60);
    total -= offset;
  }
  if (out) {
    *out = total;
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
  df->has_index = 0;
  df->index_col = 0;
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

int cp_df_shape(const CpDataFrame *df,
                size_t *out_rows,
                size_t *out_cols,
                CpError *err) {
  if (!df || (!out_rows && !out_cols)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid shape output");
    return 0;
  }
  if (out_rows) {
    *out_rows = df->nrows;
  }
  if (out_cols) {
    *out_cols = df->ncols;
  }
  return 1;
}

size_t cp_df_size(const CpDataFrame *df) {
  return df ? df->nrows * df->ncols : 0;
}

size_t cp_df_ndim(const CpDataFrame *df) {
  return df ? 2 : 0;
}

int cp_df_columns(const CpDataFrame *df,
                  const char **out,
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
    out[i] = df->cols[i]->name;
  }
  return 1;
}

CpDataFrame *cp_df_copy(const CpDataFrame *df, CpError *err) {
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
  CpDataFrame *out = cp_df_create(ncols, names, dtypes, df->nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }
  if (df->has_index && df->index_col < ncols) {
    out->has_index = 1;
    out->index_col = df->index_col;
  }
  const CpSeries **src_cols = (const CpSeries **)df->cols;
  for (size_t row = 0; row < df->nrows; ++row) {
    if (!cp_df_append_row_from_sources(out, src_cols, ncols, row, err)) {
      cp_df_free(out);
      return NULL;
    }
  }
  return out;
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

static const char *cp_join_format_name(const char *base,
                                       const char **existing,
                                       size_t existing_count,
                                       const char *suffix,
                                       int force_suffix,
                                       int *owned,
                                       CpError *err) {
  if (!owned || !suffix) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid name ownership");
    return NULL;
  }
  *owned = 0;
  const char *safe = base ? base : "";
  int base_in_list = cp_name_in_list(safe, existing, existing_count);
  if (!force_suffix && !base_in_list) {
    return safe;
  }

  size_t base_len = strlen(safe);
  size_t suffix_len = (force_suffix || base_in_list) ? strlen(suffix) : 0;
  size_t base_with_suffix_len = base_len + suffix_len;
  char *base_with_suffix = (char *)malloc(base_with_suffix_len + 1);
  if (!base_with_suffix) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  memcpy(base_with_suffix, safe, base_len);
  if (suffix_len > 0) {
    memcpy(base_with_suffix + base_len, suffix, suffix_len);
  }
  base_with_suffix[base_with_suffix_len] = '\0';

  if (!cp_name_in_list(base_with_suffix, existing, existing_count)) {
    *owned = 1;
    return base_with_suffix;
  }

  for (size_t attempt = 2;; ++attempt) {
    char num_buf[32];
    snprintf(num_buf, sizeof(num_buf), "%zu", attempt);
    size_t len = base_with_suffix_len + strlen(num_buf);
    char *name = (char *)malloc(len + 1);
    if (!name) {
      free(base_with_suffix);
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    memcpy(name, base_with_suffix, base_with_suffix_len);
    memcpy(name + base_with_suffix_len, num_buf, strlen(num_buf));
    name[len] = '\0';
    if (!cp_name_in_list(name, existing, existing_count)) {
      free(base_with_suffix);
      *owned = 1;
      return name;
    }
    free(name);
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

static int cp_join_keys_any_null(const CpSeries **keys,
                                 size_t key_count,
                                 size_t row) {
  if (!keys || key_count == 0) {
    return 1;
  }
  for (size_t i = 0; i < key_count; ++i) {
    if (cp_join_key_is_null(keys[i], row)) {
      return 1;
    }
  }
  return 0;
}

static int cp_join_keys_equal(const CpSeries **left_keys,
                              const CpSeries **right_keys,
                              size_t key_count,
                              size_t left_row,
                              size_t right_row) {
  if (!left_keys || !right_keys || key_count == 0) {
    return 0;
  }
  for (size_t i = 0; i < key_count; ++i) {
    const CpSeries *lkey = left_keys[i];
    const CpSeries *rkey = right_keys[i];
    if (!lkey || !rkey) {
      return 0;
    }
    if (lkey->dtype == CP_DTYPE_INT64) {
      if (lkey->data.i64[left_row] != rkey->data.i64[right_row]) {
        return 0;
      }
    } else {
      const char *lhs = lkey->data.str[left_row];
      const char *rhs = rkey->data.str[right_row];
      if (!lhs || !rhs || strcmp(lhs, rhs) != 0) {
        return 0;
      }
    }
  }
  return 1;
}

static int cp_join_compare_lr(const CpSeries **left_keys,
                              const CpSeries **right_keys,
                              size_t key_count,
                              size_t left_row,
                              size_t right_row) {
  if (!left_keys || !right_keys || key_count == 0) {
    return 0;
  }
  for (size_t i = 0; i < key_count; ++i) {
    const CpSeries *lkey = left_keys[i];
    const CpSeries *rkey = right_keys[i];
    if (!lkey || !rkey) {
      return 0;
    }
    if (lkey->dtype == CP_DTYPE_INT64) {
      int64_t lhs = lkey->data.i64[left_row];
      int64_t rhs = rkey->data.i64[right_row];
      if (lhs < rhs) {
        return -1;
      }
      if (lhs > rhs) {
        return 1;
      }
    } else {
      const char *lhs = lkey->data.str[left_row];
      const char *rhs = rkey->data.str[right_row];
      if (!lhs && !rhs) {
        continue;
      }
      if (!lhs) {
        return -1;
      }
      if (!rhs) {
        return 1;
      }
      int cmp = strcmp(lhs, rhs);
      if (cmp != 0) {
        return cmp < 0 ? -1 : 1;
      }
    }
  }
  return 0;
}

static size_t cp_join_lower_bound(const CpSeries **left_keys,
                                  const CpSeries **right_keys,
                                  size_t key_count,
                                  size_t left_row,
                                  const size_t *right_sorted,
                                  size_t right_count) {
  size_t lo = 0;
  size_t hi = right_count;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    size_t rrow = right_sorted[mid];
    int cmp = cp_join_compare_lr(left_keys, right_keys, key_count, left_row, rrow);
    if (cmp <= 0) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

typedef struct {
  uint64_t hash;
  size_t *rows;
  size_t count;
  size_t capacity;
  int in_use;
} CpJoinBucket;

typedef struct {
  CpJoinBucket *buckets;
  size_t bucket_count;
  size_t mask;
} CpJoinIndex;

static uint64_t cp_hash_bytes(uint64_t hash,
                              const unsigned char *data,
                              size_t len) {
  const uint64_t prime = 1099511628211ULL;
  for (size_t i = 0; i < len; ++i) {
    hash ^= (uint64_t)data[i];
    hash *= prime;
  }
  return hash;
}

static uint64_t cp_hash_int64(uint64_t hash, int64_t value) {
  unsigned char bytes[sizeof(int64_t)];
  memcpy(bytes, &value, sizeof(int64_t));
  return cp_hash_bytes(hash, bytes, sizeof(int64_t));
}

static uint64_t cp_hash_size(uint64_t hash, size_t value) {
  unsigned char bytes[sizeof(size_t)];
  memcpy(bytes, &value, sizeof(size_t));
  return cp_hash_bytes(hash, bytes, sizeof(size_t));
}

static uint64_t cp_join_hash_keys(const CpSeries **keys,
                                  size_t key_count,
                                  size_t row) {
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < key_count; ++i) {
    const CpSeries *series = keys[i];
    unsigned char marker = series->dtype == CP_DTYPE_INT64 ? 'i' : 's';
    hash = cp_hash_bytes(hash, &marker, 1);
    if (series->dtype == CP_DTYPE_INT64) {
      hash = cp_hash_int64(hash, series->data.i64[row]);
    } else {
      const char *value = series->data.str[row];
      size_t len = value ? strlen(value) : 0;
      hash = cp_hash_size(hash, len);
      if (len > 0) {
        hash = cp_hash_bytes(hash, (const unsigned char *)value, len);
      }
    }
  }
  return hash;
}

static int cp_join_bucket_push(CpJoinBucket *bucket,
                               size_t row,
                               CpError *err) {
  if (!bucket) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join bucket");
    return 0;
  }
  if (bucket->count == bucket->capacity) {
    size_t new_cap = bucket->capacity == 0 ? 4 : bucket->capacity * 2;
    size_t *next = (size_t *)realloc(bucket->rows, new_cap * sizeof(size_t));
    if (!next) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return 0;
    }
    bucket->rows = next;
    bucket->capacity = new_cap;
  }
  bucket->rows[bucket->count++] = row;
  return 1;
}

static int cp_join_index_init(CpJoinIndex *index,
                              size_t row_count,
                              CpError *err) {
  if (!index) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join index");
    return 0;
  }
  index->buckets = NULL;
  index->bucket_count = 0;
  index->mask = 0;
  if (row_count == 0) {
    return 1;
  }
  if (row_count > SIZE_MAX / 2) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
    return 0;
  }
  size_t target = row_count * 2;
  if (target < 8) {
    target = 8;
  }
  size_t bucket_count = 1;
  while (bucket_count < target) {
    if (bucket_count > SIZE_MAX / 2) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
      return 0;
    }
    bucket_count <<= 1;
  }

  CpJoinBucket *buckets =
      (CpJoinBucket *)calloc(bucket_count, sizeof(CpJoinBucket));
  if (!buckets) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return 0;
  }
  index->buckets = buckets;
  index->bucket_count = bucket_count;
  index->mask = bucket_count - 1;
  return 1;
}

static void cp_join_index_free(CpJoinIndex *index) {
  if (!index || !index->buckets) {
    return;
  }
  for (size_t i = 0; i < index->bucket_count; ++i) {
    free(index->buckets[i].rows);
  }
  free(index->buckets);
  index->buckets = NULL;
  index->bucket_count = 0;
  index->mask = 0;
}

static int cp_join_index_add(CpJoinIndex *index,
                             uint64_t hash,
                             size_t row,
                             CpError *err) {
  if (!index || !index->buckets || index->bucket_count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join index");
    return 0;
  }
  size_t mask = index->mask;
  size_t idx = (size_t)hash & mask;
  for (size_t probe = 0; probe < index->bucket_count; ++probe) {
    CpJoinBucket *bucket = &index->buckets[idx];
    if (!bucket->in_use) {
      bucket->in_use = 1;
      bucket->hash = hash;
      return cp_join_bucket_push(bucket, row, err);
    }
    if (bucket->hash == hash) {
      return cp_join_bucket_push(bucket, row, err);
    }
    idx = (idx + 1) & mask;
  }
  cp_error_set(err, CP_ERR_INVALID, 0, 0, "join hash table full");
  return 0;
}

static const CpJoinBucket *cp_join_index_find(const CpJoinIndex *index,
                                              uint64_t hash) {
  if (!index || !index->buckets || index->bucket_count == 0) {
    return NULL;
  }
  size_t mask = index->mask;
  size_t idx = (size_t)hash & mask;
  for (size_t probe = 0; probe < index->bucket_count; ++probe) {
    const CpJoinBucket *bucket = &index->buckets[idx];
    if (!bucket->in_use) {
      return NULL;
    }
    if (bucket->hash == hash) {
      return bucket;
    }
    idx = (idx + 1) & mask;
  }
  return NULL;
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

static int cp_dtype_in_list(const CpDType *list,
                            size_t count,
                            CpDType dtype) {
  if (!list || count == 0) {
    return 0;
  }
  for (size_t i = 0; i < count; ++i) {
    if (list[i] == dtype) {
      return 1;
    }
  }
  return 0;
}

CpDataFrame *cp_df_select_dtypes(const CpDataFrame *df,
                                 const CpDType *include,
                                 size_t include_count,
                                 const CpDType *exclude,
                                 size_t exclude_count,
                                 CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid selection");
    return NULL;
  }
  if ((include_count > 0 && !include) || (exclude_count > 0 && !exclude)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid selection");
    return NULL;
  }
  if (include_count == 0 && exclude_count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no selection criteria");
    return NULL;
  }

  size_t ncols = df->ncols;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **sel_names = (const char **)malloc(ncols * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(ncols * sizeof(const CpSeries *));
  if (!dtypes || !sel_names || !src_cols) {
    free(dtypes);
    free(sel_names);
    free(src_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  size_t count = 0;
  for (size_t col = 0; col < ncols; ++col) {
    const CpSeries *series = df->cols[col];
    if (!series) {
      free(dtypes);
      free(sel_names);
      free(src_cols);
      cp_error_set(err, CP_ERR_INVALID, 0, col, "column not found");
      return NULL;
    }
    int include_ok = include_count == 0 ||
                     cp_dtype_in_list(include, include_count, series->dtype);
    int exclude_ok = exclude_count == 0 ||
                     !cp_dtype_in_list(exclude, exclude_count, series->dtype);
    if (include_ok && exclude_ok) {
      dtypes[count] = series->dtype;
      sel_names[count] = series->name;
      src_cols[count] = series;
      count += 1;
    }
  }

  if (count == 0) {
    free(dtypes);
    free(sel_names);
    free(src_cols);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns selected");
    return NULL;
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

int cp_df_isna_mask(const CpDataFrame *df,
                    uint8_t *out,
                    size_t out_len,
                    CpError *err) {
  return cp_df_isnull_mask(df, out, out_len, err);
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

static int cp_df_append_value_count_row(CpDataFrame *out,
                                        const CpSeries *series,
                                        size_t row,
                                        int64_t count,
                                        CpError *err) {
  if (!out || !series || out->ncols != 2) {
    cp_error_set(err, CP_ERR_INVALID, row, 0, "invalid value_counts output");
    return 0;
  }
  if (!cp_series_append_from(out->cols[0], series, row, err)) {
    return 0;
  }
  if (!cp_series_append_int64(out->cols[1], count, 0, err)) {
    cp_series_pop(out->cols[0]);
    return 0;
  }
  out->nrows += 1;
  return 1;
}

CpDataFrame *cp_df_unique(const CpDataFrame *df,
                          const char *name,
                          CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }

  size_t nrows = series->length;
  const char *names[1] = {series->name};
  CpDType dtypes[1] = {series->dtype};
  CpDataFrame *out = cp_df_create(1, names, dtypes, nrows, err);
  if (!out) {
    return NULL;
  }

  size_t *indices = NULL;
  size_t count = 0;
  size_t cap = 0;
  const CpSeries *src_cols[1] = {series};

  for (size_t row = 0; row < nrows; ++row) {
    if (cp_series_find_value(series, indices, count, row) != SIZE_MAX) {
      continue;
    }
    if (!cp_df_append_row_from_sources(out, src_cols, 1, row, err)) {
      cp_df_free(out);
      out = NULL;
      break;
    }
    if (count + 1 > cap) {
      size_t new_cap = cap == 0 ? 8 : cap * 2;
      size_t *new_indices =
          (size_t *)realloc(indices, new_cap * sizeof(size_t));
      if (!new_indices) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        cp_df_free(out);
        out = NULL;
        break;
      }
      indices = new_indices;
      cap = new_cap;
    }
    indices[count++] = row;
  }

  free(indices);
  return out;
}

int cp_df_nunique(const CpDataFrame *df,
                  const char *name,
                  size_t *out,
                  CpError *err) {
  if (!out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid output");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }

  size_t *indices = NULL;
  size_t count = 0;
  size_t cap = 0;

  for (size_t row = 0; row < series->length; ++row) {
    if (series->is_null[row] || cp_series_is_nan(series, row)) {
      continue;
    }
    if (cp_series_find_value(series, indices, count, row) != SIZE_MAX) {
      continue;
    }
    if (count + 1 > cap) {
      size_t new_cap = cap == 0 ? 8 : cap * 2;
      size_t *new_indices =
          (size_t *)realloc(indices, new_cap * sizeof(size_t));
      if (!new_indices) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        free(indices);
        return 0;
      }
      indices = new_indices;
      cap = new_cap;
    }
    indices[count++] = row;
  }

  free(indices);
  *out = count;
  return 1;
}

CpDataFrame *cp_df_value_counts(const CpDataFrame *df,
                                const char *name,
                                CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }

  size_t *indices = NULL;
  size_t *counts = NULL;
  size_t count = 0;
  size_t cap = 0;

  for (size_t row = 0; row < series->length; ++row) {
    if (series->is_null[row] || cp_series_is_nan(series, row)) {
      continue;
    }
    size_t pos = cp_series_find_value(series, indices, count, row);
    if (pos != SIZE_MAX) {
      counts[pos] += 1;
      continue;
    }
    if (count + 1 > cap) {
      size_t new_cap = cap == 0 ? 8 : cap * 2;
      size_t *new_indices = (size_t *)realloc(indices,
                                              new_cap * sizeof(size_t));
      if (!new_indices) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        free(indices);
        free(counts);
        return NULL;
      }
      indices = new_indices;
      size_t *new_counts =
          (size_t *)realloc(counts, new_cap * sizeof(size_t));
      if (!new_counts) {
        free(indices);
        free(counts);
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        return NULL;
      }
      counts = new_counts;
      cap = new_cap;
    }
    indices[count] = row;
    counts[count] = 1;
    count += 1;
  }

  const char *value_name = series->name;
  char count_name[32];
  const char *count_col = "count";
  if (strcmp(value_name, "count") == 0) {
    snprintf(count_name, sizeof(count_name), "count_1");
    count_col = count_name;
  }
  const char *names[2] = {value_name, count_col};
  CpDType dtypes[2] = {series->dtype, CP_DTYPE_INT64};
  CpDataFrame *out = cp_df_create(2, names, dtypes, count, err);
  if (!out) {
    free(indices);
    free(counts);
    return NULL;
  }

  for (size_t i = 0; i < count; ++i) {
    if (counts[i] > (size_t)INT64_MAX) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "count overflow");
      cp_df_free(out);
      out = NULL;
      break;
    }
    if (!cp_df_append_value_count_row(out,
                                      series,
                                      indices[i],
                                      (int64_t)counts[i],
                                      err)) {
      cp_df_free(out);
      out = NULL;
      break;
    }
  }

  free(indices);
  free(counts);

  if (!out) {
    return NULL;
  }

  CpDataFrame *sorted = cp_df_sort_values(out, count_col, 0, err);
  cp_df_free(out);
  return sorted;
}

int cp_df_duplicated(const CpDataFrame *df,
                     const char *name,
                     CpDuplicateKeep keep,
                     uint8_t *out,
                     size_t out_len,
                     CpError *err) {
  if (!out) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid output");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  size_t nrows = series->length;
  if (out_len < nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "output buffer too small");
    return 0;
  }
  if (nrows == 0) {
    return 1;
  }
  memset(out, 0, nrows * sizeof(uint8_t));

  if (keep == CP_DUP_KEEP_FIRST || keep == CP_DUP_KEEP_LAST) {
    size_t *indices = NULL;
    size_t count = 0;
    size_t cap = 0;
    if (keep == CP_DUP_KEEP_FIRST) {
      for (size_t row = 0; row < nrows; ++row) {
        if (cp_series_find_value(series, indices, count, row) != SIZE_MAX) {
          out[row] = 1;
          continue;
        }
        if (count + 1 > cap) {
          size_t new_cap = cap == 0 ? 8 : cap * 2;
          size_t *new_indices =
              (size_t *)realloc(indices, new_cap * sizeof(size_t));
          if (!new_indices) {
            cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
            free(indices);
            return 0;
          }
          indices = new_indices;
          cap = new_cap;
        }
        indices[count++] = row;
      }
    } else {
      for (size_t row = nrows; row-- > 0;) {
        if (cp_series_find_value(series, indices, count, row) != SIZE_MAX) {
          out[row] = 1;
          continue;
        }
        if (count + 1 > cap) {
          size_t new_cap = cap == 0 ? 8 : cap * 2;
          size_t *new_indices =
              (size_t *)realloc(indices, new_cap * sizeof(size_t));
          if (!new_indices) {
            cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
            free(indices);
            return 0;
          }
          indices = new_indices;
          cap = new_cap;
        }
        indices[count++] = row;
      }
    }
    free(indices);
    return 1;
  }

  if (keep != CP_DUP_KEEP_NONE) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid duplicate keep mode");
    return 0;
  }

  size_t *indices = NULL;
  size_t *counts = NULL;
  size_t count = 0;
  size_t cap = 0;

  for (size_t row = 0; row < nrows; ++row) {
    size_t pos = cp_series_find_value(series, indices, count, row);
    if (pos != SIZE_MAX) {
      counts[pos] += 1;
      continue;
    }
    if (count + 1 > cap) {
      size_t new_cap = cap == 0 ? 8 : cap * 2;
      size_t *new_indices = (size_t *)realloc(indices,
                                              new_cap * sizeof(size_t));
      if (!new_indices) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        free(indices);
        free(counts);
        return 0;
      }
      indices = new_indices;
      size_t *new_counts =
          (size_t *)realloc(counts, new_cap * sizeof(size_t));
      if (!new_counts) {
        free(indices);
        free(counts);
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        return 0;
      }
      counts = new_counts;
      cap = new_cap;
    }
    indices[count] = row;
    counts[count] = 1;
    count += 1;
  }

  for (size_t row = 0; row < nrows; ++row) {
    size_t pos = cp_series_find_value(series, indices, count, row);
    if (pos == SIZE_MAX) {
      cp_error_set(err, CP_ERR_INVALID, row, 0, "duplicate lookup failed");
      free(indices);
      free(counts);
      return 0;
    }
    out[row] = counts[pos] > 1 ? 1 : 0;
  }

  free(indices);
  free(counts);
  return 1;
}

CpDataFrame *cp_df_drop_duplicates(const CpDataFrame *df,
                                   const char *name,
                                   CpDuplicateKeep keep,
                                   CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  size_t nrows = df->nrows;
  if (nrows == 0) {
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  if (!cp_df_duplicated(df, name, keep, mask, nrows, err)) {
    free(mask);
    return NULL;
  }
  for (size_t i = 0; i < nrows; ++i) {
    mask[i] = mask[i] ? 0 : 1;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  return out;
}

static CpDataFrame *cp_df_apply_mask(const CpDataFrame *df,
                                     const uint8_t *mask,
                                     size_t mask_len,
                                     const char **values,
                                     size_t count,
                                     int invert,
                                     CpError *err) {
  if (!df || !mask) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid mask");
    return NULL;
  }
  if (mask_len != df->nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "mask length mismatch");
    return NULL;
  }
  if (df->nrows == 0) {
    return cp_df_empty_like(df, err);
  }

  size_t ncols = df->ncols;
  int *rep_is_null = (int *)calloc(ncols, sizeof(int));
  int64_t *rep_i64 = (int64_t *)calloc(ncols, sizeof(int64_t));
  double *rep_f64 = (double *)calloc(ncols, sizeof(double));
  const char **rep_str = (const char **)calloc(ncols, sizeof(const char *));
  if (!rep_is_null || !rep_i64 || !rep_f64 || !rep_str) {
    free(rep_is_null);
    free(rep_i64);
    free(rep_f64);
    free(rep_str);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  if (!cp_prepare_replacements(df, values, count, rep_is_null, rep_i64,
                               rep_f64, rep_str, err)) {
    free(rep_is_null);
    free(rep_i64);
    free(rep_f64);
    free(rep_str);
    return NULL;
  }

  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    free(rep_is_null);
    free(rep_i64);
    free(rep_f64);
    free(rep_str);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t col = 0; col < ncols; ++col) {
    dtypes[col] = df->cols[col]->dtype;
    names[col] = df->cols[col]->name;
  }

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, df->nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    free(rep_is_null);
    free(rep_i64);
    free(rep_f64);
    free(rep_str);
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    int keep = mask[row] ? 1 : 0;
    if (invert) {
      keep = keep ? 0 : 1;
    }
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *src = df->cols[col];
      int ok = 0;
      if (keep) {
        ok = cp_series_append_from(dest, src, row, err);
      } else if (rep_is_null[col]) {
        ok = cp_series_append_null(dest, err);
      } else {
        switch (src->dtype) {
          case CP_DTYPE_INT64:
            ok = cp_series_append_int64(dest, rep_i64[col], 0, err);
            break;
          case CP_DTYPE_FLOAT64:
            ok = cp_series_append_float64(dest, rep_f64[col], 0, err);
            break;
          case CP_DTYPE_STRING:
            ok = cp_series_append_string(dest, rep_str[col], 0, err);
            break;
          default:
            cp_error_set(err, CP_ERR_INVALID, row, col, "unknown dtype");
            ok = 0;
            break;
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

  free(rep_is_null);
  free(rep_i64);
  free(rep_f64);
  free(rep_str);
  return out;
}

CpDataFrame *cp_df_where(const CpDataFrame *df,
                         const uint8_t *mask,
                         size_t mask_len,
                         const char **values,
                         size_t count,
                         CpError *err) {
  return cp_df_apply_mask(df, mask, mask_len, values, count, 0, err);
}

CpDataFrame *cp_df_mask(const CpDataFrame *df,
                        const uint8_t *mask,
                        size_t mask_len,
                        const char **values,
                        size_t count,
                        CpError *err) {
  return cp_df_apply_mask(df, mask, mask_len, values, count, 1, err);
}

CpDataFrame *cp_df_clip(const CpDataFrame *df,
                        const char *name,
                        double lower,
                        double upper,
                        CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
    return NULL;
  }
  if (isnan(lower) || isnan(upper)) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid clip bounds");
    return NULL;
  }
  if (lower > upper) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "clip lower > upper");
    return NULL;
  }

  size_t target_idx = 0;
  for (; target_idx < df->ncols; ++target_idx) {
    if (df->cols[target_idx] == series) {
      break;
    }
  }
  if (target_idx >= df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }

  if (series->dtype == CP_DTYPE_INT64) {
    if (lower < (double)INT64_MIN || lower > (double)INT64_MAX ||
        upper < (double)INT64_MIN || upper > (double)INT64_MAX) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "clip bounds out of range");
      return NULL;
    }
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t col = 0; col < ncols; ++col) {
    dtypes[col] = df->cols[col]->dtype;
    names[col] = df->cols[col]->name;
  }
  CpDataFrame *out = cp_df_create(ncols, names, dtypes, nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }

  int64_t lower_i = (int64_t)lower;
  int64_t upper_i = (int64_t)upper;
  if (series->dtype == CP_DTYPE_INT64 && lower_i > upper_i) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "clip bounds invalid");
    cp_df_free(out);
    return NULL;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *src = df->cols[col];
      int ok = 0;
      if (col != target_idx) {
        ok = cp_series_append_from(dest, src, row, err);
      } else if (src->is_null[row]) {
        ok = cp_series_append_null(dest, err);
      } else if (src->dtype == CP_DTYPE_FLOAT64 &&
                 cp_series_is_nan(src, row)) {
        ok = cp_series_append_float64(dest, src->data.f64[row], 0, err);
      } else if (src->dtype == CP_DTYPE_INT64) {
        int64_t v = src->data.i64[row];
        if (v < lower_i) {
          v = lower_i;
        } else if (v > upper_i) {
          v = upper_i;
        }
        ok = cp_series_append_int64(dest, v, 0, err);
      } else {
        double v = src->data.f64[row];
        if (v < lower) {
          v = lower;
        } else if (v > upper) {
          v = upper;
        }
        ok = cp_series_append_float64(dest, v, 0, err);
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
  return out;
}

CpDataFrame *cp_df_replace(const CpDataFrame *df,
                           const char *name,
                           const char *old_value,
                           const char *new_value,
                           CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }

  size_t target_idx = 0;
  for (; target_idx < df->ncols; ++target_idx) {
    if (df->cols[target_idx] == series) {
      break;
    }
  }
  if (target_idx >= df->ncols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }

  int old_is_null = 0;
  int new_is_null = 0;
  int old_is_nan = 0;
  int64_t old_i64 = 0;
  int64_t new_i64 = 0;
  double old_f64 = 0.0;
  double new_f64 = 0.0;
  const char *old_str = NULL;
  const char *new_str = NULL;

  switch (series->dtype) {
    case CP_DTYPE_INT64:
      if (!old_value) {
        old_is_null = 1;
      } else if (!cp_parse_int64(old_value, &old_i64, &old_is_null, err, 0,
                                 target_idx)) {
        return NULL;
      }
      if (!new_value) {
        new_is_null = 1;
      } else if (!cp_parse_int64(new_value, &new_i64, &new_is_null, err, 0,
                                 target_idx)) {
        return NULL;
      }
      break;
    case CP_DTYPE_FLOAT64:
      if (!old_value) {
        old_is_null = 1;
      } else if (!cp_parse_float64(old_value, &old_f64, &old_is_null, err, 0,
                                   target_idx)) {
        return NULL;
      }
      if (!old_is_null && isnan(old_f64)) {
        old_is_nan = 1;
      }
      if (!new_value) {
        new_is_null = 1;
      } else if (!cp_parse_float64(new_value, &new_f64, &new_is_null, err, 0,
                                   target_idx)) {
        return NULL;
      }
      break;
    case CP_DTYPE_STRING:
      if (!cp_parse_string(old_value, &old_str, &old_is_null)) {
        cp_error_set(err, CP_ERR_INVALID, 0, target_idx, "invalid string");
        return NULL;
      }
      if (!cp_parse_string(new_value, &new_str, &new_is_null)) {
        cp_error_set(err, CP_ERR_INVALID, 0, target_idx, "invalid string");
        return NULL;
      }
      break;
    default:
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
      return NULL;
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t col = 0; col < ncols; ++col) {
    dtypes[col] = df->cols[col]->dtype;
    names[col] = df->cols[col]->name;
  }
  CpDataFrame *out = cp_df_create(ncols, names, dtypes, nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *src = df->cols[col];
      int ok = 0;
      if (col != target_idx) {
        ok = cp_series_append_from(dest, src, row, err);
      } else {
        int match = 0;
        if (old_is_null) {
          match = src->is_null[row] ? 1 : 0;
        } else if (series->dtype == CP_DTYPE_FLOAT64 && old_is_nan) {
          match = (!src->is_null[row] && isnan(src->data.f64[row])) ? 1 : 0;
        } else if (!src->is_null[row]) {
          switch (series->dtype) {
            case CP_DTYPE_INT64:
              match = src->data.i64[row] == old_i64;
              break;
            case CP_DTYPE_FLOAT64:
              match = src->data.f64[row] == old_f64;
              break;
            case CP_DTYPE_STRING:
              match = src->data.str[row] &&
                      strcmp(src->data.str[row], old_str) == 0;
              break;
            default:
              match = 0;
              break;
          }
        }

        if (match) {
          if (new_is_null) {
            ok = cp_series_append_null(dest, err);
          } else {
            switch (series->dtype) {
              case CP_DTYPE_INT64:
                ok = cp_series_append_int64(dest, new_i64, 0, err);
                break;
              case CP_DTYPE_FLOAT64:
                ok = cp_series_append_float64(dest, new_f64, 0, err);
                break;
              case CP_DTYPE_STRING:
                ok = cp_series_append_string(dest, new_str, 0, err);
                break;
              default:
                ok = 0;
                break;
            }
          }
        } else {
          ok = cp_series_append_from(dest, src, row, err);
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

  return out;
}

CpDataFrame *cp_df_astype(const CpDataFrame *df,
                          const char *name,
                          CpDType dtype,
                          CpError *err) {
  if (!df || !name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid astype");
    return NULL;
  }
  size_t target = 0;
  if (!cp_df_find_col_index(df, name, &target, err)) {
    return NULL;
  }
  const CpSeries *src = df->cols[target];
  if (!src) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }
  if (src->dtype == dtype) {
    return cp_df_copy(df, err);
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t col = 0; col < ncols; ++col) {
    dtypes[col] = df->cols[col]->dtype;
    names[col] = df->cols[col]->name;
  }
  dtypes[target] = dtype;

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }
  if (df->has_index && df->index_col < ncols) {
    out->has_index = 1;
    out->index_col = df->index_col;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *col_src = df->cols[col];
      int ok = 0;
      if (col != target) {
        ok = cp_series_append_from(dest, col_src, row, err);
      } else {
        if (col_src->is_null[row]) {
          ok = cp_series_append_null(dest, err);
        } else {
          switch (col_src->dtype) {
            case CP_DTYPE_INT64:
              if (dtype == CP_DTYPE_FLOAT64) {
                ok = cp_series_append_float64(dest,
                                              (double)col_src->data.i64[row],
                                              0,
                                              err);
              } else if (dtype == CP_DTYPE_STRING) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%" PRId64, col_src->data.i64[row]);
                ok = cp_series_append_string(dest, buf, 0, err);
              } else {
                cp_error_set(err, CP_ERR_INVALID, row, col, "invalid cast");
                ok = 0;
              }
              break;
            case CP_DTYPE_FLOAT64:
              if (dtype == CP_DTYPE_INT64) {
                double v = col_src->data.f64[row];
                if (isnan(v)) {
                  ok = cp_series_append_null(dest, err);
                  break;
                }
                if (!isfinite(v) ||
                    v < (double)INT64_MIN ||
                    v > (double)INT64_MAX) {
                  cp_error_set(err, CP_ERR_INVALID, row, col,
                               "float64 out of int64 range");
                  ok = 0;
                  break;
                }
                double intpart = 0.0;
                double frac = modf(v, &intpart);
                if (fabs(frac) > 1e-9) {
                  cp_error_set(err, CP_ERR_INVALID, row, col,
                               "float64 has fractional part");
                  ok = 0;
                  break;
                }
                ok = cp_series_append_int64(dest, (int64_t)intpart, 0, err);
              } else if (dtype == CP_DTYPE_STRING) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.17g", col_src->data.f64[row]);
                ok = cp_series_append_string(dest, buf, 0, err);
              } else {
                cp_error_set(err, CP_ERR_INVALID, row, col, "invalid cast");
                ok = 0;
              }
              break;
            case CP_DTYPE_STRING:
              if (dtype == CP_DTYPE_INT64) {
                int64_t v = 0;
                int is_null = 0;
                if (!cp_parse_int64(col_src->data.str[row], &v, &is_null,
                                    err, row, col)) {
                  ok = 0;
                } else if (is_null) {
                  ok = cp_series_append_null(dest, err);
                } else {
                  ok = cp_series_append_int64(dest, v, 0, err);
                }
              } else if (dtype == CP_DTYPE_FLOAT64) {
                double v = 0.0;
                int is_null = 0;
                if (!cp_parse_float64(col_src->data.str[row], &v, &is_null,
                                      err, row, col)) {
                  ok = 0;
                } else if (is_null) {
                  ok = cp_series_append_null(dest, err);
                } else {
                  ok = cp_series_append_float64(dest, v, 0, err);
                }
              } else if (dtype == CP_DTYPE_STRING) {
                const char *v = col_src->data.str[row];
                ok = cp_series_append_string(dest, v, 0, err);
              } else {
                cp_error_set(err, CP_ERR_INVALID, row, col, "invalid cast");
                ok = 0;
              }
              break;
            default:
              cp_error_set(err, CP_ERR_INVALID, row, col, "invalid dtype");
              ok = 0;
              break;
          }
        }
      }
      if (!ok) {
        for (size_t j = 0; j < col; ++j) {
          cp_series_pop(out->cols[j]);
        }
        cp_df_free(out);
        return NULL;
      }
    }
    out->nrows += 1;
  }

  return out;
}

CpDataFrame *cp_df_to_numeric(const CpDataFrame *df,
                              const char *name,
                              CpError *err) {
  return cp_df_astype(df, name, CP_DTYPE_FLOAT64, err);
}

CpDataFrame *cp_df_to_datetime(const CpDataFrame *df,
                               const char *name,
                               CpError *err) {
  if (!df || !name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid to_datetime");
    return NULL;
  }
  size_t target = 0;
  if (!cp_df_find_col_index(df, name, &target, err)) {
    return NULL;
  }
  const CpSeries *src = df->cols[target];
  if (!src) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }
  if (src->dtype == CP_DTYPE_INT64) {
    return cp_df_copy(df, err);
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t col = 0; col < ncols; ++col) {
    dtypes[col] = df->cols[col]->dtype;
    names[col] = df->cols[col]->name;
  }
  dtypes[target] = CP_DTYPE_INT64;

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }
  if (df->has_index && df->index_col < ncols) {
    out->has_index = 1;
    out->index_col = df->index_col;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *col_src = df->cols[col];
      int ok = 0;
      if (col != target) {
        ok = cp_series_append_from(dest, col_src, row, err);
      } else {
        if (col_src->is_null[row]) {
          ok = cp_series_append_null(dest, err);
        } else {
          switch (col_src->dtype) {
            case CP_DTYPE_INT64:
              ok = cp_series_append_int64(dest, col_src->data.i64[row], 0, err);
              break;
            case CP_DTYPE_FLOAT64: {
              double v = col_src->data.f64[row];
              if (isnan(v)) {
                ok = cp_series_append_null(dest, err);
                break;
              }
              if (!isfinite(v) ||
                  v < (double)INT64_MIN ||
                  v > (double)INT64_MAX) {
                cp_error_set(err, CP_ERR_INVALID, row, col,
                             "float64 out of int64 range");
                ok = 0;
                break;
              }
              double intpart = 0.0;
              double frac = modf(v, &intpart);
              if (fabs(frac) > 1e-9) {
                cp_error_set(err, CP_ERR_INVALID, row, col,
                             "float64 has fractional part");
                ok = 0;
                break;
              }
              ok = cp_series_append_int64(dest, (int64_t)intpart, 0, err);
              break;
            }
            case CP_DTYPE_STRING: {
              int64_t v = 0;
              int is_null = 0;
              if (!cp_parse_datetime(col_src->data.str[row], &v, &is_null, err,
                                     row, col)) {
                ok = 0;
              } else if (is_null) {
                ok = cp_series_append_null(dest, err);
              } else {
                ok = cp_series_append_int64(dest, v, 0, err);
              }
              break;
            }
            default:
              cp_error_set(err, CP_ERR_INVALID, row, col,
                           "invalid datetime dtype");
              ok = 0;
              break;
          }
        }
      }
      if (!ok) {
        for (size_t j = 0; j < col; ++j) {
          cp_series_pop(out->cols[j]);
        }
        cp_df_free(out);
        return NULL;
      }
    }
    out->nrows += 1;
  }

  return out;
}

CpDataFrame *cp_df_set_index(const CpDataFrame *df,
                             const char *name,
                             CpError *err) {
  if (!df || !name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid set_index");
    return NULL;
  }
  size_t idx = 0;
  if (!cp_df_find_col_index(df, name, &idx, err)) {
    return NULL;
  }
  const CpSeries *series = df->cols[idx];
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported index dtype");
    return NULL;
  }

  CpDataFrame *out = cp_df_copy(df, err);
  if (!out) {
    return NULL;
  }
  out->has_index = 1;
  out->index_col = idx;
  return out;
}

CpDataFrame *cp_df_reset_index(const CpDataFrame *df,
                               CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid reset_index");
    return NULL;
  }
  CpDataFrame *out = cp_df_copy(df, err);
  if (!out) {
    return NULL;
  }
  out->has_index = 0;
  out->index_col = 0;
  return out;
}

int cp_df_at_int64(const CpDataFrame *df,
                   const char *row_label,
                   const char *col_name,
                   int64_t *out,
                   int *is_null,
                   CpError *err) {
  if (!df || !row_label || !col_name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid at request");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, col_name, err);
  if (!series) {
    return 0;
  }
  if (series->dtype != CP_DTYPE_INT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  size_t row = 0;
  if (!cp_df_find_row_label(df, row_label, &row, err)) {
    return 0;
  }
  if (!cp_series_get_int64(series, row, out, is_null)) {
    cp_error_set(err, CP_ERR_INVALID, row, 0, "row index out of range");
    return 0;
  }
  return 1;
}

int cp_df_at_float64(const CpDataFrame *df,
                     const char *row_label,
                     const char *col_name,
                     double *out,
                     int *is_null,
                     CpError *err) {
  if (!df || !row_label || !col_name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid at request");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, col_name, err);
  if (!series) {
    return 0;
  }
  if (series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  size_t row = 0;
  if (!cp_df_find_row_label(df, row_label, &row, err)) {
    return 0;
  }
  if (!cp_series_get_float64(series, row, out, is_null)) {
    cp_error_set(err, CP_ERR_INVALID, row, 0, "row index out of range");
    return 0;
  }
  return 1;
}

int cp_df_at_string(const CpDataFrame *df,
                    const char *row_label,
                    const char *col_name,
                    const char **out,
                    int *is_null,
                    CpError *err) {
  if (!df || !row_label || !col_name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid at request");
    return 0;
  }
  const CpSeries *series = cp_df_require_col(df, col_name, err);
  if (!series) {
    return 0;
  }
  if (series->dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dtype mismatch");
    return 0;
  }
  size_t row = 0;
  if (!cp_df_find_row_label(df, row_label, &row, err)) {
    return 0;
  }
  if (!cp_series_get_string(series, row, out, is_null)) {
    cp_error_set(err, CP_ERR_INVALID, row, 0, "row index out of range");
    return 0;
  }
  return 1;
}

CpDataFrame *cp_df_apply(const CpDataFrame *df,
                         CpDType out_dtype,
                         const char *out_name,
                         CpApplyFn func,
                         void *user_data,
                         CpError *err) {
  if (!df || !func) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid apply");
    return NULL;
  }
  if (out_dtype != CP_DTYPE_INT64 &&
      out_dtype != CP_DTYPE_FLOAT64 &&
      out_dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid output dtype");
    return NULL;
  }
  const char *name = out_name ? out_name : "apply";
  CpDType dtype = out_dtype;
  CpDataFrame *out = cp_df_create(1, &name, &dtype, df->nrows, err);
  if (!out) {
    return NULL;
  }

  for (size_t row = 0; row < df->nrows; ++row) {
    CpValue value;
    memset(&value, 0, sizeof(value));
    if (!func(df, row, user_data, &value, err)) {
      if (err && err->code == CP_OK) {
        cp_error_set(err, CP_ERR_INVALID, row, 0, "apply failed");
      }
      cp_df_free(out);
      return NULL;
    }
    if (!cp_series_append_value(out->cols[0], out_dtype, &value, row, 0, err)) {
      cp_df_free(out);
      return NULL;
    }
    out->nrows += 1;
  }

  return out;
}

CpDataFrame *cp_df_transform(const CpDataFrame *df,
                             const char *name,
                             CpDType out_dtype,
                             CpTransformFn func,
                             void *user_data,
                             CpError *err) {
  if (!df || !name || !func) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid transform");
    return NULL;
  }
  if (out_dtype != CP_DTYPE_INT64 &&
      out_dtype != CP_DTYPE_FLOAT64 &&
      out_dtype != CP_DTYPE_STRING) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid output dtype");
    return NULL;
  }
  size_t target = 0;
  if (!cp_df_find_col_index(df, name, &target, err)) {
    return NULL;
  }
  const CpSeries *src = df->cols[target];
  if (!src) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "column not found");
    return NULL;
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
  const char **names = (const char **)malloc(ncols * sizeof(const char *));
  if (!dtypes || !names) {
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  for (size_t col = 0; col < ncols; ++col) {
    dtypes[col] = df->cols[col]->dtype;
    names[col] = df->cols[col]->name;
  }
  dtypes[target] = out_dtype;

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, nrows, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }
  if (df->has_index && df->index_col < ncols) {
    out->has_index = 1;
    out->index_col = df->index_col;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      CpSeries *dest = out->cols[col];
      const CpSeries *col_src = df->cols[col];
      int ok = 0;
      if (col != target) {
        ok = cp_series_append_from(dest, col_src, row, err);
      } else {
        CpValue value;
        memset(&value, 0, sizeof(value));
        if (!func(src, row, user_data, &value, err)) {
          if (err && err->code == CP_OK) {
            cp_error_set(err, CP_ERR_INVALID, row, col, "transform failed");
          }
          ok = 0;
        } else {
          ok = cp_series_append_value(dest, out_dtype, &value, row, col, err);
        }
      }
      if (!ok) {
        for (size_t j = 0; j < col; ++j) {
          cp_series_pop(out->cols[j]);
        }
        cp_df_free(out);
        return NULL;
      }
    }
    out->nrows += 1;
  }

  return out;
}

int cp_df_iterrows(const CpDataFrame *df,
                   CpIterRowFn func,
                   void *user_data,
                   CpError *err) {
  if (!df || !func) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid iterrows");
    return 0;
  }
  for (size_t row = 0; row < df->nrows; ++row) {
    if (!func(df, row, user_data, err)) {
      if (err && err->code == CP_OK) {
        cp_error_set(err, CP_ERR_INVALID, row, 0, "iterrows failed");
      }
      return 0;
    }
  }
  return 1;
}

int cp_df_iteritems(const CpDataFrame *df,
                    CpIterItemFn func,
                    void *user_data,
                    CpError *err) {
  if (!df || !func) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid iteritems");
    return 0;
  }
  for (size_t col = 0; col < df->ncols; ++col) {
    const CpSeries *series = df->cols[col];
    if (!series) {
      cp_error_set(err, CP_ERR_INVALID, 0, col, "column not found");
      return 0;
    }
    if (!func(series, col, user_data, err)) {
      if (err && err->code == CP_OK) {
        cp_error_set(err, CP_ERR_INVALID, 0, col, "iteritems failed");
      }
      return 0;
    }
  }
  return 1;
}

CpDataFrame *cp_df_diff(const CpDataFrame *df,
                        const char *name,
                        CpError *err) {
  if (!df || !name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid diff");
    return NULL;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
    return NULL;
  }

  size_t nrows = df->nrows;
  const char *names[1] = {series->name};
  CpDType dtypes[1] = {series->dtype};
  CpDataFrame *out = cp_df_create(1, names, dtypes, nrows, err);
  if (!out) {
    return NULL;
  }
  if (df->has_index && df->index_col < df->ncols &&
      df->cols[df->index_col] == series) {
    out->has_index = 1;
    out->index_col = 0;
  }

  for (size_t row = 0; row < nrows; ++row) {
    int ok = 0;
    if (row == 0) {
      ok = cp_series_append_null(out->cols[0], err);
    } else if (series->dtype == CP_DTYPE_INT64) {
      if (series->is_null[row] || series->is_null[row - 1]) {
        ok = cp_series_append_null(out->cols[0], err);
      } else {
        int64_t curr = series->data.i64[row];
        int64_t prev = series->data.i64[row - 1];
        if ((prev > 0 && curr < INT64_MIN + prev) ||
            (prev < 0 && curr > INT64_MAX + prev)) {
          cp_error_set(err, CP_ERR_INVALID, row, 0, "int64 diff overflow");
          ok = 0;
        } else {
          ok = cp_series_append_int64(out->cols[0], curr - prev, 0, err);
        }
      }
    } else {
      if (series->is_null[row] || series->is_null[row - 1] ||
          cp_series_is_nan(series, row) || cp_series_is_nan(series, row - 1)) {
        ok = cp_series_append_null(out->cols[0], err);
      } else {
        double diff = series->data.f64[row] - series->data.f64[row - 1];
        ok = cp_series_append_float64(out->cols[0], diff, 0, err);
      }
    }
    if (!ok) {
      cp_df_free(out);
      return NULL;
    }
    out->nrows += 1;
  }

  return out;
}

CpDataFrame *cp_df_rank(const CpDataFrame *df,
                        const char *name,
                        CpError *err) {
  if (!df || !name) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid rank");
    return NULL;
  }
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
    return NULL;
  }

  size_t nrows = df->nrows;
  const char *names[1] = {series->name};
  CpDType dtypes[1] = {CP_DTYPE_FLOAT64};
  CpDataFrame *out = cp_df_create(1, names, dtypes, nrows, err);
  if (!out) {
    return NULL;
  }
  if (df->has_index && df->index_col < df->ncols &&
      df->cols[df->index_col] == series) {
    out->has_index = 1;
    out->index_col = 0;
  }

  size_t *indices = NULL;
  size_t *tmp = NULL;
  double *ranks = NULL;
  uint8_t *valid = NULL;
  if (nrows > 0) {
    indices = (size_t *)malloc(nrows * sizeof(size_t));
    tmp = (size_t *)malloc(nrows * sizeof(size_t));
    ranks = (double *)malloc(nrows * sizeof(double));
    valid = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  }
  if ((nrows > 0 && (!indices || !tmp || !ranks || !valid))) {
    free(indices);
    free(tmp);
    free(ranks);
    free(valid);
    cp_df_free(out);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  size_t count = 0;
  for (size_t row = 0; row < nrows; ++row) {
    if (!cp_series_is_valid_numeric(series, row)) {
      continue;
    }
    indices[count++] = row;
    valid[row] = 1;
  }

  if (count > 1) {
    cp_sort_indices_merge(indices, tmp, 0, count, series, 1);
  }

  size_t rank_pos = 1;
  size_t pos = 0;
  while (pos < count) {
    size_t start = pos;
    size_t end = pos;
    while (end + 1 < count &&
           cp_series_value_equal(series, indices[end], indices[end + 1])) {
      end += 1;
    }
    size_t span = end - start + 1;
    double avg_rank = ((double)rank_pos +
                       (double)(rank_pos + span - 1)) / 2.0;
    for (size_t i = start; i <= end; ++i) {
      ranks[indices[i]] = avg_rank;
    }
    rank_pos += span;
    pos = end + 1;
  }

  for (size_t row = 0; row < nrows; ++row) {
    int ok = 0;
    if (!valid || !valid[row]) {
      ok = cp_series_append_null(out->cols[0], err);
    } else {
      ok = cp_series_append_float64(out->cols[0], ranks[row], 0, err);
    }
    if (!ok) {
      free(indices);
      free(tmp);
      free(ranks);
      free(valid);
      cp_df_free(out);
      return NULL;
    }
    out->nrows += 1;
  }

  free(indices);
  free(tmp);
  free(ranks);
  free(valid);

  return out;
}

static int cp_series_pair_stat(const CpSeries *left,
                               const CpSeries *right,
                               int want_corr,
                               double *out) {
  if (!left || !right || !out) {
    return 0;
  }
  if ((left->dtype != CP_DTYPE_INT64 && left->dtype != CP_DTYPE_FLOAT64) ||
      (right->dtype != CP_DTYPE_INT64 && right->dtype != CP_DTYPE_FLOAT64)) {
    return 0;
  }
  size_t nrows = left->length < right->length ? left->length : right->length;
  size_t count = 0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xy = 0.0;
  double sum_x2 = 0.0;
  double sum_y2 = 0.0;
  for (size_t row = 0; row < nrows; ++row) {
    double x = 0.0;
    double y = 0.0;
    if (!cp_series_get_numeric(left, row, &x) ||
        !cp_series_get_numeric(right, row, &y)) {
      continue;
    }
    count += 1;
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_x2 += x * x;
    sum_y2 += y * y;
  }
  if (count < 2) {
    return 0;
  }
  double mean_x = sum_x / (double)count;
  double mean_y = sum_y / (double)count;
  double cov = (sum_xy - (double)count * mean_x * mean_y) /
               (double)(count - 1);
  if (!want_corr) {
    *out = cov;
    return 1;
  }
  double var_x = (sum_x2 - (double)count * mean_x * mean_x) /
                 (double)(count - 1);
  double var_y = (sum_y2 - (double)count * mean_y * mean_y) /
                 (double)(count - 1);
  if (var_x <= 0.0 || var_y <= 0.0) {
    return 0;
  }
  double denom = sqrt(var_x * var_y);
  if (denom == 0.0) {
    return 0;
  }
  *out = cov / denom;
  return 1;
}

static CpDataFrame *cp_df_corr_cov_internal(const CpDataFrame *df,
                                            int want_corr,
                                            CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  size_t ncols = df->ncols;
  if (ncols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns");
    return NULL;
  }
  size_t *num_cols = (size_t *)malloc(ncols * sizeof(size_t));
  const char **num_names = (const char **)malloc(ncols * sizeof(const char *));
  if (!num_cols || !num_names) {
    free(num_cols);
    free(num_names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  size_t num_count = 0;
  for (size_t col = 0; col < ncols; ++col) {
    CpSeries *series = df->cols[col];
    if (!series) {
      continue;
    }
    if (series->dtype == CP_DTYPE_INT64 || series->dtype == CP_DTYPE_FLOAT64) {
      num_cols[num_count] = col;
      num_names[num_count] = series->name ? series->name : "";
      num_count += 1;
    }
  }
  if (num_count == 0) {
    free(num_cols);
    free(num_names);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no numeric columns");
    return NULL;
  }

  const char *header = "column";
  char header_buf[64];
  if (cp_name_in_list(header, num_names, num_count)) {
    int suffix = 1;
    do {
      snprintf(header_buf, sizeof(header_buf), "column_%d", suffix++);
    } while (cp_name_in_list(header_buf, num_names, num_count));
    header = header_buf;
  }

  size_t out_cols = num_count + 1;
  CpDType *dtypes = (CpDType *)malloc(out_cols * sizeof(CpDType));
  const char **names = (const char **)malloc(out_cols * sizeof(const char *));
  if (!dtypes || !names) {
    free(num_cols);
    free(num_names);
    free(dtypes);
    free(names);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  names[0] = header;
  dtypes[0] = CP_DTYPE_STRING;
  for (size_t i = 0; i < num_count; ++i) {
    names[i + 1] = num_names[i];
    dtypes[i + 1] = CP_DTYPE_FLOAT64;
  }

  CpDataFrame *out = cp_df_create(out_cols, names, dtypes, num_count, err);
  free(dtypes);
  free(names);
  if (!out) {
    free(num_cols);
    free(num_names);
    return NULL;
  }

  for (size_t i = 0; i < num_count; ++i) {
    const CpSeries *row_series = df->cols[num_cols[i]];
    if (!cp_series_append_string(out->cols[0], row_series->name, 0, err)) {
      cp_df_free(out);
      free(num_cols);
      free(num_names);
      return NULL;
    }
    for (size_t j = 0; j < num_count; ++j) {
      const CpSeries *col_series = df->cols[num_cols[j]];
      double stat = 0.0;
      int ok = cp_series_pair_stat(row_series, col_series, want_corr, &stat);
      if (ok) {
        if (!cp_series_append_float64(out->cols[j + 1], stat, 0, err)) {
          cp_df_free(out);
          free(num_cols);
          free(num_names);
          return NULL;
        }
      } else {
        if (!cp_series_append_null(out->cols[j + 1], err)) {
          cp_df_free(out);
          free(num_cols);
          free(num_names);
          return NULL;
        }
      }
    }
    out->nrows += 1;
  }

  free(num_cols);
  free(num_names);
  return out;
}

CpDataFrame *cp_df_corr(const CpDataFrame *df, CpError *err) {
  return cp_df_corr_cov_internal(df, 1, err);
}

CpDataFrame *cp_df_cov(const CpDataFrame *df, CpError *err) {
  return cp_df_corr_cov_internal(df, 0, err);
}

CpDataFrame *cp_df_query(const CpDataFrame *df,
                         const char *expr,
                         CpError *err) {
  if (!df || !expr) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid query");
    return NULL;
  }

  const char *cursor = expr;
  CpQueryNode *root = cp_query_parse_expr(df, &cursor, err);
  if (!root) {
    return NULL;
  }
  cursor = cp_skip_space(cursor);
  if (cursor && *cursor != '\0') {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unexpected query content");
    cp_query_node_free(root);
    return NULL;
  }

  size_t nrows = df->nrows;
  if (nrows == 0) {
    cp_query_node_free(root);
    return cp_df_empty_like(df, err);
  }
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    cp_query_node_free(root);
    return NULL;
  }
  for (size_t row = 0; row < nrows; ++row) {
    int match = 0;
    if (!cp_query_eval_node(root, row, &match, err)) {
      free(mask);
      cp_query_node_free(root);
      return NULL;
    }
    mask[row] = match ? 1 : 0;
  }
  CpDataFrame *out = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  cp_query_node_free(root);
  return out;
}

CpDataFrame *cp_df_concat(const CpDataFrame **dfs,
                          size_t count,
                          CpConcatAxis axis,
                          CpError *err) {
  if (!dfs || count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid concat input");
    return NULL;
  }
  if (axis != CP_CONCAT_ROWS && axis != CP_CONCAT_COLS) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid concat axis");
    return NULL;
  }

  const CpDataFrame *base = dfs[0];
  if (!base) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }

  if (axis == CP_CONCAT_ROWS) {
    size_t total_rows = 0;
    for (size_t i = 0; i < count; ++i) {
      const CpDataFrame *df = dfs[i];
      if (!df) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
        return NULL;
      }
      if (!cp_df_schema_matches(base, df, err)) {
        return NULL;
      }
      if (df->nrows > SIZE_MAX - total_rows) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
        return NULL;
      }
      total_rows += df->nrows;
    }

    size_t ncols = base->ncols;
    if (ncols == 0) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns");
      return NULL;
    }
    CpDType *dtypes = (CpDType *)malloc(ncols * sizeof(CpDType));
    const char **names = (const char **)malloc(ncols * sizeof(const char *));
    if (!dtypes || !names) {
      free(dtypes);
      free(names);
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      return NULL;
    }
    for (size_t col = 0; col < ncols; ++col) {
      dtypes[col] = base->cols[col]->dtype;
      names[col] = base->cols[col]->name;
    }
    CpDataFrame *out = cp_df_create(ncols, names, dtypes, total_rows, err);
    free(dtypes);
    free(names);
    if (!out) {
      return NULL;
    }

    for (size_t i = 0; i < count; ++i) {
      const CpDataFrame *df = dfs[i];
      const CpSeries **src_cols = (const CpSeries **)df->cols;
      for (size_t row = 0; row < df->nrows; ++row) {
        if (!cp_df_append_row_from_sources(out, src_cols, ncols, row, err)) {
          cp_df_free(out);
          return NULL;
        }
      }
    }
    return out;
  }

  size_t total_cols = 0;
  size_t total_rows = base->nrows;
  for (size_t i = 0; i < count; ++i) {
    const CpDataFrame *df = dfs[i];
    if (!df) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
      return NULL;
    }
    if (df->nrows != total_rows) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count mismatch");
      return NULL;
    }
    if (df->ncols > SIZE_MAX - total_cols) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "column count overflow");
      return NULL;
    }
    total_cols += df->ncols;
  }
  if (total_cols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "no columns");
    return NULL;
  }

  CpDType *dtypes = (CpDType *)malloc(total_cols * sizeof(CpDType));
  const char **names =
      (const char **)malloc(total_cols * sizeof(const char *));
  const CpSeries **src_cols =
      (const CpSeries **)malloc(total_cols * sizeof(const CpSeries *));
  if (!dtypes || !names || !src_cols) {
    free(dtypes);
    free(names);
    free(src_cols);
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  size_t idx = 0;
  for (size_t i = 0; i < count; ++i) {
    const CpDataFrame *df = dfs[i];
    for (size_t col = 0; col < df->ncols; ++col) {
      names[idx] = df->cols[col]->name;
      dtypes[idx] = df->cols[col]->dtype;
      src_cols[idx] = df->cols[col];
      idx += 1;
    }
  }

  if (cp_names_have_duplicates(names, total_cols)) {
    free(dtypes);
    free(names);
    free(src_cols);
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "duplicate column names");
    return NULL;
  }

  CpDataFrame *out = cp_df_create(total_cols, names, dtypes, total_rows, err);
  free(dtypes);
  free(names);
  if (!out) {
    free(src_cols);
    return NULL;
  }

  for (size_t row = 0; row < total_rows; ++row) {
    if (!cp_df_append_row_from_sources(out, src_cols, total_cols, row, err)) {
      cp_df_free(out);
      free(src_cols);
      return NULL;
    }
  }

  free(src_cols);
  return out;
}

CpDataFrame *cp_df_sample(const CpDataFrame *df,
                          size_t n,
                          int replace,
                          uint32_t seed,
                          CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (n == 0) {
    return cp_df_empty_like(df, err);
  }
  size_t nrows = df->nrows;
  if (nrows == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "empty dataframe");
    return NULL;
  }
  if (!replace && n > nrows) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "sample size exceeds rows");
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

  CpDataFrame *out = cp_df_create(ncols, names, dtypes, n, err);
  free(dtypes);
  free(names);
  if (!out) {
    return NULL;
  }

  const CpSeries **src_cols = (const CpSeries **)df->cols;
  uint32_t state = seed;
  if (replace) {
    for (size_t i = 0; i < n; ++i) {
      size_t row = (size_t)(cp_rand_next(&state) % nrows);
      if (!cp_df_append_row_from_sources(out, src_cols, ncols, row, err)) {
        cp_df_free(out);
        return NULL;
      }
    }
    return out;
  }

  size_t *indices = (size_t *)malloc(nrows * sizeof(size_t));
  if (!indices) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    cp_df_free(out);
    return NULL;
  }
  for (size_t i = 0; i < nrows; ++i) {
    indices[i] = i;
  }
  for (size_t i = 0; i < n; ++i) {
    size_t span = nrows - i;
    size_t j = i + (size_t)(cp_rand_next(&state) % span);
    size_t tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
    if (!cp_df_append_row_from_sources(out, src_cols, ncols, indices[i], err)) {
      free(indices);
      cp_df_free(out);
      return NULL;
    }
  }
  free(indices);
  return out;
}

CpDataFrame *cp_df_nlargest(const CpDataFrame *df,
                            const char *name,
                            size_t n,
                            CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
    return NULL;
  }
  if (n == 0 || df->nrows == 0) {
    return cp_df_empty_like(df, err);
  }

  size_t nrows = df->nrows;
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  size_t valid = 0;
  for (size_t row = 0; row < nrows; ++row) {
    if (cp_series_is_valid_numeric(series, row)) {
      mask[row] = 1;
      valid += 1;
    }
  }
  if (valid == 0) {
    free(mask);
    return cp_df_empty_like(df, err);
  }
  CpDataFrame *filtered = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  if (!filtered) {
    return NULL;
  }
  CpDataFrame *sorted = cp_df_sort_values(filtered, name, 0, err);
  cp_df_free(filtered);
  if (!sorted) {
    return NULL;
  }
  CpDataFrame *out = cp_df_head(sorted, n, err);
  cp_df_free(sorted);
  return out;
}

CpDataFrame *cp_df_nsmallest(const CpDataFrame *df,
                             const char *name,
                             size_t n,
                             CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return NULL;
  }
  if (series->dtype != CP_DTYPE_INT64 && series->dtype != CP_DTYPE_FLOAT64) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported dtype");
    return NULL;
  }
  if (n == 0 || df->nrows == 0) {
    return cp_df_empty_like(df, err);
  }

  size_t nrows = df->nrows;
  uint8_t *mask = (uint8_t *)calloc(nrows, sizeof(uint8_t));
  if (!mask) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }
  size_t valid = 0;
  for (size_t row = 0; row < nrows; ++row) {
    if (cp_series_is_valid_numeric(series, row)) {
      mask[row] = 1;
      valid += 1;
    }
  }
  if (valid == 0) {
    free(mask);
    return cp_df_empty_like(df, err);
  }
  CpDataFrame *filtered = cp_df_filter_mask(df, mask, nrows, err);
  free(mask);
  if (!filtered) {
    return NULL;
  }
  CpDataFrame *sorted = cp_df_sort_values(filtered, name, 1, err);
  cp_df_free(filtered);
  if (!sorted) {
    return NULL;
  }
  CpDataFrame *out = cp_df_head(sorted, n, err);
  cp_df_free(sorted);
  return out;
}

static size_t cp_series_value_len(const CpSeries *series, size_t row) {
  if (!series || row >= series->length || series->is_null[row]) {
    return 4; /* "null" */
  }
  switch (series->dtype) {
    case CP_DTYPE_INT64: {
      char buf[32];
      int len = snprintf(buf, sizeof(buf), "%" PRId64, series->data.i64[row]);
      return len > 0 ? (size_t)len : 0;
    }
    case CP_DTYPE_FLOAT64: {
      char buf[64];
      int len = snprintf(buf, sizeof(buf), "%.17g", series->data.f64[row]);
      return len > 0 ? (size_t)len : 0;
    }
    case CP_DTYPE_STRING: {
      const char *value = series->data.str[row];
      return value ? strlen(value) : 0;
    }
    default:
      return 0;
  }
}

static const char *cp_series_value_repr(const CpSeries *series,
                                        size_t row,
                                        char *buf,
                                        size_t buf_len,
                                        size_t *out_len) {
  if (!series || row >= series->length || series->is_null[row]) {
    if (out_len) {
      *out_len = 4;
    }
    return "null";
  }
  switch (series->dtype) {
    case CP_DTYPE_INT64: {
      int len = snprintf(buf, buf_len, "%" PRId64, series->data.i64[row]);
      if (out_len) {
        *out_len = len > 0 ? (size_t)len : 0;
      }
      return buf;
    }
    case CP_DTYPE_FLOAT64: {
      int len = snprintf(buf, buf_len, "%.17g", series->data.f64[row]);
      if (out_len) {
        *out_len = len > 0 ? (size_t)len : 0;
      }
      return buf;
    }
    case CP_DTYPE_STRING: {
      const char *value = series->data.str[row];
      if (out_len) {
        *out_len = value ? strlen(value) : 0;
      }
      return value ? value : "";
    }
    default:
      if (out_len) {
        *out_len = 0;
      }
      return "";
  }
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

char *cp_df_to_string(const CpDataFrame *df, CpError *err) {
  if (!df) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid dataframe");
    return NULL;
  }
  if (df->ncols == 0) {
    char *empty = cp_strdup("");
    if (!empty) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    }
    return empty;
  }

  size_t ncols = df->ncols;
  size_t nrows = df->nrows;
  size_t *widths = (size_t *)calloc(ncols, sizeof(size_t));
  if (!widths) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    return NULL;
  }

  for (size_t col = 0; col < ncols; ++col) {
    const char *name = df->cols[col]->name ? df->cols[col]->name : "";
    widths[col] = strlen(name);
  }
  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      size_t len = cp_series_value_len(df->cols[col], row);
      if (len > widths[col]) {
        widths[col] = len;
      }
    }
  }

  size_t row_width = 0;
  for (size_t col = 0; col < ncols; ++col) {
    row_width += widths[col];
    if (col + 1 < ncols) {
      row_width += 1;
    }
  }
  size_t capacity = (row_width + 1) * (nrows + 1) + 1;
  CpStrBuf buf;
  if (!cp_strbuf_init(&buf, capacity, err)) {
    free(widths);
    return NULL;
  }

  for (size_t col = 0; col < ncols; ++col) {
    if (col > 0) {
      if (!cp_strbuf_append_char(&buf, ' ', err)) {
        cp_strbuf_free(&buf);
        free(widths);
        return NULL;
      }
    }
    const char *name = df->cols[col]->name ? df->cols[col]->name : "";
    size_t name_len = strlen(name);
    if (!cp_strbuf_append_padded(&buf, name, name_len, widths[col], 0, err)) {
      cp_strbuf_free(&buf);
      free(widths);
      return NULL;
    }
  }
  if (!cp_strbuf_append_char(&buf, '\n', err)) {
    cp_strbuf_free(&buf);
    free(widths);
    return NULL;
  }

  for (size_t row = 0; row < nrows; ++row) {
    for (size_t col = 0; col < ncols; ++col) {
      if (col > 0) {
        if (!cp_strbuf_append_char(&buf, ' ', err)) {
          cp_strbuf_free(&buf);
          free(widths);
          return NULL;
        }
      }
      char tmp[64];
      size_t val_len = 0;
      const char *val =
          cp_series_value_repr(df->cols[col], row, tmp, sizeof(tmp), &val_len);
      int right_align = df->cols[col]->dtype != CP_DTYPE_STRING;
      if (!cp_strbuf_append_padded(&buf, val, val_len, widths[col],
                                   right_align, err)) {
        cp_strbuf_free(&buf);
        free(widths);
        return NULL;
      }
    }
    if (!cp_strbuf_append_char(&buf, '\n', err)) {
      cp_strbuf_free(&buf);
      free(widths);
      return NULL;
    }
  }

  free(widths);
  return buf.data;
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
                              int has_left,
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
      if (has_left) {
        ok = cp_series_append_from(out->cols[col], sources[col], left_row, err);
      } else {
        ok = cp_series_append_null(out->cols[col], err);
      }
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

CpDataFrame *cp_df_join_multi_with_strategy(const CpDataFrame *left,
                                            const CpDataFrame *right,
                                            const char **left_keys,
                                            const char **right_keys,
                                            size_t key_count,
                                            CpJoinType how,
                                            const char *left_suffix,
                                            const char *right_suffix,
                                            CpJoinStrategy strategy,
                                            CpError *err) {
  if (!left || !right || !left_keys || !right_keys || key_count == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join arguments");
    return NULL;
  }
  if (left->ncols == 0 || right->ncols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "empty schema");
    return NULL;
  }
  if (how != CP_JOIN_INNER && how != CP_JOIN_LEFT &&
      how != CP_JOIN_RIGHT && how != CP_JOIN_OUTER) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported join type");
    return NULL;
  }
  if (strategy != CP_JOIN_STRATEGY_AUTO &&
      strategy != CP_JOIN_STRATEGY_NESTED &&
      strategy != CP_JOIN_STRATEGY_HASH &&
      strategy != CP_JOIN_STRATEGY_SORTED) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported join strategy");
    return NULL;
  }

  const char *left_suffix_safe = left_suffix ? left_suffix : "";
  const char *right_suffix_safe =
      (right_suffix && right_suffix[0] != '\0') ? right_suffix : "_right";

  CpDataFrame *out = NULL;
  const CpSeries **left_key_series = NULL;
  const CpSeries **right_key_series = NULL;
  unsigned char *right_include = NULL;
  const char **right_names = NULL;
  const char **left_names = NULL;
  const char **out_names = NULL;
  CpDType *out_dtypes = NULL;
  const CpSeries **out_sources = NULL;
  unsigned char *out_from_right = NULL;
  unsigned char *name_owned = NULL;
  unsigned char *right_matched = NULL;
  size_t right_include_count = 0;
  size_t out_cols = 0;
  size_t out_idx = 0;
  size_t total_rows = 0;
  CpJoinIndex hash_index;
  int use_hash = 0;
  size_t *right_sorted = NULL;
  size_t *right_tmp = NULL;
  size_t right_sorted_count = 0;
  int *sort_asc = NULL;
  int use_index = 0;

  memset(&hash_index, 0, sizeof(hash_index));

  left_key_series =
      (const CpSeries **)calloc(key_count, sizeof(const CpSeries *));
  right_key_series =
      (const CpSeries **)calloc(key_count, sizeof(const CpSeries *));
  if (!left_key_series || !right_key_series) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    goto cleanup;
  }

  for (size_t i = 0; i < key_count; ++i) {
    const CpSeries *lkey = cp_df_require_col(left, left_keys[i], err);
    if (!lkey) {
      goto cleanup;
    }
    const CpSeries *rkey = cp_df_require_col(right, right_keys[i], err);
    if (!rkey) {
      goto cleanup;
    }
    if (lkey->dtype != rkey->dtype) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "join key dtype mismatch");
      goto cleanup;
    }
    if (lkey->dtype != CP_DTYPE_INT64 && lkey->dtype != CP_DTYPE_STRING) {
      cp_error_set(err, CP_ERR_INVALID, 0, 0, "unsupported join key dtype");
      goto cleanup;
    }
    left_key_series[i] = lkey;
    right_key_series[i] = rkey;
  }

  right_include =
      (unsigned char *)calloc(right->ncols, sizeof(unsigned char));
  if (!right_include) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    goto cleanup;
  }

  for (size_t col = 0; col < right->ncols; ++col) {
    const char *name = right->cols[col]->name;
    int drop = 0;
    if (cp_name_in_list(name, right_keys, key_count) &&
        cp_name_in_list(name, left_keys, key_count)) {
      drop = 1;
    }
    if (!drop) {
      right_include[col] = 1;
      right_include_count += 1;
    }
  }

  if (right_include_count > 0) {
    right_names =
        (const char **)malloc(right_include_count * sizeof(const char *));
    if (!right_names) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      goto cleanup;
    }
    size_t idx = 0;
    for (size_t col = 0; col < right->ncols; ++col) {
      if (right_include[col]) {
        right_names[idx++] = right->cols[col]->name;
      }
    }
  }

  left_names = (const char **)malloc(left->ncols * sizeof(const char *));
  if (!left_names) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    goto cleanup;
  }
  for (size_t col = 0; col < left->ncols; ++col) {
    left_names[col] = left->cols[col]->name;
  }

  out_cols = left->ncols + right_include_count;
  if (out_cols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join schema");
    goto cleanup;
  }

  out_names = (const char **)malloc(out_cols * sizeof(const char *));
  out_dtypes = (CpDType *)malloc(out_cols * sizeof(CpDType));
  out_sources = (const CpSeries **)malloc(out_cols * sizeof(const CpSeries *));
  out_from_right = (unsigned char *)calloc(out_cols, sizeof(unsigned char));
  name_owned = (unsigned char *)calloc(out_cols, sizeof(unsigned char));
  if (!out_names || !out_dtypes || !out_sources || !out_from_right || !name_owned) {
    cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
    goto cleanup;
  }

  out_idx = 0;
  for (size_t col = 0; col < left->ncols; ++col) {
    const char *base = left->cols[col]->name;
    int collision = cp_name_in_list(base, right_names, right_include_count);
    int force_suffix = collision && left_suffix_safe[0] != '\0';
    const char *suffix = force_suffix ? left_suffix_safe : "";
    int owned = 0;
    const char *name = cp_join_format_name(base,
                                           out_names,
                                           out_idx,
                                           suffix,
                                           force_suffix,
                                           &owned,
                                           err);
    if (!name) {
      goto cleanup;
    }
    out_names[out_idx] = name;
    if (owned) {
      name_owned[out_idx] = 1;
    }
    out_dtypes[out_idx] = left->cols[col]->dtype;
    out_sources[out_idx] = left->cols[col];
    out_from_right[out_idx] = 0;
    out_idx += 1;
  }

  for (size_t col = 0; col < right->ncols; ++col) {
    if (!right_include[col]) {
      continue;
    }
    const char *base = right->cols[col]->name;
    int collision = cp_name_in_list(base, left_names, left->ncols);
    int force_suffix = collision;
    const char *suffix = collision ? right_suffix_safe : "";
    int owned = 0;
    const char *name = cp_join_format_name(base,
                                           out_names,
                                           out_idx,
                                           suffix,
                                           force_suffix,
                                           &owned,
                                           err);
    if (!name) {
      goto cleanup;
    }
    out_names[out_idx] = name;
    out_dtypes[out_idx] = right->cols[col]->dtype;
    out_sources[out_idx] = right->cols[col];
    out_from_right[out_idx] = 1;
    if (owned) {
      name_owned[out_idx] = 1;
    }
    out_idx += 1;
  }

  if (out_idx != out_cols) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "join schema mismatch");
    goto cleanup;
  }

  if (right->nrows > 0) {
    right_matched =
        (unsigned char *)calloc(right->nrows, sizeof(unsigned char));
    if (!right_matched) {
      cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
      goto cleanup;
    }
  }

  if (strategy == CP_JOIN_STRATEGY_HASH) {
    use_hash = 1;
  } else if (strategy == CP_JOIN_STRATEGY_SORTED) {
    if (left->nrows > 0 && right->nrows > 0) {
      use_index = 1;
    }
  } else if (strategy == CP_JOIN_STRATEGY_AUTO) {
    if (left->nrows > 0 && right->nrows > 0) {
      size_t threshold = 1024;
      if (left->nrows > threshold / right->nrows) {
        use_hash = 1;
      } else {
        use_index = 1;
      }
    }
  }

  if (use_hash) {
    if (!cp_join_index_init(&hash_index, right->nrows, err)) {
      goto cleanup;
    }
    for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
      if (cp_join_keys_any_null(right_key_series, key_count, rrow)) {
        continue;
      }
      uint64_t hash = cp_join_hash_keys(right_key_series, key_count, rrow);
      if (!cp_join_index_add(&hash_index, hash, rrow, err)) {
        goto cleanup;
      }
    }
  }

  if (use_index) {
    for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
      if (cp_join_keys_any_null(right_key_series, key_count, rrow)) {
        continue;
      }
      right_sorted_count += 1;
    }
    if (right_sorted_count > 0) {
      right_sorted = (size_t *)malloc(right_sorted_count * sizeof(size_t));
      right_tmp = (size_t *)malloc(right_sorted_count * sizeof(size_t));
      sort_asc = (int *)malloc(key_count * sizeof(int));
      if (!right_sorted || !right_tmp || !sort_asc) {
        cp_error_set(err, CP_ERR_OOM, 0, 0, "out of memory");
        goto cleanup;
      }
      for (size_t i = 0; i < key_count; ++i) {
        sort_asc[i] = 1;
      }
      size_t idx = 0;
      for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
        if (cp_join_keys_any_null(right_key_series, key_count, rrow)) {
          continue;
        }
        right_sorted[idx++] = rrow;
      }
      if (right_sorted_count > 1) {
        cp_sort_indices_merge_multi(right_sorted,
                                    right_tmp,
                                    0,
                                    right_sorted_count,
                                    right_key_series,
                                    sort_asc,
                                    key_count);
      }
    }
  }

  total_rows = 0;
  for (size_t lrow = 0; lrow < left->nrows; ++lrow) {
    if (cp_join_keys_any_null(left_key_series, key_count, lrow)) {
      if (how == CP_JOIN_LEFT || how == CP_JOIN_OUTER) {
        if (total_rows == SIZE_MAX) {
          cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
          goto cleanup;
        }
        total_rows += 1;
      }
      continue;
    }
    size_t matches = 0;
    if (use_hash) {
      uint64_t hash = cp_join_hash_keys(left_key_series, key_count, lrow);
      const CpJoinBucket *bucket = cp_join_index_find(&hash_index, hash);
      if (bucket) {
        for (size_t i = 0; i < bucket->count; ++i) {
          size_t rrow = bucket->rows[i];
          if (cp_join_keys_equal(left_key_series,
                                 right_key_series,
                                 key_count,
                                 lrow,
                                 rrow)) {
            matches += 1;
            if (right_matched) {
              right_matched[rrow] = 1;
            }
          }
        }
      }
    } else if (use_index) {
      size_t start = cp_join_lower_bound(left_key_series,
                                         right_key_series,
                                         key_count,
                                         lrow,
                                         right_sorted,
                                         right_sorted_count);
      for (size_t pos = start; pos < right_sorted_count; ++pos) {
        size_t rrow = right_sorted[pos];
        int cmp = cp_join_compare_lr(left_key_series,
                                     right_key_series,
                                     key_count,
                                     lrow,
                                     rrow);
        if (cmp != 0) {
          break;
        }
        matches += 1;
        if (right_matched) {
          right_matched[rrow] = 1;
        }
      }
    } else {
      for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
        if (cp_join_keys_any_null(right_key_series, key_count, rrow)) {
          continue;
        }
        if (cp_join_keys_equal(left_key_series,
                               right_key_series,
                               key_count,
                               lrow,
                               rrow)) {
          matches += 1;
          if (right_matched) {
            right_matched[rrow] = 1;
          }
        }
      }
    }
    if (matches == 0) {
      if (how == CP_JOIN_LEFT || how == CP_JOIN_OUTER) {
        if (total_rows == SIZE_MAX) {
          cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
          goto cleanup;
        }
        total_rows += 1;
      }
    } else {
      if (total_rows > SIZE_MAX - matches) {
        cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
        goto cleanup;
      }
      total_rows += matches;
    }
  }

  if (how == CP_JOIN_RIGHT || how == CP_JOIN_OUTER) {
    for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
      if (!right_matched || right_matched[rrow] == 0) {
        if (total_rows == SIZE_MAX) {
          cp_error_set(err, CP_ERR_INVALID, 0, 0, "row count overflow");
          goto cleanup;
        }
        total_rows += 1;
      }
    }
  }

  out = cp_df_create(out_cols, out_names, out_dtypes, total_rows, err);
  for (size_t j = 0; j < out_cols; ++j) {
    if (name_owned[j]) {
      free((char *)out_names[j]);
    }
  }
  free(out_names);
  free(out_dtypes);
  free(name_owned);
  out_names = NULL;
  out_dtypes = NULL;
  name_owned = NULL;

  if (!out) {
    goto cleanup;
  }

  if (right_matched) {
    memset(right_matched, 0, right->nrows);
  }

  for (size_t lrow = 0; lrow < left->nrows; ++lrow) {
    if (cp_join_keys_any_null(left_key_series, key_count, lrow)) {
      if (how == CP_JOIN_LEFT || how == CP_JOIN_OUTER) {
        if (!cp_join_append_row(out,
                                out_sources,
                                out_from_right,
                                out_cols,
                                lrow,
                                0,
                                1,
                                0,
                                err)) {
          cp_df_free(out);
          out = NULL;
          goto cleanup;
        }
      }
      continue;
    }

    int matched = 0;
    if (use_hash) {
      uint64_t hash = cp_join_hash_keys(left_key_series, key_count, lrow);
      const CpJoinBucket *bucket = cp_join_index_find(&hash_index, hash);
      if (bucket) {
        for (size_t i = 0; i < bucket->count; ++i) {
          size_t rrow = bucket->rows[i];
          if (cp_join_keys_equal(left_key_series,
                                 right_key_series,
                                 key_count,
                                 lrow,
                                 rrow)) {
            matched = 1;
            if (right_matched) {
              right_matched[rrow] = 1;
            }
            if (!cp_join_append_row(out,
                                    out_sources,
                                    out_from_right,
                                    out_cols,
                                    lrow,
                                    rrow,
                                    1,
                                    1,
                                    err)) {
              cp_df_free(out);
              out = NULL;
              goto cleanup;
            }
          }
        }
      }
    } else if (use_index) {
      size_t start = cp_join_lower_bound(left_key_series,
                                         right_key_series,
                                         key_count,
                                         lrow,
                                         right_sorted,
                                         right_sorted_count);
      for (size_t pos = start; pos < right_sorted_count; ++pos) {
        size_t rrow = right_sorted[pos];
        int cmp = cp_join_compare_lr(left_key_series,
                                     right_key_series,
                                     key_count,
                                     lrow,
                                     rrow);
        if (cmp != 0) {
          break;
        }
        matched = 1;
        if (right_matched) {
          right_matched[rrow] = 1;
        }
        if (!cp_join_append_row(out,
                                out_sources,
                                out_from_right,
                                out_cols,
                                lrow,
                                rrow,
                                1,
                                1,
                                err)) {
          cp_df_free(out);
          out = NULL;
          goto cleanup;
        }
      }
    } else {
      for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
        if (cp_join_keys_any_null(right_key_series, key_count, rrow)) {
          continue;
        }
        if (cp_join_keys_equal(left_key_series,
                               right_key_series,
                               key_count,
                               lrow,
                               rrow)) {
          matched = 1;
          if (right_matched) {
            right_matched[rrow] = 1;
          }
          if (!cp_join_append_row(out,
                                  out_sources,
                                  out_from_right,
                                  out_cols,
                                  lrow,
                                  rrow,
                                  1,
                                  1,
                                  err)) {
            cp_df_free(out);
            out = NULL;
            goto cleanup;
          }
        }
      }
    }

    if (!matched && (how == CP_JOIN_LEFT || how == CP_JOIN_OUTER)) {
      if (!cp_join_append_row(out,
                              out_sources,
                              out_from_right,
                              out_cols,
                              lrow,
                              0,
                              1,
                              0,
                              err)) {
        cp_df_free(out);
        out = NULL;
        goto cleanup;
      }
    }
  }

  if (out && (how == CP_JOIN_RIGHT || how == CP_JOIN_OUTER)) {
    for (size_t rrow = 0; rrow < right->nrows; ++rrow) {
      if (right_matched && right_matched[rrow]) {
        continue;
      }
      if (!cp_join_append_row(out,
                              out_sources,
                              out_from_right,
                              out_cols,
                              0,
                              rrow,
                              0,
                              1,
                              err)) {
        cp_df_free(out);
        out = NULL;
        break;
      }
    }
  }

cleanup:
  cp_join_index_free(&hash_index);
  free(right_sorted);
  free(right_tmp);
  free(sort_asc);
  free(left_key_series);
  free(right_key_series);
  free(right_include);
  free(right_names);
  free(left_names);
  free(out_sources);
  free(out_from_right);
  free(right_matched);
  if (out_names && name_owned) {
    for (size_t j = 0; j < out_cols; ++j) {
      if (name_owned[j]) {
        free((char *)out_names[j]);
      }
    }
  }
  free(out_names);
  free(out_dtypes);
  free(name_owned);
  return out;
}

CpDataFrame *cp_df_join_multi(const CpDataFrame *left,
                              const CpDataFrame *right,
                              const char **left_keys,
                              const char **right_keys,
                              size_t key_count,
                              CpJoinType how,
                              const char *left_suffix,
                              const char *right_suffix,
                              CpError *err) {
  return cp_df_join_multi_with_strategy(left,
                                        right,
                                        left_keys,
                                        right_keys,
                                        key_count,
                                        how,
                                        left_suffix,
                                        right_suffix,
                                        CP_JOIN_STRATEGY_AUTO,
                                        err);
}

CpDataFrame *cp_df_join_with_strategy(const CpDataFrame *left,
                                      const CpDataFrame *right,
                                      const char *left_key,
                                      const char *right_key,
                                      CpJoinType how,
                                      CpJoinStrategy strategy,
                                      CpError *err) {
  if (!left || !right || !left_key || !right_key) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid join arguments");
    return NULL;
  }
  const char *left_keys[] = {left_key};
  const char *right_keys[] = {right_key};
  return cp_df_join_multi_with_strategy(left,
                                        right,
                                        left_keys,
                                        right_keys,
                                        1,
                                        how,
                                        "",
                                        "_right",
                                        strategy,
                                        err);
}

CpDataFrame *cp_df_join(const CpDataFrame *left,
                        const CpDataFrame *right,
                        const char *left_key,
                        const char *right_key,
                        CpJoinType how,
                        CpError *err) {
  return cp_df_join_with_strategy(left,
                                  right,
                                  left_key,
                                  right_key,
                                  how,
                                  CP_JOIN_STRATEGY_AUTO,
                                  err);
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

int cp_df_median(const CpDataFrame *df,
                 const char *name,
                 double *out,
                 size_t *out_count,
                 size_t *out_nulls,
                 CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_median(series, out, out_count, out_nulls, err);
}

int cp_df_std(const CpDataFrame *df,
              const char *name,
              double *out,
              size_t *out_count,
              size_t *out_nulls,
              CpError *err) {
  const CpSeries *series = cp_df_require_col(df, name, err);
  if (!series) {
    return 0;
  }
  return cp_series_std(series, out, out_count, out_nulls, err);
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

int cp_df_median_at(const CpDataFrame *df,
                    size_t col_idx,
                    double *out,
                    size_t *out_count,
                    size_t *out_nulls,
                    CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_median(series, out, out_count, out_nulls, err);
}

int cp_df_std_at(const CpDataFrame *df,
                 size_t col_idx,
                 double *out,
                 size_t *out_count,
                 size_t *out_nulls,
                 CpError *err) {
  const CpSeries *series = cp_df_require_col_index(df, col_idx, err);
  if (!series) {
    return 0;
  }
  return cp_series_std(series, out, out_count, out_nulls, err);
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
