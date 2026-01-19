#ifndef CPANDAS_H
#define CPANDAS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

typedef enum {
  CP_DTYPE_INT64 = 0,
  CP_DTYPE_FLOAT64 = 1,
  CP_DTYPE_STRING = 2
} CpDType;

typedef enum {
  CP_OK = 0,
  CP_ERR_OOM = 1,
  CP_ERR_IO = 2,
  CP_ERR_PARSE = 3,
  CP_ERR_INVALID = 4
} CpErrCode;

typedef enum {
  CP_OP_EQ = 0,
  CP_OP_NE = 1,
  CP_OP_LT = 2,
  CP_OP_LE = 3,
  CP_OP_GT = 4,
  CP_OP_GE = 5
} CpCompareOp;

typedef enum {
  CP_ARITH_ADD = 0,
  CP_ARITH_SUB = 1,
  CP_ARITH_MUL = 2,
  CP_ARITH_DIV = 3
} CpArithOp;

typedef enum {
  CP_AGG_COUNT = 0,
  CP_AGG_SUM = 1,
  CP_AGG_MEAN = 2,
  CP_AGG_MIN = 3,
  CP_AGG_MAX = 4
} CpAggOp;

typedef enum {
  CP_JOIN_INNER = 0,
  CP_JOIN_LEFT = 1,
  CP_JOIN_RIGHT = 2,
  CP_JOIN_OUTER = 3
} CpJoinType;

typedef enum {
  CP_JOIN_STRATEGY_AUTO = 0,
  CP_JOIN_STRATEGY_NESTED = 1,
  CP_JOIN_STRATEGY_HASH = 2,
  CP_JOIN_STRATEGY_SORTED = 3
} CpJoinStrategy;

typedef enum {
  CP_DUP_KEEP_FIRST = 0,
  CP_DUP_KEEP_LAST = 1,
  CP_DUP_KEEP_NONE = 2
} CpDuplicateKeep;

typedef enum {
  CP_CONCAT_ROWS = 0,
  CP_CONCAT_COLS = 1
} CpConcatAxis;

typedef struct {
  CpErrCode code;
  char message[256];
  size_t row;
  size_t col;
} CpError;

typedef struct {
  int64_t value;
  size_t count;
  size_t nulls;
} CpAggInt64;

typedef struct {
  double value;
  size_t count;
  size_t nulls;
} CpAggFloat64;

typedef struct {
  int is_null;
  union {
    int64_t i64;
    double f64;
    const char *str;
  } value;
} CpValue;

typedef struct CpSeries CpSeries;
typedef struct CpDataFrame CpDataFrame;

typedef int (*CpApplyFn)(const CpDataFrame *df,
                         size_t row,
                         void *user_data,
                         CpValue *out,
                         CpError *err);
typedef int (*CpTransformFn)(const CpSeries *series,
                             size_t row,
                             void *user_data,
                             CpValue *out,
                             CpError *err);
typedef int (*CpIterRowFn)(const CpDataFrame *df,
                           size_t row,
                           void *user_data,
                           CpError *err);
typedef int (*CpIterItemFn)(const CpSeries *series,
                            size_t col,
                            void *user_data,
                            CpError *err);

void cp_error_clear(CpError *err);

CpDataFrame *cp_df_create(size_t ncols,
                          const char **names,
                          const CpDType *dtypes,
                          size_t capacity,
                          CpError *err);
void cp_df_free(CpDataFrame *df);

size_t cp_df_nrows(const CpDataFrame *df);
size_t cp_df_ncols(const CpDataFrame *df);
int cp_df_shape(const CpDataFrame *df,
                size_t *out_rows,
                size_t *out_cols,
                CpError *err);
size_t cp_df_size(const CpDataFrame *df);
size_t cp_df_ndim(const CpDataFrame *df);
int cp_df_columns(const CpDataFrame *df,
                  const char **out,
                  size_t out_len,
                  CpError *err);
CpDataFrame *cp_df_copy(const CpDataFrame *df, CpError *err);
const CpSeries *cp_df_get_col(const CpDataFrame *df, const char *name);

CpDataFrame *cp_df_select_cols(const CpDataFrame *df,
                               const char **names,
                               size_t count,
                               CpError *err);
CpDataFrame *cp_df_select_dtypes(const CpDataFrame *df,
                                 const CpDType *include,
                                 size_t include_count,
                                 const CpDType *exclude,
                                 size_t exclude_count,
                                 CpError *err);
CpDataFrame *cp_df_filter_mask(const CpDataFrame *df,
                               const uint8_t *mask,
                               size_t mask_len,
                               CpError *err);
CpDataFrame *cp_df_sort_values(const CpDataFrame *df,
                               const char *name,
                               int ascending,
                               CpError *err);
CpDataFrame *cp_df_sort_values_multi(const CpDataFrame *df,
                                     const char **names,
                                     size_t count,
                                     const int *ascending,
                                     CpError *err);
int cp_df_info(const CpDataFrame *df, FILE *out, CpError *err);
char *cp_df_to_string(const CpDataFrame *df, CpError *err);
CpDataFrame *cp_df_describe(const CpDataFrame *df, CpError *err);
CpDataFrame *cp_df_groupby_agg(const CpDataFrame *df,
                               const char *key,
                               const char **value_cols,
                               const CpAggOp *ops,
                               size_t count,
                               CpError *err);
CpDataFrame *cp_df_pivot_table(const CpDataFrame *df,
                               const char *index,
                               const char *columns,
                               const char *values,
                               CpAggOp op,
                               CpError *err);
CpDataFrame *cp_df_join(const CpDataFrame *left,
                        const CpDataFrame *right,
                        const char *left_key,
                        const char *right_key,
                        CpJoinType how,
                        CpError *err);
CpDataFrame *cp_df_join_with_strategy(const CpDataFrame *left,
                                      const CpDataFrame *right,
                                      const char *left_key,
                                      const char *right_key,
                                      CpJoinType how,
                                      CpJoinStrategy strategy,
                                      CpError *err);
CpDataFrame *cp_df_join_multi(const CpDataFrame *left,
                              const CpDataFrame *right,
                              const char **left_keys,
                              const char **right_keys,
                              size_t key_count,
                              CpJoinType how,
                              const char *left_suffix,
                              const char *right_suffix,
                              CpError *err);
CpDataFrame *cp_df_join_multi_with_strategy(const CpDataFrame *left,
                                            const CpDataFrame *right,
                                            const char **left_keys,
                                            const char **right_keys,
                                            size_t key_count,
                                            CpJoinType how,
                                            const char *left_suffix,
                                            const char *right_suffix,
                                            CpJoinStrategy strategy,
                                            CpError *err);
int cp_df_mask_int64(const CpDataFrame *df,
                     const char *name,
                     CpCompareOp op,
                     int64_t value,
                     uint8_t *out,
                     size_t out_len,
                     CpError *err);
int cp_df_mask_float64(const CpDataFrame *df,
                       const char *name,
                       CpCompareOp op,
                       double value,
                       uint8_t *out,
                       size_t out_len,
                       CpError *err);
int cp_df_mask_string(const CpDataFrame *df,
                      const char *name,
                      CpCompareOp op,
                      const char *value,
                      uint8_t *out,
                      size_t out_len,
                      CpError *err);
int cp_df_mask_cols(const CpDataFrame *df,
                    const char *left,
                    CpCompareOp op,
                    const char *right,
                    uint8_t *out,
                    size_t out_len,
                    CpError *err);
CpDataFrame *cp_df_filter_int64(const CpDataFrame *df,
                                const char *name,
                                CpCompareOp op,
                                int64_t value,
                                CpError *err);
CpDataFrame *cp_df_filter_float64(const CpDataFrame *df,
                                  const char *name,
                                  CpCompareOp op,
                                  double value,
                                  CpError *err);
CpDataFrame *cp_df_filter_string(const CpDataFrame *df,
                                 const char *name,
                                 CpCompareOp op,
                                 const char *value,
                                 CpError *err);
CpDataFrame *cp_df_iloc(const CpDataFrame *df,
                        const size_t *row_indices,
                        size_t row_count,
                        const size_t *col_indices,
                        size_t col_count,
                        CpError *err);
CpDataFrame *cp_df_loc(const CpDataFrame *df,
                       const size_t *row_indices,
                       size_t row_count,
                       const char **names,
                       size_t name_count,
                       CpError *err);
CpDataFrame *cp_df_loc_labels(const CpDataFrame *df,
                              const char **row_labels,
                              size_t row_count,
                              const char **names,
                              size_t name_count,
                              CpError *err);
CpDataFrame *cp_df_loc_slice(const CpDataFrame *df,
                             const char *start_label,
                             const char *end_label,
                             const char **names,
                             size_t name_count,
                             CpError *err);
CpDataFrame *cp_df_head(const CpDataFrame *df, size_t n, CpError *err);
CpDataFrame *cp_df_tail(const CpDataFrame *df, size_t n, CpError *err);
int cp_df_dtypes(const CpDataFrame *df,
                 CpDType *out,
                 size_t out_len,
                 CpError *err);
CpDataFrame *cp_df_drop_cols(const CpDataFrame *df,
                             const char **names,
                             size_t count,
                             CpError *err);
CpDataFrame *cp_df_rename_cols(const CpDataFrame *df,
                               const char **old_names,
                               const char **new_names,
                               size_t count,
                               CpError *err);
int cp_df_isnull_mask(const CpDataFrame *df,
                      uint8_t *out,
                      size_t out_len,
                      CpError *err);
int cp_df_isna_mask(const CpDataFrame *df,
                    uint8_t *out,
                    size_t out_len,
                    CpError *err);
CpDataFrame *cp_df_dropna(const CpDataFrame *df, CpError *err);
CpDataFrame *cp_df_fillna(const CpDataFrame *df,
                          const char **values,
                          size_t count,
                          CpError *err);
CpDataFrame *cp_df_unique(const CpDataFrame *df,
                          const char *name,
                          CpError *err);
int cp_df_nunique(const CpDataFrame *df,
                  const char *name,
                  size_t *out,
                  CpError *err);
CpDataFrame *cp_df_value_counts(const CpDataFrame *df,
                                const char *name,
                                CpError *err);
int cp_df_duplicated(const CpDataFrame *df,
                     const char *name,
                     CpDuplicateKeep keep,
                     uint8_t *out,
                     size_t out_len,
                     CpError *err);
CpDataFrame *cp_df_drop_duplicates(const CpDataFrame *df,
                                   const char *name,
                                   CpDuplicateKeep keep,
                                   CpError *err);
CpDataFrame *cp_df_where(const CpDataFrame *df,
                         const uint8_t *mask,
                         size_t mask_len,
                         const char **values,
                         size_t count,
                         CpError *err);
CpDataFrame *cp_df_mask(const CpDataFrame *df,
                        const uint8_t *mask,
                        size_t mask_len,
                        const char **values,
                        size_t count,
                        CpError *err);
CpDataFrame *cp_df_clip(const CpDataFrame *df,
                        const char *name,
                        double lower,
                        double upper,
                        CpError *err);
CpDataFrame *cp_df_replace(const CpDataFrame *df,
                           const char *name,
                           const char *old_value,
                           const char *new_value,
                           CpError *err);
CpDataFrame *cp_df_astype(const CpDataFrame *df,
                          const char *name,
                          CpDType dtype,
                          CpError *err);
CpDataFrame *cp_df_to_numeric(const CpDataFrame *df,
                              const char *name,
                              CpError *err);
CpDataFrame *cp_df_to_datetime(const CpDataFrame *df,
                               const char *name,
                               CpError *err);
CpDataFrame *cp_df_set_index(const CpDataFrame *df,
                             const char *name,
                             CpError *err);
CpDataFrame *cp_df_reset_index(const CpDataFrame *df,
                               CpError *err);
int cp_df_at_int64(const CpDataFrame *df,
                   const char *row_label,
                   const char *col_name,
                   int64_t *out,
                   int *is_null,
                   CpError *err);
int cp_df_at_float64(const CpDataFrame *df,
                     const char *row_label,
                     const char *col_name,
                     double *out,
                     int *is_null,
                     CpError *err);
int cp_df_at_string(const CpDataFrame *df,
                    const char *row_label,
                    const char *col_name,
                    const char **out,
                    int *is_null,
                    CpError *err);
CpDataFrame *cp_df_apply(const CpDataFrame *df,
                         CpDType out_dtype,
                         const char *out_name,
                         CpApplyFn func,
                         void *user_data,
                         CpError *err);
CpDataFrame *cp_df_transform(const CpDataFrame *df,
                             const char *name,
                             CpDType out_dtype,
                             CpTransformFn func,
                             void *user_data,
                             CpError *err);
int cp_df_iterrows(const CpDataFrame *df,
                   CpIterRowFn func,
                   void *user_data,
                   CpError *err);
int cp_df_iteritems(const CpDataFrame *df,
                    CpIterItemFn func,
                    void *user_data,
                    CpError *err);
CpDataFrame *cp_df_arith_scalar(const CpDataFrame *df,
                                const char *name,
                                CpArithOp op,
                                double value,
                                const char *out_name,
                                CpError *err);
CpDataFrame *cp_df_arith_cols(const CpDataFrame *df,
                              const char *left,
                              const char *right,
                              CpArithOp op,
                              const char *out_name,
                              CpError *err);
CpDataFrame *cp_df_diff(const CpDataFrame *df,
                        const char *name,
                        CpError *err);
CpDataFrame *cp_df_rank(const CpDataFrame *df,
                        const char *name,
                        CpError *err);
CpDataFrame *cp_df_corr(const CpDataFrame *df,
                        CpError *err);
CpDataFrame *cp_df_cov(const CpDataFrame *df,
                       CpError *err);
CpDataFrame *cp_df_query(const CpDataFrame *df,
                         const char *expr,
                         CpError *err);
CpDataFrame *cp_df_concat(const CpDataFrame **dfs,
                          size_t count,
                          CpConcatAxis axis,
                          CpError *err);
CpDataFrame *cp_df_sample(const CpDataFrame *df,
                          size_t n,
                          int replace,
                          uint32_t seed,
                          CpError *err);
CpDataFrame *cp_df_nlargest(const CpDataFrame *df,
                            const char *name,
                            size_t n,
                            CpError *err);
CpDataFrame *cp_df_nsmallest(const CpDataFrame *df,
                             const char *name,
                             size_t n,
                             CpError *err);

int cp_df_append_row(CpDataFrame *df,
                     const char **values,
                     size_t nvalues,
                     CpError *err);

CpDataFrame *cp_df_read_csv(const char *path,
                            char delimiter,
                            int has_header,
                            const CpDType *dtypes,
                            size_t dtype_count,
                            CpError *err);
CpDataFrame *cp_df_read_csv_with_na(const char *path,
                                    char delimiter,
                                    int has_header,
                                    const CpDType *dtypes,
                                    size_t dtype_count,
                                    const char **na_values,
                                    size_t na_count,
                                    CpError *err);
CpDataFrame *cp_df_read_tsv(const char *path,
                            int has_header,
                            const CpDType *dtypes,
                            size_t dtype_count,
                            CpError *err);
CpDataFrame *cp_df_read_tsv_with_na(const char *path,
                                    int has_header,
                                    const CpDType *dtypes,
                                    size_t dtype_count,
                                    const char **na_values,
                                    size_t na_count,
                                    CpError *err);
CpDataFrame *cp_df_read_json(const char *path,
                             const CpDType *dtypes,
                             size_t dtype_count,
                             CpError *err);
CpDataFrame *cp_df_read_ndjson(const char *path,
                               const CpDType *dtypes,
                               size_t dtype_count,
                               CpError *err);
CpDataFrame *cp_df_read_cpd(const char *path, CpError *err);
CpDataFrame *cp_df_read_parquet(const char *path, CpError *err);
int cp_df_write_csv(const CpDataFrame *df,
                    const char *path,
                    char delimiter,
                    int include_header,
                    CpError *err);
int cp_df_write_tsv(const CpDataFrame *df,
                    const char *path,
                    int include_header,
                    CpError *err);
int cp_df_write_json(const CpDataFrame *df,
                     const char *path,
                     CpError *err);
int cp_df_write_ndjson(const CpDataFrame *df,
                       const char *path,
                       CpError *err);
int cp_df_write_cpd(const CpDataFrame *df,
                    const char *path,
                    CpError *err);
int cp_df_write_parquet(const CpDataFrame *df,
                        const char *path,
                        CpError *err);
int cp_df_to_excel(const CpDataFrame *df,
                   const char *path,
                   CpError *err);
int cp_df_to_sql(const CpDataFrame *df,
                 const char *path,
                 const char *table,
                 CpError *err);
int cp_df_plot(const CpDataFrame *df,
               const char *path,
               CpError *err);

const char *cp_series_name(const CpSeries *s);
CpDType cp_series_dtype(const CpSeries *s);
size_t cp_series_len(const CpSeries *s);
int cp_series_get_int64(const CpSeries *s, size_t idx, int64_t *out, int *is_null);
int cp_series_get_float64(const CpSeries *s, size_t idx, double *out, int *is_null);
int cp_series_get_string(const CpSeries *s, size_t idx, const char **out, int *is_null);

int cp_series_count(const CpSeries *s, size_t *out, size_t *out_nulls, CpError *err);
int cp_series_sum_int64(const CpSeries *s,
                        int64_t *out,
                        size_t *out_count,
                        size_t *out_nulls,
                        CpError *err);
int cp_series_sum_float64(const CpSeries *s,
                          double *out,
                          size_t *out_count,
                          size_t *out_nulls,
                          CpError *err);
int cp_series_mean(const CpSeries *s,
                   double *out,
                   size_t *out_count,
                   size_t *out_nulls,
                   CpError *err);
int cp_series_min_int64(const CpSeries *s,
                        int64_t *out,
                        size_t *out_nulls,
                        CpError *err);
int cp_series_max_int64(const CpSeries *s,
                        int64_t *out,
                        size_t *out_nulls,
                        CpError *err);
int cp_series_min_float64(const CpSeries *s,
                          double *out,
                          size_t *out_nulls,
                          CpError *err);
int cp_series_max_float64(const CpSeries *s,
                          double *out,
                          size_t *out_nulls,
                          CpError *err);

int cp_df_count(const CpDataFrame *df,
                const char *name,
                size_t *out,
                size_t *out_nulls,
                CpError *err);
int cp_df_sum_int64(const CpDataFrame *df,
                    const char *name,
                    int64_t *out,
                    size_t *out_count,
                    size_t *out_nulls,
                    CpError *err);
int cp_df_sum_float64(const CpDataFrame *df,
                      const char *name,
                      double *out,
                      size_t *out_count,
                      size_t *out_nulls,
                      CpError *err);
int cp_df_mean(const CpDataFrame *df,
               const char *name,
               double *out,
               size_t *out_count,
               size_t *out_nulls,
               CpError *err);
int cp_df_median(const CpDataFrame *df,
                 const char *name,
                 double *out,
                 size_t *out_count,
                 size_t *out_nulls,
                 CpError *err);
int cp_df_std(const CpDataFrame *df,
              const char *name,
              double *out,
              size_t *out_count,
              size_t *out_nulls,
              CpError *err);
int cp_df_min_int64(const CpDataFrame *df,
                    const char *name,
                    int64_t *out,
                    size_t *out_nulls,
                    CpError *err);
int cp_df_max_int64(const CpDataFrame *df,
                    const char *name,
                    int64_t *out,
                    size_t *out_nulls,
                    CpError *err);
int cp_df_min_float64(const CpDataFrame *df,
                      const char *name,
                      double *out,
                      size_t *out_nulls,
                      CpError *err);
int cp_df_max_float64(const CpDataFrame *df,
                      const char *name,
                      double *out,
                      size_t *out_nulls,
                      CpError *err);

int cp_df_count_at(const CpDataFrame *df,
                   size_t col_idx,
                   size_t *out,
                   size_t *out_nulls,
                   CpError *err);
int cp_df_sum_int64_at(const CpDataFrame *df,
                       size_t col_idx,
                       int64_t *out,
                       size_t *out_count,
                       size_t *out_nulls,
                       CpError *err);
int cp_df_sum_float64_at(const CpDataFrame *df,
                         size_t col_idx,
                         double *out,
                         size_t *out_count,
                         size_t *out_nulls,
                         CpError *err);
int cp_df_mean_at(const CpDataFrame *df,
                  size_t col_idx,
                  double *out,
                  size_t *out_count,
                  size_t *out_nulls,
                  CpError *err);
int cp_df_median_at(const CpDataFrame *df,
                    size_t col_idx,
                    double *out,
                    size_t *out_count,
                    size_t *out_nulls,
                    CpError *err);
int cp_df_std_at(const CpDataFrame *df,
                 size_t col_idx,
                 double *out,
                 size_t *out_count,
                 size_t *out_nulls,
                 CpError *err);
int cp_df_min_int64_at(const CpDataFrame *df,
                       size_t col_idx,
                       int64_t *out,
                       size_t *out_nulls,
                       CpError *err);
int cp_df_max_int64_at(const CpDataFrame *df,
                       size_t col_idx,
                       int64_t *out,
                       size_t *out_nulls,
                       CpError *err);
int cp_df_min_float64_at(const CpDataFrame *df,
                         size_t col_idx,
                         double *out,
                         size_t *out_nulls,
                         CpError *err);
int cp_df_max_float64_at(const CpDataFrame *df,
                         size_t col_idx,
                         double *out,
                         size_t *out_nulls,
                         CpError *err);

int cp_df_sum_int64_result(const CpDataFrame *df,
                           const char *name,
                           CpAggInt64 *out,
                           CpError *err);
int cp_df_sum_float64_result(const CpDataFrame *df,
                             const char *name,
                             CpAggFloat64 *out,
                             CpError *err);
int cp_df_mean_result(const CpDataFrame *df,
                      const char *name,
                      CpAggFloat64 *out,
                      CpError *err);
int cp_df_min_int64_result(const CpDataFrame *df,
                           const char *name,
                           CpAggInt64 *out,
                           CpError *err);
int cp_df_max_int64_result(const CpDataFrame *df,
                           const char *name,
                           CpAggInt64 *out,
                           CpError *err);
int cp_df_min_float64_result(const CpDataFrame *df,
                             const char *name,
                             CpAggFloat64 *out,
                             CpError *err);
int cp_df_max_float64_result(const CpDataFrame *df,
                             const char *name,
                             CpAggFloat64 *out,
                             CpError *err);

int cp_df_sum_int64_result_at(const CpDataFrame *df,
                              size_t col_idx,
                              CpAggInt64 *out,
                              CpError *err);
int cp_df_sum_float64_result_at(const CpDataFrame *df,
                                size_t col_idx,
                                CpAggFloat64 *out,
                                CpError *err);
int cp_df_mean_result_at(const CpDataFrame *df,
                         size_t col_idx,
                         CpAggFloat64 *out,
                         CpError *err);
int cp_df_min_int64_result_at(const CpDataFrame *df,
                              size_t col_idx,
                              CpAggInt64 *out,
                              CpError *err);
int cp_df_max_int64_result_at(const CpDataFrame *df,
                              size_t col_idx,
                              CpAggInt64 *out,
                              CpError *err);
int cp_df_min_float64_result_at(const CpDataFrame *df,
                                size_t col_idx,
                                CpAggFloat64 *out,
                                CpError *err);
int cp_df_max_float64_result_at(const CpDataFrame *df,
                                size_t col_idx,
                                CpAggFloat64 *out,
                                CpError *err);

#ifdef __cplusplus
}
#endif

#endif
