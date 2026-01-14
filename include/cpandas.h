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

typedef struct CpSeries CpSeries;
typedef struct CpDataFrame CpDataFrame;

void cp_error_clear(CpError *err);

CpDataFrame *cp_df_create(size_t ncols,
                          const char **names,
                          const CpDType *dtypes,
                          size_t capacity,
                          CpError *err);
void cp_df_free(CpDataFrame *df);

size_t cp_df_nrows(const CpDataFrame *df);
size_t cp_df_ncols(const CpDataFrame *df);
const CpSeries *cp_df_get_col(const CpDataFrame *df, const char *name);

CpDataFrame *cp_df_select_cols(const CpDataFrame *df,
                               const char **names,
                               size_t count,
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
CpDataFrame *cp_df_dropna(const CpDataFrame *df, CpError *err);
CpDataFrame *cp_df_fillna(const CpDataFrame *df,
                          const char **values,
                          size_t count,
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
int cp_df_write_csv(const CpDataFrame *df,
                    const char *path,
                    char delimiter,
                    int include_header,
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
