#ifndef CPANDAS_H
#define CPANDAS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
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
