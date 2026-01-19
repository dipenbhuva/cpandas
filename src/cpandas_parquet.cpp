#include "cpandas.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <cstdarg>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

static bool cp_arrow_type_to_dtype(const std::shared_ptr<arrow::DataType> &type,
                                   CpDType *out,
                                   std::string *err_msg) {
  if (!type || !out) {
    if (err_msg) {
      *err_msg = "invalid arrow type";
    }
    return false;
  }
  switch (type->id()) {
    case arrow::Type::BOOL:
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
      *out = CP_DTYPE_INT64;
      return true;
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      *out = CP_DTYPE_FLOAT64;
      return true;
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
      *out = CP_DTYPE_STRING;
      return true;
    default:
      if (err_msg) {
        *err_msg = "unsupported parquet column type: " + type->ToString();
      }
      return false;
  }
}

static bool cp_arrow_index_value(const std::shared_ptr<arrow::Array> &indices,
                                 int64_t row,
                                 int64_t *out,
                                 CpError *err,
                                 size_t row_idx,
                                 size_t col_idx) {
  if (!indices || !out) {
    cp_error_set(err, CP_ERR_INVALID, row_idx, col_idx,
                 "invalid dictionary indices");
    return false;
  }
  switch (indices->type_id()) {
    case arrow::Type::INT8: {
      auto arr = std::static_pointer_cast<arrow::Int8Array>(indices);
      *out = static_cast<int64_t>(arr->Value(row));
      return true;
    }
    case arrow::Type::INT16: {
      auto arr = std::static_pointer_cast<arrow::Int16Array>(indices);
      *out = static_cast<int64_t>(arr->Value(row));
      return true;
    }
    case arrow::Type::INT32: {
      auto arr = std::static_pointer_cast<arrow::Int32Array>(indices);
      *out = static_cast<int64_t>(arr->Value(row));
      return true;
    }
    case arrow::Type::INT64: {
      auto arr = std::static_pointer_cast<arrow::Int64Array>(indices);
      *out = static_cast<int64_t>(arr->Value(row));
      return true;
    }
    default:
      cp_error_set(err, CP_ERR_INVALID, row_idx, col_idx,
                   "unsupported dictionary index type");
      return false;
  }
}

static bool cp_arrow_value_to_string(const std::shared_ptr<arrow::Array> &array,
                                     int64_t row,
                                     std::string *out,
                                     CpError *err,
                                     size_t row_idx,
                                     size_t col_idx) {
  if (!array || !out) {
    cp_error_set(err, CP_ERR_INVALID, row_idx, col_idx,
                 "invalid parquet value");
    return false;
  }
  switch (array->type_id()) {
    case arrow::Type::BOOL: {
      auto arr = std::static_pointer_cast<arrow::BooleanArray>(array);
      *out = arr->Value(row) ? "1" : "0";
      return true;
    }
    case arrow::Type::INT8: {
      auto arr = std::static_pointer_cast<arrow::Int8Array>(array);
      *out = std::to_string(static_cast<int64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::INT16: {
      auto arr = std::static_pointer_cast<arrow::Int16Array>(array);
      *out = std::to_string(static_cast<int64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::INT32: {
      auto arr = std::static_pointer_cast<arrow::Int32Array>(array);
      *out = std::to_string(static_cast<int64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::INT64: {
      auto arr = std::static_pointer_cast<arrow::Int64Array>(array);
      *out = std::to_string(static_cast<int64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::UINT8: {
      auto arr = std::static_pointer_cast<arrow::UInt8Array>(array);
      *out = std::to_string(static_cast<uint64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::UINT16: {
      auto arr = std::static_pointer_cast<arrow::UInt16Array>(array);
      *out = std::to_string(static_cast<uint64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::UINT32: {
      auto arr = std::static_pointer_cast<arrow::UInt32Array>(array);
      *out = std::to_string(static_cast<uint64_t>(arr->Value(row)));
      return true;
    }
    case arrow::Type::UINT64: {
      auto arr = std::static_pointer_cast<arrow::UInt64Array>(array);
      uint64_t value = arr->Value(row);
      if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        cp_error_set(err, CP_ERR_PARSE, row_idx, col_idx,
                     "uint64 value out of int64 range");
        return false;
      }
      *out = std::to_string(static_cast<int64_t>(value));
      return true;
    }
    case arrow::Type::FLOAT: {
      auto arr = std::static_pointer_cast<arrow::FloatArray>(array);
      double value = static_cast<double>(arr->Value(row));
      char buf[64];
      snprintf(buf, sizeof(buf), "%.17g", value);
      *out = buf;
      return true;
    }
    case arrow::Type::DOUBLE: {
      auto arr = std::static_pointer_cast<arrow::DoubleArray>(array);
      double value = arr->Value(row);
      char buf[64];
      snprintf(buf, sizeof(buf), "%.17g", value);
      *out = buf;
      return true;
    }
    case arrow::Type::STRING: {
      auto arr = std::static_pointer_cast<arrow::StringArray>(array);
      auto view = arr->GetView(row);
      out->assign(view.data(), view.size());
      return true;
    }
    case arrow::Type::LARGE_STRING: {
      auto arr = std::static_pointer_cast<arrow::LargeStringArray>(array);
      auto view = arr->GetView(row);
      out->assign(view.data(), view.size());
      return true;
    }
    case arrow::Type::DICTIONARY: {
      auto arr = std::static_pointer_cast<arrow::DictionaryArray>(array);
      if (arr->IsNull(row)) {
        out->clear();
        return true;
      }
      int64_t idx = 0;
      if (!cp_arrow_index_value(arr->indices(), row, &idx, err,
                                row_idx, col_idx)) {
        return false;
      }
      if (idx < 0 || idx >= arr->dictionary()->length()) {
        cp_error_set(err, CP_ERR_PARSE, row_idx, col_idx,
                     "dictionary index out of range");
        return false;
      }
      return cp_arrow_value_to_string(arr->dictionary(), idx, out, err,
                                      row_idx, col_idx);
    }
    default:
      cp_error_set(err, CP_ERR_INVALID, row_idx, col_idx,
                   "unsupported parquet type: %s",
                   array->type()->ToString().c_str());
      return false;
  }
}

extern "C" CpDataFrame *cp_df_read_parquet(const char *path, CpError *err) {
  if (!path) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "path is required");
    return NULL;
  }

  auto input_result = arrow::io::ReadableFile::Open(path);
  if (!input_result.ok()) {
    cp_error_set(err, CP_ERR_IO, 0, 0, "%s",
                 input_result.status().ToString().c_str());
    return NULL;
  }
  std::shared_ptr<arrow::io::ReadableFile> input = *input_result;

  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto status =
      parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &reader);
  if (!status.ok()) {
    cp_error_set(err, CP_ERR_PARSE, 0, 0, "%s",
                 status.ToString().c_str());
    return NULL;
  }

  std::shared_ptr<arrow::Table> table;
  status = reader->ReadTable(&table);
  if (!status.ok() || !table) {
    cp_error_set(err, CP_ERR_PARSE, 0, 0, "%s",
                 status.ToString().c_str());
    return NULL;
  }

  if (table->num_columns() == 0) {
    cp_error_set(err, CP_ERR_PARSE, 0, 0,
                 "parquet file has no columns");
    return NULL;
  }

  auto combined_result = table->CombineChunks(arrow::default_memory_pool());
  if (!combined_result.ok()) {
    cp_error_set(err, CP_ERR_PARSE, 0, 0, "%s",
                 combined_result.status().ToString().c_str());
    return NULL;
  }
  table = *combined_result;

  int ncols = table->num_columns();
  std::vector<CpDType> dtypes;
  std::vector<std::string> names;
  dtypes.resize(static_cast<size_t>(ncols));
  names.reserve(static_cast<size_t>(ncols));

  for (int i = 0; i < ncols; ++i) {
    auto field = table->schema()->field(i);
    std::string err_msg;
    if (!cp_arrow_type_to_dtype(field->type(), &dtypes[i], &err_msg)) {
      cp_error_set(err, CP_ERR_INVALID, 0, static_cast<size_t>(i),
                   "%s", err_msg.c_str());
      return NULL;
    }
    names.push_back(field->name());
  }

  std::vector<const char *> name_ptrs;
  name_ptrs.reserve(names.size());
  for (const auto &name : names) {
    name_ptrs.push_back(name.c_str());
  }

  CpDataFrame *df =
      cp_df_create(static_cast<size_t>(ncols), name_ptrs.data(),
                   dtypes.data(), 0, err);
  if (!df) {
    return NULL;
  }

  int64_t nrows = table->num_rows();
  if (nrows == 0) {
    return df;
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  arrays.resize(static_cast<size_t>(ncols));
  for (int i = 0; i < ncols; ++i) {
    auto column = table->column(i);
    if (column->num_chunks() != 1) {
      cp_error_set(err, CP_ERR_PARSE, 0, static_cast<size_t>(i),
                   "unexpected parquet chunk count");
      cp_df_free(df);
      return NULL;
    }
    arrays[static_cast<size_t>(i)] = column->chunk(0);
  }

  std::vector<std::string> value_storage;
  std::vector<const char *> values;
  value_storage.resize(static_cast<size_t>(ncols));
  values.resize(static_cast<size_t>(ncols));

  for (int64_t row = 0; row < nrows; ++row) {
    for (int col = 0; col < ncols; ++col) {
      const auto &array = arrays[static_cast<size_t>(col)];
      if (array->IsNull(row)) {
        values[static_cast<size_t>(col)] = "";
        continue;
      }
      if (!cp_arrow_value_to_string(array, row,
                                    &value_storage[static_cast<size_t>(col)],
                                    err,
                                    static_cast<size_t>(row),
                                    static_cast<size_t>(col))) {
        cp_df_free(df);
        return NULL;
      }
      values[static_cast<size_t>(col)] =
          value_storage[static_cast<size_t>(col)].c_str();
    }
    if (!cp_df_append_row(df, values.data(),
                          static_cast<size_t>(ncols), err)) {
      cp_df_free(df);
      return NULL;
    }
  }

  return df;
}

extern "C" int cp_df_write_parquet(const CpDataFrame *df,
                                   const char *path,
                                   CpError *err) {
  if (!df || !path) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "invalid arguments");
    return 0;
  }
  size_t ncols = cp_df_ncols(df);
  if (ncols == 0) {
    cp_error_set(err, CP_ERR_INVALID, 0, 0, "dataframe has no columns");
    return 0;
  }

  std::vector<const char *> names;
  names.resize(ncols);
  if (!cp_df_columns(df, names.data(), ncols, err)) {
    return 0;
  }

  size_t nrows = cp_df_nrows(df);
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  fields.reserve(ncols);
  arrays.reserve(ncols);

  for (size_t col = 0; col < ncols; ++col) {
    const CpSeries *series = cp_df_get_col(df, names[col]);
    if (!series) {
      cp_error_set(err, CP_ERR_INVALID, 0, col, "unknown column");
      return 0;
    }
    CpDType dtype = cp_series_dtype(series);
    switch (dtype) {
      case CP_DTYPE_INT64: {
        arrow::Int64Builder builder;
        for (size_t row = 0; row < nrows; ++row) {
          int64_t value = 0;
          int is_null = 0;
          if (!cp_series_get_int64(series, row, &value, &is_null)) {
            cp_error_set(err, CP_ERR_INVALID, row, col,
                         "invalid int64 value");
            return 0;
          }
          if (is_null) {
            if (!builder.AppendNull().ok()) {
              cp_error_set(err, CP_ERR_IO, row, col,
                           "failed to append parquet value");
              return 0;
            }
          } else if (!builder.Append(value).ok()) {
            cp_error_set(err, CP_ERR_IO, row, col,
                         "failed to append parquet value");
            return 0;
          }
        }
        std::shared_ptr<arrow::Array> array;
        if (!builder.Finish(&array).ok()) {
          cp_error_set(err, CP_ERR_IO, 0, col,
                       "failed to build parquet array");
          return 0;
        }
        fields.push_back(arrow::field(names[col], arrow::int64()));
        arrays.push_back(array);
        break;
      }
      case CP_DTYPE_FLOAT64: {
        arrow::DoubleBuilder builder;
        for (size_t row = 0; row < nrows; ++row) {
          double value = 0.0;
          int is_null = 0;
          if (!cp_series_get_float64(series, row, &value, &is_null)) {
            cp_error_set(err, CP_ERR_INVALID, row, col,
                         "invalid float64 value");
            return 0;
          }
          if (is_null) {
            if (!builder.AppendNull().ok()) {
              cp_error_set(err, CP_ERR_IO, row, col,
                           "failed to append parquet value");
              return 0;
            }
          } else if (!builder.Append(value).ok()) {
            cp_error_set(err, CP_ERR_IO, row, col,
                         "failed to append parquet value");
            return 0;
          }
        }
        std::shared_ptr<arrow::Array> array;
        if (!builder.Finish(&array).ok()) {
          cp_error_set(err, CP_ERR_IO, 0, col,
                       "failed to build parquet array");
          return 0;
        }
        fields.push_back(arrow::field(names[col], arrow::float64()));
        arrays.push_back(array);
        break;
      }
      case CP_DTYPE_STRING: {
        arrow::StringBuilder builder;
        for (size_t row = 0; row < nrows; ++row) {
          const char *value = NULL;
          int is_null = 0;
          if (!cp_series_get_string(series, row, &value, &is_null)) {
            cp_error_set(err, CP_ERR_INVALID, row, col,
                         "invalid string value");
            return 0;
          }
          if (is_null) {
            if (!builder.AppendNull().ok()) {
              cp_error_set(err, CP_ERR_IO, row, col,
                           "failed to append parquet value");
              return 0;
            }
          } else if (!builder.Append(value ? value : "").ok()) {
            cp_error_set(err, CP_ERR_IO, row, col,
                         "failed to append parquet value");
            return 0;
          }
        }
        std::shared_ptr<arrow::Array> array;
        if (!builder.Finish(&array).ok()) {
          cp_error_set(err, CP_ERR_IO, 0, col,
                       "failed to build parquet array");
          return 0;
        }
        fields.push_back(arrow::field(names[col], arrow::utf8()));
        arrays.push_back(array);
        break;
      }
      default:
        cp_error_set(err, CP_ERR_INVALID, 0, col,
                     "unsupported dtype for parquet");
        return 0;
    }
  }

  auto schema = arrow::schema(fields);
  auto table =
      arrow::Table::Make(schema, arrays, static_cast<int64_t>(nrows));

  auto out_result = arrow::io::FileOutputStream::Open(path);
  if (!out_result.ok()) {
    cp_error_set(err, CP_ERR_IO, 0, 0, "%s",
                 out_result.status().ToString().c_str());
    return 0;
  }
  std::shared_ptr<arrow::io::FileOutputStream> out = *out_result;

  int64_t chunk_size = nrows > 0 ? static_cast<int64_t>(nrows) : 1;
  auto status =
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                 out, chunk_size);
  if (!status.ok()) {
    cp_error_set(err, CP_ERR_IO, 0, 0, "%s",
                 status.ToString().c_str());
    return 0;
  }
  return 1;
}
