/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/error.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/optional.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup fast Fast custom operations
 */
/**@{*/

typedef struct mlx_fast_cuda_kernel_config_ {
  void* ctx;
} mlx_fast_cuda_kernel_config;
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_uint32(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    uint32_t value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_cuda_kernel_ {
  void* ctx;
} mlx_fast_cuda_kernel;

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_config_ {
  void* ctx;
} mlx_fast_metal_kernel_config;
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_uint32(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    uint32_t value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s);
int mlx_fast_quantized_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array key_scales,
    const mlx_array key_biases /* may be null */,
    const mlx_array values,
    const mlx_array value_scales,
    const mlx_array value_biases /* may be null */,
    float scale,
    const mlx_array mask /* may be null */,
    const mlx_array sinks /* may be null */,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    bool causal,
    const mlx_stream s);
int mlx_fast_mixed_quantized_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array key_scales,
    const mlx_array key_biases,
    const mlx_array values,
    const mlx_array value_scales,
    const mlx_array value_biases,
    float scale,
    const mlx_array mask /* may be null */,
    const mlx_array sinks /* may be null */,
    int key_group_size,
    int key_bits,
    int value_group_size,
    int value_bits,
    bool causal,
    const mlx_stream s);

typedef struct mlx_fast_mixed_quantized_attention_options_ {
  float scale;
  bool causal;
  int key_group_size;
  int key_bits;
  int value_group_size;
  int value_bits;
  float sparse_v_threshold;
  int reserved[4];
} mlx_fast_mixed_quantized_attention_options;

mlx_status mlx_fast_mixed_quantized_scaled_dot_product_attention_with_options(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array key_scales,
    const mlx_array key_biases,
    const mlx_array values,
    const mlx_array value_scales,
    const mlx_array value_biases,
    const mlx_array mask /* may be null */,
    const mlx_array sinks /* may be null */,
    mlx_fast_mixed_quantized_attention_options options,
    const mlx_stream s);

mlx_status
mlx_fast_mixed_quantized_scaled_dot_product_attention_with_options_and_diagnostics(
    mlx_array* res,
    mlx_array* diagnostics,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array key_scales,
    const mlx_array key_biases,
    const mlx_array values,
    const mlx_array value_scales,
    const mlx_array value_biases,
    const mlx_array mask /* may be null */,
    const mlx_array sinks /* may be null */,
    mlx_fast_mixed_quantized_attention_options options,
    const mlx_stream s);

typedef struct mlx_fast_turbo_quant_attention_layout_descriptor_ {
  int layout_version;
  int batch_size;
  int kv_head_count;
  int capacity;
  int logical_length;
  int ring_offset;
  int pinned_prefix_length;
  int head_dimension;
  int groups_per_vector;
  int magnitude_words_per_group;
  int bitset_words_per_group;
} mlx_fast_turbo_quant_attention_layout_descriptor;

typedef struct mlx_fast_turbo_quant_precision_policy_descriptor_ {
  int preset;
  int group_size;
  int key_base_bits;
  int key_high_bits;
  int high_precision_numerator;
  int high_precision_denominator;
  int value_bits;
  int key_scales_per_group;
  int value_scales_per_group;
  int value_magnitude_words_per_group;
  uint64_t key_seed;
  uint64_t value_seed;
} mlx_fast_turbo_quant_precision_policy_descriptor;

typedef struct mlx_fast_turbo_quant_attention_options_ {
  float scale;
  bool causal;
  int split_k_blocks;
  float sparse_v_threshold;
  // Sparse-V mode: 0 off, 1 token threshold, 2 top-k,
  // 3 cumulative mass, 4 cumulative mass plus max top-k,
  // 5 block threshold, 6 page top-k, 7 candidate sparse.
  int sparse_v_selection_mode;
  int sparse_v_top_k;
  float sparse_v_cumulative_mass;
  int sparse_v_max_top_k;
  int sparse_v_recent_tokens;
  int sparse_v_candidate_pages;
  bool diagnostics;
  int backend_version;
} mlx_fast_turbo_quant_attention_options;

typedef enum mlx_fast_turbo_quant_segmented_attention_backend_ {
  MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_UNAVAILABLE = 0,
  MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_EXPERIMENTAL_JIT = 1,
  MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_NATIVE_FUSED = 2,
} mlx_fast_turbo_quant_segmented_attention_backend;

typedef enum mlx_fast_turbo_quant_segmented_attention_codec_ {
  MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_CODEC_POLAR_QJL = 0,
  MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_CODEC_POLAR_WHT = 1,
  MLX_FAST_TURBO_QUANT_SEGMENTED_ATTENTION_CODEC_HYBRID_K8_POLAR_WHT_VALUE = 2,
} mlx_fast_turbo_quant_segmented_attention_codec;

mlx_status mlx_fast_turbo_quant_segmented_attention_get_backend(
    mlx_fast_turbo_quant_segmented_attention_backend* backend,
    bool allow_experimental_jit,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention_is_available(
    bool* available,
    bool allow_experimental_jit,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention_get_backend_for_codec(
    mlx_fast_turbo_quant_segmented_attention_backend* backend,
    mlx_fast_turbo_quant_segmented_attention_codec codec,
    bool allow_experimental_jit,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention_is_available_for_codec(
    bool* available,
    mlx_fast_turbo_quant_segmented_attention_codec codec,
    bool allow_experimental_jit,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention_with_diagnostics(
    mlx_vector_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention_with_page_summaries(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    const mlx_array key_page_summary,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

mlx_status
mlx_fast_turbo_quant_segmented_attention_with_page_summaries_and_diagnostics(
    mlx_vector_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    const mlx_array key_page_summary,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

mlx_status mlx_fast_turbo_quant_segmented_attention_with_candidate_sketches(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    const mlx_array key_candidate_sketch,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

mlx_status
mlx_fast_turbo_quant_segmented_attention_with_candidate_sketches_and_diagnostics(
    mlx_vector_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    const mlx_array key_candidate_sketch,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

int mlx_fast_turbo_quant_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

int mlx_fast_turbo_quant_scaled_dot_product_attention_with_diagnostics(
    mlx_vector_array* res,
    const mlx_array queries,
    const mlx_array key_packed,
    const mlx_array key_signs,
    const mlx_array key_high_precision_mask,
    const mlx_array key_residual_signs,
    const mlx_array key_scales,
    const mlx_array value_packed,
    const mlx_array value_signs,
    const mlx_array value_high_precision_mask,
    const mlx_array value_residual_signs,
    const mlx_array value_scales,
    mlx_fast_turbo_quant_attention_layout_descriptor layout,
    mlx_fast_turbo_quant_precision_policy_descriptor precision,
    mlx_fast_turbo_quant_attention_options options,
    const mlx_stream s);

int mlx_fast_prefault(mlx_array x);

int mlx_fast_pread_into(
    mlx_array dst,
    const char* safetensors_path,
    const char* tensor_name,
    uint32_t expert_index);

int mlx_fast_pread_into_offset(
    mlx_array dst,
    const char* safetensors_path,
    const char* tensor_name,
    uint32_t expert_index,
    size_t dst_offset);

/**@}*/

typedef struct MlxSSDMetricsSnapshot {
  double throughput_mb_per_s;
  uint64_t total_bytes_read;
  uint64_t total_chunks;
  double avg_chunk_latency_ms;
} MlxSSDMetricsSnapshot;

void mlx_ssd_metrics_snapshot(MlxSSDMetricsSnapshot* out);

#ifdef __cplusplus
}
#endif

#endif
