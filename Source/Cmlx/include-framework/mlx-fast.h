#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <Cmlx/mlx-api.h>
#include <Cmlx/mlx-utils.h>

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

/** Computes: `O = softmax(Q @ K.T) @ V` where K and V are quantized. **/
MLX_API array quantized_scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& key_scales,
    const std::optional<array>& key_biases,
    const array& values,
    const array& value_scales,
    const std::optional<array>& value_biases,
    const float scale,
    const std::optional<array>& mask = std::nullopt,
    const std::optional<array>& sinks = std::nullopt,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "mxfp4",
    bool causal = false,
    StreamOrDevice s = {});

struct TurboQuantAttentionLayoutDescriptor {
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
};

struct TurboQuantPrecisionPolicyDescriptor {
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
};

struct TurboQuantAttentionOptions {
  float scale;
  bool causal;
  int split_k_blocks;
  float sparse_v_threshold;
  bool diagnostics;
  int backend_version;
};

class MLX_API TurboQuantNativeAttentionUnavailable : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class TurboQuantSegmentedAttentionBackend : int {
  Unavailable = 0,
  ExperimentalJit = 1,
  NativeFused = 2,
};

MLX_API TurboQuantSegmentedAttentionBackend
turbo_quant_segmented_attention_backend(
    bool allow_experimental_jit = false,
    StreamOrDevice s = {});

MLX_API bool turbo_quant_segmented_attention_is_available(
    bool allow_experimental_jit = false,
    StreamOrDevice s = {});

/** Computes: `O = softmax(Q @ K.T) @ V` from TurboQuant compressed K/V planes. **/
MLX_API array turbo_quant_segmented_attention(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s = {});

/** Returns `[output, diagnostics]` for TurboQuant compressed segmented attention. **/
MLX_API std::vector<array> turbo_quant_segmented_attention_with_diagnostics(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s = {});

/** Compatibility name for `turbo_quant_segmented_attention`. **/
MLX_API array turbo_quant_scaled_dot_product_attention(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s = {});

/** Compatibility name for `turbo_quant_segmented_attention_with_diagnostics`. **/
MLX_API std::vector<array> turbo_quant_scaled_dot_product_attention_with_diagnostics(
    const array& queries,
    const array& key_packed,
    const array& key_signs,
    const array& key_high_precision_mask,
    const array& key_residual_signs,
    const array& key_scales,
    const array& value_packed,
    const array& value_signs,
    const array& value_high_precision_mask,
    const array& value_residual_signs,
    const array& value_scales,
    const TurboQuantAttentionLayoutDescriptor& layout,
    const TurboQuantPrecisionPolicyDescriptor& precision,
    const TurboQuantAttentionOptions& options,
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, uint32_t, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
#endif
