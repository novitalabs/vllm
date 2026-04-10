// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include "core/registration.h"

// Fused AllReduce + RMSNorm + Residual Add (+ optional FP8/FP4 quantization)
void novita_allreduce_fusion(
    torch::Tensor& allreduce_in, torch::Tensor& residual_in,
    torch::Tensor& residual_out, torch::Tensor& workspace_ptrs,
    int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_dim, double rms_eps, torch::Tensor& rms_gamma,
    bool launch_with_pdl, bool fp32_acc, bool use_oneshot,
    int64_t pattern_code, torch::Tensor& norm_out, torch::Tensor& quant_out,
    torch::Tensor& scale_out, torch::Tensor& scale_factor);

// Fused QK RMS Norm with TP variance all-reduce via NVLink
void novita_qk_rmsnorm_tp(
    torch::Tensor& q_in, torch::Tensor& k_in,
    torch::Tensor& q_out, torch::Tensor& k_out,
    torch::Tensor& q_weight, torch::Tensor& k_weight,
    double q_eps, double k_eps,
    torch::Tensor& workspace_ptrs,
    int64_t world_size, int64_t world_rank,
    int64_t max_tokens, int64_t epoch);

void novita_clear_qk_norm_workspace(
    torch::Tensor& workspace_ptrs, int64_t world_size,
    int64_t world_rank, int64_t max_tokens);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "allreduce_fusion(Tensor! allreduce_in, Tensor! residual_in, "
      "Tensor! residual_out, Tensor workspace_ptrs, "
      "int world_size, int world_rank, "
      "int token_num, int hidden_dim, float rms_eps, "
      "Tensor rms_gamma, bool launch_with_pdl, bool fp32_acc, "
      "bool use_oneshot, int pattern_code, "
      "Tensor! norm_out, Tensor! quant_out, "
      "Tensor! scale_out, Tensor scale_factor) -> ()");
  ops.impl("allreduce_fusion", torch::kCUDA, &novita_allreduce_fusion);

  ops.def(
      "qk_rmsnorm_tp(Tensor! q_in, Tensor! k_in, "
      "Tensor! q_out, Tensor! k_out, "
      "Tensor q_weight, Tensor k_weight, "
      "float q_eps, float k_eps, "
      "Tensor workspace_ptrs, "
      "int world_size, int world_rank, "
      "int max_tokens, int epoch) -> ()");
  ops.impl("qk_rmsnorm_tp", torch::kCUDA, &novita_qk_rmsnorm_tp);

  ops.def(
      "clear_qk_norm_workspace(Tensor! workspace_ptrs, "
      "int world_size, int world_rank, int max_tokens) -> ()");
  ops.impl("clear_qk_norm_workspace", torch::kCUDA, &novita_clear_qk_norm_workspace);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
