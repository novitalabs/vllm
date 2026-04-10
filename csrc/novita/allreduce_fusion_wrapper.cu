// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PyTorch C++ wrapper for the novita fused allreduce + RMSNorm + add kernel.
// Adapted from AKO4ALL's TVM FFI wrapper to use PyTorch tensor interface.

#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "allreduce_fusion_kernel.cuh"

using namespace flashinfer::trtllm_allreduce_fusion;

namespace {

// Helper: extract void* from an optional tensor (0-numel tensor means None)
inline void* optional_ptr(const torch::Tensor& t) {
  return t.numel() > 0 ? t.data_ptr() : nullptr;
}

inline float* optional_float_ptr(const torch::Tensor& t) {
  return t.numel() > 0 ? static_cast<float*>(t.data_ptr()) : nullptr;
}

template <typename T>
void launch_allreduce_fusion(
    torch::Tensor& allreduce_in, torch::Tensor& residual_in,
    torch::Tensor& residual_out, torch::Tensor& workspace_ptrs,
    int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_dim, double rms_eps, torch::Tensor& rms_gamma,
    bool launch_with_pdl, bool fp32_acc, bool use_oneshot,
    int64_t pattern_code, torch::Tensor& norm_out, torch::Tensor& quant_out,
    torch::Tensor& scale_out, torch::Tensor& scale_factor,
    cudaStream_t stream) {
  AllReduceFusionParams<T> params;
  params.nranks = static_cast<int>(world_size);
  params.rank = static_cast<int>(world_rank);
  params.size = static_cast<int>(token_num * hidden_dim);
  params.hidden_dim = static_cast<int>(hidden_dim);
  params.workspace = reinterpret_cast<void**>(workspace_ptrs.data_ptr());
  params.allreduce_in = allreduce_in.data_ptr();
  params.allreduce_out = nullptr;  // Not used for AR+RMS patterns
  params.residual_in = residual_in.data_ptr();
  params.residual_out = residual_out.data_ptr();
  params.norm_out = optional_ptr(norm_out);
  params.quant_out = optional_ptr(quant_out);
  params.scale_out = optional_ptr(scale_out);
  params.rms_gamma = rms_gamma.data_ptr();
  params.rms_eps = static_cast<float>(rms_eps);
  params.scale_factor = optional_float_ptr(scale_factor);
  params.use_oneshot = use_oneshot;
  params.layout = QuantizationSFLayout::SWIZZLED_128x4;
  params.pattern = static_cast<AllReduceFusionPattern>(pattern_code);
  params.trigger_completion_at_end = launch_with_pdl;
  params.stream = stream;

  auto status = allreduce_fusion_op(params, launch_with_pdl, fp32_acc);
  TORCH_CHECK(status == cudaSuccess,
              "novita allreduce_fusion_op failed: ",
              cudaGetErrorString(status));
}

}  // namespace

void novita_allreduce_fusion(
    torch::Tensor& allreduce_in,    // [token_num * hidden_dim] flat
    torch::Tensor& residual_in,     // [token_num * hidden_dim] flat
    torch::Tensor& residual_out,    // [token_num * hidden_dim] flat
    torch::Tensor& workspace_ptrs,  // [N] int64 IPC pointer array
    int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_dim, double rms_eps, torch::Tensor& rms_gamma,
    bool launch_with_pdl, bool fp32_acc, bool use_oneshot,
    int64_t pattern_code,
    // Optional tensors (pass 0-numel tensor for None):
    torch::Tensor& norm_out, torch::Tensor& quant_out,
    torch::Tensor& scale_out, torch::Tensor& scale_factor) {

  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(allreduce_in.device().index());

  auto dtype = allreduce_in.scalar_type();
  switch (dtype) {
    case at::ScalarType::Half:
      launch_allreduce_fusion<half>(
          allreduce_in, residual_in, residual_out, workspace_ptrs,
          world_size, world_rank, token_num, hidden_dim, rms_eps, rms_gamma,
          launch_with_pdl, fp32_acc, use_oneshot, pattern_code,
          norm_out, quant_out, scale_out, scale_factor, stream);
      break;
    case at::ScalarType::BFloat16:
      launch_allreduce_fusion<__nv_bfloat16>(
          allreduce_in, residual_in, residual_out, workspace_ptrs,
          world_size, world_rank, token_num, hidden_dim, rms_eps, rms_gamma,
          launch_with_pdl, fp32_acc, use_oneshot, pattern_code,
          norm_out, quant_out, scale_out, scale_factor, stream);
      break;
    case at::ScalarType::Float:
      launch_allreduce_fusion<float>(
          allreduce_in, residual_in, residual_out, workspace_ptrs,
          world_size, world_rank, token_num, hidden_dim, rms_eps, rms_gamma,
          launch_with_pdl, fp32_acc, use_oneshot, pattern_code,
          norm_out, quant_out, scale_out, scale_factor, stream);
      break;
    default:
      TORCH_CHECK(false, "novita_allreduce_fusion: unsupported dtype ",
                  dtype);
  }
}
