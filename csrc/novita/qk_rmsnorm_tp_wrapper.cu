// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PyTorch C++ wrapper for the fused QK RMS Norm with TP variance all-reduce.

#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "qk_rmsnorm_tp_kernel.cuh"

using namespace flashinfer::qk_rmsnorm_tp;

namespace {

template <typename T>
void launch_qk_rmsnorm_tp(torch::Tensor& q_in, torch::Tensor& k_in, torch::Tensor& q_out,
                           torch::Tensor& k_out, torch::Tensor& q_weight, torch::Tensor& k_weight,
                           double q_eps, double k_eps, torch::Tensor& workspace_ptrs,
                           int64_t world_size, int64_t world_rank, int64_t max_tokens,
                           int64_t epoch, cudaStream_t stream) {
  int num_tokens = q_in.size(0);
  int D_q_local = q_in.size(1);
  int D_k_local = k_in.size(1);

  auto status = dispatch_fused_qk_rmsnorm_tp<T>(
      reinterpret_cast<T*>(q_in.data_ptr()), reinterpret_cast<T*>(k_in.data_ptr()),
      reinterpret_cast<T*>(q_out.data_ptr()), reinterpret_cast<T*>(k_out.data_ptr()),
      reinterpret_cast<const T*>(q_weight.data_ptr()),
      reinterpret_cast<const T*>(k_weight.data_ptr()), static_cast<float>(q_eps),
      static_cast<float>(k_eps), D_q_local, D_k_local, static_cast<int>(world_size), num_tokens,
      static_cast<int>(max_tokens), reinterpret_cast<void**>(workspace_ptrs.data_ptr()),
      static_cast<int>(world_rank), static_cast<int>(epoch), stream);

  TORCH_CHECK(status == cudaSuccess, "fused_qk_rmsnorm_tp failed: ", cudaGetErrorString(status));
}

}  // namespace

void novita_qk_rmsnorm_tp(torch::Tensor& q_in,              // [num_tokens, D_q_local]
                           torch::Tensor& k_in,              // [num_tokens, D_k_local]
                           torch::Tensor& q_out,             // [num_tokens, D_q_local]
                           torch::Tensor& k_out,             // [num_tokens, D_k_local]
                           torch::Tensor& q_weight,          // [D_q_local]
                           torch::Tensor& k_weight,          // [D_k_local]
                           double q_eps, double k_eps,
                           torch::Tensor& workspace_ptrs,    // [world_size] int64 ptrs
                           int64_t world_size, int64_t world_rank,
                           int64_t max_tokens, int64_t epoch) {
  TORCH_CHECK(q_in.is_cuda(), "q_in must be on CUDA");
  TORCH_CHECK(q_in.dim() == 2, "q_in must be 2D [num_tokens, D_q_local]");
  TORCH_CHECK(k_in.dim() == 2, "k_in must be 2D [num_tokens, D_k_local]");
  TORCH_CHECK(q_in.size(0) == k_in.size(0), "q and k must have same num_tokens");

  constexpr int kBytesPerAccess = 16;
  int vec_size = kBytesPerAccess / q_in.element_size();
  TORCH_CHECK(q_in.size(1) % vec_size == 0,
              "D_q_local must be divisible by vec_size (", vec_size, ")");
  TORCH_CHECK(k_in.size(1) % vec_size == 0,
              "D_k_local must be divisible by vec_size (", vec_size, ")");

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q_in.device().index());

  auto dtype = q_in.scalar_type();
  switch (dtype) {
    case at::ScalarType::Half:
      launch_qk_rmsnorm_tp<half>(q_in, k_in, q_out, k_out, q_weight, k_weight, q_eps, k_eps,
                                  workspace_ptrs, world_size, world_rank, max_tokens, epoch,
                                  stream);
      break;
    case at::ScalarType::BFloat16:
      launch_qk_rmsnorm_tp<__nv_bfloat16>(q_in, k_in, q_out, k_out, q_weight, k_weight, q_eps,
                                            k_eps, workspace_ptrs, world_size, world_rank,
                                            max_tokens, epoch, stream);
      break;
    default:
      TORCH_CHECK(false, "novita_qk_rmsnorm_tp: unsupported dtype ", dtype);
  }
}

void novita_clear_qk_norm_workspace(torch::Tensor& workspace_ptrs, int64_t world_size,
                                     int64_t world_rank, int64_t max_tokens) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(workspace_ptrs.device().index());
  void** ws = reinterpret_cast<void**>(workspace_ptrs.data_ptr());
  int total = static_cast<int>(world_size * max_tokens * 3);
  int block = 256;
  int grid = std::min((total + block - 1) / block, 128);
  clear_qk_norm_workspace_kernel<<<grid, block, 0, stream>>>(ws, static_cast<int>(world_rank),
                                                              static_cast<int>(world_size),
                                                              static_cast<int>(max_tokens));
}
