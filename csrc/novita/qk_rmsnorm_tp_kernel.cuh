// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused QK RMS Norm with TP variance all-reduce.
//
// Replaces the MiniMaxText01RMSNormTP.forward_qk pattern:
//   q,k -> cast fp32 -> local variance -> NCCL all-reduce variance
//        -> rsqrt+scale -> cast back
// with a single kernel that does local variance computation, in-kernel
// NVLink P2P variance exchange, and normalization. Eliminates ~14 kernel
// launches + 1 NCCL call per layer.
//
// Workspace layout (per rank's buffer, float32):
//   For each (src_rank, token) pair, 3 floats:
//     [q_sum_of_squares, k_sum_of_squares, epoch_flag(as int)]
//   Index: src_rank * max_tokens * 3 + token_id * 3 + {0,1,2}
//
// Synchronization: epoch-based flag. The Python wrapper bumps the single-value
// CUDA tensor `epoch_state` using a capture-safe `Tensor.add_(1)` before launch.
// Writers store data then epoch via __threadfence_system(). Readers poll for a
// matching epoch value.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "include/flashinfer/vec_dtypes.cuh"

namespace flashinfer {
namespace qk_rmsnorm_tp {

using flashinfer::vec_t;

// ====================== Block Reduction ======================

#define QK_FINAL_MASK 0xffffffff

template <int NUM>
__inline__ __device__ void warpReduceSum(float* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(QK_FINAL_MASK, val[i], mask, 32);
  }
}

template <int NUM>
__inline__ __device__ void blockReduceSum(float* val) {
  static __shared__ float shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }
  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : 0.0f;
  }
  warpReduceSum<NUM>(val);
}

// ====================== Clear Workspace ======================

__global__ void clear_qk_norm_workspace_kernel(void** workspace, int rank, int world_size,
                                               int max_tokens) {
  float* my_buf = reinterpret_cast<float*>(workspace[rank]);
  int total = world_size * max_tokens * 3;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
    my_buf[i] = 0.0f;
  }
}

// ====================== Main Kernel ======================

template <typename T, int NRanks>
__global__ void fused_qk_rmsnorm_tp_kernel(T* __restrict__ q_in, T* __restrict__ k_in,
                                           T* __restrict__ q_out, T* __restrict__ k_out,
                                           const T* __restrict__ q_weight,
                                           const T* __restrict__ k_weight, float q_eps, float k_eps,
                                           int D_q_local, int D_k_local, int D_q_full, int D_k_full,
                                           int num_tokens, int max_tokens,
                                           void** __restrict__ workspace, int rank,
                                           int const* __restrict__ epoch_state) {
  static constexpr int VEC_SIZE = 16 / sizeof(T);

  int token_id = blockIdx.x;
  if (token_id >= num_tokens) return;

  int tid = threadIdx.x;
  int const epoch = epoch_state[0];

  // ============ Phase 1: Compute local sum of squares ============
  float q_ss = 0.0f, k_ss = 0.0f;

  int num_q_vecs = D_q_local / VEC_SIZE;
  for (int i = tid; i < num_q_vecs; i += blockDim.x) {
    vec_t<T, VEC_SIZE> v;
    v.load(q_in + token_id * D_q_local + i * VEC_SIZE);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float fv = static_cast<float>(v[j]);
      q_ss += fv * fv;
    }
  }

  int num_k_vecs = D_k_local / VEC_SIZE;
  for (int i = tid; i < num_k_vecs; i += blockDim.x) {
    vec_t<T, VEC_SIZE> v;
    v.load(k_in + token_id * D_k_local + i * VEC_SIZE);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float fv = static_cast<float>(v[j]);
      k_ss += fv * fv;
    }
  }

  float vals[2] = {q_ss, k_ss};
  blockReduceSum<2>(vals);
  q_ss = vals[0];
  k_ss = vals[1];

  // ============ Phase 2: Cross-GPU variance exchange ============
  __shared__ float s_q_inv_rms, s_k_inv_rms;

  if (tid == 0) {
    if constexpr (NRanks == 1) {
      // No communication needed for single GPU
      s_q_inv_rms = rsqrtf(q_ss / D_q_full + q_eps);
      s_k_inv_rms = rsqrtf(k_ss / D_k_full + k_eps);
    } else {
      // Push partial sums to all peers' buffers via NVLink
#pragma unroll
      for (int r = 0; r < NRanks; r++) {
        volatile float* peer_buf = reinterpret_cast<volatile float*>(workspace[r]);
        int base = rank * max_tokens * 3 + token_id * 3;
        peer_buf[base + 0] = q_ss;
        peer_buf[base + 1] = k_ss;
      }
      __threadfence_system();
      // Write epoch flag after data is globally visible
#pragma unroll
      for (int r = 0; r < NRanks; r++) {
        volatile int* peer_flag =
            reinterpret_cast<volatile int*>(reinterpret_cast<volatile float*>(workspace[r]) +
                                            rank * max_tokens * 3 + token_id * 3 + 2);
        *peer_flag = epoch;
      }

      // Poll own buffer for all peers' data
      float total_q_ss = 0.0f, total_k_ss = 0.0f;
#pragma unroll
      for (int r = 0; r < NRanks; r++) {
        int base = r * max_tokens * 3 + token_id * 3;
        // Wait for epoch flag from rank r
        volatile int* my_flag =
            reinterpret_cast<volatile int*>(reinterpret_cast<volatile float*>(workspace[rank]) +
                                            base + 2);
        while (*my_flag != epoch) {
        }
        // Data is guaranteed visible (writer did threadfence_system before flag)
        volatile float* my_buf = reinterpret_cast<volatile float*>(workspace[rank]);
        total_q_ss += my_buf[base + 0];
        total_k_ss += my_buf[base + 1];
      }

      s_q_inv_rms = rsqrtf(total_q_ss / D_q_full + q_eps);
      s_k_inv_rms = rsqrtf(total_k_ss / D_k_full + k_eps);
    }
  }
  __syncthreads();

  // ============ Phase 3: Normalize q and k ============
  float q_scale = s_q_inv_rms;
  for (int i = tid; i < num_q_vecs; i += blockDim.x) {
    vec_t<T, VEC_SIZE> v, w;
    v.load(q_in + token_id * D_q_local + i * VEC_SIZE);
    w.load(q_weight + i * VEC_SIZE);
    vec_t<T, VEC_SIZE> out;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      out[j] = static_cast<T>(static_cast<float>(v[j]) * q_scale * static_cast<float>(w[j]));
    }
    out.store(q_out + token_id * D_q_local + i * VEC_SIZE);
  }

  float k_scale = s_k_inv_rms;
  for (int i = tid; i < num_k_vecs; i += blockDim.x) {
    vec_t<T, VEC_SIZE> v, w;
    v.load(k_in + token_id * D_k_local + i * VEC_SIZE);
    w.load(k_weight + i * VEC_SIZE);
    vec_t<T, VEC_SIZE> out;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      out[j] = static_cast<T>(static_cast<float>(v[j]) * k_scale * static_cast<float>(w[j]));
    }
    out.store(k_out + token_id * D_k_local + i * VEC_SIZE);
  }
}

// ====================== Launcher ======================

template <typename T, int NRanks>
cudaError_t launch_fused_qk_rmsnorm_tp(T* q_in, T* k_in, T* q_out, T* k_out, const T* q_weight,
                                        const T* k_weight, float q_eps, float k_eps, int D_q_local,
                                        int D_k_local, int D_q_full, int D_k_full, int num_tokens,
                                        int max_tokens, void** workspace, int rank,
                                        int* epoch_state,
                                        cudaStream_t stream) {
  constexpr int VEC_SIZE = 16 / sizeof(T);
  int threads_needed = D_q_local / VEC_SIZE;
  int block_size = ((threads_needed + 31) / 32) * 32;
  block_size = max(32, min(block_size, 1024));
  int grid_size = num_tokens;

  fused_qk_rmsnorm_tp_kernel<T, NRanks><<<grid_size, block_size, 0, stream>>>(
      q_in, k_in, q_out, k_out, q_weight, k_weight, q_eps, k_eps, D_q_local, D_k_local, D_q_full,
      D_k_full, num_tokens, max_tokens, workspace, rank, epoch_state);

  return cudaGetLastError();
}

template <typename T>
cudaError_t dispatch_fused_qk_rmsnorm_tp(T* q_in, T* k_in, T* q_out, T* k_out, const T* q_weight,
                                          const T* k_weight, float q_eps, float k_eps,
                                          int D_q_local, int D_k_local, int world_size,
                                          int num_tokens, int max_tokens, void** workspace,
                                          int rank, int* epoch_state,
                                          cudaStream_t stream) {
  int D_q_full = D_q_local * world_size;
  int D_k_full = D_k_local * world_size;

#define LAUNCH_QK_NORM(NRANKS)                                                                 \
  return launch_fused_qk_rmsnorm_tp<T, NRANKS>(q_in, k_in, q_out, k_out, q_weight, k_weight,  \
                                                q_eps, k_eps, D_q_local, D_k_local, D_q_full,  \
                                                D_k_full, num_tokens, max_tokens, workspace,    \
                                                rank, epoch_state, stream)

  switch (world_size) {
    case 1:
      LAUNCH_QK_NORM(1);
    case 2:
      LAUNCH_QK_NORM(2);
    case 4:
      LAUNCH_QK_NORM(4);
    case 8:
      LAUNCH_QK_NORM(8);
    default:
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_QK_NORM
}

}  // namespace qk_rmsnorm_tp
}  // namespace flashinfer
