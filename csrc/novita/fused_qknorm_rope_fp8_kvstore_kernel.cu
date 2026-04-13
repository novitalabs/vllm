/*
 * Fused QK-Norm (with TP variance exchange) + RoPE + FP8 Cast + KV-Cache-Store
 *
 * Combines the following per-layer pipeline into a single GPU kernel:
 *   Phase 1: Local sum-of-squares for Q and K (across ALL local heads)
 *   Phase 2: Cross-GPU P2P variance exchange via NVLink (epoch-based)
 *   Phase 3: RMS normalize + RoPE + FP8 quantize + write Q output / KV cache
 *
 * Grid: one block per token.
 * Block: 8 warps (256 threads) — one warp per local head (6Q + 1K + 1V).
 *
 * Adapted from:
 *   - csrc/novita/qk_rmsnorm_tp_kernel.cuh  (P2P variance exchange)
 *   - feature/novita-combined-optimizations branch kernel (warp-per-head style)
 */

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>


#define FINAL_MASK 0xffffffff

// ============================================================================
// Utility helpers
// ============================================================================

namespace novita_fused {

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace novita_fused

// ============================================================================
// Main fused kernel
// ============================================================================

/*
 * Template parameters:
 *   HEAD_DIM  - dimension per head (64, 128, ...)
 *   NRanks    - TP world size (1, 2, 4, 8)
 *   IS_NEOX   - true for NeoX-style RoPE, false for interleaved
 *
 * Block layout (HEAD_DIM=64, TP8 MiniMax M2):
 *   Warp 0..5: Q heads 0..5  (norm + RoPE + FP8 -> q_output)
 *   Warp 6:    K head 0       (norm + RoPE + FP8 -> k_cache)
 *   Warp 7:    V head 0       (FP8 -> v_cache, no norm/rope)
 */
template <int HEAD_DIM, int NRanks, bool IS_NEOX>
__global__ void fusedQKNormRopeFP8KVStoreKernel(
    __nv_bfloat16 const* __restrict__ qkv,   // [num_tokens, (nq+nk+nv)*hd]
    int const num_heads_q,                    // local Q heads (after TP split)
    int const num_heads_k,                    // local K heads
    int const num_heads_v,                    // local V heads
    int const num_heads_q_total,              // total Q heads (before TP split)
    int const num_heads_k_total,              // total K heads
    float const eps,
    __nv_bfloat16 const* __restrict__ q_weight,  // [num_heads_q * HEAD_DIM]
    __nv_bfloat16 const* __restrict__ k_weight,  // [num_heads_k * HEAD_DIM]
    int64_t const* __restrict__ position_ids,
    int const num_tokens,
    int const rotary_dim,
    __nv_bfloat16 const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim]
    __nv_fp8_e4m3* __restrict__ q_output,     // [num_tokens, nq*hd] FP8
    float const* __restrict__ q_scale_ptr,
    int64_t const q_output_stride,
    __nv_fp8_e4m3* __restrict__ k_cache,      // paged KV cache (flat)
    __nv_fp8_e4m3* __restrict__ v_cache,
    int64_t const* __restrict__ slot_mapping,
    float const* __restrict__ k_scale_ptr,
    float const* __restrict__ v_scale_ptr,
    int64_t const kv_cache_stride,
    void** __restrict__ workspace,
    int const rank,
    int const max_tokens,
    int const* __restrict__ epoch_state) {

  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;
  int const tokenIdx = blockIdx.x;
  int const epoch = epoch_state[0];
  if (tokenIdx >= num_tokens) return;

  int const total_local_heads = num_heads_q + num_heads_k + num_heads_v;

  // Determine which head this warp handles
  enum HeadType { Q_HEAD, K_HEAD, V_HEAD, IDLE };
  HeadType headType = IDLE;
  int headIdx = 0;

  if (warpId < num_heads_q) {
    headType = Q_HEAD;
    headIdx = warpId;
  } else if (warpId < num_heads_q + num_heads_k) {
    headType = K_HEAD;
    headIdx = warpId - num_heads_q;
  } else if (warpId < total_local_heads) {
    headType = V_HEAD;
    headIdx = warpId - num_heads_q - num_heads_k;
  }

  // ====== Load head data from QKV buffer ======
  static_assert(HEAD_DIM % 64 == 0, "HEAD_DIM must be multiple of 64");
  constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
  constexpr int ELEM_BYTES = ELEMS_PER_THREAD * sizeof(__nv_bfloat16);
  static_assert(ELEM_BYTES % 4 == 0);
  constexpr int VEC_INTS = ELEM_BYTES / 4;

  float elements[ELEMS_PER_THREAD];

  if (headType != IDLE) {
    int const all_heads = num_heads_q + num_heads_k + num_heads_v;
    int64_t const tokenOff =
        static_cast<int64_t>(tokenIdx) * all_heads * HEAD_DIM;
    int64_t warpOff;
    if (headType == Q_HEAD) {
      warpOff = tokenOff + static_cast<int64_t>(headIdx) * HEAD_DIM;
    } else if (headType == K_HEAD) {
      warpOff =
          tokenOff + static_cast<int64_t>(num_heads_q + headIdx) * HEAD_DIM;
    } else {
      warpOff = tokenOff +
                static_cast<int64_t>(num_heads_q + num_heads_k + headIdx) *
                    HEAD_DIM;
    }
    int64_t threadOff = warpOff + laneId * ELEMS_PER_THREAD;

    // Vectorized BF16 load
    using vec_T = typename std::conditional<VEC_INTS == 1, uint,
                  typename std::conditional<VEC_INTS == 2, uint2, uint4>::type
                  >::type;
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[threadOff]);
#pragma unroll
    for (int i = 0; i < VEC_INTS; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(
          reinterpret_cast<uint const*>(&vec) + i));
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) elements[i] = 0.0f;
  }

  // NOTE: V heads skip norm/RoPE but must NOT return early — they need
  // to participate in __syncthreads() during Phases 1-2. V store happens
  // after Phase 2.

  // ====== Phase 1: Per-warp sum of squares + block reduce ======

  float local_ss = 0.0f;
  if (headType == Q_HEAD || headType == K_HEAD) {
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      local_ss += elements[i] * elements[i];
    }
    local_ss = novita_fused::warpReduceSum(local_ss);
  }

  // Block-level reduce: sum per-warp SS into total Q SS and K SS
  __shared__ float s_warp_ss[32];  // room for up to 32 warps (1024 threads)
  __shared__ float s_q_inv_rms, s_k_inv_rms;

  if (laneId == 0 && headType != IDLE) {
    s_warp_ss[warpId] = local_ss;
  }
  __syncthreads();

  // Thread 0 aggregates all warp SS into total Q and K SS
  if (threadIdx.x == 0) {
    float total_q_ss = 0.0f;
    for (int w = 0; w < num_heads_q; w++) {
      total_q_ss += s_warp_ss[w];
    }
    float total_k_ss = 0.0f;
    for (int w = num_heads_q; w < num_heads_q + num_heads_k; w++) {
      total_k_ss += s_warp_ss[w];
    }

    // ====== Phase 2: Cross-GPU P2P variance exchange ======
    int const D_q_full = num_heads_q_total * HEAD_DIM;
    int const D_k_full = num_heads_k_total * HEAD_DIM;

    if constexpr (NRanks == 1) {
      s_q_inv_rms = rsqrtf(total_q_ss / static_cast<float>(D_q_full) + eps);
      s_k_inv_rms = rsqrtf(total_k_ss / static_cast<float>(D_k_full) + eps);
    } else {
      // Push partial sums to all peers' buffers via NVLink
#pragma unroll
      for (int r = 0; r < NRanks; r++) {
        volatile float* peer_buf =
            reinterpret_cast<volatile float*>(workspace[r]);
        int base = rank * max_tokens * 3 + tokenIdx * 3;
        peer_buf[base + 0] = total_q_ss;
        peer_buf[base + 1] = total_k_ss;
      }
      __threadfence_system();
#pragma unroll
      for (int r = 0; r < NRanks; r++) {
        volatile int* peer_flag = reinterpret_cast<volatile int*>(
            reinterpret_cast<volatile float*>(workspace[r]) +
            rank * max_tokens * 3 + tokenIdx * 3 + 2);
        *peer_flag = epoch;
      }

      // Poll own buffer for all peers' data
      float agg_q_ss = 0.0f, agg_k_ss = 0.0f;
#pragma unroll
      for (int r = 0; r < NRanks; r++) {
        int base = r * max_tokens * 3 + tokenIdx * 3;
        volatile int* my_flag = reinterpret_cast<volatile int*>(
            reinterpret_cast<volatile float*>(workspace[rank]) + base + 2);
        while (*my_flag != epoch) {
        }
        volatile float* my_buf =
            reinterpret_cast<volatile float*>(workspace[rank]);
        agg_q_ss += my_buf[base + 0];
        agg_k_ss += my_buf[base + 1];
      }

      s_q_inv_rms = rsqrtf(agg_q_ss / static_cast<float>(D_q_full) + eps);
      s_k_inv_rms = rsqrtf(agg_k_ss / static_cast<float>(D_k_full) + eps);
    }
  }
  __syncthreads();

  // ====== Phase 3: Normalize + RoPE + FP8 write ======

  // V heads: no norm/RoPE, just FP8 + cache store
  if (headType == V_HEAD) {
    int64_t const cacheSlot = slot_mapping[tokenIdx];
    if (cacheSlot >= 0) {
      int64_t const cacheOffset = cacheSlot * kv_cache_stride +
                                  static_cast<int64_t>(headIdx) * HEAD_DIM +
                                  laneId * ELEMS_PER_THREAD;
      float v_scale = *v_scale_ptr;
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / v_scale);
        v_cache[cacheOffset + i] = fp8;
      }
    }
    return;
  }

  if (headType == IDLE) return;

  // Apply RMSNorm weights
  float rms_rcp = (headType == Q_HEAD) ? s_q_inv_rms : s_k_inv_rms;
  {
    __nv_bfloat16 const* wptr = (headType == Q_HEAD) ? q_weight : k_weight;
    int const weightOff = headIdx * HEAD_DIM + laneId * ELEMS_PER_THREAD;

    using vec_T = typename std::conditional<VEC_INTS == 1, uint,
                  typename std::conditional<VEC_INTS == 2, uint2, uint4>::type
                  >::type;
    vec_T wvec = *reinterpret_cast<vec_T const*>(&wptr[weightOff]);
#pragma unroll
    for (int i = 0; i < VEC_INTS; i++) {
      float2 wvals = __bfloat1622float2(
          *reinterpret_cast<__nv_bfloat162 const*>(
              reinterpret_cast<uint const*>(&wvec) + i));
      elements[2 * i] *= rms_rcp * wvals.x;
      elements[2 * i + 1] *= rms_rcp * wvals.y;
    }
  }

  // Apply RoPE (NeoX style: first half <-> second half via shfl_xor)
  int const rotary_lanes = rotary_dim / ELEMS_PER_THREAD;
  bool const applyRotary = (laneId < rotary_lanes);

  if (applyRotary) {
    int64_t const pos = position_ids[tokenIdx];
    int const half_rotary = rotary_dim / 2;
    __nv_bfloat16 const* cache_row =
        cos_sin_cache + static_cast<int64_t>(pos) * rotary_dim;

    if constexpr (IS_NEOX) {
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (rotary_lanes >= 32) ? 0xFFFFFFFFu : ((1u << rotary_lanes) - 1);
      int base_half = (laneId * ELEMS_PER_THREAD) % half_rotary;

      // Vectorized cos/sin load (ELEMS_PER_THREAD BF16 values)
      float cos_arr[ELEMS_PER_THREAD];
      float sin_arr[ELEMS_PER_THREAD];
#pragma unroll
      for (int i = 0; i < VEC_INTS; i++) {
        __nv_bfloat162 cp = *reinterpret_cast<__nv_bfloat162 const*>(
            &cache_row[base_half + 2 * i]);
        float2 cf = __bfloat1622float2(cp);
        cos_arr[2 * i] = cf.x;
        cos_arr[2 * i + 1] = cf.y;

        __nv_bfloat162 sp = *reinterpret_cast<__nv_bfloat162 const*>(
            &cache_row[half_rotary + base_half + 2 * i]);
        float2 sf = __bfloat1622float2(sp);
        sin_arr[2 * i] = sf.x;
        sin_arr[2 * i + 1] = sf.y;
      }

#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        float e2 =
            __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          e2 = -e2;
        }
        elements[i] = elements[i] * cos_arr[i] + e2 * sin_arr[i];
      }
    } else {
      // Interleaved style
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        float e2 = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        int half_dim = (laneId * ELEMS_PER_THREAD + i) / 2;
        float cos_val = __bfloat162float(cache_row[half_dim]);
        float sin_val = __bfloat162float(cache_row[half_rotary + half_dim]);
        elements[i] = elements[i] * cos_val + e2 * sin_val;
      }
    }
  }

  // FP8 quantize + store
  if (headType == Q_HEAD) {
    float q_scale = *q_scale_ptr;
    int64_t const outOffset =
        static_cast<int64_t>(tokenIdx) * q_output_stride +
        static_cast<int64_t>(headIdx) * HEAD_DIM +
        laneId * ELEMS_PER_THREAD;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / q_scale);
      q_output[outOffset + i] = fp8;
    }
  } else {
    // K head -> paged KV cache
    int64_t const cacheSlot = slot_mapping[tokenIdx];
    if (cacheSlot < 0) return;

    float k_scale = *k_scale_ptr;
    int64_t const cacheOffset = cacheSlot * kv_cache_stride +
                                static_cast<int64_t>(headIdx) * HEAD_DIM +
                                laneId * ELEMS_PER_THREAD;

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / k_scale);
      k_cache[cacheOffset + i] = fp8;
    }
  }
}

// ============================================================================
// Launcher
// ============================================================================

static void launchFusedQKNormRopeFP8KVStore(
    void const* qkv, int num_tokens, int num_heads_q, int num_heads_k,
    int num_heads_v, int num_heads_q_total, int num_heads_k_total,
    int head_dim, float eps, void const* q_weight, void const* k_weight,
    bool is_neox, int64_t const* position_ids, int rotary_dim,
    __nv_bfloat16 const* cos_sin_cache, void* q_output,
    float const* q_scale, int64_t q_output_stride, void* k_cache,
    void* v_cache, int64_t const* slot_mapping, float const* k_scale,
    float const* v_scale, int64_t kv_cache_stride, void** workspace,
    int rank, int max_tokens, int* epoch_state, int world_size,
    cudaStream_t stream) {

  int const total_heads = num_heads_q + num_heads_k + num_heads_v;
  int const blockSize = total_heads * 32;
  TORCH_CHECK(blockSize <= 1024,
              "Too many local heads for single block: ", total_heads);
  int const gridSize = num_tokens;

#define NOVITA_LAUNCH(HD, NRANKS, NEOX)                                      \
  fusedQKNormRopeFP8KVStoreKernel<HD, NRANKS, NEOX>                          \
      <<<gridSize, blockSize, 0, stream>>>(                                  \
          reinterpret_cast<__nv_bfloat16 const*>(qkv), num_heads_q,          \
          num_heads_k, num_heads_v, num_heads_q_total, num_heads_k_total,    \
          eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),             \
          reinterpret_cast<__nv_bfloat16 const*>(k_weight), position_ids,    \
          num_tokens, rotary_dim, cos_sin_cache,                             \
          reinterpret_cast<__nv_fp8_e4m3*>(q_output), q_scale,              \
          q_output_stride, reinterpret_cast<__nv_fp8_e4m3*>(k_cache),       \
          reinterpret_cast<__nv_fp8_e4m3*>(v_cache), slot_mapping, k_scale, \
          v_scale, kv_cache_stride, workspace, rank, max_tokens, epoch_state)

#define NOVITA_DISPATCH_RANKS(HD, NEOX)         \
  switch (world_size) {                          \
    case 1: NOVITA_LAUNCH(HD, 1, NEOX); break;  \
    case 2: NOVITA_LAUNCH(HD, 2, NEOX); break;  \
    case 4: NOVITA_LAUNCH(HD, 4, NEOX); break;  \
    case 8: NOVITA_LAUNCH(HD, 8, NEOX); break;  \
    default:                                     \
      TORCH_CHECK(false, "Unsupported world_size: ", world_size); \
  }

#define NOVITA_DISPATCH_NEOX(HD)     \
  if (is_neox) {                     \
    NOVITA_DISPATCH_RANKS(HD, true)  \
  } else {                           \
    NOVITA_DISPATCH_RANKS(HD, false) \
  }

  switch (head_dim) {
    case 64:  NOVITA_DISPATCH_NEOX(64);  break;
    case 128: NOVITA_DISPATCH_NEOX(128); break;
    case 256: NOVITA_DISPATCH_NEOX(256); break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);
  }

#undef NOVITA_DISPATCH_NEOX
#undef NOVITA_DISPATCH_RANKS
#undef NOVITA_LAUNCH
}

// ============================================================================
// Torch C++ entry point
// ============================================================================

void fused_qk_norm_rope_fp8_kvstore(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t num_heads_q_total, int64_t num_heads_k_total,
    int64_t head_dim, double eps, torch::Tensor& q_weight,
    torch::Tensor& k_weight, bool is_neox, torch::Tensor& position_ids,
    int64_t rotary_dim, torch::Tensor& cos_sin_cache,
    torch::Tensor& q_output, torch::Tensor& q_scale, torch::Tensor& k_cache,
    torch::Tensor& v_cache, torch::Tensor& slot_mapping,
    torch::Tensor& k_scale, torch::Tensor& v_scale,
    torch::Tensor& workspace_ptrs, int64_t world_size, int64_t world_rank,
    int64_t max_tokens, torch::Tensor& epoch_state) {

  TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous());
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(position_ids.is_cuda() && position_ids.is_contiguous());
  TORCH_CHECK(position_ids.scalar_type() == torch::kInt64);
  TORCH_CHECK(q_weight.is_cuda() && q_weight.is_contiguous());
  TORCH_CHECK(k_weight.is_cuda() && k_weight.is_contiguous());
  TORCH_CHECK(cos_sin_cache.is_cuda() && cos_sin_cache.is_contiguous());
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.is_contiguous());
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64);
  TORCH_CHECK(q_scale.numel() == 1 && q_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(k_scale.numel() == 1 && k_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(v_scale.numel() == 1 && v_scale.scalar_type() == torch::kFloat32);
  TORCH_CHECK(epoch_state.is_cuda() && epoch_state.is_contiguous());
  TORCH_CHECK(epoch_state.scalar_type() == torch::kInt32);
  TORCH_CHECK(epoch_state.numel() == 1);

  int64_t num_tokens = qkv.size(0);
  int64_t q_output_stride = num_heads_q * head_dim;
  int64_t kv_cache_stride = num_heads_k * head_dim;

  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  void** ws_ptr = reinterpret_cast<void**>(workspace_ptrs.data_ptr<int64_t>());

  launchFusedQKNormRopeFP8KVStore(
      qkv.data_ptr(), static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v), static_cast<int>(num_heads_q_total),
      static_cast<int>(num_heads_k_total), static_cast<int>(head_dim),
      static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
      is_neox,
      reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
      static_cast<int>(rotary_dim),
      reinterpret_cast<__nv_bfloat16 const*>(cos_sin_cache.data_ptr()),
      q_output.data_ptr(),
      reinterpret_cast<float const*>(q_scale.data_ptr()), q_output_stride,
      k_cache.data_ptr(), v_cache.data_ptr(),
      reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
      reinterpret_cast<float const*>(k_scale.data_ptr()),
      reinterpret_cast<float const*>(v_scale.data_ptr()), kv_cache_stride,
      ws_ptr, static_cast<int>(world_rank), static_cast<int>(max_tokens),
      reinterpret_cast<int*>(epoch_state.data_ptr<int32_t>()),
      static_cast<int>(world_size), stream);
}
