# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness test for fused QK-Norm + RoPE + FP8 + KV-Cache-Store kernel.

Verifies that the single fused kernel produces results matching the reference
pipeline: MiniMaxText01RMSNormTP.forward_qk (NCCL all-reduce) -> RoPE ->
FP8 quant -> KV cache scatter write.

Usage:
    pytest tests/unit/test_fused_qknorm_rope_fp8_kvstore.py -v
    python tests/unit/test_fused_qknorm_rope_fp8_kvstore.py
"""

import ctypes
import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# CUDA IPC helpers
# ---------------------------------------------------------------------------

_cuda_rt = ctypes.CDLL("libcudart.so")


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


def _ipc_get_handle(ptr: int) -> bytes:
    handle = _CudaIpcMemHandle()
    ret = _cuda_rt.cudaIpcGetMemHandle(
        ctypes.byref(handle), ctypes.c_void_p(ptr)
    )
    assert ret == 0, f"cudaIpcGetMemHandle failed with error {ret}"
    return bytes(handle.reserved)


def _ipc_open_handle(handle_bytes: bytes) -> int:
    handle = _CudaIpcMemHandle()
    handle.reserved[:] = handle_bytes
    ptr = ctypes.c_void_p()
    ret = _cuda_rt.cudaIpcOpenMemHandle(
        ctypes.byref(ptr), handle, 1,
    )
    assert ret == 0, f"cudaIpcOpenMemHandle failed with error {ret}"
    return ptr.value


def _ipc_close_handle(ptr: int) -> None:
    _cuda_rt.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))


def create_workspace(world_size, rank, max_tokens, group=None):
    buf_size = world_size * max_tokens * 3
    local_buf = torch.zeros(buf_size, dtype=torch.float32, device="cuda")
    handle = _ipc_get_handle(local_buf.data_ptr())
    all_handles = [None] * world_size
    dist.all_gather_object(all_handles, handle, group=group)
    peer_ptrs = []
    opened = []
    for r in range(world_size):
        if r == rank:
            peer_ptrs.append(local_buf.data_ptr())
        else:
            ptr = _ipc_open_handle(all_handles[r])
            peer_ptrs.append(ptr)
            opened.append(ptr)
    ws = torch.tensor(peer_ptrs, dtype=torch.int64, device="cuda")
    return ws, local_buf, opened


def destroy_workspace(opened):
    for ptr in opened:
        _ipc_close_handle(ptr)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def reference_qk_rmsnorm_tp(q, k, q_w, k_w, eps, world_size, group):
    """Cross-TP RMSNorm on Q and K (same as MiniMaxText01RMSNormTP.forward_qk)."""
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    q_ss = q32.pow(2).sum(dim=-1, keepdim=True)
    k_ss = k32.pow(2).sum(dim=-1, keepdim=True)
    if world_size > 1:
        qk = torch.cat([q_ss, k_ss], dim=-1)
        dist.all_reduce(qk, group=group)
        q_ss, k_ss = qk.chunk(2, dim=-1)
    D_q_full = q.size(-1) * world_size
    D_k_full = k.size(-1) * world_size
    q_out = (q32 * torch.rsqrt(q_ss / D_q_full + eps) * q_w.float()).to(q.dtype)
    k_out = (k32 * torch.rsqrt(k_ss / D_k_full + eps) * k_w.float()).to(k.dtype)
    return q_out, k_out


def reference_rope_neox(q, k, positions, cos_sin_cache, head_dim, rotary_dim):
    """NeoX-style RoPE on Q [N, nq*hd] and K [N, nk*hd]."""

    def _apply(x, nheads):
        N = x.shape[0]
        x = x.view(N, nheads, head_dim)
        embed_dim = rotary_dim // 2
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x1 = x_rot[..., :embed_dim]
        x2 = x_rot[..., embed_dim:]
        cos = cos_sin_cache[positions, :embed_dim].unsqueeze(1).float()
        sin = cos_sin_cache[positions, embed_dim:rotary_dim].unsqueeze(1).float()
        o1 = x1.float() * cos - x2.float() * sin
        o2 = x2.float() * cos + x1.float() * sin
        out = torch.cat([o1, o2, x_pass.float()], dim=-1)
        return out.to(x.dtype).view(N, -1)

    nq = q.shape[-1] // head_dim
    nk = k.shape[-1] // head_dim
    return _apply(q, nq), _apply(k, nk)


def reference_fp8_quant(x, scale):
    """Simulate per-tensor FP8 E4M3 quantization."""
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    x_scaled = x.float() / scale
    x_clamped = x_scaled.clamp(fp8_info.min, fp8_info.max)
    return x_clamped.to(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

MAX_TOKENS = 256
HEAD_DIM = 64
NUM_Q_HEADS_TOTAL = 48
NUM_KV_HEADS_TOTAL = 8
ROTARY_DIM = 64
MAX_POSITION = 8192
EPS = 1e-6


def _run_worker(
    world_size: int,
    rank: int,
    distributed_init_port: int,
    gpu_offset: int = 0,
):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    import vllm._novita_C  # noqa: F401

    nq = NUM_Q_HEADS_TOTAL // world_size
    nk = NUM_KV_HEADS_TOTAL // world_size
    nv = nk

    workspace_ptrs, local_buf, opened = create_workspace(
        world_size, rank, MAX_TOKENS, group
    )
    torch.ops._novita_C.clear_qk_norm_workspace(
        workspace_ptrs, world_size, rank, MAX_TOKENS
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    cos_sin_cache = torch.randn(
        MAX_POSITION, ROTARY_DIM, dtype=torch.bfloat16, device=device
    )

    q_weight = torch.randn(nq * HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(nk * HEAD_DIM, dtype=torch.bfloat16, device=device)
    q_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    token_nums = [1, 4, 16, 64, 128]
    epoch = 0

    num_blocks = 1024
    block_size = 16

    try:
        for num_tokens in token_nums:
            dist.barrier(group=group)
            torch.manual_seed(42 + rank)

            qkv_dim = (nq + nk + nv) * HEAD_DIM
            qkv = torch.randn(
                num_tokens, qkv_dim, dtype=torch.bfloat16, device=device
            )
            q_raw = qkv[:, : nq * HEAD_DIM].clone()
            k_raw = qkv[:, nq * HEAD_DIM : (nq + nk) * HEAD_DIM].clone()
            v_raw = qkv[:, (nq + nk) * HEAD_DIM :].clone()

            positions = torch.randint(
                0, MAX_POSITION, (num_tokens,), dtype=torch.int64, device=device
            )
            slot_mapping = torch.arange(
                num_tokens, dtype=torch.int64, device=device
            )

            # --- Reference pipeline ---
            ref_q, ref_k = reference_qk_rmsnorm_tp(
                q_raw, k_raw, q_weight, k_weight, EPS, world_size, group
            )
            ref_q_roped, ref_k_roped = reference_rope_neox(
                ref_q, ref_k, positions, cos_sin_cache, HEAD_DIM, ROTARY_DIM
            )

            # Compute dynamic scales from reference output
            q_absmax = ref_q_roped.float().abs().max().item()
            k_absmax = ref_k_roped.float().abs().max().item()
            v_absmax = v_raw.float().abs().max().item()
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            q_scale_val = max(q_absmax / fp8_max, 1e-12)
            k_scale_val = max(k_absmax / fp8_max, 1e-12)
            v_scale_val = max(v_absmax / fp8_max, 1e-12)
            q_scale.fill_(q_scale_val)
            k_scale.fill_(k_scale_val)
            v_scale.fill_(v_scale_val)

            ref_q_fp8 = reference_fp8_quant(ref_q_roped, q_scale_val)
            ref_k_fp8 = reference_fp8_quant(ref_k_roped, k_scale_val)
            ref_v_fp8 = reference_fp8_quant(v_raw, v_scale_val)

            # --- Fused kernel ---
            epoch += 1
            q_output = torch.zeros(
                num_tokens, nq * HEAD_DIM,
                dtype=torch.float8_e4m3fn, device=device,
            )
            k_cache = torch.zeros(
                num_blocks * block_size, nk * HEAD_DIM,
                dtype=torch.float8_e4m3fn, device=device,
            )
            v_cache = torch.zeros(
                num_blocks * block_size, nv * HEAD_DIM,
                dtype=torch.float8_e4m3fn, device=device,
            )

            torch.ops._novita_C.fused_qk_norm_rope_fp8_kvstore(
                qkv, nq, nk, nv,
                NUM_Q_HEADS_TOTAL, NUM_KV_HEADS_TOTAL,
                HEAD_DIM, EPS,
                q_weight, k_weight,
                True,  # is_neox
                positions, ROTARY_DIM, cos_sin_cache,
                q_output, q_scale,
                k_cache, v_cache,
                slot_mapping, k_scale, v_scale,
                workspace_ptrs, world_size, rank,
                MAX_TOKENS, epoch,
            )
            torch.cuda.synchronize()

            # --- Compare ---
            fused_q = q_output.float() * q_scale_val
            ref_q_f = ref_q_fp8.float() * q_scale_val

            fused_k_flat = k_cache[:num_tokens].float() * k_scale_val
            ref_k_f = ref_k_fp8.float() * k_scale_val

            fused_v_flat = v_cache[:num_tokens].float() * v_scale_val
            ref_v_f = ref_v_fp8.float() * v_scale_val

            q_maxdiff = (fused_q - ref_q_f).abs().max().item()
            k_maxdiff = (fused_k_flat - ref_k_f).abs().max().item()
            v_maxdiff = (fused_v_flat - ref_v_f).abs().max().item()

            # FP8 E4M3 has limited precision; allow some tolerance
            atol = 0.15
            rtol = 0.05

            try:
                torch.testing.assert_close(
                    fused_q, ref_q_f, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    fused_k_flat, ref_k_f, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    fused_v_flat, ref_v_f, atol=atol, rtol=rtol
                )
            except AssertionError:
                print(
                    f"RANK {rank}: FAILED tokens={num_tokens} "
                    f"q_diff={q_maxdiff:.4f} k_diff={k_maxdiff:.4f} "
                    f"v_diff={v_maxdiff:.4f}"
                )
                raise

            dist.barrier(group=group)
            if rank == 0:
                print(
                    f"  tokens={num_tokens:4d}  "
                    f"q_err={q_maxdiff:.4e}  k_err={k_maxdiff:.4e}  "
                    f"v_err={v_maxdiff:.4e}  PASSED"
                )

    finally:
        dist.barrier(group=group)
        destroy_workspace(opened)
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Multi-process launcher
# ---------------------------------------------------------------------------


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    test_target: Any,
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = []
    for i in range(world_size):
        p = mp.Process(
            target=test_target,
            args=(world_size, i, port, gpu_offset),
            name=f"Worker-{i}",
        )
        p.start()
        procs.append(p)
    for i, p in enumerate(procs):
        p.join()
        assert p.exitcode == 0, f"Process {i} failed with exit code {p.exitcode}"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

WORLD_SIZE = 8


def test_fused_qknorm_rope_fp8_kvstore():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    available = torch.cuda.device_count()
    if WORLD_SIZE > available:
        pytest.skip(f"Need {WORLD_SIZE} GPUs, have {available}")

    print(
        f"\n=== Fused QK-Norm+RoPE+FP8+KVStore: tp={WORLD_SIZE} "
        f"heads={NUM_Q_HEADS_TOTAL}Q/{NUM_KV_HEADS_TOTAL}KV hd={HEAD_DIM} ==="
    )
    multi_process_parallel(WORLD_SIZE, _run_worker)
    print("=== ALL PASSED ===\n")


if __name__ == "__main__":
    available = torch.cuda.device_count()
    if available < WORLD_SIZE:
        print(f"Need >= {WORLD_SIZE} GPUs, have {available}. Skipping.")
    else:
        test_fused_qknorm_rope_fp8_kvstore()
