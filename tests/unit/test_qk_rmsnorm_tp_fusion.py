# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness test for the fused QK RMS Norm with TP variance all-reduce.

Tests that the novita fused kernel produces results matching the reference
Python implementation (MiniMaxText01RMSNormTP.forward_qk) which uses
NCCL all-reduce for variance exchange.

Usage:
    pytest tests/unit/test_qk_rmsnorm_tp_fusion.py -v
    # or directly:
    python tests/unit/test_qk_rmsnorm_tp_fusion.py
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
# CUDA IPC helpers (standalone, no flashinfer dependency)
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
        ctypes.byref(ptr),
        handle,
        1,  # cudaIpcMemLazyEnablePeerAccess
    )
    assert ret == 0, f"cudaIpcOpenMemHandle failed with error {ret}"
    return ptr.value


def _ipc_close_handle(ptr: int) -> None:
    _cuda_rt.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))


# ---------------------------------------------------------------------------
# Workspace creation via CUDA IPC
# ---------------------------------------------------------------------------


def create_qk_norm_workspace(
    world_size: int, rank: int, max_tokens: int, group=None
):
    """Create NVLink-accessible workspace for QK norm variance exchange.

    Each rank allocates a float buffer. IPC handles are exchanged so every
    rank can read/write every other rank's buffer via NVLink.

    Returns:
        workspace_ptrs: [world_size] int64 tensor on GPU with device pointers
        local_buf: the local CUDA tensor (must stay alive)
        opened_handles: list of opened IPC pointers (for cleanup)
    """
    buf_size = world_size * max_tokens * 3  # floats: (q_ss, k_ss, epoch) per src per token
    local_buf = torch.zeros(buf_size, dtype=torch.float32, device="cuda")

    handle = _ipc_get_handle(local_buf.data_ptr())
    all_handles = [None] * world_size
    dist.all_gather_object(all_handles, handle, group=group)

    peer_ptrs = []
    opened_handles = []
    for r in range(world_size):
        if r == rank:
            peer_ptrs.append(local_buf.data_ptr())
        else:
            ptr = _ipc_open_handle(all_handles[r])
            peer_ptrs.append(ptr)
            opened_handles.append(ptr)

    workspace_ptrs = torch.tensor(peer_ptrs, dtype=torch.int64, device="cuda")
    return workspace_ptrs, local_buf, opened_handles


def destroy_qk_norm_workspace(opened_handles):
    for ptr in opened_handles:
        _ipc_close_handle(ptr)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def reference_qk_rmsnorm_tp(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    world_size: int,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference QK RMS norm with NCCL all-reduce for variance."""
    orig_dtype = q.dtype
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)

    q_var_local = q_fp32.pow(2).sum(dim=-1, keepdim=True)
    k_var_local = k_fp32.pow(2).sum(dim=-1, keepdim=True)

    if world_size > 1:
        qk_var = torch.cat([q_var_local, k_var_local], dim=-1)
        dist.all_reduce(qk_var, group=group)
        q_var_local, k_var_local = qk_var.chunk(2, dim=-1)

    D_q_full = q.size(1) * world_size
    D_k_full = k.size(1) * world_size
    q_var = q_var_local / D_q_full
    k_var = k_var_local / D_k_full

    q_out = (
        q_fp32 * torch.rsqrt(q_var + q_eps) * q_weight.to(torch.float32)
    ).to(orig_dtype)
    k_out = (
        k_fp32 * torch.rsqrt(k_var + k_eps) * k_weight.to(torch.float32)
    ).to(orig_dtype)

    return q_out, k_out


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

MAX_TOKENS = 128


def _run_correctness_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    D_q_total: int,
    D_k_total: int,
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

    # Import novita ops
    import vllm._novita_C  # noqa: F401

    D_q_local = D_q_total // world_size
    D_k_local = D_k_total // world_size

    workspace_ptrs = None
    local_buf = None
    opened_handles = None

    try:
        workspace_ptrs, local_buf, opened_handles = create_qk_norm_workspace(
            world_size, rank, MAX_TOKENS, group=group
        )

        # Clear workspace (initialize epoch flags to 0)
        torch.ops._novita_C.clear_qk_norm_workspace(
            workspace_ptrs, world_size, rank, MAX_TOKENS
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)

        token_nums = [1, 4, 16, 64, 128]
        eps = 1e-6
        epoch = 0

        for token_num in token_nums:
            if token_num > MAX_TOKENS:
                continue

            dist.barrier(group=group)
            torch.manual_seed(42 + rank)

            q = torch.randn(
                token_num, D_q_local, dtype=dtype, device=device
            )
            k = torch.randn(
                token_num, D_k_local, dtype=dtype, device=device
            )
            q_weight = torch.randn(D_q_local, dtype=dtype, device=device)
            k_weight = torch.randn(D_k_local, dtype=dtype, device=device)

            q_clone = q.clone()
            k_clone = k.clone()

            # --- Reference (NCCL) ---
            ref_q, ref_k = reference_qk_rmsnorm_tp(
                q_clone, k_clone, q_weight, k_weight, eps, eps, world_size,
                group=group,
            )

            # --- Fused kernel ---
            epoch += 1
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)

            torch.ops._novita_C.qk_rmsnorm_tp(
                q, k, q_out, k_out,
                q_weight, k_weight,
                eps, eps,
                workspace_ptrs,
                world_size, rank,
                MAX_TOKENS, epoch,
            )
            torch.cuda.synchronize()

            # --- Compare ---
            atol = 1e-2 if dtype == torch.float16 else 5e-2
            rtol = 1e-2

            try:
                torch.testing.assert_close(
                    q_out.to(torch.float32),
                    ref_q.to(torch.float32),
                    atol=atol,
                    rtol=rtol,
                )
                torch.testing.assert_close(
                    k_out.to(torch.float32),
                    ref_k.to(torch.float32),
                    atol=atol,
                    rtol=rtol,
                )
            except AssertionError as e:  # noqa: F841
                q_diff = (
                    (q_out.float() - ref_q.float()).abs().max().item()
                )
                k_diff = (
                    (k_out.float() - ref_k.float()).abs().max().item()
                )
                print(
                    f"RANK {rank}: FAILED tokens={token_num} "
                    f"D_q={D_q_total} D_k={D_k_total} "
                    f"q_max_diff={q_diff:.6f} k_max_diff={k_diff:.6f}"
                )
                raise

            dist.barrier(group=group)
            if rank == 0:
                q_diff = (
                    (q_out.float() - ref_q.float()).abs().max().item()
                )
                k_diff = (
                    (k_out.float() - ref_k.float()).abs().max().item()
                )
                print(
                    f"  tokens={token_num:4d}  "
                    f"q_max_err={q_diff:.2e}  k_max_err={k_diff:.2e}  "
                    f"PASSED"
                )

    finally:
        dist.barrier(group=group)
        if opened_handles is not None:
            destroy_qk_norm_workspace(opened_handles)
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
    dtype: torch.dtype,
    D_q_total: int,
    D_k_total: int,
    test_target: Any,
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)

    distributed_init_port = get_open_port()
    procs = []
    for i in range(world_size):
        proc = mp.Process(
            target=test_target,
            args=(
                world_size, i, dtype,
                D_q_total, D_k_total,
                distributed_init_port, gpu_offset,
            ),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# MiniMax-Text-01 config: heads=48, kv_heads=8, head_dim=64, hidden=3072, TP=8
D_Q_TOTAL = 3072
D_K_TOTAL = 512
WORLD_SIZE = 8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qk_rmsnorm_tp_fusion(dtype):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    available_gpus = torch.cuda.device_count()
    if WORLD_SIZE > available_gpus:
        pytest.skip(
            f"world_size {WORLD_SIZE} > available GPUs {available_gpus}"
        )

    print(
        f"\n=== QK RMS Norm TP Fusion: tp={WORLD_SIZE} dtype={dtype} "
        f"D_q={D_Q_TOTAL} D_k={D_K_TOTAL} ==="
    )
    multi_process_parallel(
        WORLD_SIZE, dtype, D_Q_TOTAL, D_K_TOTAL,
        _run_correctness_worker,
    )
    print(f"=== PASSED ===\n")


if __name__ == "__main__":
    available = torch.cuda.device_count()
    if available < WORLD_SIZE:
        print(f"Need >= {WORLD_SIZE} GPUs, have {available}. Skipping.")
    else:
        print(f"Running with world_size={WORLD_SIZE}")
        test_qk_rmsnorm_tp_fusion(torch.bfloat16)
