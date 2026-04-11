# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Novita fused kernels (vllm._novita_C):
  1) TP QK-Norm + RoPE + FP8 quantization + KV cache store
  2) AllReduce + RMSNorm + Residual Add (+ optional FP8/FP4 quantization)

Isolated in a separate .so from the main vLLM C extensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_novita_available = False
if current_platform.is_cuda():
    try:
        import vllm._novita_C  # noqa: F401

        _novita_available = True
    except ImportError:
        pass


def is_novita_available() -> bool:
    return _novita_available


def register_novita_ops() -> None:
    """Register the novita custom op with vLLM's torch library.

    Must be called after vllm._novita_C is imported.
    """
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="novita_fused_attn",
        op_func=novita_fused_attn,
        mutates_args=["output"],
        fake_impl=novita_fused_attn_fake,
    )

    direct_register_custom_op(
        op_name="novita_fused_allreduce_norm",
        op_func=call_novita_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
        fake_impl=call_novita_fused_allreduce_norm_fake,
    )


# ---------------------------------------------------------------------------
# Custom op: novita_fused_attn
#
# Encapsulates get_attention_context + fused CUDA kernel + attention-only
# forward into a single opaque op so torch.compile / cudagraph can trace
# through the MiniMaxM2 fused path without graph breaks.
# ---------------------------------------------------------------------------


@dataclass
class _QKNormTPWorkspace:
    workspace_ptrs: torch.Tensor
    local_buf: torch.Tensor
    max_tokens: int
    world_size: int
    rank: int
    epoch: int = 0

    def next_epoch(self) -> int:
        self.epoch += 1
        return self.epoch


_qk_norm_tp_workspaces: dict[tuple[int, int, int, int], _QKNormTPWorkspace] = {}
_logged_fused_attn_layers: set[str] = set()


def _create_qk_norm_tp_workspace(
    *,
    device: torch.device,
    group: object,
    world_size: int,
    rank: int,
    max_tokens: int,
) -> _QKNormTPWorkspace:
    from vllm.distributed.device_communicators.cuda_wrapper import (
        CudaRTLibrary,
        cudaIpcMemHandle_t,
    )

    buf_size = world_size * max_tokens * 3
    local_buf = torch.zeros(buf_size, dtype=torch.float32, device=device)

    if world_size == 1:
        workspace_ptrs = torch.tensor(
            [local_buf.data_ptr()], dtype=torch.int64, device=device
        )
        return _QKNormTPWorkspace(
            workspace_ptrs=workspace_ptrs,
            local_buf=local_buf,
            max_tokens=max_tokens,
            world_size=world_size,
            rank=rank,
        )

    cudart = CudaRTLibrary()
    local_handle = bytes(cudart.cudaIpcGetMemHandle(local_buf.data_ptr()).internal)
    gathered_handles: list[bytes | None] = [None] * world_size
    dist.all_gather_object(gathered_handles, local_handle, group=group)

    peer_ptrs: list[int] = []
    for peer_rank, handle_bytes in enumerate(gathered_handles):
        assert handle_bytes is not None
        if peer_rank == rank:
            peer_ptrs.append(local_buf.data_ptr())
            continue

        handle = cudaIpcMemHandle_t()
        handle.internal[:] = handle_bytes
        peer_ptrs.append(cudart.cudaIpcOpenMemHandle(handle).value)

    workspace_ptrs = torch.tensor(peer_ptrs, dtype=torch.int64, device=device)
    return _QKNormTPWorkspace(
        workspace_ptrs=workspace_ptrs,
        local_buf=local_buf,
        max_tokens=max_tokens,
        world_size=world_size,
        rank=rank,
    )


def _get_qk_norm_tp_workspace(
    *,
    device: torch.device,
    group: object,
    world_size: int,
    rank: int,
    max_tokens: int,
) -> _QKNormTPWorkspace:
    cache_key = (id(group), device.index or 0, world_size, max_tokens)
    workspace = _qk_norm_tp_workspaces.get(cache_key)
    if workspace is not None:
        return workspace

    workspace = _create_qk_norm_tp_workspace(
        device=device,
        group=group,
        world_size=world_size,
        rank=rank,
        max_tokens=max_tokens,
    )
    novita_clear_qk_norm_workspace(
        workspace.workspace_ptrs, world_size, rank, max_tokens
    )
    _qk_norm_tp_workspaces[cache_key] = workspace
    return workspace


def _tp_qk_rmsnorm_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    group: object,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    q_var = q_fp32.pow(2).mean(dim=-1, keepdim=True)
    k_var = k_fp32.pow(2).mean(dim=-1, keepdim=True)

    if world_size > 1:
        qk_var = torch.cat([q_var, k_var], dim=-1)
        dist.all_reduce(qk_var, group=group)
        qk_var /= world_size
        q_var, k_var = qk_var.chunk(2, dim=-1)

    q_out = q_fp32 * torch.rsqrt(q_var + eps) * q_weight.to(torch.float32)
    k_out = k_fp32 * torch.rsqrt(k_var + eps) * k_weight.to(torch.float32)
    return q_out.to(orig_dtype), k_out.to(orig_dtype)


def novita_fused_attn(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    num_heads_q_total: int,
    num_heads_k_total: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    eps: float,
    rotary_dim: int,
    max_tokens: int,
    is_neox: bool = True,
) -> None:
    """Fused TP QK-norm + RoPE + FP8 quant + KV-cache-store + attention."""
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
        unified_attention_with_output,
        unified_kv_cache_update,
    )

    tp_group = get_tp_group()
    world_size = tp_group.world_size
    rank = tp_group.rank_in_group
    num_tokens = qkv.shape[0]
    if num_tokens > max_tokens:
        raise ValueError(
            f"novita_fused_attn got {num_tokens=} > configured {max_tokens=}"
        )

    _, _, kv_cache, slot_mapping = get_attention_context(layer_name)

    if kv_cache.numel() == 0 or slot_mapping is None:
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = _tp_qk_rmsnorm_fallback(
            q=q,
            k=k,
            q_weight=q_weight,
            k_weight=k_weight,
            eps=eps,
            group=tp_group.device_group,
            world_size=world_size,
        )
        torch.ops._C.rotary_embedding(
            positions, q, k, head_dim, cos_sin_cache, is_neox
        )

        q_view = q.view(-1, num_heads_q, head_dim)
        k_view = k.view(-1, num_heads_k, head_dim)
        v_view = v.view(-1, num_heads_v, head_dim)
        output_view = output.view(-1, num_heads_q, head_dim)

        kv_dep = unified_kv_cache_update(k_view, v_view, layer_name)
        unified_attention_with_output(
            q_view,
            k_view,
            v_view,
            output_view,
            layer_name,
            kv_cache_dummy_dep=kv_dep,
        )
        return

    key_cache, value_cache = kv_cache.unbind(0)
    workspace = _get_qk_norm_tp_workspace(
        device=qkv.device,
        group=tp_group.cpu_group,
        world_size=world_size,
        rank=rank,
        max_tokens=max_tokens,
    )
    epoch = workspace.next_epoch()

    if layer_name not in _logged_fused_attn_layers:
        logger.info(
            "novita fused kernel ACTIVE for layer %s (tp=%d, max_tokens=%d)",
            layer_name,
            world_size,
            max_tokens,
        )
        _logged_fused_attn_layers.add(layer_name)

    q_output = torch.empty(
        num_tokens, q_size, dtype=torch.float8_e4m3fn, device=qkv.device
    )
    torch.ops._novita_C.fused_qk_norm_rope_fp8_kvstore(
        qkv.contiguous(),
        num_heads_q,
        num_heads_k,
        num_heads_v,
        num_heads_q_total,
        num_heads_k_total,
        head_dim,
        eps,
        q_weight,
        k_weight,
        is_neox,
        positions,
        rotary_dim,
        cos_sin_cache,
        q_output,
        q_scale,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        workspace.workspace_ptrs,
        world_size,
        rank,
        max_tokens,
        epoch,
    )

    q_view = q_output.view(-1, num_heads_q, head_dim)
    output_view = output.view(-1, num_heads_q, head_dim)
    unified_attention_with_output(q_view, q_view, q_view, output_view, layer_name)


def novita_fused_attn_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    num_heads_q_total: int,
    num_heads_k_total: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    eps: float,
    rotary_dim: int,
    max_tokens: int,
    is_neox: bool = True,
) -> None:
    return


# ---------------------------------------------------------------------------
# Custom op: novita_fused_allreduce_norm
#
# Drop-in replacement for flashinfer_trtllm_fused_allreduce_norm.
# Same signature so the graph pattern matching pass can swap them.
# Uses flashinfer workspace for IPC memory, but our optimized CUDA kernel.
# ---------------------------------------------------------------------------

_NOVITA_ONE_SHOT_MAX_SIZES_MB: dict[int, dict[int, float]] = {
    90: {2: 32, 4: 2, 8: 0.5},
    100: {2: 32, 4: 4, 8: 1},
}

MiB = 1024 * 1024

# Token-count range where novita kernel is slower than flashinfer.
# Benchmarked on H200 / tp=8 / hidden=7168:
#   tokens ≤ 128  → novita wins  (up to 2.9×)
#   tokens 160-768 → flashinfer wins
#   tokens ≥ 896  → novita wins again
NOVITA_AR_FALLBACK_MIN_TOKENS = 160
NOVITA_AR_FALLBACK_MAX_TOKENS = 768

_logged_fallback = False


def _flashinfer_allreduce_fallback(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    workspace: object,
    pattern_code: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    use_oneshot: bool,
    norm_out: torch.Tensor | None,
    quant_out: torch.Tensor | None,
    scale_out: torch.Tensor | None,
    scale_factor: torch.Tensor | None,
) -> None:
    """Call the native flashinfer allreduce_fusion kernel directly."""
    import flashinfer.comm as fi_comm

    if norm_out is None:
        actual_norm_out = allreduce_in
        residual_out = residual
    else:
        actual_norm_out = norm_out
        residual_out = allreduce_in

    layout_code = None
    if workspace.backend == "trtllm":  # type: ignore[union-attr]
        layout_code = fi_comm.QuantizationSFLayout.SWIZZLED_128x4

    fi_comm.allreduce_fusion(
        input=allreduce_in,
        workspace=workspace,
        pattern=pattern_code,
        launch_with_pdl=launch_with_pdl,
        output=None,
        residual_out=residual_out,
        norm_out=actual_norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        residual_in=residual,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
        use_oneshot=use_oneshot,
        fp32_acc=fp32_acc,
    )


def call_novita_fused_allreduce_norm(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_factor: torch.Tensor | None = None,
) -> None:
    """Fused allreduce + RMSNorm + residual add using novita kernel.

    Mirrors the flashinfer_trtllm_fused_allreduce_norm signature exactly.
    Uses flashinfer workspace for IPC shared memory management.

    When token count falls in the novita-kernel disadvantage zone
    (NOVITA_AR_FALLBACK_MIN_TOKENS .. NOVITA_AR_FALLBACK_MAX_TOKENS),
    falls back to the native flashinfer kernel automatically.
    """
    from vllm.distributed import get_tp_group
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_quant_workspace,
        get_fi_ar_workspace,
    )
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
    )

    num_tokens, hidden_size = allreduce_in.shape
    element_size = allreduce_in.element_size()
    current_tensor_size = num_tokens * hidden_size * element_size

    curr_device = current_platform.get_device_capability()
    device_capability = (
        curr_device.to_int() if curr_device is not None else None
    )
    max_one_shot_size = _NOVITA_ONE_SHOT_MAX_SIZES_MB.get(
        device_capability, {}  # type: ignore[arg-type]
    ).get(world_size, None)
    use_oneshot = (
        max_one_shot_size is None
        or current_tensor_size <= max_one_shot_size * MiB
    )

    rank = get_tensor_model_parallel_rank()
    group = get_tp_group().device_group
    workspace_kwargs = dict(
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=allreduce_in.dtype,
        group=group,
    )

    try:
        import flashinfer.comm as _fi_comm

        ar_patterns = _fi_comm.AllReduceFusionPattern
        is_quant = pattern_code in (
            ar_patterns.kARResidualRMSNormFP8Quant,
            ar_patterns.kARResidualRMSNormFP4Quant,
        )
    except ImportError:
        is_quant = False

    workspace = (
        get_fi_ar_quant_workspace(**workspace_kwargs)
        if is_quant
        else get_fi_ar_workspace(**workspace_kwargs)
    )
    assert workspace is not None, (
        "Flashinfer workspace must be initialized for novita allreduce fusion"
    )

    # --- token-count based dispatch ---
    if NOVITA_AR_FALLBACK_MIN_TOKENS <= num_tokens <= NOVITA_AR_FALLBACK_MAX_TOKENS:
        global _logged_fallback
        if not _logged_fallback:
            logger.info(
                "novita AR fallback to flashinfer for %d tokens "
                "(disadvantage zone %d-%d)",
                num_tokens,
                NOVITA_AR_FALLBACK_MIN_TOKENS,
                NOVITA_AR_FALLBACK_MAX_TOKENS,
            )
            _logged_fallback = True
        _flashinfer_allreduce_fallback(
            allreduce_in=allreduce_in,
            residual=residual,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            workspace=workspace,
            pattern_code=pattern_code,
            launch_with_pdl=launch_with_pdl,
            fp32_acc=fp32_acc,
            use_oneshot=use_oneshot,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            scale_factor=scale_factor,
        )
        return

    # --- novita fused kernel path ---
    if norm_out is None:
        actual_norm_out = allreduce_in
        residual_out = residual
    else:
        actual_norm_out = norm_out
        residual_out = allreduce_in

    def _flatten(t: torch.Tensor) -> torch.Tensor:
        assert t.is_contiguous(), f"Tensor must be contiguous, got {t.shape}"
        return t.view(-1)

    empty = torch.empty(0, device=allreduce_in.device, dtype=torch.float32)

    torch.ops._novita_C.allreduce_fusion(
        _flatten(allreduce_in),
        _flatten(residual),
        _flatten(residual_out),
        workspace.workspace_tensor,
        world_size,
        rank,
        num_tokens,
        hidden_size,
        rms_eps,
        rms_gamma,
        launch_with_pdl,
        fp32_acc,
        use_oneshot,
        pattern_code,
        _flatten(actual_norm_out) if actual_norm_out.numel() > 0 else empty,
        _flatten(quant_out) if quant_out is not None else empty,
        _flatten(scale_out) if scale_out is not None else empty,
        scale_factor if scale_factor is not None else empty,
    )


def call_novita_fused_allreduce_norm_fake(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_factor: torch.Tensor | None = None,
) -> None:
    pass


# ---------------------------------------------------------------------------
# Fused QK RMS Norm with TP variance all-reduce
# ---------------------------------------------------------------------------

_qk_norm_epoch = 0


def novita_fused_qk_rmsnorm_tp(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    workspace_ptrs: torch.Tensor,
    world_size: int,
    rank: int,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK RMS Norm with TP variance all-reduce via NVLink.

    Replaces MiniMaxText01RMSNormTP.forward_qk with a single CUDA kernel
    that fuses cast + variance + cross-GPU exchange + normalize.

    Args:
        q: [num_tokens, D_q_local] input query tensor (bf16/fp16)
        k: [num_tokens, D_k_local] input key tensor (bf16/fp16)
        q_weight: [D_q_local] RMS norm weight for q
        k_weight: [D_k_local] RMS norm weight for k
        q_eps: epsilon for q RMS norm
        k_eps: epsilon for k RMS norm
        workspace_ptrs: [world_size] int64 tensor of NVLink-accessible buffer pointers
        world_size: tensor parallel world size
        rank: current rank
        max_tokens: maximum tokens the workspace was allocated for

    Returns:
        (q_out, k_out) normalized tensors with same shape and dtype as inputs
    """
    global _qk_norm_epoch
    _qk_norm_epoch += 1

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    torch.ops._novita_C.qk_rmsnorm_tp(
        q, k, q_out, k_out,
        q_weight, k_weight,
        q_eps, k_eps,
        workspace_ptrs,
        world_size, rank,
        max_tokens, _qk_norm_epoch,
    )

    return q_out, k_out


def novita_clear_qk_norm_workspace(
    workspace_ptrs: torch.Tensor,
    world_size: int,
    rank: int,
    max_tokens: int,
) -> None:
    """Clear the QK norm workspace (call once before first use)."""
    torch.ops._novita_C.clear_qk_norm_workspace(
        workspace_ptrs, world_size, rank, max_tokens,
    )


if _novita_available:
    register_novita_ops()
