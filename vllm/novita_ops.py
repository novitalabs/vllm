# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Novita fused kernels (vllm._novita_C):
  AllReduce + RMSNorm + Residual Add (+ optional FP8/FP4 quantization)

Isolated in a separate .so from the main vLLM C extensions.
"""

from __future__ import annotations

import torch

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
        if pattern_code in (
            ar_patterns.kARResidualRMSNormFP8Quant,
            ar_patterns.kARResidualRMSNormFP4Quant,
        ):
            workspace = get_fi_ar_quant_workspace(**workspace_kwargs)
        else:
            workspace = get_fi_ar_workspace(**workspace_kwargs)
    except ImportError:
        workspace = get_fi_ar_workspace(**workspace_kwargs)

    assert workspace is not None, (
        "Flashinfer workspace must be initialized for novita allreduce fusion"
    )

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


if _novita_available:
    register_novita_ops()
