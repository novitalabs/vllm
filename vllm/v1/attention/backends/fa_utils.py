# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Track whether upstream flash-attn is available on ROCm.
# Set during module initialization and never modified afterwards.
# This module-level flag avoids repeated import attempts and ensures
# consistent behavior (similar to IS_AITER_FOUND in _aiter_ops.py).
_ROCM_FLASH_ATTN_AVAILABLE = False

if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash

    # Try sgl_kernel FA3 first (faster kernel implementation)
    _USE_SGL_KERNEL_FA = False
    try:
        from sgl_kernel.flash_attn import (
            flash_attn_varlen_func as _sgl_flash_attn_varlen_func,
        )
        _USE_SGL_KERNEL_FA = True
        logger.info("Using sgl_kernel FlashAttention (faster FA3)")
    except ImportError:
        pass

    if _USE_SGL_KERNEL_FA:
        def flash_attn_varlen_func(  # type: ignore[misc]
            q, k, v,
            max_seqlen_q=None, cu_seqlens_q=None,
            max_seqlen_k=None, cu_seqlens_k=None,
            seqused_k=None, q_v=None,
            dropout_p=0.0, softmax_scale=None, causal=False,
            window_size=None, softcap=0.0, alibi_slopes=None,
            deterministic=False, return_attn_probs=False,
            block_table=None, return_softmax_lse=False, out=None,
            scheduler_metadata=None,
            q_descale=None, k_descale=None, v_descale=None,
            num_splits=0, fa_version=3, s_aux=None,
            cp_world_size=1, cp_rank=0, cp_tot_seqused_k=None,
        ):
            # Map vLLM call signature to sgl_kernel C++ op directly
            import torch as _torch
            sgl_window = tuple(window_size) if window_size is not None else (-1, -1)
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            _num_splits = num_splits if num_splits > 0 else 1

            # Two modes: paged (block_table != None) vs varlen (contiguous K/V)
            if block_table is not None:
                # Paged KV cache mode — mirrors flash_attn_with_kvcache
                # cu_seqlens_k=None, max_seqlen_k=None, seqused_k=cache_seqlens
                result_out, result_lse, *rest = _torch.ops.sgl_kernel.fwd.default(
                    q,                      # q
                    k,                      # k_cache
                    v,                      # v_cache
                    None,                   # k_new
                    None,                   # v_new
                    q_v,                    # qv
                    None,                   # out
                    cu_seqlens_q,           # cu_seqlens_q
                    None,                   # cu_seqlens_k (None for paged)
                    None,                   # cu_seqlens_k_new
                    None,                   # seqused_q
                    seqused_k,              # cache_seqlens
                    max_seqlen_q,           # max_seqlen_q
                    None,                   # max_seqlen_k (None for paged)
                    block_table,            # page_table
                    None,                   # kv_batch_idx
                    None,                   # leftpad_k
                    None,                   # rotary cos
                    None,                   # rotary sin
                    None,                   # seqlens_rotary
                    q_descale,
                    k_descale,
                    v_descale,
                    softmax_scale,
                    causal,
                    sgl_window[0],
                    sgl_window[1],
                    0,                      # attention_chunk
                    softcap,
                    is_rotary_interleaved=False,
                    scheduler_metadata=None,
                    num_splits=_num_splits,
                    pack_gqa=None,
                    sm_margin=0,
                    sinks=s_aux,
                )
            else:
                # Varlen mode — contiguous K/V (prefill without paged cache)
                result_out, result_lse, *rest = _torch.ops.sgl_kernel.fwd.default(
                    q, k, v,
                    None, None, q_v, None,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    None, None,
                    seqused_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    None,                   # no page_table
                    None, None, None, None, None,
                    q_descale, k_descale, v_descale,
                    softmax_scale, causal,
                    sgl_window[0], sgl_window[1],
                    0, softcap,
                    is_rotary_interleaved=False,
                    scheduler_metadata=None,
                    num_splits=_num_splits,
                    pack_gqa=None, sm_margin=0, sinks=s_aux,
                )
            result = (result_out, result_lse)
            if out is not None:
                # sgl_kernel allocates output internally; copy to vLLM's buffer
                if isinstance(result, tuple):
                    out.copy_(result[0])
                else:
                    out.copy_(result)
            if return_softmax_lse and isinstance(result, tuple):
                return result  # (out, lse)
            if isinstance(result, tuple):
                return result[0]
            return result

        def get_scheduler_metadata(*args, **kwargs):  # type: ignore[misc]
            # sgl_kernel FA3 doesn't use scheduler_metadata
            return None
    else:
        from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
            flash_attn_varlen_func,
            get_scheduler_metadata,
        )

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        # Mark that upstream flash-attn is available on ROCm
        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            raise ImportError(
                "ROCm platform requires upstream flash-attn "
                "to be installed. Please install flash-attn first."
            )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major >= 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 4 if supported, otherwise FA2."
            )
            fa_version = 4 if is_fa_version_supported(4) else 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        # FA4 currently uses batch-shape-dependent scheduling
        # heuristics on SM100+, which breaks batch invariance.
        if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
            logger.warning_once(
                "Cannot use FA version 4 with batch invariance, "
                "defaulting to FA version 2.",
                scope="local",
            )
            fa_version = 2

        # FA4 on SM100 (Blackwell) has TMEM capacity limits that restrict
        # supported head dimensions.
        # See: https://github.com/Dao-AILab/flash-attention/issues/1959
        # Exception: hdim 192 is supported for MLA's diff-headdim case
        # (qk=192, v=128), added upstream in commits 1a15733e/1b36ab19.
        if (
            fa_version == 4
            and device_capability.major >= 10
            and head_size is not None
            and head_size > 128
            and head_size != 192
        ):
            logger.warning_once(
                "FA4 on Blackwell does not support head_size=%d due to TMEM "
                "capacity limits, defaulting to FA version 2.",
                head_size,
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)

            # NOTE(Lucas): FA4 CuteDSL does NOT currently support MLA's non-standard
            # head dimensions (576 for qk, 512 for v) due to TMEM capacity limits.

        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
