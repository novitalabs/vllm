"""Unit tests for MiniMaxM2 novita fused attention integration."""

import types

import pytest
import torch

import vllm.distributed as distributed_module
import vllm.model_executor.models.minimax_m2 as minimax_m2
import vllm.novita_ops as novita_ops
from vllm.model_executor.layers.attention import attention as attention_module

pytestmark = pytest.mark.cpu_test


class _FakeQKVParallelLinear:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        qkv = torch.arange(
            num_tokens * 32, dtype=hidden_states.dtype, device=hidden_states.device
        ).view(num_tokens, 32)
        return qkv, None


class _FakeRowParallelLinear:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, hidden_states):
        return hidden_states, None


class _FakeAttentionLayer:
    def __init__(self, *args, prefix: str = "", **kwargs):
        self.kv_cache_dtype = "fp8"
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        self.layer_name = prefix

    def __call__(self, q, k, v):
        return q


class _FakeRMSNormTP:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.variance_epsilon = eps

    @staticmethod
    def forward_qk(q_norm, k_norm, q, k):
        return q, k


class _FakeRotaryEmbedding:
    def __init__(self):
        self.cos_sin_cache = torch.ones((32, 4), dtype=torch.bfloat16)
        self.is_neox_style = True

    def __call__(self, positions, q, k):
        return q, k


def _make_fake_vllm_config(enable_fused: bool):
    return types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            pass_config=types.SimpleNamespace(
                enable_qk_norm_rope_fp8_kvstore_fusion=enable_fused
            )
        ),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=64),
    )


def test_minimax_m2_attention_prefetches_novita_workspace(monkeypatch):
    monkeypatch.setattr(
        minimax_m2, "get_current_vllm_config", lambda: _make_fake_vllm_config(True)
    )
    monkeypatch.setattr(
        minimax_m2, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(minimax_m2, "QKVParallelLinear", _FakeQKVParallelLinear)
    monkeypatch.setattr(minimax_m2, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(minimax_m2, "Attention", _FakeAttentionLayer)
    monkeypatch.setattr(minimax_m2, "MiniMaxText01RMSNormTP", _FakeRMSNormTP)
    monkeypatch.setattr(minimax_m2, "get_rope", lambda *args, **kwargs: _FakeRotaryEmbedding())
    monkeypatch.setattr(novita_ops, "is_novita_available", lambda: True)

    init_calls = []

    def _fake_init_workspace(*, device, max_tokens):
        init_calls.append((device, max_tokens))

    monkeypatch.setattr(
        minimax_m2,
        "ensure_novita_qk_workspace_initialized",
        _fake_init_workspace,
    )

    minimax_m2.MiniMaxM2Attention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        rotary_dim=4,
        head_dim=4,
        prefix="model.layers.0.self_attn",
    )

    assert init_calls == [(None, 64)]


def test_qk_workspace_cache_reuses_shared_state(monkeypatch):
    workspace = novita_ops._QKNormTPWorkspace(
        workspace_ptrs=torch.tensor([123], dtype=torch.int64),
        local_buf=torch.zeros(3, dtype=torch.float32),
        epoch_state=torch.zeros(1, dtype=torch.int32),
        max_tokens=64,
        world_size=1,
        rank=0,
    )
    create_calls = []
    clear_calls = []

    monkeypatch.setattr(novita_ops, "_qk_norm_tp_workspaces", {})

    def _fake_create(**kwargs):
        create_calls.append(kwargs)
        return workspace

    monkeypatch.setattr(novita_ops, "_create_qk_norm_tp_workspace", _fake_create)
    monkeypatch.setattr(
        novita_ops,
        "novita_clear_qk_norm_workspace",
        lambda *args: clear_calls.append(args),
    )

    args = dict(
        device=torch.device("cuda:0"),
        group=object(),
        world_size=1,
        rank=0,
        max_tokens=64,
    )
    first = novita_ops._get_qk_norm_tp_workspace(**args)
    second = novita_ops._get_qk_norm_tp_workspace(**args)

    assert first is workspace
    assert second is workspace
    assert len(create_calls) == 1
    assert len(clear_calls) == 1


def test_minimax_m2_attention_uses_novita_fused_path(monkeypatch):
    monkeypatch.setattr(
        minimax_m2, "get_current_vllm_config", lambda: _make_fake_vllm_config(True)
    )
    monkeypatch.setattr(
        minimax_m2, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(minimax_m2, "QKVParallelLinear", _FakeQKVParallelLinear)
    monkeypatch.setattr(minimax_m2, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(minimax_m2, "Attention", _FakeAttentionLayer)
    monkeypatch.setattr(minimax_m2, "MiniMaxText01RMSNormTP", _FakeRMSNormTP)
    monkeypatch.setattr(minimax_m2, "get_rope", lambda *args, **kwargs: _FakeRotaryEmbedding())
    monkeypatch.setattr(novita_ops, "is_novita_available", lambda: True)

    fused_calls = []

    def _fake_fused_attn(*args):
        fused_calls.append(args)
        output = args[8]
        output.copy_(torch.full_like(output, 3.0))

    monkeypatch.setattr(
        minimax_m2.torch.ops.vllm,
        "novita_fused_attn",
        _fake_fused_attn,
        raising=False,
    )

    attn = minimax_m2.MiniMaxM2Attention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        rotary_dim=4,
        head_dim=4,
        prefix="model.layers.0.self_attn",
    )

    hidden_states = torch.randn(2, 16)
    positions = torch.arange(2, dtype=torch.long)
    output = attn(positions, hidden_states)

    assert fused_calls, "expected MiniMaxM2Attention to call novita_fused_attn"
    fused_args = fused_calls[0]
    assert fused_args[9] == "model.layers.0.self_attn.attn"
    assert fused_args[10:15] == (4, 2, 2, 4, 2)
    assert fused_args[20] == 64
    assert fused_args[21] is True
    assert output.shape == (2, 16)


def test_novita_fused_attn_passes_device_epoch_state(monkeypatch):
    epoch_state = torch.zeros(1, dtype=torch.int32)
    workspace = novita_ops._QKNormTPWorkspace(
        workspace_ptrs=torch.tensor([123], dtype=torch.int64),
        local_buf=torch.zeros(3, dtype=torch.float32),
        epoch_state=epoch_state,
        max_tokens=64,
        world_size=1,
        rank=0,
    )
    fused_kernel_args = []

    monkeypatch.setattr(
        distributed_module, "get_tp_group", lambda: types.SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            cpu_group=object(),
            device_group=object(),
        )
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (
            None,
            None,
            torch.ones((2, 4, 8), dtype=torch.float8_e4m3fn),
            torch.arange(2, dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        novita_ops,
        "_get_qk_norm_tp_workspace",
        lambda **kwargs: workspace,
    )

    def _fake_fused_kernel(*args, **kwargs):
        fused_kernel_args.append(args)
        q_output = args[14]
        q_output.copy_(torch.zeros_like(q_output))

    monkeypatch.setattr(
        attention_module,
        "unified_attention_with_output",
        lambda q, k, v, output, layer_name, **kwargs: output.zero_(),
    )
    monkeypatch.setattr(
        novita_ops.torch.ops._novita_C,
        "fused_qk_norm_rope_fp8_kvstore",
        _fake_fused_kernel,
        raising=False,
    )

    novita_ops.novita_fused_attn(
        qkv=torch.randn(2, 32),
        positions=torch.arange(2, dtype=torch.long),
        q_weight=torch.ones(16),
        k_weight=torch.ones(8),
        cos_sin_cache=torch.ones((32, 4), dtype=torch.bfloat16),
        q_scale=torch.tensor(1.0),
        k_scale=torch.tensor(1.0),
        v_scale=torch.tensor(1.0),
        output=torch.empty(2, 16),
        layer_name="model.layers.0.self_attn.attn",
        num_heads_q=4,
        num_heads_k=2,
        num_heads_v=2,
        num_heads_q_total=4,
        num_heads_k_total=2,
        head_dim=4,
        q_size=16,
        kv_size=8,
        eps=1e-6,
        rotary_dim=4,
        max_tokens=64,
        is_neox=True,
    )

    assert fused_kernel_args, "expected fused kernel to be invoked"
    assert fused_kernel_args[0][25] is epoch_state


def test_novita_fused_attn_falls_back_for_empty_kv_cache(monkeypatch):
    monkeypatch.setattr(
        distributed_module, "get_tp_group", lambda: types.SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            cpu_group=object(),
            device_group=object(),
        )
    )

    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (
            None,
            None,
            torch.empty(0),
            None,
        ),
    )

    fallback_calls = {"kv_update": 0, "attn": 0, "rotary": 0, "fused_kernel": 0}

    def _fake_kv_update(k, v, layer_name):
        fallback_calls["kv_update"] += 1
        return torch.empty(0)

    def _fake_unified_attention_with_output(q, k, v, output, layer_name, **kwargs):
        fallback_calls["attn"] += 1
        output.copy_(q)

    def _fake_rotary_embedding(positions, q, k, head_dim, cos_sin_cache, is_neox):
        fallback_calls["rotary"] += 1

    def _fake_fused_kernel(*args, **kwargs):
        fallback_calls["fused_kernel"] += 1

    monkeypatch.setattr(attention_module, "unified_kv_cache_update", _fake_kv_update)
    monkeypatch.setattr(
        attention_module,
        "unified_attention_with_output",
        _fake_unified_attention_with_output,
    )
    monkeypatch.setattr(
        novita_ops.torch.ops._C,
        "rotary_embedding",
        _fake_rotary_embedding,
        raising=False,
    )
    monkeypatch.setattr(
        novita_ops.torch.ops._novita_C,
        "fused_qk_norm_rope_fp8_kvstore",
        _fake_fused_kernel,
        raising=False,
    )

    qkv = torch.randn(2, 32)
    output = torch.empty(2, 16)
    novita_ops.novita_fused_attn(
        qkv=qkv,
        positions=torch.arange(2, dtype=torch.long),
        q_weight=torch.ones(16),
        k_weight=torch.ones(8),
        cos_sin_cache=torch.ones((32, 4), dtype=torch.bfloat16),
        q_scale=torch.tensor(1.0),
        k_scale=torch.tensor(1.0),
        v_scale=torch.tensor(1.0),
        output=output,
        layer_name="model.layers.0.self_attn.attn",
        num_heads_q=4,
        num_heads_k=2,
        num_heads_v=2,
        num_heads_q_total=4,
        num_heads_k_total=2,
        head_dim=4,
        q_size=16,
        kv_size=8,
        eps=1e-6,
        rotary_dim=4,
        max_tokens=64,
        is_neox=True,
    )

    assert fallback_calls["rotary"] == 1
    assert fallback_calls["kv_update"] == 1
    assert fallback_calls["attn"] == 1
    assert fallback_calls["fused_kernel"] == 0
