# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang port of FLA's fused_recurrent_dplr_delta_rule (forward only).

Recurrence per timestep (per sequence, head):
    h_new = exp(gk) * h + outer(b, a^T h)        (DPLR update)
    h_new = h_new + outer(k, v)
    o     = h_new^T @ q

Internal layout is unified to (N_tokens, H, K) / (N_tokens, H, V) — both the
rectangular path (cu_seqlens=None) and the varlen path go through the same
kernel, which loops over each sequence using `cu_seqlens` to find bos/eos.

The kernel keeps the state in registers and reads the per-step vectors
straight from global memory (broadcast within each warp); the only
cross-thread communication is the two K-axis reductions per step. Staging
the vectors through shared memory measured consistently slower.

RWKV7_NEGKK mode takes RWKV7-native (kk, a) inputs and forms a = -kk,
b = kk * a in-register, saving two elementwise passes per call — at decode
(T=1) latencies of a few microseconds those launches cost as much as the
kernel itself.

The public entry point bypasses torch.library.custom_op on purpose: decode is
inference-only (the verifier rejects grad-enabled calls), and the custom-op
dispatch overhead (~6us/call) exceeds the kernel time at small batches.
"""

import tilelang
import tilelang.language as T
import torch

from fla.utils import get_device_capability


def _recurrent_config(T_: int, V: int, n_bh: int = 0) -> dict[str, int]:
    # Measured on sm_90/sm_120; small tiles win the serial T>1 chain (more blocks,
    # lower latency per step). At T=1 decode the workload is pure state IO:
    # batch-1 grids stay latency-bound and want tiny v-tiles, larger batches
    # prefer 32-wide v-tiles with few threads (64) to keep the block count per
    # wave down; the linear grid order (see kernel) does the rest.
    if T_ == 1:
        if V <= 64 and n_bh <= 64:
            return {"BV": 16, "threads": 64}
        if V <= 64:
            return {"BV": 32, "threads": 64}
        return {"BV": 32, "threads": 128}
    if V <= 64:
        # sm_90 favors 64 threads per block on the serial chain, sm_120 favors 32
        threads = 64 if get_device_capability()[0] == 9 else 32
    else:
        threads = 64
    return {"BV": 16, "threads": threads}


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _fused_recurrent_dplr_fwd_kernel(
    H, K, V,
    in_dtype, state_dtype,
    scale_value: float,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
    REVERSE: bool,
    RWKV7_NEGKK: bool = False,
    BV: int = 32,
    threads: int = 128,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_ht, n_h0 = T.dynamic("n_tokens, n_seq_plus_one, n_ht, n_h0")
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def fused_recurrent_dplr_fwd_tl(
        q: T.Tensor((n_tokens, H, K), in_dtype),
        k: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        a: T.Tensor((n_tokens, H, K), in_dtype),
        b: T.Tensor((n_tokens, H, K), in_dtype),
        gk: T.Tensor((n_tokens, H, K), in_dtype),
        o: T.Tensor((n_tokens, H, V), in_dtype),
        ht: T.Tensor((n_ht, H, K, V), state_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        h0: T.Tensor((n_h0, H, K, V), state_dtype),
    ):
        # Grid order: v-tile fastest, then head, then seq — consecutive blocks
        # walk the state tensor linearly (matches the Triton kernel's pid
        # decomposition). Striding by seq first measured ~10% off copy-peak on
        # state-IO-bound T=1 decode (profile/rwkv7-tl-opt/OPT_LOG.md).
        # Keep head on grid.y (65535 limit) rather than seq: seq counts above
        # 1023 exceed the y limit at H=64. n_seqs on grid.z is capped at 65535,
        # which serving batches never approach.
        with T.Kernel(T.ceildiv(V, BV), H, n_seqs, threads=threads) as (i_v, i_h, i_n):
            h_frag = T.alloc_fragment((K, BV), acc_dtype)
            ah_prod = T.alloc_fragment((K, BV), acc_dtype)
            ah_sum = T.alloc_fragment((BV,), acc_dtype)
            hq_prod = T.alloc_fragment((K, BV), acc_dtype)
            hq_sum = T.alloc_fragment((BV,), acc_dtype)
            scale_v = T.Cast(acc_dtype, scale_value)

            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            t_len = eos - bos

            # Init state
            if USE_INITIAL_STATE:
                for kk_, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        h_frag[kk_, vv] = T.Cast(acc_dtype, h0[i_n, i_h, kk_, g_v])
                    else:
                        h_frag[kk_, vv] = 0.0
            else:
                for kk_, vv in T.Parallel(K, BV):
                    h_frag[kk_, vv] = 0.0

            # Time loop
            for t_step in T.serial(t_len):
                if REVERSE:
                    t_local = t_len - 1 - t_step
                else:
                    t_local = t_step
                g_t = bos + t_local

                # ah_sum[vv] = sum_k(a[k] * h[k, vv]); per-step vectors are read
                # directly from global memory (broadcast across the warp).
                for kk_, vv in T.Parallel(K, BV):
                    if RWKV7_NEGKK:
                        ah_prod[kk_, vv] = (-T.Cast(acc_dtype, a[g_t, i_h, kk_])) * h_frag[kk_, vv]
                    else:
                        ah_prod[kk_, vv] = T.Cast(acc_dtype, a[g_t, i_h, kk_]) * h_frag[kk_, vv]
                T.reduce_sum(ah_prod, ah_sum, dim=0, clear=True)

                # h = exp(gk) * h + outer(b, ah_sum) + outer(k, v)
                for kk_, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if RWKV7_NEGKK:
                        # a holds kk, b holds a; a_dplr = -kk, b_dplr = kk * a
                        kk_val = T.Cast(acc_dtype, a[g_t, i_h, kk_])
                        b_val = kk_val * T.Cast(acc_dtype, b[g_t, i_h, kk_])
                    else:
                        b_val = T.Cast(acc_dtype, b[g_t, i_h, kk_])
                    h_frag[kk_, vv] = (
                        T.exp(T.Cast(acc_dtype, gk[g_t, i_h, kk_])) * h_frag[kk_, vv]
                        + b_val * ah_sum[vv]
                        + T.Cast(acc_dtype, k[g_t, i_h, kk_]) * T.Cast(acc_dtype, v[g_t, i_h, g_v])
                    )

                    # o[vv] = sum_k(h_new[k, vv] * q[k]).  Form the product
                    # while the updated state element is already hot in the
                    # state-update loop instead of launching a second KxV pass.
                    hq_prod[kk_, vv] = h_frag[kk_, vv] * (T.Cast(acc_dtype, q[g_t, i_h, kk_]) * scale_v)
                T.reduce_sum(hq_prod, hq_sum, dim=0, clear=True)

                for vv in T.Parallel(BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        o[g_t, i_h, g_v] = T.Cast(in_dtype, hq_sum[vv])

            # Store final state
            if STORE_FINAL_STATE:
                for kk_, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        ht[i_n, i_h, kk_, g_v] = T.Cast(state_dtype, h_frag[kk_, vv])

    return fused_recurrent_dplr_fwd_tl


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


def _rect_cu_seqlens(B: int, T_: int, device) -> torch.Tensor:
    # deliberately not cached at module level: a persistent tensor
    # first-allocated during a cudagraph-trees warmup run lands in the
    # cudagraph-private pool and trips "not tracked as outputs"; a per-call
    # temporary dies inside the op and is capture-safe (and free at replay)
    return torch.arange(0, (B + 1) * T_, T_, device=device, dtype=torch.int32)


def fused_recurrent_dplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    rwkv7_negkk: bool = False,
):
    B, T_, H, K = k.shape
    V = v.shape[-1]
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1
        active_nseq = cu_seqlens.shape[0] - 1
        cu_int32 = cu_seqlens.to(dtype=torch.int32)
        if not cu_int32.is_contiguous():
            cu_int32 = cu_int32.contiguous()
    else:
        active_nseq = B
        cu_int32 = _rect_cu_seqlens(B, T_, k.device)

    token_rows = B * T_

    use_h0 = initial_state is not None
    store_ht = output_final_state
    n_ht = active_nseq if store_ht else 1
    n_h0 = active_nseq if use_h0 else 1
    if scale is None:
        scale = K ** -0.5

    # Flatten to (N_tokens, H, K/V); contiguous inputs reshape to views.
    q_f = q.reshape(token_rows, H, K)
    k_f = k.reshape(token_rows, H, K)
    v_f = v.reshape(token_rows, H, V)
    a_f = a.reshape(token_rows, H, K)
    b_f = b.reshape(token_rows, H, K)
    gk_f = gk.reshape(token_rows, H, K)
    if not (q_f.is_contiguous() and k_f.is_contiguous() and v_f.is_contiguous()
            and a_f.is_contiguous() and b_f.is_contiguous() and gk_f.is_contiguous()):
        q_f, k_f, v_f = q_f.contiguous(), k_f.contiguous(), v_f.contiguous()
        a_f, b_f, gk_f = a_f.contiguous(), b_f.contiguous(), gk_f.contiguous()

    if use_h0:
        h0 = initial_state
        if h0.dtype != torch.float32:
            h0 = h0.to(torch.float32)
        if not h0.is_contiguous():
            h0 = h0.contiguous()
    else:
        h0 = torch.empty((n_h0, H, K, V), dtype=torch.float32, device=k.device)

    in_dtype = _dtype_str(k)
    state_dtype = _dtype_str(h0)

    kernel = _fused_recurrent_dplr_fwd_kernel(
        H, K, V,
        in_dtype, state_dtype, float(scale),
        use_h0, store_ht, reverse, rwkv7_negkk,
        **_recurrent_config(T_ if not is_varlen else 0, V, 0 if is_varlen else B * H),
    )
    ht_out = torch.empty((n_ht, H, K, V), dtype=torch.float32, device=k.device)
    o_f = kernel(q_f, k_f, v_f, a_f, b_f, gk_f, ht_out, cu_int32, h0)

    o = o_f.view(B, T_, H, V)
    if store_ht:
        return o, ht_out
    return o, None


def fused_recurrent_dplr_delta_rule_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return fused_recurrent_dplr_delta_rule_fwd(
        q, k, v, a, b, gk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


def fused_mul_recurrent_rwkv7_tilelang(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RWKV7 fused recurrent forward.

    Args:
        r, w, k:    (B, T, H, K)   — receptance, log-decay (= gk), key
        v:          (B, T, H, V)
        kk:         (B, T, H, K)   — normalized k_k * k
        a:          (B, T, H, K)   — sigmoid output of a_lora
        Other args mirror FLA's API.

    The kernel consumes (kk, a) directly (RWKV7_NEGKK mode) instead of
    materializing a_dplr = -kk and b_dplr = kk * a.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if (
        r.shape[1] == 1
        and not reverse
        and cu_seqlens is None
        and r.is_contiguous() and w.is_contiguous() and k.is_contiguous()
        and v.is_contiguous() and kk.is_contiguous() and a.is_contiguous()
        and (initial_state is None
             or (initial_state.dtype == torch.float32 and initial_state.is_contiguous()))
    ):
        return _decode_fast_call(r, w, k, v, kk, a, float(scale), initial_state, output_final_state)
    return fused_recurrent_dplr_delta_rule_fwd(
        q=r,
        k=k,
        v=v,
        a=kk,
        b=a,
        gk=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        rwkv7_negkk=True,
    )


# (kernel, cu_seqlens) per decode key; the kernel handle is already cached by
# tilelang.jit, this cache skips the remaining per-call Python (config
# derivation, JIT cache lookup, reshapes) that dominates T == 1 decode on slow
# host CPUs. Measured: -0.6 ms/token e2e vs the generic wrapper on H800.
_decode_fast_cache: dict[tuple, tuple] = {}


def _decode_fast_call(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, _, H, K = k.shape
    V = v.shape[-1]
    key = (B, H, K, V, k.dtype, scale, initial_state is not None, output_final_state, k.device.index)
    ent = _decode_fast_cache.get(key)
    if ent is None:
        kernel = _fused_recurrent_dplr_fwd_kernel(
            H, K, V,
            _dtype_str(k), 'float32', scale,
            initial_state is not None, output_final_state, False, True,
            **_recurrent_config(1, V, B * H),
        )
        ent = (kernel, _rect_cu_seqlens(B, 1, k.device))
        _decode_fast_cache[key] = ent
    kernel, cu = ent
    if initial_state is not None:
        h0 = initial_state
    else:
        h0 = torch.empty((1, H, K, V), dtype=torch.float32, device=k.device)
    ht = torch.empty((B if output_final_state else 1, H, K, V), dtype=torch.float32, device=k.device)
    # NEGKK slot mapping: a <- kk, b <- a, gk <- w (mirrors the generic wrapper).
    o = kernel(
        r.view(B, H, K), k.view(B, H, K), v.view(B, H, V),
        kk.view(B, H, K), a.view(B, H, K), w.view(B, H, K),
        ht, cu, h0,
    )
    o = o.view(B, 1, H, V)
    if output_final_state:
        return o, ht
    return o, None
