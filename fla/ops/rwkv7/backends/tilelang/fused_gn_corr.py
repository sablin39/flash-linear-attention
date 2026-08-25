# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused GroupNorm + gate_output_correction for RWKV7 in TileLang.

One kernel replaces the unfused sequence
    o = GroupNorm(o, num_groups=H)(...)       # per-(token, head), fp32 stats
    o = gate_output_correction(o, r, k, r_k, v, g)
computing per (token, head):
    mean  = sum(o) / D
    rstd  = rsqrt(sum((o - mean)^2) / D + eps)
    corr  = sum(r * k * r_k)
    out   = ((o - mean) * rstd * weight + bias + corr * v) * g
which matches nn.GroupNorm (additive eps, fp32 accumulation) followed by the
correction. This saves a full read+write of o and one kernel launch, and —
being opaque custom ops in both directions — keeps the norm inside a
torch.compile fullgraph region, which the Triton fla GroupNorm cannot do.

The backward mirrors the forward structure: a main kernel computes all
per-position grads plus (NB, H, D) fp32 segment partials for the affine
weights and r_k, and a small reduce kernel sums the segments.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
from torch import Tensor


def _fwd_cfg(D: int) -> dict[str, int]:
    # vec2-friendly: each thread owns 2 contiguous d elements
    return {"BT": 8}


@tilelang.jit(
    out_idx=[-1],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _gn_corr_fwd_kernel(B, T_, H, D, in_dtype, eps: float, BT: int = 8):
    acc_dtype = "float32"
    threads = BT * (D // 2)

    @T.prim_func
    def gn_corr_fwd_tl(
        o: T.Tensor((B, T_, H, D), in_dtype),
        r: T.Tensor((B, T_, H, D), in_dtype),
        k: T.Tensor((B, T_, H, D), in_dtype),
        r_k: T.Tensor((H, D), in_dtype),
        v: T.Tensor((B, T_, H, D), in_dtype),
        g: T.Tensor((B, T_, H, D), in_dtype),
        weight: T.Tensor((H, D), in_dtype),
        bias: T.Tensor((H, D), in_dtype),
        out: T.Tensor((B, T_, H, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(T_, BT), B, H, threads=threads) as (i_tb, i_b, i_h):
            o_shared = T.alloc_shared((BT, D), in_dtype)
            r_shared = T.alloc_shared((BT, D), in_dtype)
            k_shared = T.alloc_shared((BT, D), in_dtype)
            v_shared = T.alloc_shared((BT, D), in_dtype)
            g_shared = T.alloc_shared((BT, D), in_dtype)
            w_shared = T.alloc_shared((D,), in_dtype)
            b_shared = T.alloc_shared((D,), in_dtype)
            rk_shared = T.alloc_shared((D,), in_dtype)

            mean_frag = T.alloc_fragment((BT,), acc_dtype)
            rstd_frag = T.alloc_fragment((BT,), acc_dtype)
            corr_frag = T.alloc_fragment((BT,), acc_dtype)

            for d in T.Parallel(D):
                w_shared[d] = weight[i_h, d]
                b_shared[d] = bias[i_h, d]
                rk_shared[d] = r_k[i_h, d]

            for kt, d in T.Parallel(BT, D):
                t = i_tb * BT + kt
                if t < T_:
                    o_shared[kt, d] = o[i_b, t, i_h, d]
                    r_shared[kt, d] = r[i_b, t, i_h, d]
                    k_shared[kt, d] = k[i_b, t, i_h, d]
                    v_shared[kt, d] = v[i_b, t, i_h, d]
                    g_shared[kt, d] = g[i_b, t, i_h, d]
                else:
                    o_shared[kt, d] = T.Cast(in_dtype, 0.0)
                    r_shared[kt, d] = T.Cast(in_dtype, 0.0)
                    k_shared[kt, d] = T.Cast(in_dtype, 0.0)
                    v_shared[kt, d] = T.Cast(in_dtype, 0.0)
                    g_shared[kt, d] = T.Cast(in_dtype, 0.0)

            sum_o = T.alloc_fragment((BT, D), acc_dtype)
            for kt, d in T.Parallel(BT, D):
                sum_o[kt, d] = T.Cast(acc_dtype, o_shared[kt, d])
            T.reduce_sum(sum_o, mean_frag, dim=1, clear=True)
            inv_D = T.Cast(acc_dtype, 1.0) / T.Cast(acc_dtype, D)
            for kt in T.Parallel(BT):
                mean_frag[kt] = mean_frag[kt] * inv_D

            var_acc = T.alloc_fragment((BT, D), acc_dtype)
            for kt, d in T.Parallel(BT, D):
                diff = T.Cast(acc_dtype, o_shared[kt, d]) - mean_frag[kt]
                var_acc[kt, d] = diff * diff
            T.reduce_sum(var_acc, rstd_frag, dim=1, clear=True)
            for kt in T.Parallel(BT):
                rstd_frag[kt] = T.rsqrt(rstd_frag[kt] * inv_D + T.Cast(acc_dtype, eps))

            corr_acc = T.alloc_fragment((BT, D), acc_dtype)
            for kt, d in T.Parallel(BT, D):
                corr_acc[kt, d] = (
                    T.Cast(acc_dtype, r_shared[kt, d])
                    * T.Cast(acc_dtype, k_shared[kt, d])
                    * T.Cast(acc_dtype, rk_shared[d])
                )
            T.reduce_sum(corr_acc, corr_frag, dim=1, clear=True)

            for kt, d in T.Parallel(BT, D):
                t = i_tb * BT + kt
                if t < T_:
                    xhat = (T.Cast(acc_dtype, o_shared[kt, d]) - mean_frag[kt]) * rstd_frag[kt]
                    normed = xhat * T.Cast(acc_dtype, w_shared[d]) + T.Cast(acc_dtype, b_shared[d])
                    out_val = (normed + corr_frag[kt] * T.Cast(acc_dtype, v_shared[kt, d])) * \
                        T.Cast(acc_dtype, g_shared[kt, d])
                    out[i_b, t, i_h, d] = T.Cast(in_dtype, out_val)

    return gn_corr_fwd_tl


@tilelang.jit(
    out_idx=[9, 10, 11, 12, 13, 14, 15, 16],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _gn_corr_bwd_main_kernel(M, H, D, NB, in_dtype, eps: float, BT: int = 8, threads: int = 256):
    acc_dtype = "float32"

    @T.prim_func
    def gn_corr_bwd_main_tl(
        dy: T.Tensor((M, H, D), in_dtype),
        o: T.Tensor((M, H, D), in_dtype),
        r: T.Tensor((M, H, D), in_dtype),
        k: T.Tensor((M, H, D), in_dtype),
        r_k: T.Tensor((H, D), in_dtype),
        v: T.Tensor((M, H, D), in_dtype),
        g: T.Tensor((M, H, D), in_dtype),
        weight: T.Tensor((H, D), in_dtype),
        bias: T.Tensor((H, D), in_dtype),
        d_o: T.Tensor((M, H, D), in_dtype),
        d_r: T.Tensor((M, H, D), in_dtype),
        d_k: T.Tensor((M, H, D), in_dtype),
        d_v: T.Tensor((M, H, D), in_dtype),
        d_g: T.Tensor((M, H, D), in_dtype),
        d_weight_partial: T.Tensor((NB, H, D), acc_dtype),
        d_bias_partial: T.Tensor((NB, H, D), acc_dtype),
        d_rk_partial: T.Tensor((NB, H, D), acc_dtype),
    ):
        with T.Kernel(NB, H, threads=threads) as (i_b, i_h):
            o_frag = T.alloc_fragment((BT, D), acc_dtype)
            r_frag = T.alloc_fragment((BT, D), acc_dtype)
            k_frag = T.alloc_fragment((BT, D), acc_dtype)
            v_frag = T.alloc_fragment((BT, D), acc_dtype)
            dy_frag = T.alloc_fragment((BT, D), acc_dtype)
            g_frag = T.alloc_fragment((BT, D), acc_dtype)
            xhat_frag = T.alloc_fragment((BT, D), acc_dtype)
            corr_terms = T.alloc_fragment((BT, D), acc_dtype)
            corr = T.alloc_fragment((BT,), acc_dtype)
            mean_terms = T.alloc_fragment((BT, D), acc_dtype)
            mean = T.alloc_fragment((BT,), acc_dtype)
            var_terms = T.alloc_fragment((BT, D), acc_dtype)
            rstd = T.alloc_fragment((BT,), acc_dtype)
            d_gated = T.alloc_fragment((BT, D), acc_dtype)
            d_corr_terms = T.alloc_fragment((BT, D), acc_dtype)
            d_corr = T.alloc_fragment((BT,), acc_dtype)
            dxhat = T.alloc_fragment((BT, D), acc_dtype)
            sum_dxhat_terms = T.alloc_fragment((BT, D), acc_dtype)
            sum_dxhat = T.alloc_fragment((BT,), acc_dtype)
            sum_dxhat_xhat_terms = T.alloc_fragment((BT, D), acc_dtype)
            sum_dxhat_xhat = T.alloc_fragment((BT,), acc_dtype)
            dw_terms = T.alloc_fragment((BT, D), acc_dtype)
            db_terms = T.alloc_fragment((BT, D), acc_dtype)
            drk_terms = T.alloc_fragment((BT, D), acc_dtype)
            dw_sum = T.alloc_fragment((D,), acc_dtype)
            db_sum = T.alloc_fragment((D,), acc_dtype)
            drk_sum = T.alloc_fragment((D,), acc_dtype)

            inv_D = T.Cast(acc_dtype, 1.0) / T.Cast(acc_dtype, D)

            for r_i, d in T.Parallel(BT, D):
                i_m = i_b * BT + r_i
                if i_m < M:
                    ov = T.Cast(acc_dtype, o[i_m, i_h, d])
                    rv = T.Cast(acc_dtype, r[i_m, i_h, d])
                    kv = T.Cast(acc_dtype, k[i_m, i_h, d])
                    vv = T.Cast(acc_dtype, v[i_m, i_h, d])
                    dyv = T.Cast(acc_dtype, dy[i_m, i_h, d])
                    gv = T.Cast(acc_dtype, g[i_m, i_h, d])
                    rkv = T.Cast(acc_dtype, r_k[i_h, d])
                    o_frag[r_i, d] = ov
                    r_frag[r_i, d] = rv
                    k_frag[r_i, d] = kv
                    v_frag[r_i, d] = vv
                    dy_frag[r_i, d] = dyv
                    g_frag[r_i, d] = gv
                    mean_terms[r_i, d] = ov
                    corr_terms[r_i, d] = rv * kv * rkv
                else:
                    o_frag[r_i, d] = 0.0
                    r_frag[r_i, d] = 0.0
                    k_frag[r_i, d] = 0.0
                    v_frag[r_i, d] = 0.0
                    dy_frag[r_i, d] = 0.0
                    g_frag[r_i, d] = 0.0
                    mean_terms[r_i, d] = 0.0
                    corr_terms[r_i, d] = 0.0

            T.reduce_sum(mean_terms, mean, dim=1, clear=True)
            T.reduce_sum(corr_terms, corr, dim=1, clear=True)
            for r_i in T.Parallel(BT):
                mean[r_i] = mean[r_i] * inv_D

            for r_i, d in T.Parallel(BT, D):
                i_m = i_b * BT + r_i
                if i_m < M:
                    centered = o_frag[r_i, d] - mean[r_i]
                    var_terms[r_i, d] = centered * centered
                else:
                    var_terms[r_i, d] = 0.0

            T.reduce_sum(var_terms, rstd, dim=1, clear=True)
            for r_i in T.Parallel(BT):
                rstd[r_i] = T.rsqrt(rstd[r_i] * inv_D + T.Cast(acc_dtype, eps))

            for r_i, d in T.Parallel(BT, D):
                i_m = i_b * BT + r_i
                if i_m < M:
                    wv = T.Cast(acc_dtype, weight[i_h, d])
                    bv = T.Cast(acc_dtype, bias[i_h, d])
                    xhat = (o_frag[r_i, d] - mean[r_i]) * rstd[r_i]
                    xhat_frag[r_i, d] = xhat
                    gated = xhat * wv + bv + corr[r_i] * v_frag[r_i, d]
                    d_g[i_m, i_h, d] = T.Cast(in_dtype, dy_frag[r_i, d] * gated)
                    dgated = dy_frag[r_i, d] * g_frag[r_i, d]
                    d_gated[r_i, d] = dgated
                    d_v[i_m, i_h, d] = T.Cast(in_dtype, dgated * corr[r_i])
                    d_corr_terms[r_i, d] = dgated * v_frag[r_i, d]
                    dxhat[r_i, d] = dgated * wv
                    sum_dxhat_terms[r_i, d] = dxhat[r_i, d]
                    sum_dxhat_xhat_terms[r_i, d] = dxhat[r_i, d] * xhat
                    dw_terms[r_i, d] = dgated * xhat
                    db_terms[r_i, d] = dgated
                    # Placeholder until d_corr is reduced below.
                    drk_terms[r_i, d] = r_frag[r_i, d] * k_frag[r_i, d]
                else:
                    xhat_frag[r_i, d] = 0.0
                    d_gated[r_i, d] = 0.0
                    d_corr_terms[r_i, d] = 0.0
                    dxhat[r_i, d] = 0.0
                    sum_dxhat_terms[r_i, d] = 0.0
                    sum_dxhat_xhat_terms[r_i, d] = 0.0
                    dw_terms[r_i, d] = 0.0
                    db_terms[r_i, d] = 0.0
                    drk_terms[r_i, d] = 0.0

            T.reduce_sum(d_corr_terms, d_corr, dim=1, clear=True)
            T.reduce_sum(sum_dxhat_terms, sum_dxhat, dim=1, clear=True)
            T.reduce_sum(sum_dxhat_xhat_terms, sum_dxhat_xhat, dim=1, clear=True)

            for r_i, d in T.Parallel(BT, D):
                i_m = i_b * BT + r_i
                if i_m < M:
                    rkv = T.Cast(acc_dtype, r_k[i_h, d])
                    dc = d_corr[r_i]
                    d_r[i_m, i_h, d] = T.Cast(in_dtype, dc * k_frag[r_i, d] * rkv)
                    d_k[i_m, i_h, d] = T.Cast(in_dtype, dc * r_frag[r_i, d] * rkv)
                    drk_terms[r_i, d] = dc * drk_terms[r_i, d]
                    d_o_val = rstd[r_i] * (
                        dxhat[r_i, d]
                        - sum_dxhat[r_i] * inv_D
                        - xhat_frag[r_i, d] * sum_dxhat_xhat[r_i] * inv_D
                    )
                    d_o[i_m, i_h, d] = T.Cast(in_dtype, d_o_val)
                else:
                    drk_terms[r_i, d] = 0.0

            T.reduce_sum(dw_terms, dw_sum, dim=0, clear=True)
            T.reduce_sum(db_terms, db_sum, dim=0, clear=True)
            T.reduce_sum(drk_terms, drk_sum, dim=0, clear=True)
            for d in T.Parallel(D):
                d_weight_partial[i_b, i_h, d] = dw_sum[d]
                d_bias_partial[i_b, i_h, d] = db_sum[d]
                d_rk_partial[i_b, i_h, d] = drk_sum[d]

    return gn_corr_bwd_main_tl


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _gn_corr_bwd_reduce_kernel(NB, H, D, out_dtype, threads: int = 128):
    acc_dtype = "float32"

    @T.prim_func
    def gn_corr_bwd_reduce_tl(
        d_weight_partial: T.Tensor((NB, H, D), acc_dtype),
        d_bias_partial: T.Tensor((NB, H, D), acc_dtype),
        d_rk_partial: T.Tensor((NB, H, D), acc_dtype),
        d_weight: T.Tensor((H, D), out_dtype),
        d_bias: T.Tensor((H, D), out_dtype),
        d_rk: T.Tensor((H, D), out_dtype),
    ):
        with T.Kernel(H, threads=threads) as i_h:
            acc_w = T.alloc_fragment((D,), acc_dtype)
            acc_b = T.alloc_fragment((D,), acc_dtype)
            acc_rk = T.alloc_fragment((D,), acc_dtype)
            for d in T.Parallel(D):
                acc_w[d] = 0.0
                acc_b[d] = 0.0
                acc_rk[d] = 0.0
                for i_b in T.serial(NB):
                    acc_w[d] = acc_w[d] + d_weight_partial[i_b, i_h, d]
                    acc_b[d] = acc_b[d] + d_bias_partial[i_b, i_h, d]
                    acc_rk[d] = acc_rk[d] + d_rk_partial[i_b, i_h, d]
                d_weight[i_h, d] = T.Cast(out_dtype, acc_w[d])
                d_bias[i_h, d] = T.Cast(out_dtype, acc_b[d])
                d_rk[i_h, d] = T.Cast(out_dtype, acc_rk[d])

    return gn_corr_bwd_reduce_tl


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


# The @tilelang.jit wrapper re-runs its cache-key hashing on every call
# (~0.1-0.25 ms), which dominates kernels this small; the resolved JITKernel
# is shape/dtype-specialized and safe to reuse, so memoize it per key.
_KERNEL_CACHE: dict = {}


def _cached_kernel(key: tuple, builder):
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = builder()
        _KERNEL_CACHE[key] = k
    return k


@torch.library.custom_op(
    "fla::gn_corr_rwkv7",
    mutates_args=(),
    schema=(
        "(Tensor o, Tensor r, Tensor k, Tensor r_k, Tensor v, Tensor g, "
        "Tensor weight, Tensor bias, float eps) -> Tensor"
    ),
)
def _gn_corr_op(
    o: Tensor,
    r: Tensor,
    k: Tensor,
    r_k: Tensor,
    v: Tensor,
    g: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor:
    B, T_, H, D = o.shape
    assert D % 2 == 0
    in_dtype = _dtype_str(o)

    if g.dim() == 3:
        g = g.view(B, T_, H, D)

    o_c = o.contiguous()
    r_c = r.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    w_c = weight.contiguous().view(H, D)
    b_c = bias.contiguous().view(H, D)

    key = ("fwd", B, T_, H, D, in_dtype, float(eps))
    kernel = _cached_kernel(
        key,
        lambda: _gn_corr_fwd_kernel(B, T_, H, D, in_dtype, float(eps), **_fwd_cfg(D)),
    )
    return kernel(o_c, r_c, k_c, r_k.contiguous(), v_c, g_c, w_c, b_c)


@_gn_corr_op.register_fake
def _gn_corr_fake(o, r, k, r_k, v, g, weight, bias, eps):
    return o.new_empty(o.shape)


def _gn_corr_setup_context(ctx, inputs, output):
    o, r, k, r_k, v, g, weight, bias, eps = inputs
    ctx.save_for_backward(o, r, k, r_k, v, g, weight, bias)
    ctx.eps = float(eps)


@torch.library.custom_op(
    "fla::gn_corr_rwkv7_bwd",
    mutates_args=(),
    schema=(
        "(Tensor dy, Tensor o, Tensor r, Tensor k, Tensor r_k, Tensor v, Tensor g, "
        "Tensor weight, Tensor bias, float eps) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    ),
)
def _gn_corr_bwd_op(
    dy: Tensor,
    o: Tensor,
    r: Tensor,
    k: Tensor,
    r_k: Tensor,
    v: Tensor,
    g: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
):
    B, T_, H, D = o.shape
    M = B * T_
    BT, threads = 8, 256
    NB = (M + BT - 1) // BT

    dy_f = dy.reshape(M, H, D).contiguous()
    o_f = o.reshape(M, H, D).contiguous()
    r_f = r.reshape(M, H, D).contiguous()
    k_f = k.reshape(M, H, D).contiguous()
    v_f = v.reshape(M, H, D).contiguous()
    g_f = g.reshape(M, H, D).contiguous()
    w_f = weight.reshape(H, D).contiguous()
    b_f = bias.reshape(H, D).contiguous()
    rk_f = r_k.reshape(H, D).contiguous()

    in_dtype = _dtype_str(o)
    main_key = ("bwd_main", M, H, D, in_dtype, float(eps))
    main_kernel = _cached_kernel(
        main_key,
        lambda: _gn_corr_bwd_main_kernel(M, H, D, NB, in_dtype, float(eps), BT=BT, threads=threads),
    )
    d_o_f, d_r_f, d_k_f, d_v_f, d_g_f, d_weight_partial, d_bias_partial, d_rk_partial = main_kernel(
        dy_f, o_f, r_f, k_f, rk_f, v_f, g_f, w_f, b_f,
    )
    reduce_key = ("bwd_reduce", NB, H, D, _dtype_str(weight))
    reduce_kernel = _cached_kernel(
        reduce_key,
        lambda: _gn_corr_bwd_reduce_kernel(NB, H, D, _dtype_str(weight)),
    )
    d_weight_f, d_bias_f, d_rk = reduce_kernel(d_weight_partial, d_bias_partial, d_rk_partial)

    return (
        d_o_f.view_as(o),
        d_r_f.view_as(r),
        d_k_f.view_as(k),
        d_rk.view_as(r_k),
        d_v_f.view_as(v),
        d_g_f.view(M, H, D).reshape(g.shape),
        d_weight_f.reshape(weight.shape),
        d_bias_f.reshape(bias.shape),
    )


@_gn_corr_bwd_op.register_fake
def _gn_corr_bwd_fake(dy, o, r, k, r_k, v, g, weight, bias, eps):
    return (
        o.new_empty(o.shape),
        r.new_empty(r.shape),
        k.new_empty(k.shape),
        r_k.new_empty(r_k.shape),
        v.new_empty(v.shape),
        g.new_empty(g.shape),
        weight.new_empty(weight.shape),
        bias.new_empty(bias.shape),
    )


def _gn_corr_backward(ctx, dy):
    o, r, k, r_k, v, g, weight, bias = ctx.saved_tensors
    d_o, d_r, d_k, d_rk, d_v, d_g, d_weight, d_bias = _gn_corr_bwd_op(
        dy, o, r, k, r_k, v, g, weight, bias, ctx.eps,
    )
    return d_o, d_r, d_k, d_rk, d_v, d_g, d_weight, d_bias, None


_gn_corr_op.register_autograd(_gn_corr_backward, setup_context=_gn_corr_setup_context)


def gn_corr_rwkv7(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused GroupNorm + gate_output_correction.

    Args:
        o: attention output, (B, T, H, D).
        r, k, v: (B, T, H, D).
        r_k: (H, D) per-head correction weight.
        g: output gate, (B, T, H*D) or (B, T, H, D).
        weight, bias: GroupNorm affine params, (H*D,) or (H, D).
        eps: GroupNorm eps.

    Returns:
        (B, T, H, D) tensor: ``(GroupNorm(o) + corr * v) * g``.
    """
    return _gn_corr_op(o, r, k, r_k, v, g, weight, bias, float(eps))
