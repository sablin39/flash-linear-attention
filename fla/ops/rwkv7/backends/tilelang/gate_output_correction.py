# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""gate_output_correction in TileLang.

Math:
    correction = (r * k * r_k).sum(-1, keepdim=True) * v
    output     = (o + correction) * g

Shapes:
    o, g       : (B, T, H*D)
    r, k, v    : (B, T, H, D)
    r_k        : (H, D)

All tensors are viewed as (M, H, D) with M = B*T so each block owns a single
head (r_k stays a plain row read) and every global access is a contiguous
D-wide vector. The backward accumulates d_r_k into (S, H, D) fp32 segment
partials in-kernel — no full-size fp32 intermediate — and a host-side sum
over S finishes it.
"""

import tilelang
import tilelang.language as T
import torch
from torch import Tensor


@tilelang.jit(
    out_idx=[6],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _gate_corr_fwd_kernel(M, H, D, in_dtype, BT: int = 8, threads: int = 128):
    acc_dtype = "float32"

    @T.prim_func
    def gate_corr_fwd_tl(
        o: T.Tensor((M, H, D), in_dtype),
        r: T.Tensor((M, H, D), in_dtype),
        k: T.Tensor((M, H, D), in_dtype),
        r_k: T.Tensor((H, D), in_dtype),
        v: T.Tensor((M, H, D), in_dtype),
        g: T.Tensor((M, H, D), in_dtype),
        out: T.Tensor((M, H, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BT), H, threads=threads) as (i_t, i_h):
            prod = T.alloc_fragment((BT, D), acc_dtype)
            corr = T.alloc_fragment((BT,), acc_dtype)
            row0 = i_t * BT

            # A predicated vectorized load with an else fill lowers to
            # tl::pack_float16x4(half_t, ...), which does not compile
            # (cutlass::half_t -> __half is explicit). Full tiles therefore
            # run unpredicated; the tail tile clamps load indices instead and
            # never stores its out-of-range rows.
            if row0 + BT <= M:
                for kt, d in T.Parallel(BT, D):
                    prod[kt, d] = (
                        T.Cast(acc_dtype, r[row0 + kt, i_h, d])
                        * T.Cast(acc_dtype, k[row0 + kt, i_h, d])
                        * T.Cast(acc_dtype, r_k[i_h, d])
                    )

                T.reduce_sum(prod, corr, dim=-1, clear=True)

                for kt, d in T.Parallel(BT, D):
                    out[row0 + kt, i_h, d] = T.Cast(
                        in_dtype,
                        (
                            T.Cast(acc_dtype, o[row0 + kt, i_h, d])
                            + corr[kt] * T.Cast(acc_dtype, v[row0 + kt, i_h, d])
                        ) * T.Cast(acc_dtype, g[row0 + kt, i_h, d]),
                    )
            else:
                for kt, d in T.Parallel(BT, D):
                    tc = T.min(row0 + kt, M - 1)
                    prod[kt, d] = (
                        T.Cast(acc_dtype, r[tc, i_h, d])
                        * T.Cast(acc_dtype, k[tc, i_h, d])
                        * T.Cast(acc_dtype, r_k[i_h, d])
                    )

                T.reduce_sum(prod, corr, dim=-1, clear=True)

                for kt, d in T.Parallel(BT, D):
                    t = row0 + kt
                    if t < M:
                        out[t, i_h, d] = T.Cast(
                            in_dtype,
                            (
                                T.Cast(acc_dtype, o[t, i_h, d])
                                + corr[kt] * T.Cast(acc_dtype, v[t, i_h, d])
                            ) * T.Cast(acc_dtype, g[t, i_h, d]),
                        )

    return gate_corr_fwd_tl


@tilelang.jit(
    out_idx=[7, 8, 9, 10, 11, 12],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _gate_corr_bwd_kernel(M, S, H, D, in_dtype, BT: int = 8, threads: int = 128):
    """Backward — dO, dR, dK, dV, dG plus segmented d_r_k partials.

    d_o = dy * g
    d_g = dy * (o + corr * v)
    d_v = dy * g * corr
    gcs = sum_d(dy * g * v)
    d_r = gcs * k * r_k ; d_k = gcs * r * r_k ; d_rk += gcs * r * k
    """
    acc_dtype = "float32"

    @T.prim_func
    def gate_corr_bwd_tl(
        dy: T.Tensor((M, H, D), in_dtype),
        o: T.Tensor((M, H, D), in_dtype),
        r: T.Tensor((M, H, D), in_dtype),
        k: T.Tensor((M, H, D), in_dtype),
        r_k: T.Tensor((H, D), in_dtype),
        v: T.Tensor((M, H, D), in_dtype),
        g: T.Tensor((M, H, D), in_dtype),
        d_o: T.Tensor((M, H, D), in_dtype),
        d_r: T.Tensor((M, H, D), in_dtype),
        d_k: T.Tensor((M, H, D), in_dtype),
        d_rk_part: T.Tensor((S, H, D), acc_dtype),
        d_v: T.Tensor((M, H, D), in_dtype),
        d_g: T.Tensor((M, H, D), in_dtype),
    ):
        with T.Kernel(S, H, threads=threads) as (i_s, i_h):
            prod = T.alloc_fragment((BT, D), acc_dtype)
            gc_prod = T.alloc_fragment((BT, D), acc_dtype)
            acc_rk = T.alloc_fragment((BT, D), acc_dtype)
            corr = T.alloc_fragment((BT,), acc_dtype)
            gcs = T.alloc_fragment((BT,), acc_dtype)
            red_rk = T.alloc_fragment((D,), acc_dtype)

            rps = T.ceildiv(M, S)
            row0 = i_s * rps
            row1 = T.min(row0 + rps, M)

            T.clear(acc_rk)
            for i0 in T.serial(T.ceildiv(rps, BT)):
                # corr = sum_d r*k*r_k ; gg = dy*g staged in gc_prod for the gcs reduce
                base = row0 + i0 * BT
                # Same predicated-vector-load compile issue as the forward:
                # full tiles run unpredicated, the tail tile clamps load
                # indices and masks only what feeds the cross-row d_r_k accum.
                if base + BT <= row1:
                    for kt, d in T.Parallel(BT, D):
                        rv = T.Cast(acc_dtype, r[base + kt, i_h, d])
                        kv = T.Cast(acc_dtype, k[base + kt, i_h, d])
                        prod[kt, d] = rv * kv * T.Cast(acc_dtype, r_k[i_h, d])
                    T.reduce_sum(prod, corr, dim=-1, clear=True)

                    for kt, d in T.Parallel(BT, D):
                        dy_v = T.Cast(acc_dtype, dy[base + kt, i_h, d])
                        g_v = T.Cast(acc_dtype, g[base + kt, i_h, d])
                        v_v = T.Cast(acc_dtype, v[base + kt, i_h, d])
                        o_v = T.Cast(acc_dtype, o[base + kt, i_h, d])
                        gg = dy_v * g_v
                        d_o[base + kt, i_h, d] = T.Cast(in_dtype, gg)
                        d_g[base + kt, i_h, d] = T.Cast(in_dtype, dy_v * (o_v + corr[kt] * v_v))
                        d_v[base + kt, i_h, d] = T.Cast(in_dtype, gg * corr[kt])
                        gc_prod[kt, d] = gg * v_v
                        # reuse prod as r*k (rk multiply applied after the reduce)
                        prod[kt, d] = T.Cast(acc_dtype, r[base + kt, i_h, d]) * T.Cast(acc_dtype, k[base + kt, i_h, d])
                    T.reduce_sum(gc_prod, gcs, dim=-1, clear=True)

                    for kt, d in T.Parallel(BT, D):
                        rkd = T.Cast(acc_dtype, r_k[i_h, d])
                        d_r[base + kt, i_h, d] = T.Cast(in_dtype, gcs[kt] * T.Cast(acc_dtype, k[base + kt, i_h, d]) * rkd)
                        d_k[base + kt, i_h, d] = T.Cast(in_dtype, gcs[kt] * T.Cast(acc_dtype, r[base + kt, i_h, d]) * rkd)
                        acc_rk[kt, d] += gcs[kt] * prod[kt, d]
                else:
                    for kt, d in T.Parallel(BT, D):
                        tc = T.min(base + kt, row1 - 1)
                        rv = T.Cast(acc_dtype, r[tc, i_h, d])
                        kv = T.Cast(acc_dtype, k[tc, i_h, d])
                        prod[kt, d] = rv * kv * T.Cast(acc_dtype, r_k[i_h, d])
                    T.reduce_sum(prod, corr, dim=-1, clear=True)

                    for kt, d in T.Parallel(BT, D):
                        t = base + kt
                        tc = T.min(t, row1 - 1)
                        dy_v = T.Cast(acc_dtype, dy[tc, i_h, d])
                        g_v = T.Cast(acc_dtype, g[tc, i_h, d])
                        v_v = T.Cast(acc_dtype, v[tc, i_h, d])
                        o_v = T.Cast(acc_dtype, o[tc, i_h, d])
                        gg = dy_v * g_v
                        if t < row1:
                            d_o[t, i_h, d] = T.Cast(in_dtype, gg)
                            d_g[t, i_h, d] = T.Cast(in_dtype, dy_v * (o_v + corr[kt] * v_v))
                            d_v[t, i_h, d] = T.Cast(in_dtype, gg * corr[kt])
                        gc_prod[kt, d] = gg * v_v
                        # reuse prod as r*k (rk multiply applied after the reduce)
                        prod[kt, d] = T.Cast(acc_dtype, r[tc, i_h, d]) * T.Cast(acc_dtype, k[tc, i_h, d])
                    T.reduce_sum(gc_prod, gcs, dim=-1, clear=True)

                    for kt, d in T.Parallel(BT, D):
                        t = base + kt
                        tc = T.min(t, row1 - 1)
                        rkd = T.Cast(acc_dtype, r_k[i_h, d])
                        if t < row1:
                            d_r[t, i_h, d] = T.Cast(in_dtype, gcs[kt] * T.Cast(acc_dtype, k[tc, i_h, d]) * rkd)
                            d_k[t, i_h, d] = T.Cast(in_dtype, gcs[kt] * T.Cast(acc_dtype, r[tc, i_h, d]) * rkd)
                        acc_rk[kt, d] += T.if_then_else(t < row1, gcs[kt] * prod[kt, d], 0.0)

            T.reduce_sum(acc_rk, red_rk, dim=0)
            for d in T.Parallel(D):
                d_rk_part[i_s, i_h, d] = red_rk[d]

    return gate_corr_bwd_tl


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


def _cfg(D: int) -> dict[str, int]:
    # 16 contiguous elements per thread so IO lowers to 256-bit ops
    return {"BT": max(1, 2048 // D), "threads": max(32, min(256, (max(1, 2048 // D) * D) // 16))}


@torch.library.custom_op("fla::gate_output_correction_rwkv7", mutates_args=())
def _gate_output_correction_op(
    o: Tensor,
    r: Tensor,
    k: Tensor,
    r_k: Tensor,
    v: Tensor,
    g: Tensor,
) -> Tensor:
    B, T_, _ = o.shape
    H, D = r.shape[-2], r.shape[-1]
    M = B * T_

    o_c, r_c, k_c, r_k_c, v_c, g_c = (
        o.contiguous(), r.contiguous(), k.contiguous(),
        r_k.contiguous(), v.contiguous(), g.contiguous(),
    )

    kernel = _gate_corr_fwd_kernel(M, H, D, _dtype_str(o), **_cfg(D))
    return kernel(
        o_c.view(M, H, D), r_c.view(M, H, D), k_c.view(M, H, D),
        r_k_c, v_c.view(M, H, D), g_c.view(M, H, D),
    ).view(B, T_, H * D)


@_gate_output_correction_op.register_fake
def _gate_output_correction_fake(o, r, k, r_k, v, g):
    return o.new_empty(o.shape)


def _gate_output_correction_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


@torch.library.custom_op("fla::gate_output_correction_rwkv7_bwd", mutates_args=())
def _gate_output_correction_bwd_op(
    dy: Tensor,
    o: Tensor,
    r: Tensor,
    k: Tensor,
    r_k: Tensor,
    v: Tensor,
    g: Tensor,
) -> Tensor:
    B, T_, H, D = r.shape
    M = B * T_
    S = max(1, min(64, M // 2048))

    dy_c = dy.contiguous().view(M, H, D)
    o_c = o.contiguous().view(M, H, D)
    r_c = r.contiguous().view(M, H, D)
    k_c = k.contiguous().view(M, H, D)
    r_k_c = r_k.contiguous()
    v_c = v.contiguous().view(M, H, D)
    g_c = g.contiguous().view(M, H, D)

    bwd_kernel = _gate_corr_bwd_kernel(M, S, H, D, _dtype_str(o), **_cfg(D))
    d_o, d_r, d_k, d_rk_part, d_v, d_g = bwd_kernel(
        dy_c, o_c, r_c, k_c, r_k_c, v_c, g_c,
    )
    d_rk = d_rk_part.sum(dim=0).to(r_k.dtype)
    # single flat output: a raw multi-output tuple crossing a torch.compile
    # cudagraph partition boundary escapes cudagraph trees' flat output
    # tracking ("tensor(s) in the cudagraph pool not tracked as outputs"), and
    # tagging the op cudagraph_unsafe trips an inductor partition codegen bug
    # at scale (phantom buffer names in the generated wrapper, torch 2.13)
    return torch.cat(tuple(t.reshape(-1) for t in (d_o, d_r, d_k, d_rk, d_v, d_g)))


@_gate_output_correction_bwd_op.register_fake
def _gate_output_correction_bwd_fake(dy, o, r, k, r_k, v, g):
    return o.new_empty(sum(t.numel() for t in (o, r, k, r_k, v, g)))


def _gate_output_correction_backward(ctx, dy):
    o, r, k, r_k, v, g = ctx.saved_tensors
    packed = _gate_output_correction_bwd_op(dy, o, r, k, r_k, v, g)
    outs = torch.split(packed, [t.numel() for t in (o, r, k, r_k, v, g)])
    return tuple(s.view_as(t) for s, t in zip(outs, (o, r, k, r_k, v, g)))


_gate_output_correction_op.register_autograd(
    _gate_output_correction_backward, setup_context=_gate_output_correction_setup_context,
)


def gate_output_correction_tilelang(
    o: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    r_k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    return _gate_output_correction_op(o, r, k, r_k, v, g)
