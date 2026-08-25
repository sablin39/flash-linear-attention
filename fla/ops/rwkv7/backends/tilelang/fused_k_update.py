# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""fused_k_rwkv7: out = k * (1 + (a - 1) * ka)

Pointwise in (B, T, D); `ka` broadcasts along (B, T). Backward gives:
    dk  = dy * (1 + (a - 1) * ka)
    da  = dy * k * ka
    dka = sum_{B,T}(dy * k * (a - 1))

The backward is a single fused kernel: it writes dk/da and accumulates
per-segment dka partials (fp32) in the same pass over dy/k/a; a host-side
sum over the S segments produces dka.
"""

import tilelang
import tilelang.language as T
import torch
from torch import Tensor


def _pointwise_config(D: int) -> dict[str, int]:
    # BD=512/threads=256 measures best on both sm_90 and sm_120 for large D;
    # each thread then owns 16 consecutive d-elements (vectorized 16B IO).
    BD = min(512, max(16, 1 << (D - 1).bit_length()))
    # 16 elements per thread (contiguous d) so global IO lowers to 256-bit ops
    BT = max(1, 8192 // BD)
    threads = min(512, max(32, BT * BD // 16))
    return {"BT": BT, "BD": BD, "threads": threads}


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _k_update_fwd_kernel(N_total, D, in_dtype, BT: int = 16, BD: int = 512, threads: int = 512):
    acc_dtype = "float32"

    @T.prim_func
    def k_update_fwd_tl(
        k: T.Tensor((N_total, D), in_dtype),
        a: T.Tensor((N_total, D), in_dtype),
        ka: T.Tensor((D,), in_dtype),
        out: T.Tensor((N_total, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N_total, BT), T.ceildiv(D, BD), threads=threads) as (i_t, i_d):
            for kt, kd in T.Parallel(BT, BD):
                t = i_t * BT + kt
                d = i_d * BD + kd
                if (t < N_total) and (d < D):
                    bk = T.Cast(acc_dtype, k[t, d])
                    ba = T.Cast(acc_dtype, a[t, d])
                    bka = T.Cast(acc_dtype, ka[d])
                    out[t, d] = T.Cast(in_dtype, bk * (1.0 + (ba - 1.0) * bka))

    return k_update_fwd_tl


@tilelang.jit(
    out_idx=[4, 5, 6],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _k_update_bwd_kernel(N_total, S, D, in_dtype, BT: int = 8, BD: int = 128, threads: int = 128):
    acc_dtype = "float32"

    @T.prim_func
    def k_update_bwd_tl(
        dy: T.Tensor((N_total, D), in_dtype),
        k: T.Tensor((N_total, D), in_dtype),
        a: T.Tensor((N_total, D), in_dtype),
        ka: T.Tensor((D,), in_dtype),
        dk: T.Tensor((N_total, D), in_dtype),
        da: T.Tensor((N_total, D), in_dtype),
        dka_part: T.Tensor((S, D), acc_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BD), S, threads=threads) as (i_d, i_s):
            acc = T.alloc_fragment((BT, BD), acc_dtype)
            dka_frag = T.alloc_fragment((BD,), acc_dtype)
            T.clear(acc)

            rps = T.ceildiv(N_total, S)
            row0 = i_s * rps
            row1 = T.min(row0 + rps, N_total)

            for i0 in T.serial(T.ceildiv(rps, BT)):
                for kt, kd in T.Parallel(BT, BD):
                    t = row0 + i0 * BT + kt
                    d = i_d * BD + kd
                    if (t < row1) and (d < D):
                        bdy = T.Cast(acc_dtype, dy[t, d])
                        bk = T.Cast(acc_dtype, k[t, d])
                        ba = T.Cast(acc_dtype, a[t, d])
                        bka = T.Cast(acc_dtype, ka[d])
                        dk[t, d] = T.Cast(in_dtype, bdy * (1.0 + (ba - 1.0) * bka))
                        da[t, d] = T.Cast(in_dtype, bdy * bk * bka)
                        acc[kt, kd] += bdy * bk * (ba - 1.0)

            T.reduce_sum(acc, dka_frag, dim=0)
            for kd in T.Parallel(BD):
                d = i_d * BD + kd
                if d < D:
                    dka_part[i_s, d] = dka_frag[kd]

    return k_update_bwd_tl


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


@torch.library.custom_op("fla::fused_k_rwkv7", mutates_args=())
def _fused_k_update_op(k: Tensor, a: Tensor, ka: Tensor) -> Tensor:
    shape = k.shape
    D = k.shape[-1]
    N_total = k.numel() // D

    k_flat = k.reshape(N_total, D).contiguous()
    a_flat = a.reshape(N_total, D).contiguous()
    ka_flat = ka.reshape(D).contiguous()

    kernel = _k_update_fwd_kernel(N_total, D, _dtype_str(k), **_pointwise_config(D))
    out_flat = kernel(k_flat, a_flat, ka_flat)
    return out_flat.view(*shape)


@_fused_k_update_op.register_fake
def _fused_k_update_fake(k, a, ka):
    return k.new_empty(k.shape)


def _fused_k_update_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


@torch.library.custom_op("fla::fused_k_rwkv7_bwd", mutates_args=())
def _fused_k_update_bwd_op(dy: Tensor, k: Tensor, a: Tensor, ka: Tensor) -> Tensor:
    D = k.shape[-1]
    N_total = k.numel() // D
    # Segment count: enough blocks to fill the GPU, but each block should
    # still loop over a meaningful number of rows.
    S = max(1, min(64, N_total // 2048))

    dy_flat = dy.reshape(N_total, D).contiguous()
    k_flat = k.reshape(N_total, D).contiguous()
    a_flat = a.reshape(N_total, D).contiguous()
    ka_flat = ka.reshape(D).contiguous()

    kernel = _k_update_bwd_kernel(N_total, S, D, _dtype_str(k))
    dk, da, dka_part = kernel(dy_flat, k_flat, a_flat, ka_flat)
    dka = dka_part.sum(dim=0).to(ka.dtype)
    # single flat output: a raw multi-output tuple crossing a torch.compile
    # cudagraph partition boundary escapes cudagraph trees' flat output
    # tracking ("tensor(s) in the cudagraph pool not tracked as outputs"), and
    # tagging the op cudagraph_unsafe trips an inductor partition codegen bug
    # at scale (phantom buffer names in the generated wrapper, torch 2.13)
    return torch.cat((dk.reshape(-1), da.reshape(-1), dka.reshape(-1)))


@_fused_k_update_bwd_op.register_fake
def _fused_k_update_bwd_fake(dy, k, a, ka):
    return k.new_empty(k.numel() + a.numel() + ka.numel())


def _fused_k_update_backward(ctx, dy: torch.Tensor):
    k, a, ka = ctx.saved_tensors
    packed = _fused_k_update_bwd_op(dy, k, a, ka)
    outs = torch.split(packed, [t.numel() for t in (k, a, ka)])
    return tuple(s.view_as(t) for s, t in zip(outs, (k, a, ka)))


_fused_k_update_op.register_autograd(
    _fused_k_update_backward, setup_context=_fused_k_update_setup_context,
)


def fused_k_rwkv7_tilelang(k: torch.Tensor, a: torch.Tensor, ka: Tensor) -> torch.Tensor:
    return _fused_k_update_op(k, a, ka)
