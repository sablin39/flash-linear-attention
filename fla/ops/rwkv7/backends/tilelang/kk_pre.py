# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused RWKV7 kk pre-processing in TileLang.

One kernel replaces the unfused sequence
    k_new  = fused_k_rwkv7(k, a, k_a)      # k * (1 + (a - 1) * k_a)
    kk     = F.normalize(k * k_k, dim=-1)  # per-(token, head) l2 norm
    neg_kk = -kk                           # DPLR a
    kka    = kk * a                        # DPLR b
saving three elementwise passes and the (k * k_k) intermediate. The
normalization follows F.normalize clamp semantics (eps = 1e-12), matching the
layer's default fuse_norm=False path and the official RWKV reference.

Forward and backward are both opaque custom ops so AOTAutograd never traces
into TileLang JIT calls; the ops are capture-safe (no input mutation, no
data-dependent shapes). Both return exactly one tensor (the forward packs
(k_new, neg_kk, kka) into (3, N, D), the backward one flat buffer): a raw
multi-output tuple crossing a cudagraph partition boundary escapes cudagraph
trees' flat output tracking.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

_EPS = 1.0e-12


def _fwd_cfg(N: int, D: int) -> dict[str, int]:
    return {"BR": 4, "threads": 256}


@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _kk_pre_fwd_kernel(N, H, D, in_dtype, BR: int = 4, threads: int = 256):
    acc_dtype = "float32"
    eps2 = _EPS * _EPS
    inv_eps = 1.0 / _EPS

    @T.prim_func
    def kk_pre_fwd_tl(
        k: T.Tensor((N, D), in_dtype),
        a: T.Tensor((N, D), in_dtype),
        k_k: T.Tensor((H, D), in_dtype),
        k_a: T.Tensor((H, D), in_dtype),
        # (k_new, neg_kk, kka) packed into one buffer: a custom op returning a
        # raw tuple can have that tuple passed across a torch.compile cudagraph
        # partition boundary, where cudagraph trees only tracks flat outputs
        # and raises "tensor(s) in the cudagraph pool not tracked as outputs"
        o: T.Tensor((3, N, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BR), threads=threads) as i_b:
            scaled = T.alloc_fragment((BR, D), acc_dtype)
            sq = T.alloc_fragment((BR, D), acc_dtype)
            ss = T.alloc_fragment((BR,), acc_dtype)

            for r, d in T.Parallel(BR, D):
                i_n = i_b * BR + r
                if i_n < N:
                    i_h = i_n % H
                    bk = T.Cast(acc_dtype, k[i_n, d])
                    ba = T.Cast(acc_dtype, a[i_n, d])
                    s = bk * T.Cast(acc_dtype, k_k[i_h, d])
                    scaled[r, d] = s
                    sq[r, d] = s * s
                    o[0, i_n, d] = T.Cast(
                        in_dtype, bk * (1.0 + (ba - 1.0) * T.Cast(acc_dtype, k_a[i_h, d])),
                    )
                else:
                    scaled[r, d] = 0.0
                    sq[r, d] = 0.0

            T.reduce_sum(sq, ss, dim=1, clear=True)

            for r, d in T.Parallel(BR, D):
                i_n = i_b * BR + r
                if i_n < N:
                    inv_norm = T.if_then_else(
                        ss[r] > T.Cast(acc_dtype, eps2),
                        T.rsqrt(ss[r]),
                        T.Cast(acc_dtype, inv_eps),
                    )
                    kk = scaled[r, d] * inv_norm
                    o[1, i_n, d] = T.Cast(in_dtype, -kk)
                    o[2, i_n, d] = T.Cast(in_dtype, kk * T.Cast(acc_dtype, a[i_n, d]))

    return kk_pre_fwd_tl


@tilelang.jit(
    out_idx=[7, 8, 9, 10],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _kk_pre_bwd_main_kernel(
    M, H, D, NB, in_dtype,
    BT: int = 16,
    threads: int = 256,
    HAS_DK_NEW: bool = True,
    HAS_DNEG_KK: bool = True,
    HAS_DKKA: bool = True,
):
    acc_dtype = "float32"
    eps2 = _EPS * _EPS
    inv_eps = 1.0 / _EPS

    @T.prim_func
    def kk_pre_bwd_main_tl(
        k: T.Tensor((M, H, D), in_dtype),
        a: T.Tensor((M, H, D), in_dtype),
        k_k: T.Tensor((H, D), in_dtype),
        k_a: T.Tensor((H, D), in_dtype),
        dk_new: T.Tensor((M, H, D), in_dtype),
        dneg_kk: T.Tensor((M, H, D), in_dtype),
        dkka: T.Tensor((M, H, D), in_dtype),
        grad_k: T.Tensor((M, H, D), in_dtype),
        grad_a: T.Tensor((M, H, D), in_dtype),
        grad_kk_w_partial: T.Tensor((NB, H, D), acc_dtype),
        grad_ka_w_partial: T.Tensor((NB, H, D), acc_dtype),
    ):
        with T.Kernel(NB, H, threads=threads) as (i_b, i_h):
            scaled = T.alloc_fragment((BT, D), acc_dtype)
            sq = T.alloc_fragment((BT, D), acc_dtype)
            ss = T.alloc_fragment((BT,), acc_dtype)
            kk_frag = T.alloc_fragment((BT, D), acc_dtype)
            gkk_frag = T.alloc_fragment((BT, D), acc_dtype)
            dot_terms = T.alloc_fragment((BT, D), acc_dtype)
            dot = T.alloc_fragment((BT,), acc_dtype)
            grad_kk_terms = T.alloc_fragment((BT, D), acc_dtype)
            grad_ka_terms = T.alloc_fragment((BT, D), acc_dtype)
            grad_kk_sum = T.alloc_fragment((D,), acc_dtype)
            grad_ka_sum = T.alloc_fragment((D,), acc_dtype)

            for r, d in T.Parallel(BT, D):
                i_m = i_b * BT + r
                if i_m < M:
                    kv = T.Cast(acc_dtype, k[i_m, i_h, d])
                    s = kv * T.Cast(acc_dtype, k_k[i_h, d])
                    scaled[r, d] = s
                    sq[r, d] = s * s
                else:
                    scaled[r, d] = 0.0
                    sq[r, d] = 0.0

            T.reduce_sum(sq, ss, dim=1, clear=True)

            for r, d in T.Parallel(BT, D):
                i_m = i_b * BT + r
                if i_m < M:
                    av = T.Cast(acc_dtype, a[i_m, i_h, d])
                    kv = T.Cast(acc_dtype, k[i_m, i_h, d])
                    kav = T.Cast(acc_dtype, k_a[i_h, d])
                    kkv = T.Cast(acc_dtype, k_k[i_h, d])
                    inv_norm = T.if_then_else(
                        ss[r] > T.Cast(acc_dtype, eps2),
                        T.rsqrt(ss[r]),
                        T.Cast(acc_dtype, inv_eps),
                    )
                    kk = scaled[r, d] * inv_norm
                    kk_frag[r, d] = kk

                    gy = T.alloc_var(acc_dtype)
                    gy = T.Cast(acc_dtype, 0.0)
                    if HAS_DK_NEW:
                        gy = T.Cast(acc_dtype, dk_new[i_m, i_h, d])
                    gm = T.alloc_var(acc_dtype)
                    gm = T.Cast(acc_dtype, 0.0)
                    if HAS_DKKA:
                        gm = T.Cast(acc_dtype, dkka[i_m, i_h, d])

                    grad_k_val = gy * (1.0 + (av - 1.0) * kav)
                    grad_a_val = T.alloc_var(acc_dtype)
                    grad_a_val = gy * kv * kav
                    grad_ka_terms[r, d] = gy * kv * (av - 1.0)

                    gkk = T.alloc_var(acc_dtype)
                    gkk = T.Cast(acc_dtype, 0.0)
                    if HAS_DNEG_KK:
                        gkk = gkk - T.Cast(acc_dtype, dneg_kk[i_m, i_h, d])
                    if HAS_DKKA:
                        gkk = gkk + gm * av
                        grad_a_val = grad_a_val + gm * kk
                    gkk_frag[r, d] = gkk
                    dot_terms[r, d] = gkk * kk

                    grad_k[i_m, i_h, d] = T.Cast(in_dtype, grad_k_val)
                    grad_a[i_m, i_h, d] = T.Cast(in_dtype, grad_a_val)
                else:
                    kk_frag[r, d] = 0.0
                    gkk_frag[r, d] = 0.0
                    dot_terms[r, d] = 0.0
                    grad_ka_terms[r, d] = 0.0

            T.reduce_sum(dot_terms, dot, dim=1, clear=True)

            for r, d in T.Parallel(BT, D):
                i_m = i_b * BT + r
                if i_m < M:
                    kv = T.Cast(acc_dtype, k[i_m, i_h, d])
                    kkv = T.Cast(acc_dtype, k_k[i_h, d])
                    inv_norm = T.if_then_else(
                        ss[r] > T.Cast(acc_dtype, eps2),
                        T.rsqrt(ss[r]),
                        T.Cast(acc_dtype, inv_eps),
                    )
                    grad_scaled = (gkk_frag[r, d] - kk_frag[r, d] * dot[r]) * inv_norm
                    grad_k[i_m, i_h, d] = T.Cast(
                        in_dtype,
                        T.Cast(acc_dtype, grad_k[i_m, i_h, d]) + grad_scaled * kkv,
                    )
                    grad_kk_terms[r, d] = grad_scaled * kv
                else:
                    grad_kk_terms[r, d] = 0.0
                    grad_ka_terms[r, d] = 0.0

            T.reduce_sum(grad_kk_terms, grad_kk_sum, dim=0, clear=True)
            T.reduce_sum(grad_ka_terms, grad_ka_sum, dim=0, clear=True)
            for d in T.Parallel(D):
                grad_kk_w_partial[i_b, i_h, d] = grad_kk_sum[d]
                grad_ka_w_partial[i_b, i_h, d] = grad_ka_sum[d]

    return kk_pre_bwd_main_tl


@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _kk_pre_bwd_reduce_kernel(NB, H, D, out_dtype, threads: int = 128):
    acc_dtype = "float32"

    @T.prim_func
    def kk_pre_bwd_reduce_tl(
        grad_kk_w_partial: T.Tensor((NB, H, D), acc_dtype),
        grad_ka_w_partial: T.Tensor((NB, H, D), acc_dtype),
        grad_kk_w: T.Tensor((H, D), out_dtype),
        grad_ka_w: T.Tensor((H, D), out_dtype),
    ):
        with T.Kernel(H, threads=threads) as i_h:
            acc_kk = T.alloc_fragment((D,), acc_dtype)
            acc_ka = T.alloc_fragment((D,), acc_dtype)
            for d in T.Parallel(D):
                acc_kk[d] = 0.0
                acc_ka[d] = 0.0
                for i_b in T.serial(NB):
                    acc_kk[d] = acc_kk[d] + grad_kk_w_partial[i_b, i_h, d]
                    acc_ka[d] = acc_ka[d] + grad_ka_w_partial[i_b, i_h, d]
                grad_kk_w[i_h, d] = T.Cast(out_dtype, acc_kk[d])
                grad_ka_w[i_h, d] = T.Cast(out_dtype, acc_ka[d])

    return kk_pre_bwd_reduce_tl


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
    "fla::kk_pre_rwkv7",
    mutates_args=(),
    schema="(Tensor k, Tensor a, Tensor k_k, Tensor k_a, int head_dim) -> Tensor",
)
def _kk_pre_op(
    k: Tensor,
    a: Tensor,
    k_k: Tensor,
    k_a: Tensor,
    head_dim: int,
) -> Tensor:
    assert k.shape == a.shape
    D = int(head_dim)
    H = k.shape[-1] // D
    N = k.numel() // D

    k_f = k.reshape(N, D).contiguous()
    a_f = a.reshape(N, D).contiguous()
    kk_w = k_k.reshape(H, D).contiguous()
    ka_w = k_a.reshape(H, D).contiguous()

    key = ("fwd", N, H, D, _dtype_str(k))
    kernel = _cached_kernel(
        key,
        lambda: _kk_pre_fwd_kernel(N, H, D, _dtype_str(k), **_fwd_cfg(N, D)),
    )
    # (3, N, D): k_new, neg_kk, kka — one buffer so the op has a single
    # (cudagraph-partition-safe) tensor output
    return kernel(k_f, a_f, kk_w, ka_w)


@_kk_pre_op.register_fake
def _kk_pre_fake(k, a, k_k, k_a, head_dim: int):
    N = k.numel() // head_dim
    return k.new_empty((3, N, head_dim))


def _kk_pre_setup_context(ctx, inputs, output):
    k, a, k_k, k_a, head_dim = inputs
    ctx.save_for_backward(k, a, k_k, k_a)
    ctx.head_dim = int(head_dim)


@torch.library.custom_op(
    "fla::kk_pre_rwkv7_bwd",
    mutates_args=(),
    schema=(
        "(Tensor dk_new, Tensor dneg_kk, Tensor dkka, Tensor k, Tensor a, "
        "Tensor k_k, Tensor k_a, int head_dim, bool has_dk_new, "
        "bool has_dneg_kk, bool has_dkka) -> Tensor"
    ),
)
def _kk_pre_bwd_op(
    dk_new: Tensor,
    dneg_kk: Tensor,
    dkka: Tensor,
    k: Tensor,
    a: Tensor,
    k_k: Tensor,
    k_a: Tensor,
    head_dim: int,
    has_dk_new: bool,
    has_dneg_kk: bool,
    has_dkka: bool,
) -> Tensor:
    D = int(head_dim)
    H = k.shape[-1] // D
    flat_shape = (-1, H, D)
    M = k.numel() // (H * D)
    BT, threads = 16, 256
    NB = (M + BT - 1) // BT

    k_f = k.reshape(flat_shape).contiguous()
    a_f = a.reshape(flat_shape).contiguous()
    kk_w = k_k.reshape(H, D).contiguous()
    ka_w = k_a.reshape(H, D).contiguous()
    dk_new_f = dk_new.reshape(flat_shape).contiguous()
    dneg_f = dneg_kk.reshape(flat_shape).contiguous()
    dkka_f = dkka.reshape(flat_shape).contiguous()

    dtype = _dtype_str(k)
    main_key = ("bwd_main", M, H, D, dtype, has_dk_new, has_dneg_kk, has_dkka)
    main_kernel = _cached_kernel(
        main_key,
        lambda: _kk_pre_bwd_main_kernel(
            M, H, D, NB, dtype, BT=BT, threads=threads,
            HAS_DK_NEW=has_dk_new, HAS_DNEG_KK=has_dneg_kk, HAS_DKKA=has_dkka,
        ),
    )
    grad_k, grad_a, grad_kk_partial, grad_ka_partial = main_kernel(
        k_f, a_f, kk_w, ka_w, dk_new_f, dneg_f, dkka_f,
    )
    reduce_key = ("bwd_reduce", NB, H, D, _dtype_str(k_k))
    reduce_kernel = _cached_kernel(
        reduce_key,
        lambda: _kk_pre_bwd_reduce_kernel(NB, H, D, _dtype_str(k_k)),
    )
    grad_kk_w, grad_ka_w = reduce_kernel(grad_kk_partial, grad_ka_partial)

    # single flat output: a raw multi-output tuple crossing a torch.compile
    # cudagraph partition boundary escapes cudagraph trees' flat output
    # tracking ("tensor(s) in the cudagraph pool not tracked as outputs"), and
    # tagging the op cudagraph_unsafe trips an inductor partition codegen bug
    # at scale (phantom buffer names in the generated wrapper, torch 2.13)
    return torch.cat((
        grad_k.reshape(-1).to(k.dtype),
        grad_a.reshape(-1).to(a.dtype),
        grad_kk_w.reshape(-1),
        grad_ka_w.reshape(-1),
    ))


@_kk_pre_bwd_op.register_fake
def _kk_pre_bwd_fake(
    dk_new, dneg_kk, dkka, k, a, k_k, k_a, head_dim,
    has_dk_new, has_dneg_kk, has_dkka,
):
    return k.new_empty(sum(t.numel() for t in (k, a, k_k, k_a)))


def _kk_pre_backward(ctx, d_o: torch.Tensor):
    k, a, k_k, k_a = ctx.saved_tensors
    # the packed (3, N, D) output always comes back fully materialized
    dk_new, dneg_kk, dkka = d_o.unbind(0)
    packed = _kk_pre_bwd_op(
        dk_new,
        dneg_kk,
        dkka,
        k,
        a,
        k_k,
        k_a,
        ctx.head_dim,
        True,
        True,
        True,
    )
    saved = (k, a, k_k, k_a)
    outs = torch.split(packed, [t.numel() for t in saved])
    grad_k, grad_a, grad_kk_w, grad_ka_w = (s.view_as(t) for s, t in zip(outs, saved))
    return grad_k, grad_a, grad_kk_w, grad_ka_w, None


_kk_pre_op.register_autograd(_kk_pre_backward, setup_context=_kk_pre_setup_context)


def kk_pre_rwkv7(
    k: torch.Tensor,
    a: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RWKV7 k-update + kk normalization.

    Args:
        k: (..., H*D) key projection output (pre-update).
        a: (..., H*D) sigmoid gate from a_lora, same shape as k.
        k_k: (H*D,) or (H, D) per-channel kk weight.
        k_a: (H*D,) or (H, D) per-channel k-update weight.
        head_dim: D.

    Returns:
        k_new (..., H*D): ``k * (1 + (a - 1) * k_a)``.
        neg_kk (..., H, D): ``-F.normalize(k * k_k, dim=-1)``.
        kka (..., H, D): ``F.normalize(k * k_k, dim=-1) * a``.
    """
    D = int(head_dim)
    H = k.shape[-1] // D
    o = _kk_pre_op(k, a, k_k, k_a, D)
    k_new, neg_kk, kka = o.unbind(0)
    head_shape = (*k.shape[:-1], H, D)
    return (
        k_new.view_as(k),
        neg_kk.view(head_shape),
        kka.view(head_shape),
    )
