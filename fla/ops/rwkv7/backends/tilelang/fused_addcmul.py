# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""fused_addcmul_rwkv7 in TileLang.

For each of the 5 or 6 mix-parameters x_*, computes
    o_x*  = hidden_states + delta * x_*
where hidden_states and delta have shape (B, T, D) and x_* has shape (1, 1, D)
(broadcast over batch and time).

A pointwise op — varlen-packed (1, T_total, D) layout works identically to
rectangular (B, T, D).

Backward:
    d_hidden = sum_* dx_*
    d_delta  = sum_* dx_* * x_*
    d_x*     = sum_{B,T} dx_* * delta        (per-channel reduction)
All of it in one fused kernel: the elementwise terms are written directly and
the six per-channel reductions accumulate into (6, S, D) fp32 segment
partials that a host-side sum finishes.
"""

import tilelang
import tilelang.language as T
import torch
from torch import Tensor


def _pointwise_config(D: int) -> dict[str, int]:
    # BD=512/threads=256 measures best on both sm_90 and sm_120 for large D;
    # each thread then owns 16 consecutive d-elements (vectorized 16B IO).
    # At D >= 2048 a full-warp block (32x512 tile, 1024 threads) edges out the
    # 16x512/512 tile by 1-3% on sm_120 (write-heavy: 2 reads, 6 outputs).
    if D >= 2048:
        return {"BT": 32, "BD": 512, "threads": 1024}
    BD = min(512, max(16, 1 << (D - 1).bit_length()))
    # 16 elements per thread (contiguous d) so global IO lowers to 256-bit ops
    BT = max(1, 8192 // BD)
    threads = min(512, max(32, BT * BD // 16))
    return {"BT": BT, "BD": BD, "threads": threads}


@tilelang.jit(
    out_idx=[8],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _addcmul_fwd_kernel(N_total, D, in_dtype, use_xg: bool, BT: int = 16, BD: int = 512, threads: int = 512):
    @T.prim_func
    def addcmul_fwd_tl(
        hidden: T.Tensor((N_total, D), in_dtype),
        delta: T.Tensor((N_total, D), in_dtype),
        x_r: T.Tensor((D,), in_dtype),
        x_w: T.Tensor((D,), in_dtype),
        x_k: T.Tensor((D,), in_dtype),
        x_v: T.Tensor((D,), in_dtype),
        x_a: T.Tensor((D,), in_dtype),
        x_g: T.Tensor((D,), in_dtype),
        # the 6 outputs share one buffer: a custom op returning a raw tuple can
        # have that tuple passed across a torch.compile cudagraph partition
        # boundary, where cudagraph trees only tracks flat outputs and raises
        # "tensor(s) in the cudagraph pool not tracked as outputs"
        o: T.Tensor((6, N_total, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N_total, BT), T.ceildiv(D, BD), threads=threads) as (i_t, i_d):
            for kt, kd in T.Parallel(BT, BD):
                t = i_t * BT + kt
                d = i_d * BD + kd
                if (t < N_total) and (d < D):
                    h = hidden[t, d]
                    de = delta[t, d]
                    o[0, t, d] = h + de * x_r[d]
                    o[1, t, d] = h + de * x_w[d]
                    o[2, t, d] = h + de * x_k[d]
                    o[3, t, d] = h + de * x_v[d]
                    o[4, t, d] = h + de * x_a[d]
                    if use_xg:
                        o[5, t, d] = h + de * x_g[d]
                    else:
                        # x_g is a zeros placeholder here; copying it keeps o[5]
                        # zero-filled without a broadcast-constant store, which
                        # breaks fp16 codegen (cutlass::half_t -> half pack)
                        o[5, t, d] = x_g[d]

    return addcmul_fwd_tl


@tilelang.jit(
    out_idx=[13, 14, 15],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _addcmul_bwd_kernel(N_total, S, D, in_dtype, use_xg: bool, BT: int = 8, BD: int = 128, threads: int = 128):
    acc_dtype = "float32"

    @T.prim_func
    def addcmul_bwd_tl(
        dxr: T.Tensor((N_total, D), in_dtype),
        dxw: T.Tensor((N_total, D), in_dtype),
        dxk: T.Tensor((N_total, D), in_dtype),
        dxv: T.Tensor((N_total, D), in_dtype),
        dxa: T.Tensor((N_total, D), in_dtype),
        dxg: T.Tensor((N_total, D), in_dtype),
        x_r: T.Tensor((D,), in_dtype),
        x_w: T.Tensor((D,), in_dtype),
        x_k: T.Tensor((D,), in_dtype),
        x_v: T.Tensor((D,), in_dtype),
        x_a: T.Tensor((D,), in_dtype),
        x_g: T.Tensor((D,), in_dtype),
        delta: T.Tensor((N_total, D), in_dtype),
        d_hidden: T.Tensor((N_total, D), in_dtype),
        d_delta: T.Tensor((N_total, D), in_dtype),
        parts: T.Tensor((6, S, D), acc_dtype),
    ):
        with T.Kernel(T.ceildiv(D, BD), S, threads=threads) as (i_d, i_s):
            acc_r = T.alloc_fragment((BT, BD), acc_dtype)
            acc_w = T.alloc_fragment((BT, BD), acc_dtype)
            acc_k = T.alloc_fragment((BT, BD), acc_dtype)
            acc_v = T.alloc_fragment((BT, BD), acc_dtype)
            acc_a = T.alloc_fragment((BT, BD), acc_dtype)
            red_r = T.alloc_fragment((BD,), acc_dtype)
            red_w = T.alloc_fragment((BD,), acc_dtype)
            red_k = T.alloc_fragment((BD,), acc_dtype)
            red_v = T.alloc_fragment((BD,), acc_dtype)
            red_a = T.alloc_fragment((BD,), acc_dtype)
            T.clear(acc_r)
            T.clear(acc_w)
            T.clear(acc_k)
            T.clear(acc_v)
            T.clear(acc_a)
            if use_xg:
                acc_g = T.alloc_fragment((BT, BD), acc_dtype)
                red_g = T.alloc_fragment((BD,), acc_dtype)
                T.clear(acc_g)

            rps = T.ceildiv(N_total, S)
            row0 = i_s * rps
            row1 = T.min(row0 + rps, N_total)

            for i0 in T.serial(T.ceildiv(rps, BT)):
                for kt, kd in T.Parallel(BT, BD):
                    t = row0 + i0 * BT + kt
                    d = i_d * BD + kd
                    if (t < row1) and (d < D):
                        ar = T.Cast(acc_dtype, dxr[t, d])
                        aw = T.Cast(acc_dtype, dxw[t, d])
                        ak = T.Cast(acc_dtype, dxk[t, d])
                        av = T.Cast(acc_dtype, dxv[t, d])
                        aa = T.Cast(acc_dtype, dxa[t, d])
                        de = T.Cast(acc_dtype, delta[t, d])
                        gh = ar + aw + ak + av + aa
                        gd = (
                            ar * T.Cast(acc_dtype, x_r[d])
                            + aw * T.Cast(acc_dtype, x_w[d])
                            + ak * T.Cast(acc_dtype, x_k[d])
                            + av * T.Cast(acc_dtype, x_v[d])
                            + aa * T.Cast(acc_dtype, x_a[d])
                        )
                        acc_r[kt, kd] += ar * de
                        acc_w[kt, kd] += aw * de
                        acc_k[kt, kd] += ak * de
                        acc_v[kt, kd] += av * de
                        acc_a[kt, kd] += aa * de
                        if use_xg:
                            ag = T.Cast(acc_dtype, dxg[t, d])
                            gh = gh + ag
                            gd = gd + ag * T.Cast(acc_dtype, x_g[d])
                            acc_g[kt, kd] += ag * de
                        d_hidden[t, d] = T.Cast(in_dtype, gh)
                        d_delta[t, d] = T.Cast(in_dtype, gd)

            T.reduce_sum(acc_r, red_r, dim=0)
            T.reduce_sum(acc_w, red_w, dim=0)
            T.reduce_sum(acc_k, red_k, dim=0)
            T.reduce_sum(acc_v, red_v, dim=0)
            T.reduce_sum(acc_a, red_a, dim=0)
            if use_xg:
                T.reduce_sum(acc_g, red_g, dim=0)
            for kd in T.Parallel(BD):
                d = i_d * BD + kd
                if d < D:
                    parts[0, i_s, d] = red_r[kd]
                    parts[1, i_s, d] = red_w[kd]
                    parts[2, i_s, d] = red_k[kd]
                    parts[3, i_s, d] = red_v[kd]
                    parts[4, i_s, d] = red_a[kd]
                    if use_xg:
                        parts[5, i_s, d] = red_g[kd]
                    else:
                        parts[5, i_s, d] = 0.0

    return addcmul_bwd_tl


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


@torch.library.custom_op("fla::fused_addcmul_rwkv7", mutates_args=())
def _fused_addcmul_op(
    hidden: Tensor,
    delta: Tensor,
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor | None,
) -> Tensor:
    B, T_, D = hidden.shape
    N_total = B * T_
    use_xg = x_g is not None

    hidden_flat = hidden.reshape(N_total, D).contiguous()
    delta_flat = delta.reshape(N_total, D).contiguous()
    xr_flat = x_r.reshape(D).contiguous()
    xw_flat = x_w.reshape(D).contiguous()
    xk_flat = x_k.reshape(D).contiguous()
    xv_flat = x_v.reshape(D).contiguous()
    xa_flat = x_a.reshape(D).contiguous()
    if use_xg:
        xg_flat = x_g.reshape(D).contiguous()
    else:
        xg_flat = torch.zeros((D,), dtype=hidden.dtype, device=hidden.device)

    kernel = _addcmul_fwd_kernel(N_total, D, _dtype_str(hidden), use_xg, **_pointwise_config(D))
    o = kernel(
        hidden_flat, delta_flat, xr_flat, xw_flat, xk_flat, xv_flat, xa_flat, xg_flat,
    )
    return o.view(6, B, T_, D)


@_fused_addcmul_op.register_fake
def _fused_addcmul_fake(hidden, delta, x_r, x_w, x_k, x_v, x_a, x_g):
    return hidden.new_empty((6,) + hidden.shape)


def _fused_addcmul_setup_context(ctx, inputs, output):
    hidden, delta, x_r, x_w, x_k, x_v, x_a, x_g = inputs
    dummy = hidden.new_empty((0,))
    ctx.save_for_backward(hidden, delta, x_r, x_w, x_k, x_v, x_a, x_g if x_g is not None else dummy)
    ctx.use_xg = x_g is not None


@torch.library.custom_op("fla::fused_addcmul_rwkv7_bwd", mutates_args=())
def _fused_addcmul_bwd_op(
    dxr: Tensor,
    dxw: Tensor,
    dxk: Tensor,
    dxv: Tensor,
    dxa: Tensor,
    dxg: Tensor,
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor,
    delta: Tensor,
    use_xg: bool,
) -> Tensor:
    B, T_, D = delta.shape
    N_total = B * T_
    S = max(1, min(64, N_total // 2048))
    in_dtype = _dtype_str(delta)
    kernel = _addcmul_bwd_kernel(N_total, S, D, in_dtype, use_xg)
    d_hidden, d_delta, parts = kernel(
        dxr.reshape(N_total, D).contiguous(),
        dxw.reshape(N_total, D).contiguous(),
        dxk.reshape(N_total, D).contiguous(),
        dxv.reshape(N_total, D).contiguous(),
        dxa.reshape(N_total, D).contiguous(),
        dxg.reshape(N_total, D).contiguous(),
        x_r.reshape(D).contiguous(),
        x_w.reshape(D).contiguous(),
        x_k.reshape(D).contiguous(),
        x_v.reshape(D).contiguous(),
        x_a.reshape(D).contiguous(),
        x_g.reshape(D).contiguous(),
        delta.reshape(N_total, D).contiguous(),
    )
    d_x = [parts.select(0, i).sum(dim=0).to(delta.dtype) for i in range(6)]
    # single flat output: a raw multi-output tuple crossing a torch.compile
    # cudagraph partition boundary escapes cudagraph trees' flat output
    # tracking ("tensor(s) in the cudagraph pool not tracked as outputs"), and
    # tagging the op cudagraph_unsafe trips an inductor partition codegen bug
    # at scale (phantom buffer names in the generated wrapper, torch 2.13)
    return torch.cat((d_hidden.reshape(-1), d_delta.reshape(-1),
                      *(d.reshape(-1) for d in d_x)))


@_fused_addcmul_bwd_op.register_fake
def _fused_addcmul_bwd_fake(dxr, dxw, dxk, dxv, dxa, dxg, x_r, x_w, x_k, x_v, x_a, x_g, delta, use_xg):
    return delta.new_empty(2 * delta.numel() + 6 * x_r.numel())


def _fused_addcmul_backward(ctx, d_o: torch.Tensor):
    hidden, delta, x_r, x_w, x_k, x_v, x_a, x_g_or_dummy = ctx.saved_tensors
    use_xg = ctx.use_xg
    # dead slices (e.g. oxg when x_g is None) come back as zeros; the
    # use_xg=False kernel never reads dxg anyway
    dxr, dxw, dxk, dxv, dxa, dxg = d_o.unbind(0)

    packed = _fused_addcmul_bwd_op(
        dxr, dxw, dxk, dxv, dxa, dxg,
        x_r, x_w, x_k, x_v, x_a,
        # placeholder for the use_xg=False kernel, which never reads it
        # (the saved dummy is 0-element and would fail the op's reshape)
        x_g_or_dummy if use_xg else x_r,
        delta,
        use_xg,
    )
    saved = (delta, delta, x_r, x_w, x_k, x_v, x_a, x_g_or_dummy if use_xg else x_r)
    outs = torch.split(packed, [t.numel() for t in saved])
    d_hidden, d_delta, d_xr, d_xw, d_xk, d_xv, d_xa, d_xg = (
        s.view_as(t) for s, t in zip(outs, saved)
    )
    return (
        d_hidden.view_as(hidden),
        d_delta,
        d_xr, d_xw, d_xk, d_xv, d_xa,
        d_xg if use_xg else None,
    )


_fused_addcmul_op.register_autograd(
    _fused_addcmul_backward, setup_context=_fused_addcmul_setup_context,
)


def fused_addcmul_rwkv7_tilelang(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    xr: torch.Tensor,
    xw: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    xa: torch.Tensor,
    xg: torch.Tensor | None = None,
):
    o = _fused_addcmul_op(hidden_states, delta, xr, xw, xk, xv, xa, xg)
    oxr, oxw, oxk, oxv, oxa, oxg = o.unbind(0)
    return oxr, oxw, oxk, oxv, oxa, oxg if xg is not None else None
