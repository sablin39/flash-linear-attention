# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang LayerNorm for the RWKV7 end-to-end fullgraph path.

Covers the two forms RWKV7 uses when ``fuse_norm=True`` (block-level
pre/attn/ffn/final norms): plain ``y = LN(x) * w + b`` and the prenorm
fused-residual form ``(y, res) = LN(x + residual) * w + b, x + residual``.
Both kernels hold a (BT, D) tile in register fragments and reduce rows with
warp shuffles (the layout that reached Triton parity in l2norm); the backward
recomputes mean/rstd in registers from the saved (cast) residual-sum instead
of materializing mean/rstd in the forward, and accumulates weight/bias grads
into per-block fp32 partials reduced deterministically in torch.
"""

import tilelang
import tilelang.language as T
import torch
from torch import Tensor, nn

# The @tilelang.jit wrapper re-runs its cache-key hashing on every call
# (~0.1-0.25 ms), which dominates kernels this small; the resolved JITKernel
# is shape/dtype-specialized and safe to reuse, so memoize it per key.
_KERNEL_CACHE: dict = {}

_BT = 8
_THREADS = 256


def _cached_kernel(key: tuple, builder):
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = builder()
        _KERNEL_CACHE[key] = k
    return k


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _ln_fwd_kernel(N, D, in_dtype, eps: float, BT: int = _BT, threads: int = _THREADS):
    acc_dtype = "float32"

    @T.prim_func
    def ln_fwd_tl(
        x: T.Tensor((N, D), in_dtype),
        weight: T.Tensor((D,), in_dtype),
        bias: T.Tensor((D,), in_dtype),
        y: T.Tensor((N, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BT), threads=threads) as i_n:
            s_frag = T.alloc_fragment((BT, D), acc_dtype)
            sq_frag = T.alloc_fragment((BT, D), acc_dtype)
            mean_frag = T.alloc_fragment((BT,), acc_dtype)
            rstd_frag = T.alloc_fragment((BT,), acc_dtype)
            w_frag = T.alloc_fragment((D,), acc_dtype)
            b_frag = T.alloc_fragment((D,), acc_dtype)

            for d in T.Parallel(D):
                w_frag[d] = T.Cast(acc_dtype, weight[d])
                b_frag[d] = T.Cast(acc_dtype, bias[d])
            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    s_frag[bt, d] = T.Cast(acc_dtype, x[i_n * BT + bt, d])
                else:
                    s_frag[bt, d] = 0.0

            T.reduce_sum(s_frag, mean_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                mean_frag[bt] = mean_frag[bt] / D
            # masked rows stay all-zero here, so they contribute nothing to var
            for bt, d in T.Parallel(BT, D):
                s_frag[bt, d] = s_frag[bt, d] - mean_frag[bt]
                sq_frag[bt, d] = s_frag[bt, d] * s_frag[bt, d]
            T.reduce_sum(sq_frag, rstd_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                rstd_frag[bt] = T.rsqrt(rstd_frag[bt] / D + eps)
            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    y[i_n * BT + bt, d] = T.Cast(
                        in_dtype,
                        s_frag[bt, d] * rstd_frag[bt] * w_frag[d] + b_frag[d],
                    )

    return ln_fwd_tl


@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _ln_prenorm_fwd_kernel(N, D, in_dtype, eps: float, BT: int = _BT, threads: int = _THREADS):
    acc_dtype = "float32"

    @T.prim_func
    def ln_prenorm_fwd_tl(
        x: T.Tensor((N, D), in_dtype),
        residual: T.Tensor((N, D), in_dtype),
        weight: T.Tensor((D,), in_dtype),
        bias: T.Tensor((D,), in_dtype),
        y: T.Tensor((N, D), in_dtype),
        res_out: T.Tensor((N, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BT), threads=threads) as i_n:
            s_frag = T.alloc_fragment((BT, D), acc_dtype)
            sq_frag = T.alloc_fragment((BT, D), acc_dtype)
            mean_frag = T.alloc_fragment((BT,), acc_dtype)
            rstd_frag = T.alloc_fragment((BT,), acc_dtype)
            w_frag = T.alloc_fragment((D,), acc_dtype)
            b_frag = T.alloc_fragment((D,), acc_dtype)

            for d in T.Parallel(D):
                w_frag[d] = T.Cast(acc_dtype, weight[d])
                b_frag[d] = T.Cast(acc_dtype, bias[d])
            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    s_frag[bt, d] = (
                        T.Cast(acc_dtype, x[i_n * BT + bt, d])
                        + T.Cast(acc_dtype, residual[i_n * BT + bt, d])
                    )
                else:
                    s_frag[bt, d] = 0.0

            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    res_out[i_n * BT + bt, d] = T.Cast(in_dtype, s_frag[bt, d])

            T.reduce_sum(s_frag, mean_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                mean_frag[bt] = mean_frag[bt] / D
            for bt, d in T.Parallel(BT, D):
                s_frag[bt, d] = s_frag[bt, d] - mean_frag[bt]
                sq_frag[bt, d] = s_frag[bt, d] * s_frag[bt, d]
            T.reduce_sum(sq_frag, rstd_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                rstd_frag[bt] = T.rsqrt(rstd_frag[bt] / D + eps)
            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    y[i_n * BT + bt, d] = T.Cast(
                        in_dtype,
                        s_frag[bt, d] * rstd_frag[bt] * w_frag[d] + b_frag[d],
                    )

    return ln_prenorm_fwd_tl


@tilelang.jit(
    out_idx=[4, 5, 6],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _ln_bwd_kernel(
    N, D, NB, in_dtype, eps: float, has_dres: bool,
    BT: int = _BT, threads: int = _THREADS,
):
    """dx + per-block fp32 wgrad partials; mean/rstd recomputed from s."""
    acc_dtype = "float32"

    @T.prim_func
    def ln_bwd_tl(
        dy: T.Tensor((N, D), in_dtype),
        s: T.Tensor((N, D), in_dtype),
        weight: T.Tensor((D,), in_dtype),
        dres: T.Tensor((N, D), in_dtype),
        dx: T.Tensor((N, D), in_dtype),
        dw_p: T.Tensor((NB, D), acc_dtype),
        db_p: T.Tensor((NB, D), acc_dtype),
    ):
        with T.Kernel(NB, threads=threads) as i_n:
            s_frag = T.alloc_fragment((BT, D), acc_dtype)
            dy_frag = T.alloc_fragment((BT, D), acc_dtype)
            prod_frag = T.alloc_fragment((BT, D), acc_dtype)
            mean_frag = T.alloc_fragment((BT,), acc_dtype)
            rstd_frag = T.alloc_fragment((BT,), acc_dtype)
            c1_frag = T.alloc_fragment((BT,), acc_dtype)
            c2_frag = T.alloc_fragment((BT,), acc_dtype)
            w_frag = T.alloc_fragment((D,), acc_dtype)
            dw_frag = T.alloc_fragment((D,), acc_dtype)
            db_frag = T.alloc_fragment((D,), acc_dtype)

            for d in T.Parallel(D):
                w_frag[d] = T.Cast(acc_dtype, weight[d])
            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    s_frag[bt, d] = T.Cast(acc_dtype, s[i_n * BT + bt, d])
                    dy_frag[bt, d] = T.Cast(acc_dtype, dy[i_n * BT + bt, d])
                else:
                    s_frag[bt, d] = 0.0
                    dy_frag[bt, d] = 0.0

            # recompute the row stats (masked rows are all-zero throughout)
            T.reduce_sum(s_frag, mean_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                mean_frag[bt] = mean_frag[bt] / D
            for bt, d in T.Parallel(BT, D):
                s_frag[bt, d] = s_frag[bt, d] - mean_frag[bt]
                prod_frag[bt, d] = s_frag[bt, d] * s_frag[bt, d]
            T.reduce_sum(prod_frag, rstd_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                rstd_frag[bt] = T.rsqrt(rstd_frag[bt] / D + eps)
            # s_frag now holds x_hat = (s - mean) * rstd
            for bt, d in T.Parallel(BT, D):
                s_frag[bt, d] = s_frag[bt, d] * rstd_frag[bt]

            # dx_hat = dy * w (kept in prod_frag so dy_frag stays raw for
            # the wgrad pass); c1 = sum(dx_hat * x_hat) / D, c2 = sum(dx_hat) / D
            for bt, d in T.Parallel(BT, D):
                prod_frag[bt, d] = dy_frag[bt, d] * w_frag[d]
            T.reduce_sum(prod_frag, c2_frag, dim=1, clear=True)
            for bt, d in T.Parallel(BT, D):
                prod_frag[bt, d] = prod_frag[bt, d] * s_frag[bt, d]
            T.reduce_sum(prod_frag, c1_frag, dim=1, clear=True)
            for bt in T.Parallel(BT):
                c1_frag[bt] = c1_frag[bt] / D
                c2_frag[bt] = c2_frag[bt] / D

            for bt, d in T.Parallel(BT, D):
                if i_n * BT + bt < N:
                    dx_v = rstd_frag[bt] * (
                        dy_frag[bt, d] * w_frag[d]
                        - c2_frag[bt] - s_frag[bt, d] * c1_frag[bt]
                    )
                    if has_dres:
                        dx[i_n * BT + bt, d] = T.Cast(
                            in_dtype, dx_v + T.Cast(acc_dtype, dres[i_n * BT + bt, d])
                        )
                    else:
                        dx[i_n * BT + bt, d] = T.Cast(in_dtype, dx_v)

            # wgrad partials from the live fragments (masked rows are zero):
            # dw = sum_rows(dy * x_hat), db = sum_rows(dy)
            for bt, d in T.Parallel(BT, D):
                prod_frag[bt, d] = dy_frag[bt, d] * s_frag[bt, d]
            T.reduce_sum(prod_frag, dw_frag, dim=0, clear=True)
            for d in T.Parallel(D):
                dw_p[i_n, d] = dw_frag[d]
            T.reduce_sum(dy_frag, db_frag, dim=0, clear=True)
            for d in T.Parallel(D):
                db_p[i_n, d] = db_frag[d]

    return ln_bwd_tl


@torch.library.custom_op(
    "fla::layer_norm_tl",
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor",
)
def _layer_norm_op(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    shape = x.shape
    D = shape[-1]
    N = x.numel() // D
    x_f = x.reshape(N, D).contiguous()
    in_dtype = _dtype_str(x)
    key = ("ln_fwd", N, D, in_dtype, float(eps))
    kernel = _cached_kernel(
        key, lambda: _ln_fwd_kernel(N, D, in_dtype, float(eps)),
    )
    return kernel(x_f, weight.contiguous(), bias.contiguous()).view(shape)


@_layer_norm_op.register_fake
def _layer_norm_fake(x, weight, bias, eps):
    return x.new_empty(x.shape)


def _layer_norm_setup_context(ctx, inputs, output):
    x, weight, bias, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = float(eps)


def _layer_norm_backward(ctx, dy):
    x, weight = ctx.saved_tensors
    dx, dw, db = _layer_norm_bwd_op(dy, x, weight, None, ctx.eps)
    return dx, dw, db, None


_layer_norm_op.register_autograd(
    _layer_norm_backward, setup_context=_layer_norm_setup_context,
)


@torch.library.custom_op(
    "fla::layer_norm_prenorm_tl",
    mutates_args=(),
    schema=(
        "(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps)"
        " -> (Tensor, Tensor)"
    ),
)
def _layer_norm_prenorm_op(
    x: Tensor, residual: Tensor, weight: Tensor, bias: Tensor, eps: float,
) -> tuple[Tensor, Tensor]:
    shape = x.shape
    D = shape[-1]
    N = x.numel() // D
    x_f = x.reshape(N, D).contiguous()
    res_f = residual.reshape(N, D).contiguous()
    in_dtype = _dtype_str(x)
    key = ("ln_prenorm_fwd", N, D, in_dtype, float(eps))
    kernel = _cached_kernel(
        key, lambda: _ln_prenorm_fwd_kernel(N, D, in_dtype, float(eps)),
    )
    y_f, res_out = kernel(x_f, res_f, weight.contiguous(), bias.contiguous())
    return y_f.view(shape), res_out.view(shape)


@_layer_norm_prenorm_op.register_fake
def _layer_norm_prenorm_fake(x, residual, weight, bias, eps):
    return x.new_empty(x.shape), x.new_empty(x.shape)


def _layer_norm_prenorm_setup_context(ctx, inputs, output):
    x, residual, weight, bias, eps = inputs
    y, res_out = output
    ctx.save_for_backward(res_out, weight)
    ctx.eps = float(eps)


def _layer_norm_prenorm_backward(ctx, dy, dres):
    res_out, weight = ctx.saved_tensors
    # s = x + residual feeds both branches, so grad_x = grad_residual
    g, dw, db = _layer_norm_bwd_op(dy, res_out, weight, dres, ctx.eps)
    return g, g, dw, db, None


_layer_norm_prenorm_op.register_autograd(
    _layer_norm_prenorm_backward,
    setup_context=_layer_norm_prenorm_setup_context,
)


@torch.library.custom_op(
    "fla::layer_norm_tl_bwd",
    mutates_args=(),
    schema=(
        "(Tensor dy, Tensor s, Tensor weight, Tensor? dres, float eps)"
        " -> (Tensor, Tensor, Tensor)"
    ),
)
def _layer_norm_bwd_op(
    dy: Tensor, s: Tensor, weight: Tensor, dres: Tensor | None, eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    shape = s.shape
    D = shape[-1]
    N = s.numel() // D
    dy_f = dy.reshape(N, D).contiguous()
    s_f = s.reshape(N, D).contiguous()
    has_dres = dres is not None
    dres_f = dres.reshape(N, D).contiguous() if has_dres else dy_f
    in_dtype = _dtype_str(s)
    NB = (N + _BT - 1) // _BT
    key = ("ln_bwd", N, D, in_dtype, float(eps), has_dres)
    kernel = _cached_kernel(
        key,
        lambda: _ln_bwd_kernel(N, D, NB, in_dtype, float(eps), has_dres),
    )
    dx_f, dw_p, db_p = kernel(dy_f, s_f, weight.contiguous(), dres_f)
    dw = dw_p.sum(dim=0).to(weight.dtype)
    db = db_p.sum(dim=0).to(weight.dtype)
    return dx_f.view(shape), dw, db


@_layer_norm_bwd_op.register_fake
def _layer_norm_bwd_fake(dy, s, weight, dres, eps):
    return (
        s.new_empty(s.shape),
        weight.new_empty(weight.shape),
        weight.new_empty(weight.shape),
    )


class TileLangLayerNorm(nn.Module):
    """fla.modules.LayerNorm drop-in on the TileLang kernels above.

    Only the forms RWKV7 uses are implemented (affine weight + bias; plain and
    prenorm fused-residual forwards). Constructed in place of fla's LayerNorm
    on the e2e path so block norms stay inside the compiled graph.
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not elementwise_affine or not bias:
            raise NotImplementedError(
                "TileLangLayerNorm requires elementwise_affine=True and "
                "bias=True (the RWKV7 fuse_norm configuration)"
            )
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual_in_fp32:
            raise NotImplementedError(
                "TileLangLayerNorm does not support residual_in_fp32"
            )
        if residual is None:
            return _layer_norm_op(x, self.weight, self.bias, self.eps)
        if not prenorm:
            raise NotImplementedError(
                "TileLangLayerNorm supports residual only with prenorm=True"
            )
        return _layer_norm_prenorm_op(x, residual, self.weight, self.bias, self.eps)
