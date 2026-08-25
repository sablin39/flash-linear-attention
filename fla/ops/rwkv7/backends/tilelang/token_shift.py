# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""token_shift in TileLang, packaged for torch.compile fullgraph use.

Math: y[t] = x[t-1] - x[t]  (with x[-1] := cache or 0 per sequence)

Two execution paths, both free of data-dependent shapes and host syncs:

- Rectangular input with rect_T >= _MIN_INKERNEL_T: a single kernel owns
  everything — sequence-start rows (t % T_seq == 0) take x[t-1] from `cache`
  (or 0), the last row of each sequence is written to `cache_out`.
- Otherwise (varlen or small T): the kernel does the bulk flat
  shift-and-subtract on the (N_total, D) layout and the wrapper applies
  per-sequence boundary fix-ups + cache_out extraction with
  index_select/index_copy_, whose shapes follow from cu_seqlens.shape alone.

Unlike fla.modules.token_shift (Triton), this path needs no
prepare_chunk_indices (data-dependent output length) and no runtime SM-count
query, so the custom op below can sit inside a fullgraph-compiled region; the
backward is itself a custom op for the same reason.
"""

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

# In-kernel boundary handling uses t % T_seq, which is only correct when every
# sequence has the same length; the autotune-free config below also assumes
# rows are long enough to matter. Smaller/irregular inputs use the torch
# fix-up path instead.
_MIN_INKERNEL_T = 512


def _cfg(D: int) -> dict[str, int]:
    # 8 contiguous elements per thread so IO lowers to 16-byte vector ops
    BD = 256 if D % 256 == 0 else 128
    return {"BT": 16, "BD": BD, "threads": 16 * BD // 8}


@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _token_shift_fwd_kernel(N_total, D, T_seq, in_dtype, HAS_CACHE: bool,
                            BT: int = 16, BD: int = 256, threads: int = 512):
    nseq = N_total // T_seq

    @T.prim_func
    def token_shift_fwd_tl(
        x: T.Tensor((N_total, D), in_dtype),
        cache: T.Tensor((nseq, D), in_dtype),
        y: T.Tensor((N_total, D), in_dtype),
        cache_out: T.Tensor((nseq, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N_total, BT), T.ceildiv(D, BD), threads=threads) as (i_t, i_d):
            for kt, kd in T.Parallel(BT, BD):
                t = i_t * BT + kt
                d = i_d * BD + kd
                if (t < N_total) and (d < D):
                    if t % T_seq == 0:
                        if HAS_CACHE:
                            y[t, d] = cache[t // T_seq, d] - x[t, d]
                        else:
                            y[t, d] = -x[t, d]
                    else:
                        y[t, d] = x[t - 1, d] - x[t, d]
                    if t % T_seq == T_seq - 1:
                        cache_out[t // T_seq, d] = x[t, d]

    return token_shift_fwd_tl


@tilelang.jit(
    out_idx=[2, 3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _token_shift_bwd_kernel(N_total, D, T_seq, in_dtype, HAS_DCACHE: bool,
                            BT: int = 16, BD: int = 256, threads: int = 512):
    nseq = N_total // T_seq

    @T.prim_func
    def token_shift_bwd_tl(
        dy: T.Tensor((N_total, D), in_dtype),
        dcache: T.Tensor((nseq, D), in_dtype),
        dx: T.Tensor((N_total, D), in_dtype),
        grad_cache: T.Tensor((nseq, D), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N_total, BT), T.ceildiv(D, BD), threads=threads) as (i_t, i_d):
            for kt, kd in T.Parallel(BT, BD):
                t = i_t * BT + kt
                d = i_d * BD + kd
                if (t < N_total) and (d < D):
                    if t % T_seq == T_seq - 1:
                        if HAS_DCACHE:
                            dx[t, d] = -dy[t, d] + dcache[t // T_seq, d]
                        else:
                            dx[t, d] = -dy[t, d]
                    else:
                        dx[t, d] = -dy[t, d] + dy[t + 1, d]
                    if t % T_seq == 0:
                        grad_cache[t // T_seq, d] = dy[t, d]

    return token_shift_bwd_tl


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


def _normalize_cu_seqlens(x: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    """Return a 1-D int64 tensor of cumulative sequence lengths on x.device."""
    if cu_seqlens is not None:
        return cu_seqlens.to(device=x.device, dtype=torch.int64)
    B, T_, _ = x.shape
    return torch.arange(0, (B + 1) * T_, T_, device=x.device, dtype=torch.int64)


# shape-correct zero dummies for compiled-out cache args (the packed ABI checks
# declared shapes even when a constexpr branch never reads the tensor).
# Deliberately NOT cached at module level: a persistent tensor first-allocated
# during a cudagraph-trees warmup run lands in the cudagraph-private pool and
# trips "tensor(s) in the cudagraph pool not tracked as outputs"; a per-call
# temporary dies inside the op and is capture-safe (and free at replay).
def _zero_dummy(nseq: int, D: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros(nseq, D, device=ref.device, dtype=ref.dtype)


@torch.library.custom_op(
    "fla::token_shift_rwkv7",
    mutates_args=(),
    schema="(Tensor x, Tensor cu_seqlens, Tensor? cache, bool output_cache, int rect_T) -> (Tensor, Tensor?)",
)
def _token_shift_op(
    x: Tensor,
    cu_seqlens: Tensor,
    cache: Tensor | None,
    output_cache: bool,
    rect_T: int,
):
    assert x.dim() == 3, "Input must be [B, T, D]"
    _, _, D = x.shape
    active_nseq = cu_seqlens.numel() - 1
    x_flat = x.reshape(-1, D).contiguous()
    N_total = x_flat.shape[0]

    in_kernel = rect_T >= _MIN_INKERNEL_T
    T_seq = rect_T if in_kernel else N_total

    if in_kernel:
        has_cache = cache is not None
        cache_flat = cache.to(x.dtype) if has_cache else _zero_dummy(N_total // T_seq, D, x_flat)
        key = ("fwd", N_total, D, T_seq, _dtype_str(x), has_cache)
        kernel = _cached_kernel(key, lambda: _token_shift_fwd_kernel(N_total, D, T_seq, _dtype_str(x), has_cache, **_cfg(D)))
        y_flat, cache_out = kernel(x_flat, cache_flat)
        y = y_flat.view(*x.shape)
        return y, (cache_out if output_cache else None)

    key = ("fwd", N_total, D, N_total, _dtype_str(x), False)
    kernel = _cached_kernel(key, lambda: _token_shift_fwd_kernel(N_total, D, N_total, _dtype_str(x), False, **_cfg(D)))
    y_flat, _ = kernel(x_flat, _zero_dummy(1, D, x_flat))

    bos = cu_seqlens[:-1].to(torch.long)
    x_bos = x_flat.index_select(0, bos)
    if cache is not None:
        cache_2d = cache.reshape(active_nseq, D).to(x.dtype)
        y_flat.index_copy_(0, bos, cache_2d - x_bos)
    else:
        y_flat.index_copy_(0, bos, -x_bos)

    y = y_flat.view(*x.shape)
    if output_cache:
        eos_minus_1 = (cu_seqlens[1:] - 1).to(torch.long)
        cache_out = x_flat.index_select(0, eos_minus_1).contiguous()
        return y, cache_out
    return y, None


@_token_shift_op.register_fake
def _token_shift_fake(x, cu_seqlens, cache, output_cache: bool, rect_T: int):
    cache_out = None
    if output_cache:
        cache_out = x.new_empty((cu_seqlens.shape[0] - 1, x.shape[-1]))
    return x.new_empty(x.shape), cache_out


@torch.library.custom_op(
    "fla::token_shift_rwkv7_bwd",
    mutates_args=(),
    schema="(Tensor dy, Tensor cu_seqlens, Tensor? dcache, bool has_cache, int rect_T) -> (Tensor, Tensor?)",
)
def _token_shift_bwd_op(
    dy: Tensor,
    cu_seqlens: Tensor,
    dcache: Tensor | None,
    has_cache: bool,
    rect_T: int,
):
    _, _, D = dy.shape
    dy_flat = dy.reshape(-1, D).contiguous()
    N_total = dy_flat.shape[0]
    dtype_str = _dtype_str(dy)

    if rect_T >= _MIN_INKERNEL_T:
        nseq = N_total // rect_T
        has_dcache = dcache is not None
        dc_flat = dcache.reshape(nseq, D).to(dy.dtype) if has_dcache else _zero_dummy(nseq, D, dy_flat)
        key = ("bwd", N_total, D, rect_T, dtype_str, has_dcache)
        kernel = _cached_kernel(key, lambda: _token_shift_bwd_kernel(N_total, D, rect_T, dtype_str, has_dcache, **_cfg(D)))
        dx_flat, grad_cache = kernel(dy_flat, dc_flat)
        return dx_flat.view_as(dy), (grad_cache if has_cache else None)

    key = ("bwd", N_total, D, N_total, dtype_str, False)
    kernel = _cached_kernel(key, lambda: _token_shift_bwd_kernel(N_total, D, N_total, dtype_str, False, **_cfg(D)))
    dx_flat, _ = kernel(dy_flat, _zero_dummy(1, D, dy_flat))

    active_nseq = cu_seqlens.numel() - 1
    eos_minus_1 = (cu_seqlens[1:] - 1).to(torch.long)
    dy_eos = dy_flat.index_select(0, eos_minus_1)
    if dcache is not None:
        dc = dcache.reshape(active_nseq, D).to(dy.dtype)
        dx_flat.index_copy_(0, eos_minus_1, -dy_eos + dc)
    else:
        dx_flat.index_copy_(0, eos_minus_1, -dy_eos)

    grad_cache = None
    if has_cache:
        bos = cu_seqlens[:-1].to(torch.long)
        grad_cache = dy_flat.index_select(0, bos).contiguous()
    return dx_flat.view_as(dy), grad_cache


@_token_shift_bwd_op.register_fake
def _token_shift_bwd_fake(dy, cu_seqlens, dcache, has_cache: bool, rect_T: int):
    grad_cache = None
    if has_cache:
        grad_cache = dy.new_empty((cu_seqlens.shape[0] - 1, dy.shape[-1]))
    return dy.new_empty(dy.shape), grad_cache


def _token_shift_setup_context(ctx, inputs, output):
    _, cu_seqlens, cache, output_cache, rect_T = inputs
    ctx.save_for_backward(cu_seqlens)
    ctx.has_cache = cache is not None
    ctx.rect_T = rect_T


def _token_shift_backward(ctx, dy: torch.Tensor, dcache_out: torch.Tensor | None):
    (cu,) = ctx.saved_tensors
    dx, grad_cache = _token_shift_bwd_op(dy, cu, dcache_out, ctx.has_cache, ctx.rect_T)
    return dx, None, grad_cache, None, None


_token_shift_op.register_autograd(
    _token_shift_backward,
    setup_context=_token_shift_setup_context,
)


def token_shift_tilelang(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    output_cache: bool = False,
    chunk_indices: torch.Tensor | None = None,  # unused; kept for API parity
):
    """Drop-in TileLang replacement for fla.modules.token_shift.token_shift.

    Args:
        x: shape (B, T, D). For varlen, shape (1, T_total, D).
        cu_seqlens: 1-D int tensor (active_nseq+1,) of cumulative lengths. None for
            rectangular (B, T, D) input.
        cache: optional (active_nseq, D) tensor seeding x[-1] for each sequence's
            first token.
        output_cache: if True, also return the (active_nseq, D) last-token tensor
            for streaming.

    Returns:
        y (and cache_out if output_cache=True).
    """
    if cu_seqlens is not None:
        assert x.dim() == 3, "Input must be [B, T, D]"
        assert x.shape[0] == 1, "Batch size must be 1 when using cu_seqlens"

    if cache is not None:
        orig_cache = cache
        cache = cache.reshape(-1, x.shape[-1])
    else:
        orig_cache = None

    cu = _normalize_cu_seqlens(x, cu_seqlens)
    rect_T = x.shape[1] if cu_seqlens is None else 0
    y, cache_out = _token_shift_op(x, cu, cache, output_cache, rect_T)
    if (orig_cache is not None and cache_out is not None and rect_T == 1
            and cache.is_contiguous() and cache.dtype == x.dtype
            and not (torch.is_grad_enabled() and cache.requires_grad)):
        # decode fast path: copy the new conv state back into the caller's
        # persistent buffer (the HF static-cache pattern), so its address
        # stays static across steps under cudagraph capture; the copy is a
        # plain aten op, so the op itself stays functional and autograd-safe
        cache.copy_(cache_out)
        cache_out = orig_cache
    elif cache_out is not None and not torch.compiler.is_compiling():
        # freshly allocated streaming state: mark it static-address so that
        # when a serving loop feeds it back as `cache` under torch.compile,
        # cudagraph trees treats it like a parameter/buffer and permits the
        # in-place copy above instead of skipping cudagraphs
        torch._dynamo.mark_static_address(cache_out)
    if output_cache:
        return y, cache_out
    return y
