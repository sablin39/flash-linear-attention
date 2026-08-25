# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv7.backends.tilelang import RWKV7TileLangBackend
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7, torch_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7, k_update_ref
from fla.ops.rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction
from fla.utils import assert_close, device

_TILELANG_USABLE = RWKV7TileLangBackend.is_available()
_DISPATCH_DISABLED = os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
_CUDA_AVAILABLE = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason='verifier device queries need CUDA',
)
requires_tilelang_route = pytest.mark.skipif(
    _DISPATCH_DISABLED or not _TILELANG_USABLE,
    reason='TileLang backend not available or dispatch disabled',
)


def _spy_on_tilelang_route(monkeypatch, func_name: str):
    calls = []
    orig = getattr(RWKV7TileLangBackend, func_name)

    def spy(self, *args, **kwargs):
        calls.append(None)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(RWKV7TileLangBackend, func_name, spy)
    return calls


def _assert_route_parity(monkeypatch, func_name, run, names, ratio=0.005):
    monkeypatch.setenv('FLA_TILELANG', '0')
    ref = run()
    calls = _spy_on_tilelang_route(monkeypatch, func_name)
    monkeypatch.setenv('FLA_TILELANG', '1')
    tri = run()
    assert calls, 'TileLang backend route was not taken'
    # cross-backend fp32 accumulation order differs slightly
    for name, r, t in zip(names, ref, tri):
        assert_close(name, r, t, ratio)


def _assert_stays_off_tilelang(monkeypatch, func_name, run, names, ratio=0.002):
    monkeypatch.setenv('FLA_TILELANG', '0')
    ref = run()
    calls = _spy_on_tilelang_route(monkeypatch, func_name)
    monkeypatch.setenv('FLA_TILELANG', '1')
    tri = run()
    assert not calls, 'TileLang backend route was unexpectedly taken'
    for name, r, t in zip(names, ref, tri):
        assert_close(name, r, t, ratio)


# ------------------------------------------------------------------ verifiers


@requires_cuda
def test_addcmul_verifier_accepts():
    backend = RWKV7TileLangBackend()
    h = torch.empty(2, 16, 64, dtype=torch.bfloat16, device=device)
    xs = [torch.empty(1, 1, 64, dtype=torch.bfloat16, device=device) for _ in range(6)]
    ok, reason = backend.verify('fused_addcmul_rwkv7', h, h, *xs)
    assert ok, reason


@requires_cuda
@pytest.mark.parametrize('case', ['cpu', 'decode', 'dtype', 'mix_dtype'])
def test_addcmul_verifier_rejects(case: str):
    backend = RWKV7TileLangBackend()
    dev = 'cpu' if case == 'cpu' else device
    T = 1 if case == 'decode' else 16
    h_dtype = torch.float32 if case == 'dtype' else torch.bfloat16
    x_dtype = torch.float32 if case == 'mix_dtype' else torch.bfloat16
    h = torch.empty(2, T, 64, dtype=h_dtype, device=dev)
    xs = [torch.empty(1, 1, 64, dtype=x_dtype, device=dev) for _ in range(6)]
    ok, reason = backend.verify('fused_addcmul_rwkv7', h, h, *xs)
    assert not ok and reason


@requires_cuda
def test_k_update_verifier_accepts():
    backend = RWKV7TileLangBackend()
    k = torch.empty(2, 16, 64, dtype=torch.bfloat16, device=device)
    ka = torch.empty(64, dtype=torch.bfloat16, device=device)
    ok, reason = backend.verify('fused_k_rwkv7', k, k, ka)
    assert ok, reason


@requires_cuda
@pytest.mark.parametrize('case', ['cpu', 'decode', 'varlen', 'dtype'])
def test_k_update_verifier_rejects(case: str):
    backend = RWKV7TileLangBackend()
    dev = 'cpu' if case == 'cpu' else device
    T = 1 if case == 'decode' else 16
    dtype = torch.float32 if case == 'dtype' else torch.bfloat16
    k = torch.empty(1, T, 64, dtype=dtype, device=dev)
    ka = torch.empty(64, dtype=dtype, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev) if case == 'varlen' else None
    ok, reason = backend.verify('fused_k_rwkv7', k, k, ka, cu)
    assert not ok and reason


@requires_cuda
def test_gate_output_correction_verifier_accepts():
    backend = RWKV7TileLangBackend()
    o = torch.empty(2, 16, 128, dtype=torch.bfloat16, device=device)
    r = torch.empty(2, 16, 2, 64, dtype=torch.bfloat16, device=device)
    r_k = torch.empty(2, 64, dtype=torch.bfloat16, device=device)
    ok, reason = backend.verify('gate_output_correction', o, r, r, r_k, r, o)
    assert ok, reason


@requires_cuda
@pytest.mark.parametrize('case', ['cpu', 'dtype', 'bad_rk', 'bad_o'])
def test_gate_output_correction_verifier_rejects(case: str):
    backend = RWKV7TileLangBackend()
    dev = 'cpu' if case == 'cpu' else device
    dtype = torch.float32 if case == 'dtype' else torch.bfloat16
    o = torch.empty(2, 16, 128, dtype=dtype, device=dev)
    r = torch.empty(2, 16, 2, 64, dtype=dtype, device=dev)
    r_k = torch.empty(2, 64, dtype=dtype, device=dev)
    if case == 'bad_rk':
        r_k = torch.empty(3, 64, dtype=dtype, device=dev)
    if case == 'bad_o':
        o = torch.empty(2, 16, 64, dtype=dtype, device=dev)
    ok, reason = backend.verify('gate_output_correction', o, r, r, r_k, r, o)
    assert not ok and reason


@requires_cuda
def test_recurrent_verifier_accepts():
    backend = RWKV7TileLangBackend()
    xs = [torch.empty(2, 16, 2, 64, dtype=torch.bfloat16, device=device) for _ in range(6)]
    ok, reason = backend.verify('fused_mul_recurrent_rwkv7', *xs)
    assert ok, reason


@requires_cuda
@pytest.mark.parametrize('case', ['cpu', 'dtype', 'kv_mismatch', 'head_dim', 'grad'])
def test_recurrent_verifier_rejects(case: str):
    backend = RWKV7TileLangBackend()
    dev = 'cpu' if case == 'cpu' else device
    dtype = torch.float32 if case == 'dtype' else torch.bfloat16
    D = 32 if case == 'head_dim' else 64
    xs = [torch.empty(2, 16, 2, D, dtype=dtype, device=dev) for _ in range(6)]
    if case == 'kv_mismatch':
        xs[3] = torch.empty(2, 16, 2, 128, dtype=dtype, device=dev)
    if case == 'grad':
        xs[0].requires_grad_(True)
        ok, reason = backend.verify('fused_mul_recurrent_rwkv7', *xs)
    else:
        with torch.no_grad():
            ok, reason = backend.verify('fused_mul_recurrent_rwkv7', *xs)
    assert not ok and reason


# -------------------------------------------------------------- route parity


@requires_tilelang_route
@pytest.mark.parametrize('use_g', [True, False])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_addcmul_tilelang_route_parity(monkeypatch, use_g: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    B, T, D = 2, 1024, 2048

    hidden = torch.randn(B, T, D, device=device, dtype=dtype)
    xx = torch.randn(B, T, D, device=device, dtype=dtype)
    xs = [torch.randn(1, 1, D, device=device, dtype=dtype) for _ in range(5)]
    x_g = torch.randn(1, 1, D, device=device, dtype=dtype) if use_g else None
    dos = [torch.randn(B, T, D, device=device, dtype=dtype) for _ in range(7)]

    def run():
        h_, x_ = hidden.clone().requires_grad_(), xx.clone().requires_grad_()
        xs_ = [x.clone().requires_grad_() for x in xs]
        xg_ = x_g.clone().requires_grad_() if use_g else None
        outs = [o for o in fused_addcmul_rwkv7(h_, x_, *xs_, xg_) if o is not None]
        torch.autograd.backward(outs, dos[:len(outs)])
        grads = [h_.grad, x_.grad] + [x.grad for x in xs_] + ([xg_.grad] if use_g else [])
        return [o.detach() for o in outs] + grads

    names = ['xr', 'xw', 'xk', 'xv', 'xa'] + (['xg'] if use_g else []) + \
        ['d_hidden', 'd_xx', 'd_ixr', 'd_ixw', 'd_ixk', 'd_ixv', 'd_ixa'] + (['d_ixg'] if use_g else [])
    _assert_route_parity(monkeypatch, 'fused_addcmul_rwkv7', run, names)


@requires_tilelang_route
@pytest.mark.parametrize('ka_shape', [1, 3])
def test_k_update_tilelang_route_parity(monkeypatch, ka_shape: int):
    torch.manual_seed(42)
    B, T, H, D = 2, 1024, 32, 64
    dtype = torch.bfloat16

    k = torch.randn(B, T, H * D, device=device).uniform_(-8, 8).to(dtype)
    a = torch.randn(B, T, H * D, device=device).uniform_(-8, 8).to(dtype)
    ka = torch.randn(H * D if ka_shape == 1 else (1, 1, H * D), device=device).uniform_(-8, 8).to(dtype)
    do = torch.randn(B, T, H * D, device=device, dtype=dtype)

    def run():
        k_, a_, ka_ = (x.clone().requires_grad_() for x in (k, a, ka))
        o = fused_k_rwkv7(k_, a_, ka_)
        o.backward(do)
        return [o.detach(), k_.grad, a_.grad, ka_.grad]

    _assert_route_parity(monkeypatch, 'fused_k_rwkv7', run, ['o', 'dk', 'da', 'dka'])


@requires_tilelang_route
# T=1000 makes M=B*T indivisible by the kernel row tile, exercising the tail path
@pytest.mark.parametrize(['T', 'dtype'], [(1024, torch.bfloat16), (1000, torch.float16)])
def test_gate_output_correction_tilelang_route_parity(monkeypatch, T, dtype):
    torch.manual_seed(42)
    B, H, D = 2, 32, 64

    o = torch.randn(B, T, H * D, device=device, dtype=dtype)
    r = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    r_k = torch.randn(H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H * D, device=device, dtype=dtype)
    do = torch.randn(B, T, H * D, device=device, dtype=dtype)

    def run():
        o_, r_, k_, rk_, v_, g_ = (x.clone().requires_grad_() for x in (o, r, k, r_k, v, g))
        out = gate_output_correction(o_, r_, k_, rk_, v_, g_)
        out.backward(do)
        return [out.detach(), o_.grad, r_.grad, k_.grad, rk_.grad, v_.grad, g_.grad]

    _assert_route_parity(
        monkeypatch, 'gate_output_correction', run,
        ['o', 'do', 'dr', 'dk', 'drk', 'dv', 'dg'],
    )


def _recurrent_inputs(B, T, H, D, dtype, cu_seqlens=None):
    def mk(): return torch.empty(B, T, H, D, device=device).uniform_(-8, -6).to(dtype)
    r, k, v, w = mk(), mk(), mk(), mk()
    kk = F.normalize(torch.empty(B, T, H, D, device=device).uniform_(-1, 1), dim=-1).to(dtype)
    a = torch.empty(B, T, H, D, device=device).uniform_(0, 0.1).to(dtype)
    if cu_seqlens is None:
        h0 = torch.randn(B, H, D, D, dtype=torch.float, device=device)
    else:
        h0 = torch.randn(cu_seqlens.shape[0] - 1, H, D, D, dtype=torch.float, device=device)
    return r, w, k, v, kk, a, h0


@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        (2, 63, 32, 64, torch.bfloat16),
        (4, 1, 32, 64, torch.bfloat16),
        (2, 16, 16, 128, torch.float16),
    ],
)
def test_recurrent_tilelang_route_parity(monkeypatch, B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    r, w, k, v, kk, a, h0 = _recurrent_inputs(B, T, H, D, dtype)

    def run():
        with torch.no_grad():
            o, ht = fused_mul_recurrent_rwkv7(
                r=r, w=w, k=k, v=v, kk=kk, a=a,
                scale=1.0, initial_state=h0, output_final_state=True,
            )
        return [o, ht]

    _assert_route_parity(monkeypatch, 'fused_mul_recurrent_rwkv7', run, ['o', 'ht'])


@requires_tilelang_route
def test_recurrent_tilelang_route_parity_varlen(monkeypatch):
    torch.manual_seed(42)
    H, D = 32, 64
    cu_seqlens = torch.tensor([0, 17, 17 + 63, 17 + 63 + 33], dtype=torch.int32, device=device)
    r, w, k, v, kk, a, h0 = _recurrent_inputs(1, cu_seqlens[-1].item(), H, D, torch.bfloat16, cu_seqlens)

    def run():
        with torch.no_grad():
            o, ht = fused_mul_recurrent_rwkv7(
                r=r, w=w, k=k, v=v, kk=kk, a=a,
                scale=1.0, initial_state=h0, output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
        return [o, ht]

    _assert_route_parity(monkeypatch, 'fused_mul_recurrent_rwkv7', run, ['o', 'ht'])


# ------------------------------------------------- verifier-driven fallbacks


@requires_cuda
def test_addcmul_decode_stays_on_torch(monkeypatch):
    torch.manual_seed(42)
    B, D = 4, 2048
    dtype = torch.bfloat16

    hidden = torch.randn(B, 1, D, device=device, dtype=dtype)
    xx = torch.randn(B, 1, D, device=device, dtype=dtype)
    xs = [torch.randn(1, 1, D, device=device, dtype=dtype) for _ in range(5)]
    x_g = torch.randn(1, 1, D, device=device, dtype=dtype)

    def run():
        return list(fused_addcmul_rwkv7(hidden, xx, *xs, x_g))

    _assert_stays_off_tilelang(
        monkeypatch, 'fused_addcmul_rwkv7', run,
        ['xr', 'xw', 'xk', 'xv', 'xa', 'xg'],
    )
    ref = torch_addcmul_rwkv7(hidden.float(), xx.float(), *[x.float() for x in xs], x_g.float())
    for name, o, r in zip(['xr', 'xw', 'xk', 'xv', 'xa', 'xg'], run(), ref):
        assert_close(name, r, o, 0.002)


@requires_cuda
def test_k_update_decode_stays_on_torch(monkeypatch):
    torch.manual_seed(42)
    B, D = 4, 2048
    dtype = torch.bfloat16

    k = torch.randn(B, 1, D, device=device).uniform_(-8, 8).to(dtype)
    a = torch.randn(B, 1, D, device=device).uniform_(-8, 8).to(dtype)
    ka = torch.randn(1, 1, D, device=device).uniform_(-8, 8).to(dtype)

    def run():
        return [fused_k_rwkv7(k, a, ka)]

    _assert_stays_off_tilelang(monkeypatch, 'fused_k_rwkv7', run, ['o'])
    assert_close('o', k_update_ref(k, a, ka), run()[0], 0.002)


@requires_cuda
def test_k_update_varlen_stays_on_triton(monkeypatch):
    torch.manual_seed(42)
    H, D = 8, 64
    dtype = torch.bfloat16
    cu_seqlens = torch.tensor([0, 13, 13 + 57], dtype=torch.int32, device=device)
    T = cu_seqlens[-1].item()

    k = torch.randn(1, T, H * D, device=device).uniform_(-8, 8).to(dtype)
    a = torch.randn(1, T, H * D, device=device).uniform_(-8, 8).to(dtype)
    ka = torch.randn(1, 1, H * D, device=device).uniform_(-8, 8).to(dtype)

    def run():
        return [fused_k_rwkv7(k, a, ka, cu_seqlens)]

    _assert_stays_off_tilelang(monkeypatch, 'fused_k_rwkv7', run, ['o'])


@requires_cuda
def test_recurrent_grad_stays_on_triton(monkeypatch):
    torch.manual_seed(42)
    B, T, H, D = 2, 16, 32, 64
    dtype = torch.bfloat16
    r, w, k, v, kk, a, h0 = _recurrent_inputs(B, T, H, D, dtype)

    def run():
        r_ = r.clone().requires_grad_()
        o, ht = fused_mul_recurrent_rwkv7(
            r=r_, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, initial_state=h0, output_final_state=True,
        )
        return [o, ht]

    _assert_stays_off_tilelang(monkeypatch, 'fused_mul_recurrent_rwkv7', run, ['o', 'ht'])


# ----------------------------------------------------------------- token_shift

# token_shift has no dispatch route (it lives in fla.modules), so parity is
# checked by calling the TileLang entry directly against the Triton default.
@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'D', 'dtype', 'use_cache'),
    [
        (2, 1024, 2048, torch.bfloat16, False),   # in-kernel boundary path
        (2, 1024, 2048, torch.bfloat16, True),
        (2, 1024, 2048, torch.float16, True),
        (4, 256, 2048, torch.bfloat16, True),     # T < 512: torch fix-up path
        (8, 1, 2048, torch.bfloat16, True),       # decode
    ],
)
def test_token_shift_tilelang_parity(B: int, T: int, D: int, dtype: torch.dtype, use_cache: bool):
    from fla.modules.token_shift import token_shift
    from fla.ops.rwkv7.backends.tilelang.token_shift import token_shift_tilelang

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    cache = torch.randn(B, D, device=device, dtype=dtype) if use_cache else None
    dy = torch.randn(B, T, D, device=device, dtype=dtype)
    dcache = torch.randn(B, D, device=device, dtype=dtype)

    def run(fn):
        x_ = x.clone().requires_grad_()
        c_ = cache.clone().requires_grad_() if use_cache else None
        y, cache_out = fn(x_, None, cache=c_, output_cache=True)
        grads = torch.autograd.grad(
            [y, cache_out], [x_] + ([c_] if use_cache else []), [dy, dcache],
        )
        return [y, cache_out, *grads]

    ref = run(token_shift)
    out = run(token_shift_tilelang)
    # pure shift-and-subtract: both paths are exact
    for name, r, t in zip(['y', 'cache_out', 'dx', 'dcache'], ref, out):
        assert_close(name, r, t, 0.001)


@requires_tilelang_route
def test_token_shift_tilelang_parity_varlen():
    from fla.modules.token_shift import token_shift
    from fla.ops.rwkv7.backends.tilelang.token_shift import token_shift_tilelang

    torch.manual_seed(42)
    D = 2048
    dtype = torch.bfloat16
    cu_seqlens = torch.tensor([0, 517, 517 + 1024, 517 + 1024 + 63], dtype=torch.int32, device=device)
    T = cu_seqlens[-1].item()
    nseq = cu_seqlens.shape[0] - 1
    x = torch.randn(1, T, D, device=device, dtype=dtype)
    cache = torch.randn(nseq, D, device=device, dtype=dtype)
    dy = torch.randn(1, T, D, device=device, dtype=dtype)
    dcache = torch.randn(nseq, D, device=device, dtype=dtype)

    def run(fn):
        x_ = x.clone().requires_grad_()
        c_ = cache.clone().requires_grad_()
        y, cache_out = fn(x_, cu_seqlens, cache=c_, output_cache=True)
        grads = torch.autograd.grad([y, cache_out], [x_, c_], [dy, dcache])
        return [y, cache_out, *grads]

    ref = run(token_shift)
    out = run(token_shift_tilelang)
    for name, r, t in zip(['y', 'cache_out', 'dx', 'dcache'], ref, out):
        assert_close(name, r, t, 0.001)


# ------------------------------------------------------- fused e2e entry points

# kk_pre / gn_corr / the decode op are e2e entry points (no dispatch route),
# so parity is checked by calling the TileLang ops directly against the
# unfused composition they replace.


def _kk_pre_ref(k, a, k_k, k_a, H, D):
    B, T, _ = k.shape
    kk = F.normalize((k * k_k).view(B, T, H, D), dim=-1)
    k_new = k.addcmul(k * (a - 1.0), k_a)
    return k_new, -kk, kk * a.view(B, T, H, D)


@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        (2, 1024, 4, 64, torch.bfloat16),
        (2, 1024, 4, 64, torch.float16),
        (32, 1, 32, 64, torch.bfloat16),  # decode shape
        (1, 517, 8, 64, torch.bfloat16),  # varlen-style packed shape
    ],
)
def test_kk_pre_tilelang_parity(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    from fla.ops.rwkv7.backends.tilelang.kk_pre import kk_pre_rwkv7

    torch.manual_seed(42)
    k = torch.randn(B, T, H * D, device=device, dtype=dtype)
    a = torch.rand(B, T, H * D, device=device, dtype=dtype)
    k_k = torch.randn(H * D, device=device, dtype=dtype)
    k_a = torch.randn(H * D, device=device, dtype=dtype)

    k_new = torch.randn(B, T, H * D, device=device, dtype=dtype)
    dneg = torch.randn(B, T, H, D, device=device, dtype=dtype)
    dkka = torch.randn(B, T, H, D, device=device, dtype=dtype)

    def run(fn):
        k_, a_, kk_, ka_ = (x.clone().requires_grad_() for x in (k, a, k_k, k_a))
        outs = fn(k_, a_, kk_, ka_)
        grads = torch.autograd.grad(outs, [k_, a_, kk_, ka_], [k_new, dneg, dkka])
        return [*outs, *grads]

    ref = run(lambda k_, a_, kk_, ka_: _kk_pre_ref(k_, a_, kk_, ka_, H, D))
    out = run(lambda k_, a_, kk_, ka_: kk_pre_rwkv7(k_, a_, kk_, ka_, D))
    names = ['k_new', 'neg_kk', 'kka', 'd_k', 'd_a', 'd_k_k', 'd_k_a']
    ratio = 0.005 if dtype == torch.bfloat16 else 0.002
    for name, r, t in zip(names, ref, out):
        assert_close(name, r, t, ratio)


def _gn_corr_ref(o, r, k, r_k, v, g, weight, bias, eps, H):
    B, T, _, D = o.shape
    normed = F.group_norm(o.view(B * T, H * D), H, weight, bias, eps).view(B, T, H, D)
    corr = (r * k * r_k.view(1, 1, H, D)).sum(-1, keepdim=True)
    return (normed + corr * v) * g.view(B, T, H, D)


@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        (2, 256, 4, 64, torch.bfloat16),
        (2, 256, 4, 64, torch.float16),
        (32, 1, 32, 64, torch.bfloat16),  # decode shape
    ],
)
def test_gn_corr_tilelang_parity(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    from fla.ops.rwkv7.backends.tilelang.fused_gn_corr import gn_corr_rwkv7

    torch.manual_seed(42)
    eps = 64e-5
    o = torch.randn(B, T, H, D, device=device, dtype=dtype)
    r = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    r_k = torch.randn(H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H * D, device=device, dtype=dtype)
    weight = torch.randn(H * D, device=device, dtype=dtype)
    bias = torch.randn(H * D, device=device, dtype=dtype)
    dy = torch.randn(B, T, H, D, device=device, dtype=dtype)

    def run(fn):
        o_, r_, k_, rk_, v_, g_, w_, b_ = (x.clone().requires_grad_() for x in (o, r, k, r_k, v, g, weight, bias))
        out = fn(o_, r_, k_, rk_, v_, g_, w_, b_)
        grads = torch.autograd.grad(out, [o_, r_, k_, rk_, v_, g_, w_, b_], dy)
        return [out, *grads]

    ref = run(lambda *args: _gn_corr_ref(*args, eps, H))
    out = run(lambda o_, r_, k_, rk_, v_, g_, w_, b_: gn_corr_rwkv7(o_, r_, k_, rk_, v_, g_, w_, b_, eps))
    names = ['out', 'd_o', 'd_r', 'd_k', 'd_r_k', 'd_v', 'd_g', 'd_w', 'd_b']
    ratio = 0.005 if dtype == torch.bfloat16 else 0.002
    for name, r, t in zip(names, ref, out):
        assert_close(name, r, t, ratio)


@requires_tilelang_route
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype', 'use_state'),
    [
        (1, 1, 32, 64, torch.bfloat16, True),    # decode
        (32, 1, 32, 64, torch.bfloat16, True),   # decode batch
        (4, 1, 32, 64, torch.float16, False),
        (4, 7, 8, 64, torch.bfloat16, True),     # short prefill
    ],
)
def test_decode_e2e_tilelang_parity(
    monkeypatch, B: int, T: int, H: int, D: int, dtype: torch.dtype, use_state: bool,
):
    from fla.ops.rwkv7.backends.tilelang.decode import fused_recurrent_rwkv7_e2e

    torch.manual_seed(42)
    r = torch.randn(B, T, H, D, device=device, dtype=dtype)
    w = -torch.rand(B, T, H, D, device=device, dtype=dtype) * 0.6
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    kk = F.normalize(torch.randn(B, T, H, D, device=device).float(), dim=-1).to(dtype)
    a = torch.rand(B, T, H, D, device=device, dtype=dtype)
    h0 = torch.randn(B, H, D, D, device=device, dtype=torch.float32) * 0.1 if use_state else None

    # reference: Triton default via dispatch with TileLang forced off
    monkeypatch.setenv('FLA_TILELANG', '0')
    with torch.no_grad():
        o_ref, ht_ref = fused_mul_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a, scale=1.0,
            initial_state=h0, output_final_state=True,
        )

    h0_tl = h0.clone() if use_state else None
    with torch.no_grad():
        o_tl, ht_tl = fused_recurrent_rwkv7_e2e(
            r=r, w=w, k=k, v=v, a=-kk, b=kk * a, scale=1.0,
            initial_state=h0_tl, output_final_state=True,
        )
    if use_state:
        # in-place decode: the returned state is the input buffer
        assert ht_tl is h0_tl
    assert_close('o', o_ref, o_tl, 0.005)
    assert_close('ht', ht_ref, ht_tl, 0.005)


@requires_tilelang_route
def test_token_shift_tilelang_decode_inplace_cache():
    from fla.ops.rwkv7.backends.tilelang.token_shift import token_shift_tilelang

    torch.manual_seed(42)
    B, D = 4, 2048
    dtype = torch.bfloat16
    x = torch.randn(B, 1, D, device=device, dtype=dtype)
    cache = torch.randn(B, D, device=device, dtype=dtype)
    cache0 = cache.clone()

    y, cache_out = token_shift_tilelang(x, None, cache=cache, output_cache=True)
    # decode path updates the cache buffer in place and returns it
    assert cache_out is cache
    assert_close('y', cache0 - x.squeeze(1), y.squeeze(1), 0.001)
    assert_close('cache', x.squeeze(1), cache, 0.001)


@requires_tilelang_route
def test_e2e_layer_fullgraph_smoke(monkeypatch):
    """The e2e-patched layer must compile fullgraph in both chunk and decode
    modes, and match its own eager numerics."""
    import fla.layers.rwkv7 as layer_mod
    from fla.models.utils import Cache
    from fla.ops.rwkv7.backends.tilelang.e2e import patch_e2e_namespace

    names = [
        "token_shift", "fused_addcmul_rwkv7", "fused_k_rwkv7",
        "gate_output_correction", "chunk_rwkv7", "kk_pre_rwkv7",
        "gn_corr_rwkv7", "fused_recurrent_rwkv7_e2e", "_TILELANG_E2E",
    ]
    for name in names:
        # register current value (or absence) for undo; absent names undo to deletion
        monkeypatch.setitem(layer_mod.__dict__, name, layer_mod.__dict__.get(name))
    monkeypatch.setenv('FLA_RWKV7_TILELANG_E2E', '1')
    patch_e2e_namespace(layer_mod.__dict__)
    assert layer_mod.__dict__['_TILELANG_E2E'] is True

    from fla.layers.rwkv7 import RWKV7Attention
    torch.manual_seed(42)
    dtype = torch.bfloat16
    layer = RWKV7Attention(
        mode='chunk', hidden_size=512, head_dim=64,
        layer_idx=0, num_hidden_layers=4, fuse_norm=False,
    ).to(device).to(dtype)

    # chunk (training) path, fwd+bwd, fullgraph
    x = torch.randn(2, 64, 512, device=device, dtype=dtype, requires_grad=True)

    def fwd(x_):
        return layer(x_)[0]

    cfwd = torch.compile(fwd, fullgraph=True)
    dy = torch.randn(2, 64, 512, device=device, dtype=dtype)

    out_e = fwd(x)
    ge, = torch.autograd.grad(out_e, x, dy)
    out_c = cfwd(x)
    gc, = torch.autograd.grad(out_c, x, dy)
    assert_close('train_out', out_e, out_c, 0.005)
    assert_close('train_dx', ge, gc, 0.005)

    # decode path with cache, fullgraph
    layer.eval()
    past_e, past_c = Cache(), Cache()
    xs = [torch.randn(2, 1, 512, device=device, dtype=dtype) for _ in range(3)]

    def dstep(x_, past):
        return layer(
            x_, attention_mask=None, past_key_values=past,
            use_cache=True, output_attentions=False, v_first=None,
        )[0]

    cdstep = torch.compile(dstep, fullgraph=True)
    with torch.no_grad():
        outs_e = [dstep(x_, past_e) for x_ in xs]
        outs_c = [cdstep(x_, past_c) for x_ in xs]
    for i, (r, t) in enumerate(zip(outs_e, outs_c)):
        assert_close(f'decode_step{i}', r, t, 0.005)
