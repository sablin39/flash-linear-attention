# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""TileLang backend for RWKV7 operations."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend
from fla.utils import find_spec_cached, has_usable_nvcc

_TILELANG_AVAILABLE = find_spec_cached("tilelang") is not None


class RWKV7TileLangBackend(BaseBackend):

    backend_type = "tilelang"
    package_name = "tilelang"
    env_var = "FLA_TILELANG"

    @classmethod
    def is_available(cls) -> bool:
        return _TILELANG_AVAILABLE and has_usable_nvcc()

    def fused_addcmul_rwkv7_verifier(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        xr: torch.Tensor,
        xw: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        xa: torch.Tensor,
        xg: torch.Tensor | None = None,
    ) -> tuple[bool, str | None]:
        if not hidden_states.is_cuda:
            return False, "TileLang backend is CUDA-only; fall back to Triton"
        if hidden_states.shape[1] == 1:
            return False, "TileLang backend skips the T == 1 decode path; fall back to the default torch path"
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return False, f"TileLang backend does not support dtype {hidden_states.dtype}; fall back to Triton"
        mix_tensors = (delta, xr, xw, xk, xv, xa) + ((xg,) if xg is not None else ())
        if not all(t.dtype == hidden_states.dtype for t in mix_tensors):
            return False, "TileLang backend requires delta/xr/xw/xk/xv/xa/xg dtypes to match hidden_states; fall back to Triton"
        return True, None

    def fused_addcmul_rwkv7(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        xr: torch.Tensor,
        xw: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        xa: torch.Tensor,
        xg: torch.Tensor | None = None,
    ):
        from fla.ops.rwkv7.backends.tilelang.fused_addcmul import fused_addcmul_rwkv7_tilelang
        return fused_addcmul_rwkv7_tilelang(hidden_states, delta, xr, xw, xk, xv, xa, xg)

    def fused_k_rwkv7_verifier(
        self,
        k: torch.Tensor,
        a: torch.Tensor,
        ka: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
    ) -> tuple[bool, str | None]:
        if not k.is_cuda:
            return False, "TileLang backend is CUDA-only; fall back to Triton"
        if cu_seqlens is not None:
            return False, "TileLang backend does not support varlen (cu_seqlens); fall back to Triton"
        if k.shape[1] == 1:
            return False, "TileLang backend skips the T == 1 decode path; fall back to the default torch path"
        if k.dtype not in (torch.float16, torch.bfloat16):
            return False, f"TileLang backend does not support dtype {k.dtype}; fall back to Triton"
        if a.dtype != k.dtype or ka.dtype != k.dtype:
            return False, "TileLang backend requires a/ka dtypes to match k.dtype; fall back to Triton"
        return True, None

    def fused_k_rwkv7(
        self,
        k: torch.Tensor,
        a: torch.Tensor,
        ka: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        from fla.ops.rwkv7.backends.tilelang.fused_k_update import fused_k_rwkv7_tilelang
        return fused_k_rwkv7_tilelang(k, a, ka)

    def gate_output_correction_verifier(
        self,
        o: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        r_k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[bool, str | None]:
        if not o.is_cuda:
            return False, "TileLang backend is CUDA-only; fall back to Triton"
        if o.shape[1] == 1:
            return False, "TileLang backend skips the T == 1 decode path; fall back to the default torch path"
        if o.dtype not in (torch.float16, torch.bfloat16):
            return False, f"TileLang backend does not support dtype {o.dtype}; fall back to Triton"
        if not all(t.dtype == o.dtype for t in (r, k, r_k, v, g)):
            return False, "TileLang backend requires r/k/r_k/v/g dtypes to match o.dtype; fall back to Triton"
        num_heads, head_dim = r.shape[-2], r.shape[-1]
        if r_k.dim() != 2 or r_k.shape[0] != num_heads or r_k.shape[1] != head_dim:
            return False, (
                f"TileLang backend requires r_k of shape [H, D] (got {tuple(r_k.shape)} for "
                f"H={num_heads}, D={head_dim}); fall back to Triton"
            )
        if o.shape[-1] != num_heads * head_dim or g.shape != o.shape:
            return False, "TileLang backend requires o/g of shape [B, T, H * D]; fall back to Triton"
        if k.shape != r.shape or v.shape != r.shape:
            return False, "TileLang backend requires r/k/v shapes to match; fall back to Triton"
        return True, None

    def gate_output_correction(
        self,
        o: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        r_k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        from fla.ops.rwkv7.backends.tilelang.gate_output_correction import gate_output_correction_tilelang
        return gate_output_correction_tilelang(o, r, k, r_k, v, g)

    def fused_mul_recurrent_rwkv7_verifier(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if not r.is_cuda:
            return False, "TileLang backend is CUDA-only; fall back to Triton"
        if r.dtype not in (torch.float16, torch.bfloat16):
            return False, f"TileLang backend does not support dtype {r.dtype}; fall back to Triton"
        if not all(t.dtype == r.dtype for t in (w, k, v, kk, a)):
            return False, "TileLang backend requires w/k/v/kk/a dtypes to match r.dtype; fall back to Triton"
        if k.shape[-1] != v.shape[-1]:
            return False, (
                f"TileLang backend requires K == V (got K={k.shape[-1]}, V={v.shape[-1]}); "
                "fall back to Triton"
            )
        if k.shape[-1] not in (64, 128):
            return False, f"TileLang backend supports head dim 64 or 128 (got {k.shape[-1]}); fall back to Triton"
        if torch.is_grad_enabled() and any(t.requires_grad for t in (r, w, k, v, kk, a)):
            return False, "TileLang backend is forward-only; fall back to Triton when gradients are required"
        return True, None

    def fused_mul_recurrent_rwkv7(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.rwkv7.backends.tilelang.fused_recurrent import fused_mul_recurrent_rwkv7_tilelang
        return fused_mul_recurrent_rwkv7_tilelang(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
