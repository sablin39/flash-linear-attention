# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""torch.compile-safe wrapper for the TileLang RWKV7 fused recurrent decode.

The dispatched entry points are torch.compiler.disable'd, so a fullgraph decode
step must reach the TileLang kernel through an opaque custom op instead. This
op mirrors the inference-only contract of fla's fused_recurrent (which raises
on backward) and takes DPLR-formed (a, b) inputs — under the e2e path those
come straight from kk_pre_rwkv7 as (neg_kk, kka), so no extra elementwise
passes are needed.

The op is capture-safe by construction — no data-dependent shapes, and small
layout tensors are allocated per call so nothing persistent can leak into the
cudagraph-private pool. The wrapper
below implements the in-place state update for serving loops: given an fp32
contiguous `initial_state` plus `output_final_state=True`, it copies the op's
final-state output back into that buffer (the HF static-cache pattern), so a
decode loop under cudagraph trees reads and writes a static state address
every step while the op itself stays functional.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fla.ops.rwkv7.backends.tilelang.fused_recurrent import (
    fused_recurrent_dplr_delta_rule_fwd,
)


@torch.library.custom_op(
    "fla::fused_recurrent_rwkv7",
    mutates_args=(),
    schema=(
        "(Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, float scale, "
        "Tensor? initial_state, bool output_final_state, Tensor? cu_seqlens) -> (Tensor, Tensor?)"
    ),
)
def _fused_recurrent_rwkv7_op(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    scale: float,
    initial_state: Tensor | None,
    output_final_state: bool,
    cu_seqlens: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    return fused_recurrent_dplr_delta_rule_fwd(
        q=r, k=k, v=v, a=a, b=b, gk=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=False,
        cu_seqlens=cu_seqlens,
        rwkv7_negkk=False,
    )


@_fused_recurrent_rwkv7_op.register_fake
def _fused_recurrent_rwkv7_fake(
    r, w, k, v, a, b, scale, initial_state, output_final_state, cu_seqlens,
):
    B, T_, H, K = k.shape
    V = v.shape[-1]
    o = r.new_empty((B, T_, H, V))
    ht = None
    if output_final_state:
        nseq = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B
        ht = k.new_empty((nseq, H, K, V), dtype=torch.float32)
    return o, ht


def fused_recurrent_rwkv7_e2e(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RWKV7 fused recurrent forward for the e2e compile path.

    Args:
        r, w, k: (B, T, H, K) — receptance, log-decay, key (post kk_pre).
        v: (B, T, H, V).
        a, b: (B, T, H, K) — DPLR-formed coefficients (neg_kk, kka from
            kk_pre_rwkv7).
        Other args mirror fla.ops.rwkv7.fused_mul_recurrent_rwkv7.

    Forward-only, mirroring fla's fused_recurrent contract.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (r, w, k, v, a, b)):
        raise RuntimeError(
            "fused_recurrent_rwkv7_e2e is forward-only (inference); "
            "use chunk_rwkv7 for training.",
        )
    o, ht = _fused_recurrent_rwkv7_op(
        r, w, k, v, a, b, float(scale), initial_state, output_final_state, cu_seqlens,
    )
    if (initial_state is not None and ht is not None
            and initial_state.dtype == torch.float32 and initial_state.is_contiguous()):
        # decode serving loops: copy the new state back into the caller's
        # persistent buffer (the HF static-cache pattern), so its address
        # stays static across steps under cudagraph capture; the copy is a
        # plain aten op, so the op itself stays functional
        initial_state.copy_(ht)
        ht = initial_state
    elif ht is not None and not torch.compiler.is_compiling():
        # freshly allocated recurrent state: mark it static-address so that
        # when a serving loop feeds it back as `initial_state` under
        # torch.compile, cudagraph trees permits the in-place copy above
        # instead of skipping cudagraphs for a mutated input
        torch._dynamo.mark_static_address(ht)
    return o, ht
