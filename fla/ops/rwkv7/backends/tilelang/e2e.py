# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in end-to-end TileLang path for torch.compile fullgraph training/decode.

The default call path is not fullgraph-safe by design: the backend dispatch
wrapper (fla.ops.backends.dispatch) and fla.modules.token_shift are
torch.compiler.disable'd, so every op call forces a graph break. With
FLA_RWKV7_TILELANG_E2E=1, `patch_e2e_namespace` rebinds the op entry points in
fla.layers.rwkv7 / fla.models.rwkv7.modeling_rwkv7 to the TileLang custom ops
directly, which are opaque (and therefore fullgraph-safe) to dynamo in both
forward and backward.

Scope: the whole RWKV7Attention forward. The chunk path rebases onto fused
kernels (kk_pre_rwkv7 folds the k-update + kk normalization + DPLR a/b
materialization into one op; gn_corr_rwkv7 folds GroupNorm + gate output
correction), so both fuse_norm modes share one compile-safe route. The T==1
decode branch is served by the inference-only fused_recurrent_rwkv7_e2e custom
op, which is cudagraph-capture-safe.

cudagraph decode (mode="reduce-overhead") contract: the conv/recurrent state
buffers are updated in place via static-address tensors, so a serving loop
must (1) run the prefill and the first decode step eagerly — this allocates
the states in the regular allocator pool and lets the wrappers mark them
static-address (a from-scratch compiled loop still runs correctly, just
without cudagraphs) — and (2) keep one Cache alive for the whole compiled
session; allocating fresh state buffers mid-session forces a costly cudagraph
re-record. This mirrors HF's StaticCache serving pattern.

Training with gradient checkpointing: every TileLang custom op returns
exactly one tensor, so no raw output tuple can escape cudagraph trees' flat
output tracking across an inductor cudagraph partition boundary ("tensor(s)
in the cudagraph pool not tracked as outputs"). The DPLR chunk ops keep
their cudagraph_unsafe tags, so training graphs always contain
cudagraph-unsafe ops — and torch 2.13's partition codegen can corrupt the
wrapper around such boundaries at scale (phantom buffer names in the
generated call glue; 24 layers + non-reentrant checkpointing +
mode="max-autotune"; disabling the partition reorder passes does not help).
patch_e2e_namespace therefore forces
torch._inductor.config.graph_partition=False unless the user set it
explicitly: graphs with unsafe ops then run outside cudagraphs under every
mode instead of being miscompiled, which for training was already the
faster configuration anyway (max-autotune-no-cudagraphs). The trade-off is
decode: without partitioning, a decode graph containing any dynamic shape
is re-recorded per size instead of capturing its static layer stack once
(31 vs 281 tok/s in the 1.5B B1 decode bench). Decode-only serving should
set torch._inductor.config.graph_partition=True explicitly — the pin
respects that override — but not in the same process as checkpointed
max-autotune training, which would be miscompiled.
"""

import os
import warnings

import torch

ENV_VAR = "FLA_RWKV7_TILELANG_E2E"


def patch_e2e_namespace(ns: dict) -> None:
    """Rebind RWKV7 op names in a module's globals() to the TileLang ops.

    No-op unless FLA_RWKV7_TILELANG_E2E=1 and the TileLang backend is usable.
    Only names already present in the namespace are touched.
    """
    if os.environ.get(ENV_VAR) != "1":
        return
    from fla.ops.rwkv7.backends.tilelang import RWKV7TileLangBackend
    if not RWKV7TileLangBackend.is_available():
        return

    # torch 2.13's inductor graph-partition codegen can corrupt the wrapper
    # around cudagraph-unsafe ops at scale (phantom buffer names in the
    # generated call glue; seen with 24 layers + non-reentrant checkpointing
    # + mode="max-autotune"; not avoided by disabling the partition reorder
    # passes). The DPLR chunk ops keep their cudagraph_unsafe tags, so
    # training graphs always contain unsafe ops; with partitioning disabled,
    # inductor excludes those graphs from cudagraphs instead of miscompiling
    # them — which for training was already the faster configuration. The
    # cost is decode: an unpartitioned decode graph with any dynamic shape
    # gets its cudagraph re-recorded per size, so decode-only users who want
    # capture should set the config back explicitly. An explicit user
    # override of the config is respected.
    import torch._inductor.config as _inductor_cfg
    if (
        "graph_partition" in getattr(_inductor_cfg, "_config", {})
        and _inductor_cfg.graph_partition
        and _inductor_cfg._is_default("graph_partition")
    ):
        _inductor_cfg.graph_partition = False
        warnings.warn(
            f"{ENV_VAR}=1: torch._inductor.config.graph_partition disabled — "
            "torch 2.13's cudagraph partition codegen can miscompile graphs "
            "with cudagraph-unsafe ops at scale. Training is unaffected; "
            "decode graphs with dynamic shapes lose cudagraph capture — "
            "set the config explicitly to override (safe for decode-only "
            "serving, unsafe combined with checkpointed training)."
        )

    from fla.ops.generalized_delta_rule.dplr.backends.tilelang.chunk import (
        chunk_dplr_delta_rule_tilelang,
    )
    from fla.ops.rwkv7.backends.tilelang.decode import (
        fused_recurrent_rwkv7_e2e,
    )
    from fla.ops.rwkv7.backends.tilelang.fused_addcmul import (
        fused_addcmul_rwkv7_tilelang,
    )
    from fla.ops.rwkv7.backends.tilelang.fused_gn_corr import (
        gn_corr_rwkv7,
    )
    from fla.ops.rwkv7.backends.tilelang.fused_k_update import (
        fused_k_rwkv7_tilelang,
    )
    from fla.ops.rwkv7.backends.tilelang.gate_output_correction import (
        gate_output_correction_tilelang,
    )
    from fla.ops.rwkv7.backends.tilelang.kk_pre import (
        kk_pre_rwkv7,
    )
    from fla.ops.rwkv7.backends.tilelang.token_shift import (
        token_shift_tilelang,
    )

    def chunk_rwkv7_tilelang(r, w, k, v, a, b, scale=1.0, initial_state=None,
                             output_final_state=False, cu_seqlens=None, **kwargs):
        o, ht = chunk_dplr_delta_rule_tilelang(
            q=r, k=k, v=v, a=a, b=b, gk=w, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, **kwargs,
        )
        if ht is not None and not torch.compiler.is_compiling():
            # a prefill-produced recurrent state feeds the decode loop as a
            # persistent buffer; mark it static-address so cudagraph trees
            # permits the decode wrapper's in-place update (see decode.py)
            torch._dynamo.mark_static_address(ht)
        return o, ht

    mapping = {
        "token_shift": token_shift_tilelang,
        "fused_addcmul_rwkv7": fused_addcmul_rwkv7_tilelang,
        "fused_k_rwkv7": fused_k_rwkv7_tilelang,
        "gate_output_correction": gate_output_correction_tilelang,
        "chunk_rwkv7": chunk_rwkv7_tilelang,
    }
    for name, fn in mapping.items():
        if name in ns:
            ns[name] = fn

    # Entry points only referenced from the layer's `_TILELANG_E2E` branches;
    # they are not part of the default namespace, so inject them outright.
    ns["kk_pre_rwkv7"] = kk_pre_rwkv7
    ns["gn_corr_rwkv7"] = gn_corr_rwkv7
    ns["fused_recurrent_rwkv7_e2e"] = fused_recurrent_rwkv7_e2e
    ns["_TILELANG_E2E"] = True
