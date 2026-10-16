# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR chunk-recurrent state forward.

Per chunk:
    for each BC sub-chunk:
        v2  = w @ b_h + u           # write to v_new
        b_hc += kg @ v + bg @ v2
    b_h = exp2(g_last) * b_h + b_hc

Adapted from kda/chunk_delta_h_fwd.py (which does the equivalent single-GEMM
recurrence). Differences from KDA:
- KDA: `v_new = u - w@h; b_h += K^T @ v_new` (one outer-product)
- DPLR: `v2 = w@h + u; b_h += kg^T @ v + bg^T @ v2` (two outer-products, no sign flip)
- DPLR sub-chunks the BC inner loop to bound shared-memory pressure
- DPLR applies the decay AFTER accumulating sub-chunks; KDA applies before
"""

import tilelang
import tilelang.language as T
import torch

from fla.utils import get_device_capability, get_device_smem_optin

from .layout import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout
from .schedules import dtype_nbytes


def _chunk_h_fwd_config(K: int, V: int, device_index: int) -> dict[str, int]:
    cap_major = get_device_capability(device_index)[0]
    if cap_major == 9 and K <= 128:
        return {"BV": 64 if V >= 64 else 32 if V >= 32 else 16, "threads": 256}
    elif cap_major == 10 and K <= 128:
        # measured on cc10.3 (L20D): the cc9 tile shape is bit-exact vs the
        # generic fallback and 15-32% faster across BT 16/32/64
        return {"BV": 64 if V >= 64 else 32 if V >= 32 else 16, "threads": 256}
    elif cap_major == 8:
        return {"BV": 32 if V >= 32 else 16, "threads": 128}
    return {"BV": 16, "threads": 64}


def _chunk_h_smem_bytes(
    BT: int, K: int, BV: int, in_dtype: str, pipelined: bool, stage_bg: bool = True,
) -> int:
    """Shared-memory footprint of _chunk_dplr_fwd_h_kernel.

    Static tiles: the (K, BV) in_dtype state cast tile, the fp32 (BT, BV) v2
    operand tile, the fp32 (BT, K) bg tile, and the fp32 (K,) last-gate row.
    Operand tiles (kg/w (BT, K), v/u (BT, BV)) are double-buffered when
    pipelined; the unpipelined variant instead carries the two fp32 (K, BV)
    b_hc staging tiles. The pipelined variant drops the kg staging tile by
    keeping that product in a fragment, and carries the bg one only when
    stage_bg is set (required on sm90; see the kernel body).
    """
    elem = dtype_nbytes(in_dtype)
    base = elem * K * BV + 4 * (BT * BV + BT * K + K)
    n_bufs = 2 if pipelined else 1
    base += n_bufs * elem * (2 * BT * K + 2 * BT * BV)
    if pipelined:
        if stage_bg:
            base += 4 * K * BV
    else:
        base += 4 * 2 * K * BV
    return base


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _chunk_dplr_fwd_h_kernel(
    H, K, V, BT, BC,
    in_dtype, state_dtype,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
    BV: int = 32,
    threads: int = 128,
    PIPELINED: bool = False,
    STAGE_BG: bool = True,
):
    if PIPELINED:
        # The double-buffered chunk loop prefetches whole BT-row tiles, so
        # the sub-chunk loop must be trivial.
        assert BC == BT, "PIPELINED requires BC == BT"
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks, n_h0, n_ht = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_h0, n_ht"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_fwd_h_tl(
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        u: T.Tensor((n_tokens, H, V), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        h0: T.Tensor((n_h0, H, K, V), acc_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_offsets: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        h: T.Tensor((n_chunks, H, K, V), in_dtype),
        v_new: T.Tensor((n_tokens, H, V), in_dtype),
        ht: T.Tensor((n_ht, H, K, V), state_dtype),
    ):
        with T.Kernel(T.ceildiv(V, BV), n_seqs, H, threads=threads) as (i_v, i_n, i_h):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            boh = chunk_offsets[i_n]
            n_chunks = chunk_offsets[i_n + 1] - boh

            b_h = T.alloc_fragment((K, BV), acc_dtype)
            b_h_shared = T.alloc_shared((K, BV), in_dtype)
            b_hc = T.alloc_fragment((K, BV), acc_dtype)
            v2_frag = T.alloc_fragment((BC, BV), acc_dtype)
            v2_shared = T.alloc_shared((BC, BV), acc_dtype)
            bg_shared = T.alloc_shared((BC, K), acc_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)
            if PIPELINED:
                n_bufs = 2
                kg_shared = T.alloc_shared((n_bufs, BC, K), in_dtype)
                w_shared = T.alloc_shared((n_bufs, BC, K), in_dtype)
                v_shared = T.alloc_shared((n_bufs, BC, BV), in_dtype)
                u_shared = T.alloc_shared((n_bufs, BC, BV), in_dtype)
                if STAGE_BG:
                    # The cc90-proven split from the fused h+o kernel:
                    # writing one fragment from both the bf16 and fp32 GEMMs
                    # infers conflicting accumulator layouts on sm90 and
                    # fails lowering there, so the fp32 bg product stages
                    # through shared memory. cc10 lowers the direct
                    # double-GEMM accumulation and measures faster with it.
                    b_hc_kg = T.alloc_fragment((K, BV), acc_dtype)
                    b_hc_bg = T.alloc_fragment((K, BV), acc_dtype)
                    b_hc_bg_shared = T.alloc_shared((K, BV), acc_dtype)
            else:
                b_hc_kg = T.alloc_fragment((K, BV), acc_dtype)
                b_hc_bg = T.alloc_fragment((K, BV), acc_dtype)
                b_hc_kg_shared = T.alloc_shared((K, BV), acc_dtype)
                b_hc_bg_shared = T.alloc_shared((K, BV), acc_dtype)
                kg_shared = T.alloc_shared((BC, K), in_dtype)
                w_shared = T.alloc_shared((BC, K), in_dtype)
                v_shared = T.alloc_shared((BC, BV), in_dtype)
                u_shared = T.alloc_shared((BC, BV), in_dtype)

            # Init state.
            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        b_h[k_idx, vv] = h0[i_n, i_h, k_idx, g_v]
                    else:
                        b_h[k_idx, vv] = 0.0
            else:
                for k_idx, vv in T.Parallel(K, BV):
                    b_h[k_idx, vv] = 0.0

            if PIPELINED:
                # Prologue: fill buffer 0 with chunk 0's operand tiles.
                if bos + BT <= eos:
                    T.copy(kg[bos: bos + BT, i_h, 0:K], kg_shared[0, :, :])
                    T.copy(w[bos: bos + BT, i_h, 0:K], w_shared[0, :, :])
                    T.copy(v[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], v_shared[0, :, :])
                    T.copy(u[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], u_shared[0, :, :])
                else:
                    for c, k_idx in T.Parallel(BT, K):
                        if bos + c < eos:
                            kg_shared[0, c, k_idx] = kg[bos + c, i_h, k_idx]
                            w_shared[0, c, k_idx] = w[bos + c, i_h, k_idx]
                        else:
                            kg_shared[0, c, k_idx] = T.Cast(in_dtype, 0.0)
                            w_shared[0, c, k_idx] = T.Cast(in_dtype, 0.0)
                    for c, vv in T.Parallel(BT, BV):
                        g_v = i_v * BV + vv
                        if (bos + c < eos) and (g_v < V):
                            v_shared[0, c, vv] = v[bos + c, i_h, g_v]
                            u_shared[0, c, vv] = u[bos + c, i_h, g_v]
                        else:
                            v_shared[0, c, vv] = T.Cast(in_dtype, 0.0)
                            u_shared[0, c, vv] = T.Cast(in_dtype, 0.0)

                for i_t in T.serial(n_chunks):
                    chunk_bos = bos + i_t * BT
                    cur = i_t % 2

                    # FLA computes w @ h with h rounded to the input dtype, while the
                    # recurrent state itself remains fp32 across chunks.
                    T.copy(b_h, b_h_shared)
                    for k_idx, vv in T.Parallel(K, BV):
                        g_v = i_v * BV + vv
                        if g_v < V:
                            h[boh + i_t, i_h, k_idx, g_v] = b_h_shared[k_idx, vv]

                    # The idle buffer's previous readers must finish before
                    # the cp.async below overwrites it.
                    T.sync_threads()
                    if i_t + 1 < n_chunks:
                        nxt = (i_t + 1) % 2
                        nxt_bos = chunk_bos + BT
                        if nxt_bos + BT <= eos:
                            # Full tiles only: 4 operand tiles, one commit
                            # group per async_copy.
                            T.async_copy(kg[nxt_bos: nxt_bos + BT, i_h, 0:K], kg_shared[nxt, :, :])
                            T.async_copy(w[nxt_bos: nxt_bos + BT, i_h, 0:K], w_shared[nxt, :, :])
                            T.async_copy(v[nxt_bos: nxt_bos + BT, i_h, i_v * BV: i_v * BV + BV],
                                         v_shared[nxt, :, :])
                            T.async_copy(u[nxt_bos: nxt_bos + BT, i_h, i_v * BV: i_v * BV + BV],
                                         u_shared[nxt, :, :])
                            # Leave the 4 just issued groups pending, wait for
                            # the 4 issued for the current buffer last round.
                            T.ptx_wait_group(4)
                        else:
                            for c, k_idx in T.Parallel(BT, K):
                                if nxt_bos + c < eos:
                                    kg_shared[nxt, c, k_idx] = kg[nxt_bos + c, i_h, k_idx]
                                    w_shared[nxt, c, k_idx] = w[nxt_bos + c, i_h, k_idx]
                                else:
                                    kg_shared[nxt, c, k_idx] = T.Cast(in_dtype, 0.0)
                                    w_shared[nxt, c, k_idx] = T.Cast(in_dtype, 0.0)
                            for c, vv in T.Parallel(BT, BV):
                                g_v = i_v * BV + vv
                                if (nxt_bos + c < eos) and (g_v < V):
                                    v_shared[nxt, c, vv] = v[nxt_bos + c, i_h, g_v]
                                    u_shared[nxt, c, vv] = u[nxt_bos + c, i_h, g_v]
                                else:
                                    v_shared[nxt, c, vv] = T.Cast(in_dtype, 0.0)
                                    u_shared[nxt, c, vv] = T.Cast(in_dtype, 0.0)
                            T.ptx_wait_group(0)
                    else:
                        T.ptx_wait_group(0)

                    for c, k_idx in T.Parallel(BT, K):
                        t = chunk_bos + c
                        if t < eos:
                            bg_shared[c, k_idx] = T.Cast(acc_dtype, bg[t, i_h, k_idx])
                        else:
                            bg_shared[c, k_idx] = 0.0

                    T.gemm(w_shared[cur, :, :], b_h_shared, v2_frag, clear_accum=True)
                    for c, vv in T.Parallel(BT, BV):
                        v2_frag[c, vv] = T.ieee_add(v2_frag[c, vv], T.Cast(acc_dtype, u_shared[cur, c, vv]))
                    T.copy(v2_frag, v2_shared)

                    for c, vv in T.Parallel(BT, BV):
                        g_v = i_v * BV + vv
                        t = chunk_bos + c
                        if (t < eos) and (g_v < V):
                            v_new[t, i_h, g_v] = T.Cast(in_dtype, v2_frag[c, vv])

                    # b_hc = kg^T @ v + bg^T @ v2.
                    if STAGE_BG:
                        # Same two rounded adds as the unpipelined path at
                        # BC == BT (bit-identical), but only the bg product
                        # pays the smem round-trip.
                        T.gemm(
                            kg_shared[cur, :, :],
                            v_shared[cur, :, :],
                            b_hc_kg,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        T.gemm(
                            bg_shared,
                            v2_shared,
                            b_hc_bg,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        T.copy(b_hc_bg, b_hc_bg_shared)
                        for k_idx, vv in T.Parallel(K, BV):
                            b_hc[k_idx, vv] = T.ieee_add(b_hc_kg[k_idx, vv], b_hc_bg_shared[k_idx, vv])
                    else:
                        T.clear(b_hc)
                        T.gemm(
                            kg_shared[cur, :, :],
                            v_shared[cur, :, :],
                            b_hc,
                            transpose_A=True,
                        )
                        T.gemm(
                            bg_shared,
                            v2_shared,
                            b_hc,
                            transpose_A=True,
                        )

                    # Apply decay and accumulate. Clamp last_idx to <= eos-1.
                    last_idx = T.min(chunk_bos + BT - 1, eos - 1)
                    for k_idx in T.Parallel(K):
                        gk_last_shared[k_idx] = gk[last_idx, i_h, k_idx]
                    # Match FLA's two-step update order:
                    #   b_h *= exp2(g_last)
                    #   b_h += b_hc
                    for k_idx, vv in T.Parallel(K, BV):
                        b_h[k_idx, vv] = T.ieee_mul(T.exp2(gk_last_shared[k_idx]), b_h[k_idx, vv])
                    for k_idx, vv in T.Parallel(K, BV):
                        b_h[k_idx, vv] = T.ieee_add(b_h[k_idx, vv], b_hc[k_idx, vv])
            else:
                # Per-chunk loop.
                for i_t in T.serial(n_chunks):
                    # FLA computes w @ h with h rounded to the input dtype, while the
                    # recurrent state itself remains fp32 across chunks.
                    T.copy(b_h, b_h_shared)
                    for k_idx, vv in T.Parallel(K, BV):
                        g_v = i_v * BV + vv
                        if g_v < V:
                            h[boh + i_t, i_h, k_idx, g_v] = b_h_shared[k_idx, vv]

                    T.clear(b_hc)

                    # Sub-chunk loop.
                    NC = T.ceildiv(BT, BC)
                    for i_c in T.serial(NC):
                        t_off = bos + i_t * BT + i_c * BC

                        # Load tiles with zero-padding for partial varlen chunks.
                        for c, k_idx in T.Parallel(BC, K):
                            if t_off + c < eos:
                                kg_shared[c, k_idx] = kg[t_off + c, i_h, k_idx]
                                bg_shared[c, k_idx] = T.Cast(acc_dtype, bg[t_off + c, i_h, k_idx])
                                w_shared[c, k_idx] = w[t_off + c, i_h, k_idx]
                            else:
                                kg_shared[c, k_idx] = T.Cast(in_dtype, 0.0)
                                bg_shared[c, k_idx] = 0.0
                                w_shared[c, k_idx] = T.Cast(in_dtype, 0.0)
                        for c, vv in T.Parallel(BC, BV):
                            g_v = i_v * BV + vv
                            if (t_off + c < eos) and (g_v < V):
                                v_shared[c, vv] = v[t_off + c, i_h, g_v]
                                u_shared[c, vv] = u[t_off + c, i_h, g_v]
                            else:
                                v_shared[c, vv] = T.Cast(in_dtype, 0.0)
                                u_shared[c, vv] = T.Cast(in_dtype, 0.0)

                        T.gemm(w_shared, b_h_shared, v2_frag, clear_accum=True)
                        for c, vv in T.Parallel(BC, BV):
                            v2_frag[c, vv] = T.ieee_add(v2_frag[c, vv], T.Cast(acc_dtype, u_shared[c, vv]))
                        T.copy(v2_frag, v2_shared)

                        for c, vv in T.Parallel(BC, BV):
                            g_v = i_v * BV + vv
                            if (t_off + c < eos) and (g_v < V):
                                v_new[t_off + c, i_h, g_v] = T.Cast(in_dtype, v2_frag[c, vv])

                        # b_hc += kg^T @ v + bg^T @ v2.  The second product stays in
                        # fp32 because FLA casts bg to fp32 and consumes fp32 v2 here.
                        T.gemm(
                            kg_shared,
                            v_shared,
                            b_hc_kg,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        T.copy(b_hc_kg, b_hc_kg_shared)
                        T.gemm(
                            bg_shared,
                            v2_shared,
                            b_hc_bg,
                            transpose_A=True,
                            clear_accum=True,
                        )
                        T.copy(b_hc_bg, b_hc_bg_shared)
                        for k_idx, vv in T.Parallel(K, BV):
                            b_hc[k_idx, vv] = T.ieee_add(
                                T.ieee_add(b_hc[k_idx, vv], b_hc_kg_shared[k_idx, vv]),
                                b_hc_bg_shared[k_idx, vv],
                            )

                    # Apply decay and accumulate. Clamp last_idx to <= eos-1.
                    last_idx = T.min(bos + (i_t + 1) * BT - 1, eos - 1)
                    for k_idx in T.Parallel(K):
                        gk_last_shared[k_idx] = gk[last_idx, i_h, k_idx]
                    # Match FLA's two-step update order:
                    #   b_h *= exp2(g_last)
                    #   b_h += b_hc
                    # Keeping these as separate statements avoids contracting the
                    # recurrent state update into a different fp32 expression.
                    for k_idx, vv in T.Parallel(K, BV):
                        b_h[k_idx, vv] = T.ieee_mul(T.exp2(gk_last_shared[k_idx]), b_h[k_idx, vv])
                    for k_idx, vv in T.Parallel(K, BV):
                        b_h[k_idx, vv] = T.ieee_add(b_h[k_idx, vv], b_hc[k_idx, vv])

            # Store final state.
            if STORE_FINAL_STATE:
                for k_idx, vv in T.Parallel(K, BV):
                    g_v = i_v * BV + vv
                    if g_v < V:
                        ht[i_n, i_h, k_idx, g_v] = T.Cast(state_dtype, b_h[k_idx, vv])

    return chunk_dplr_fwd_h_tl


def chunk_dplr_fwd_h(
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,        # = gi from cumsum, fp32
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_, H, K = kg.shape
    V = v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, kg.device)
    chunk_rows = layout.chunk_indices.shape[0]
    active_nseq = layout.cu_seqlens.shape[0] - 1
    token_rows = B * T_

    # Match FLA's Hopper path for the target H800 shape: K=64 uses the whole
    # BT=64 chunk in one H recurrence sub-block. Splitting into two BC=32
    # halves changes the fp32 accumulation order and was enough to create rare
    # one-ULP bf16 output spikes at long varlen sequence lengths.
    cap_major = get_device_capability(kg.device.index)[0]
    if cap_major in (9, 10):
        # cc10.3 (L20D) also measures 41-49% faster with the unsplit
        # sub-block; the accumulation reorder is within suite tolerance
        BC = min(BT, 64 if K <= 128 else 32)
    elif cap_major == 8:
        BC = min(BT, 32)
    else:
        BC = min(BT, 16)

    in_dtype = str(kg.dtype).split(".")[-1]
    state_dtype = "float32"
    use_h0 = initial_state is not None
    store_ht = output_final_state
    n_ht = active_nseq if store_ht else 1

    if use_h0:
        h0 = initial_state
    else:
        h0 = torch.empty((1, H, K, V), dtype=torch.float32, device=kg.device)
    if h0.dtype != torch.float32:
        h0 = h0.to(torch.float32)

    kg_f = kg.reshape(token_rows, H, K).contiguous()
    v_f = v.reshape(token_rows, H, V).contiguous()
    w_f = w.reshape(token_rows, H, K).contiguous()
    bg_f = bg.reshape(token_rows, H, K).contiguous()
    u_f = u.reshape(token_rows, H, V).contiguous()
    gk_f = gk.reshape(token_rows, H, K).contiguous()

    config = _chunk_h_fwd_config(K, V, kg.device.index)
    # Double-buffered cp.async chunk loop: pays wherever it launches (the
    # unpipelined walk exposes every operand tile's load latency on the
    # serial chain). Restricted to BC == BT (trivial sub-chunk loop), where
    # it is bit-identical to the unpipelined schedule, and BT >= 64 for
    # now; the tuned small-chunk paths keep their validated schedule.
    # sm90 must stage the bg product through smem (see the kernel body);
    # cc10 lowers the direct accumulation and measures ~5% faster fwd.
    stage_bg = cap_major == 9
    pipelined = (
        BC == BT
        and BT >= 64
        and cap_major in (9, 10)
        and _chunk_h_smem_bytes(BT, K, config["BV"], in_dtype, pipelined=True, stage_bg=stage_bg)
        <= get_device_smem_optin(kg.device.index)
    )
    kernel = _chunk_dplr_fwd_h_kernel(
        H, K, V, BT, BC,
        in_dtype, state_dtype, use_h0, store_ht,
        PIPELINED=pipelined,
        STAGE_BG=stage_bg,
        **config,
    )

    h_flat = torch.empty((chunk_rows, H, K, V), dtype=kg.dtype, device=kg.device)
    v_new_flat = torch.empty((token_rows, H, V), dtype=v.dtype, device=v.device)
    ht = torch.empty((n_ht, H, K, V), dtype=torch.float32, device=kg.device)
    kernel(
        kg_f, v_f, w_f, bg_f, u_f, gk_f, h0, layout.cu_seqlens,
        layout.chunk_offsets, layout.chunk_indices, h_flat, v_new_flat, ht,
    )

    # FLA returns (B, NT, H, K, V); varlen packs all sequences' chunks flat.
    h_out = (
        h_flat.view(1, chunk_rows, H, K, V)
        if is_varlen
        else h_flat.view(B, chunk_rows // B, H, K, V)
    )
    v_new = v_new_flat.view(B, T_, H, V)
    final_state = ht if store_ht else None
    return h_out, v_new, final_state
