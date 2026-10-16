# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Two-pass DPLR stream backward: serial dh scan + chunk-parallel consumers.

The fused stream backward (`chunk_stream_bwd.py`) couples the reverse `dh`
scan with the q/o-side consumer GEMMs on a `(n_seqs, H)` grid, which cannot
pipeline at K=V=128, BT=64 (291KB > 228KB optin) and starves small batches.
Pass 1 here runs the scan alone, materializing each chunk's pre-update state
as a global `(n_chunks, H, K, V)` `dhu` tensor, so it fits a double-buffered
cp.async schedule (197KB at BT=64 D128); pass 2 runs the consumer math with
one block per chunk.  Both passes mirror the fused kernel's per-chunk math
exactly, including its rounding points: pass 1 writes `dhu[i]` from the same
cast tile its own `dv2` GEMM reads, so pass 2 consumes bit-identical `dh`
operands to the fused kernel.
"""

import tilelang
import tilelang.language as T
import torch

from fla.utils import get_device_capability, get_device_smem_optin

from .layout import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout
from .schedules import stream_consumer_smem_bytes, stream_scan_smem_bytes


def stream_bwd_twopass_config_or_none(
    *,
    K: int,
    V: int,
    BT: int,
    in_dtype: str,
    device: torch.device,
) -> dict[str, int] | None:
    """Two-pass stream-backward launch config, or None to keep the fused stream.

    The split only pays off at K=V=128 on cc10 devices whose optin fits
    both passes. cc90 keeps the fused path on purpose: there the fused
    route's wins were measured against a collapsing Triton varlen path,
    and an unmeasured two-pass could forfeit them; cc120's sub-228KB cap
    keeps the fused path too (and every D64 case everywhere).  Threads
    follow the fused kernel's rule: 256 at BT>=32, 128 at BT=16 (the
    (K, V) fp32 state fragment spills with 64).
    """
    if K != 128 or V != 128:
        return None
    index = device.index or 0
    if get_device_capability(index)[0] != 10:
        return None
    smem_cap = get_device_smem_optin(index)
    num_stages = 2 if stream_scan_smem_bytes(K, V, BT, in_dtype, 2) <= smem_cap else 0
    if stream_scan_smem_bytes(K, V, BT, in_dtype, num_stages) > smem_cap:
        return None
    if stream_consumer_smem_bytes(K, V, BT, in_dtype) > smem_cap:
        return None
    return {"threads": 256 if BT >= 32 else 128, "num_stages": num_stages}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_bwd_scan_kernel(
    H, K, V, BT,
    in_dtype, state_dtype,
    USE_FINAL_STATE_GRADIENT: bool,
    USE_INITIAL_STATE: bool,
    threads: int = 256,
    num_stages: int = 2,
):
    acc_dtype = "float32"
    # Same cp.async idiom as the fused stream backward: the last (only
    # possibly ragged) chunk of each sequence is staged with predicated
    # scalar loads before the loop, and the loop itself only ever prefetches
    # full interior chunks.
    PIPELINED = num_stages >= 2
    n_bufs = 2 if PIPELINED else 1
    n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0 = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks, n_dht, n_dh0"
    )
    n_seqs = n_seq_plus_one - 1

    @T.prim_func
    def chunk_dplr_bwd_scan_tl(
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        do: T.Tensor((n_tokens, H, V), in_dtype),
        A_qb: T.Tensor((n_tokens, H, BT), in_dtype),
        dht: T.Tensor((n_dht, H, K, V), state_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_offsets: T.Tensor((n_seq_plus_one,), "int32"),
        dhu: T.Tensor((n_chunks, H, K, V), in_dtype),
        dv2: T.Tensor((n_tokens, H, V), in_dtype),
        dh0: T.Tensor((n_dh0, H, K, V), state_dtype),
    ):
        with T.Kernel(n_seqs, H, threads=threads) as (i_n, i_h):
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            boh = chunk_offsets[i_n]
            n_chunks = chunk_offsets[i_n + 1] - boh

            b_dh = T.alloc_fragment((K, V), acc_dtype)
            b_dh_tmp = T.alloc_fragment((K, V), acc_dtype)
            dhu_shared = T.alloc_shared((K, V), in_dtype)
            qg_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            bg_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            w_shared = T.alloc_shared((n_bufs, BT, K), in_dtype)
            do_shared = T.alloc_shared((n_bufs, BT, V), in_dtype)
            A_qb_shared = T.alloc_shared((n_bufs, BT, BT), in_dtype)

            dv_intra_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv2_shared = T.alloc_shared((BT, V), in_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)

            if USE_FINAL_STATE_GRADIENT:
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = dht[i_n, i_h, k_idx, vv]
            else:
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.float32(0.0)

            if PIPELINED:
                t_pro = bos + (n_chunks - 1) * BT
                for r, c in T.Parallel(BT, K):
                    t = t_pro + r
                    if t < eos:
                        qg_shared[0, r, c] = qg[t, i_h, c]
                        bg_shared[0, r, c] = bg[t, i_h, c]
                        w_shared[0, r, c] = w[t, i_h, c]
                    else:
                        qg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        bg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                        w_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(BT, V):
                    t = t_pro + r
                    if t < eos:
                        do_shared[0, r, c] = do[t, i_h, c]
                    else:
                        do_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(BT, BT):
                    t = t_pro + r
                    if (t < eos) and (r >= c):
                        A_qb_shared[0, r, c] = A_qb[t, i_h, c]
                    else:
                        A_qb_shared[0, r, c] = T.Cast(in_dtype, 0.0)

            for i_t_rev in T.serial(n_chunks):
                i_t = n_chunks - 1 - i_t_rev
                t_off = bos + i_t * BT
                chunk_row = boh + i_t
                cur = i_t_rev % 2 if PIPELINED else 0

                T.clear(b_dh_tmp)

                if PIPELINED:
                    # The current buffer must be fully consumed before the
                    # cp.async below overwrites the idle one.
                    T.sync_threads()
                    if i_t_rev + 1 < n_chunks:
                        nxt = (i_t_rev + 1) % 2
                        t_nxt = t_off - BT
                        T.async_copy(qg[t_nxt: t_nxt + BT, i_h, 0:K], qg_shared[nxt, :, :])
                        T.async_copy(bg[t_nxt: t_nxt + BT, i_h, 0:K], bg_shared[nxt, :, :])
                        T.async_copy(w[t_nxt: t_nxt + BT, i_h, 0:K], w_shared[nxt, :, :])
                        T.async_copy(do[t_nxt: t_nxt + BT, i_h, 0:V], do_shared[nxt, :, :])
                        # Stored A matrices are already causally masked.
                        T.async_copy(A_qb[t_nxt: t_nxt + BT, i_h, 0:BT], A_qb_shared[nxt, :, :])
                        # One commit group per async_copy: leave the just
                        # issued groups pending, wait for the ones from the
                        # previous iteration (the current buffer).
                        T.ptx_wait_group(5)
                    else:
                        T.ptx_wait_group(0)
                else:
                    full_tile = t_off + BT <= eos
                    if full_tile:
                        # Bulk vectorized copies for interior chunks.
                        T.copy(qg[t_off: t_off + BT, i_h, 0:K], qg_shared[0, :, :])
                        T.copy(bg[t_off: t_off + BT, i_h, 0:K], bg_shared[0, :, :])
                        T.copy(w[t_off: t_off + BT, i_h, 0:K], w_shared[0, :, :])
                        T.copy(do[t_off: t_off + BT, i_h, 0:V], do_shared[0, :, :])
                        # Stored A matrices are already causally masked.
                        T.copy(A_qb[t_off: t_off + BT, i_h, 0:BT], A_qb_shared[0, :, :])
                    else:
                        for r, c in T.Parallel(BT, K):
                            t = t_off + r
                            if t < eos:
                                qg_shared[0, r, c] = qg[t, i_h, c]
                                bg_shared[0, r, c] = bg[t, i_h, c]
                                w_shared[0, r, c] = w[t, i_h, c]
                            else:
                                qg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                bg_shared[0, r, c] = T.Cast(in_dtype, 0.0)
                                w_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                        for r, c in T.Parallel(BT, V):
                            t = t_off + r
                            if t < eos:
                                do_shared[0, r, c] = do[t, i_h, c]
                            else:
                                do_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                        for r, c in T.Parallel(BT, BT):
                            t = t_off + r
                            if (t < eos) and (r >= c):
                                A_qb_shared[0, r, c] = A_qb[t, i_h, c]
                            else:
                                A_qb_shared[0, r, c] = T.Cast(in_dtype, 0.0)

                # dhu[i] holds the pre-update state, written from the same
                # cast tile the dv2 GEMM reads — bit-identical to the dh
                # operand the fused kernel's consumer GEMMs see.
                T.copy(b_dh, dhu_shared)
                T.copy(dhu_shared, dhu[chunk_row, i_h, 0:K, 0:V])

                # dv2 = A_qb^T @ do + bg @ dh
                T.gemm(A_qb_shared[cur, :, :], do_shared[cur, :, :],
                       dv_intra_frag, transpose_A=True, clear_accum=True)
                T.gemm(bg_shared[cur, :, :], dhu_shared, dv2_frag, clear_accum=True)
                for r, vv in T.Parallel(BT, V):
                    t = t_off + r
                    dv2_frag[r, vv] = dv2_frag[r, vv] + dv_intra_frag[r, vv]
                    if t < eos:
                        dv2[t, i_h, vv] = T.Cast(in_dtype, dv2_frag[r, vv])
                T.copy(dv2_frag, dv2_shared)

                last_idx = T.min(t_off + BT - 1, eos - 1)
                for c in T.Parallel(K):
                    gk_last_shared[c] = gk[last_idx, i_h, c]

                # Update dh for the previous chunk.
                T.gemm(qg_shared[cur, :, :], do_shared[cur, :, :], b_dh_tmp, transpose_A=True)
                T.gemm(w_shared[cur, :, :], dv2_shared, b_dh_tmp, transpose_A=True)
                for k_idx, vv in T.Parallel(K, V):
                    b_dh[k_idx, vv] = T.exp2(gk_last_shared[k_idx]) * b_dh[k_idx, vv] + b_dh_tmp[k_idx, vv]

            if USE_INITIAL_STATE:
                for k_idx, vv in T.Parallel(K, V):
                    dh0[i_n, i_h, k_idx, vv] = T.Cast(state_dtype, b_dh[k_idx, vv])

    return chunk_dplr_bwd_scan_tl


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_bwd_consumer_kernel(
    H, K, V, BT,
    in_dtype,
    threads: int = 256,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic("n_tokens, n_seq_plus_one, n_chunks")

    @T.prim_func
    def chunk_dplr_bwd_consumer_tl(
        v: T.Tensor((n_tokens, H, V), in_dtype),
        v_new: T.Tensor((n_tokens, H, V), in_dtype),
        kg: T.Tensor((n_tokens, H, K), in_dtype),
        bg: T.Tensor((n_tokens, H, K), in_dtype),
        gk: T.Tensor((n_tokens, H, K), acc_dtype),
        do: T.Tensor((n_tokens, H, V), in_dtype),
        h: T.Tensor((n_chunks, H, K, V), in_dtype),
        dhu: T.Tensor((n_chunks, H, K, V), in_dtype),
        A_qk: T.Tensor((n_tokens, H, BT), in_dtype),
        dv2: T.Tensor((n_tokens, H, V), in_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        dq_out: T.Tensor((n_tokens, H, K), in_dtype),
        dk_out: T.Tensor((n_tokens, H, K), in_dtype),
        dw_out: T.Tensor((n_tokens, H, K), in_dtype),
        db_out: T.Tensor((n_tokens, H, K), in_dtype),
        dgk_last: T.Tensor((n_chunks, H, K), acc_dtype),
        dv_full: T.Tensor((n_tokens, H, V), in_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_row = i_n >= 0
            bos = T.if_then_else(is_valid_row, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_row, eos_raw, T.int32(0))
            is_valid_chunk = is_valid_row and bos < eos
            # Sentinel chunk slots take the zero-padded scalar path and guard
            # every store; the clamp only keeps their gk read in bounds.
            last_idx = T.max(T.min(bos + BT - 1, eos - 1), 0)

            h_shared = T.alloc_shared((K, V), in_dtype)
            dhu_shared = T.alloc_shared((K, V), in_dtype)
            v_shared = T.alloc_shared((BT, V), in_dtype)
            v_new_shared = T.alloc_shared((BT, V), in_dtype)
            do_shared = T.alloc_shared((BT, V), in_dtype)
            dv2_shared = T.alloc_shared((BT, V), in_dtype)
            kg_shared = T.alloc_shared((BT, K), in_dtype)
            bg_shared = T.alloc_shared((BT, K), in_dtype)
            A_qk_shared = T.alloc_shared((BT, BT), in_dtype)

            dq_frag = T.alloc_fragment((BT, K), acc_dtype)
            dk_frag = T.alloc_fragment((BT, K), acc_dtype)
            dw_frag = T.alloc_fragment((BT, K), acc_dtype)
            db_frag = T.alloc_fragment((BT, K), acc_dtype)
            dv_full_frag = T.alloc_fragment((BT, V), acc_dtype)
            dv_full_shared = T.alloc_shared((BT, V), in_dtype)
            # One (BT, K) tile stages the four outputs sequentially, folding
            # the dgk reduction into the dk/db passes (the fused mid
            # schedule's idiom); separate tiles would exceed the 228KB cap.
            out_shared = T.alloc_shared((BT, K), in_dtype)
            dgk_last_frag = T.alloc_fragment((K,), acc_dtype)
            hdh_frag = T.alloc_fragment((K, V), acc_dtype)
            dgk_part = T.alloc_fragment((4, K), acc_dtype)
            dgk_part_shared = T.alloc_shared((4, K), acc_dtype)
            gk_last_shared = T.alloc_shared((K,), acc_dtype)

            T.clear(dq_frag)
            T.clear(dk_frag)
            T.clear(dw_frag)
            T.clear(db_frag)
            for k_idx in T.Parallel(K):
                dgk_last_frag[k_idx] = T.float32(0.0)
            for gg, c in T.Parallel(4, K):
                dgk_part[gg, c] = T.float32(0.0)

            # h/dhu rows are defined for every chunk slot (no ragged edge).
            T.copy(h[i_c, i_h, 0:K, 0:V], h_shared)
            T.copy(dhu[i_c, i_h, 0:K, 0:V], dhu_shared)

            full_tile = is_valid_chunk and (bos + BT <= eos)
            if full_tile:
                # Bulk vectorized copies for interior chunks.
                T.copy(v[bos: bos + BT, i_h, 0:V], v_shared)
                T.copy(v_new[bos: bos + BT, i_h, 0:V], v_new_shared)
                T.copy(do[bos: bos + BT, i_h, 0:V], do_shared)
                T.copy(dv2[bos: bos + BT, i_h, 0:V], dv2_shared)
                T.copy(kg[bos: bos + BT, i_h, 0:K], kg_shared)
                T.copy(bg[bos: bos + BT, i_h, 0:K], bg_shared)
                # Stored A matrices are already causally masked.
                T.copy(A_qk[bos: bos + BT, i_h, 0:BT], A_qk_shared)
            else:
                for r, c in T.Parallel(BT, V):
                    t = bos + r
                    if t < eos:
                        v_shared[r, c] = v[t, i_h, c]
                        v_new_shared[r, c] = v_new[t, i_h, c]
                        do_shared[r, c] = do[t, i_h, c]
                        dv2_shared[r, c] = dv2[t, i_h, c]
                    else:
                        v_shared[r, c] = T.Cast(in_dtype, 0.0)
                        v_new_shared[r, c] = T.Cast(in_dtype, 0.0)
                        do_shared[r, c] = T.Cast(in_dtype, 0.0)
                        dv2_shared[r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        kg_shared[r, c] = kg[t, i_h, c]
                        bg_shared[r, c] = bg[t, i_h, c]
                    else:
                        kg_shared[r, c] = T.Cast(in_dtype, 0.0)
                        bg_shared[r, c] = T.Cast(in_dtype, 0.0)

                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if (t < eos) and (r >= c):
                        A_qk_shared[r, c] = A_qk[t, i_h, c]
                    else:
                        A_qk_shared[r, c] = T.Cast(in_dtype, 0.0)

            # The h*dhu product goes through a dedicated fragment so the smem
            # loads vectorize and the row reduction stays lowerable.
            for k_idx, vv in T.Parallel(K, V):
                hdh_frag[k_idx, vv] = (
                    T.Cast(acc_dtype, h_shared[k_idx, vv])
                    * T.Cast(acc_dtype, dhu_shared[k_idx, vv])
                )
            T.reduce_sum(hdh_frag, dgk_last_frag, dim=1, clear=False)

            T.gemm(do_shared, h_shared, dq_frag, transpose_B=True)
            T.gemm(v_shared, dhu_shared, dk_frag, transpose_B=True)
            T.gemm(v_new_shared, dhu_shared, db_frag, transpose_B=True)
            T.gemm(dv2_shared, h_shared, dw_frag, transpose_B=True)
            T.gemm(kg_shared, dhu_shared, dv_full_frag, clear_accum=True)
            T.gemm(A_qk_shared, do_shared, dv_full_frag, transpose_A=True)

            T.copy(dv_full_frag, dv_full_shared)
            for r, vv in T.Parallel(BT, V):
                t = bos + r
                if t < eos:
                    dv_full[t, i_h, vv] = T.Cast(in_dtype, dv_full_shared[r, vv])

            T.copy(dk_frag, out_shared)
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    dk_out[t, i_h, c] = out_shared[r, c]
            # Split the serial over-BT dgk reduction across 4 lane groups so
            # every thread participates.
            for r_local in T.serial(BT // 4):
                for gg, c in T.Parallel(4, K):
                    r = gg * (BT // 4) + r_local
                    dgk_part[gg, c] = (
                        dgk_part[gg, c]
                        + T.Cast(acc_dtype, kg_shared[r, c])
                        * T.Cast(acc_dtype, out_shared[r, c])
                    )
            T.copy(db_frag, out_shared)
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    db_out[t, i_h, c] = out_shared[r, c]
            for r_local in T.serial(BT // 4):
                for gg, c in T.Parallel(4, K):
                    r = gg * (BT // 4) + r_local
                    dgk_part[gg, c] = (
                        dgk_part[gg, c]
                        + T.Cast(acc_dtype, bg_shared[r, c])
                        * T.Cast(acc_dtype, out_shared[r, c])
                    )
            T.copy(dq_frag, out_shared)
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    dq_out[t, i_h, c] = out_shared[r, c]
            T.copy(dw_frag, out_shared)
            for r, c in T.Parallel(BT, K):
                t = bos + r
                if t < eos:
                    dw_out[t, i_h, c] = out_shared[r, c]

            for c in T.Parallel(K):
                gk_last_shared[c] = gk[last_idx, i_h, c]
                dgk_last_frag[c] = dgk_last_frag[c] * T.exp2(gk_last_shared[c])
            for gg, c in T.Parallel(4, K):
                dgk_part_shared[gg, c] = dgk_part[gg, c]
            T.sync_threads()
            if is_valid_chunk:
                for c in T.Parallel(K):
                    dgk_last[i_c, i_h, c] = (
                        dgk_last_frag[c]
                        + dgk_part_shared[0, c]
                        + dgk_part_shared[1, c]
                        + dgk_part_shared[2, c]
                        + dgk_part_shared[3, c]
                    )

    return chunk_dplr_bwd_consumer_tl


def chunk_dplr_bwd_stream_twopass_into(
    qg: torch.Tensor,
    bg: torch.Tensor,
    w: torch.Tensor,
    kg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor | None,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    A_qb_for_dv: torch.Tensor,
    A_qk: torch.Tensor,
    dq_out: torch.Tensor,
    dk_out: torch.Tensor,
    dw_out: torch.Tensor,
    db_out: torch.Tensor,
    dgk_last_out: torch.Tensor,
    dv2_out: torch.Tensor,
    dv_full_out: torch.Tensor,
    dh0_out: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass stream backward with the same output contract as
    `chunk_dplr_bwd_stream_into`; `dv2_out` is the pass-1 -> pass-2 conduit."""
    B, T_, H, K = qg.shape
    V = do.shape[-1]
    BT = int(chunk_size)
    is_varlen = cu_seqlens is not None
    for out in (dq_out, dk_out, dw_out, db_out, dgk_last_out, dv2_out, dv_full_out, dh0_out):
        assert out.is_contiguous(), "chunk_dplr_bwd_stream_twopass_into requires contiguous outputs"
    if V != K:
        raise NotImplementedError("The two-pass DPLR stream backward requires K == V.")
    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, qg.device)
    n_chunks = layout.chunk_indices.shape[0]
    n_seqs = layout.cu_seqlens.shape[0] - 1
    n_tokens = B * T_
    in_dtype = str(qg.dtype).split(".")[-1]
    state_dtype = "float32"
    use_dht = dht is not None
    use_dh0 = h0 is not None
    n_dh0 = n_seqs if use_dh0 else 1

    qg_f = qg.reshape(n_tokens, H, K).contiguous()
    bg_f = bg.reshape(n_tokens, H, K).contiguous()
    w_f = w.reshape(n_tokens, H, K).contiguous()
    kg_f = kg.reshape(n_tokens, H, K).contiguous()
    v_f = v.reshape(n_tokens, H, V).contiguous()
    v_new_f = v_new.reshape(n_tokens, H, V).contiguous()
    gk_f = gk.reshape(n_tokens, H, K).contiguous()
    do_f = do.reshape(n_tokens, H, V).contiguous()
    h_f = h.reshape(n_chunks, H, K, V).contiguous()
    A_qb_f = A_qb_for_dv.reshape(n_tokens, H, BT).contiguous()
    A_qk_f = A_qk.reshape(n_tokens, H, BT).contiguous()

    if use_dht:
        dht_f = dht.reshape(n_seqs, H, K, V).contiguous().to(torch.float32)
    else:
        dht_f = torch.empty((1, H, K, V), dtype=torch.float32, device=qg.device)

    dq_f = dq_out.reshape(n_tokens, H, K).contiguous()
    dk_f = dk_out.reshape(n_tokens, H, K).contiguous()
    dw_f = dw_out.reshape(n_tokens, H, K).contiguous()
    db_f = db_out.reshape(n_tokens, H, K).contiguous()
    dgk_last_f = dgk_last_out.reshape(n_chunks, H, K).contiguous()
    dv2_f = dv2_out.reshape(n_tokens, H, V).contiguous()
    dv_full_f = dv_full_out.reshape(n_tokens, H, V).contiguous()
    dh0_f = dh0_out.reshape(n_dh0, H, K, V).contiguous()

    dhu_f = torch.empty((n_chunks, H, K, V), dtype=qg.dtype, device=qg.device)

    config = stream_bwd_twopass_config_or_none(
        K=K, V=V, BT=BT, in_dtype=in_dtype, device=qg.device,
    )
    if config is None:
        name = torch.cuda.get_device_properties(qg.device.index or 0).name
        raise RuntimeError(
            f"No launchable two-pass DPLR stream backward for K={K}, V={V}, BT={BT}, "
            f"dtype={in_dtype} on {name}"
        )
    scan_kernel = _chunk_dplr_bwd_scan_kernel(
        H, K, V, BT,
        in_dtype, state_dtype, use_dht, use_dh0,
        **config,
    )
    scan_kernel(
        qg_f, bg_f, w_f, gk_f, do_f, A_qb_f, dht_f,
        layout.cu_seqlens, layout.chunk_offsets,
        dhu_f, dv2_f, dh0_f,
    )
    consumer_kernel = _chunk_dplr_bwd_consumer_kernel(
        H, K, V, BT,
        in_dtype, config["threads"],
    )
    consumer_kernel(
        v_f, v_new_f, kg_f, bg_f, gk_f, do_f, h_f, dhu_f, A_qk_f, dv2_f,
        layout.cu_seqlens, layout.chunk_indices,
        dq_f, dk_f, dw_f, db_f, dgk_last_f, dv_full_f,
    )
    dh0 = dh0_out if use_dh0 else None
    return dq_out, dk_out, dw_out, db_out, dgk_last_out, dv2_out, dv_full_out, dh0
