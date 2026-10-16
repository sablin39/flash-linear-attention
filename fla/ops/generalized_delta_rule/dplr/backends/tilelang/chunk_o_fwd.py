# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR chunk-parallel output forward (the o side of the split h+o pass).

The fused h+o forward (`chunk_ho_fwd.py`) carries the per-chunk output GEMMs
on its serial chunk walk; at BT=64, K=V=128 it cannot pipeline (the BV=64
two-stage footprint exceeds the 232448B optin) and its (V/BV, N, H) grid
underfills wide-SM devices.  This kernel is the chunk-parallel half of the
split: `chunk_h_fwd.py` materializes each chunk's pre-update state as a
global (n_chunks, H, K, V) tensor (and v_new), and here one block per
(chunk, head, V-tile) computes

    o = qg @ h_pre + A_qk @ v + A_qb @ v_new

mirroring the fused kernel's o GEMMs and their operand rounding exactly:
h_pre is the same in_dtype cast tile the fused kernel's qg GEMM reads, and
v_new is the in_dtype cast of the fused kernel's v2 accumulator, so the
split consumes bit-identical GEMM operands.
"""

import tilelang
import tilelang.language as T
import torch

from .layout import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout


def _chunk_o_fwd_config(V: int) -> dict[str, int]:
    # The (n_chunks, H, V/BV) grid is wide even at BV=64; smem stays <= 96KB
    # at BT=64, K=V=128, so any cc >= 8 device launches it.
    return {"BV": 64 if V >= 64 else 32, "threads": 256}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
    },
)
def _chunk_dplr_fwd_o_kernel(
    H, K, V, BT,
    in_dtype,
    BV: int = 64,
    threads: int = 256,
):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic("n_tokens, n_seq_plus_one, n_chunks")

    @T.prim_func
    def chunk_dplr_fwd_o_tl(
        qg: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        v_new: T.Tensor((n_tokens, H, V), in_dtype),
        A_qk: T.Tensor((n_tokens, H, BT), in_dtype),
        A_qb: T.Tensor((n_tokens, H, BT), in_dtype),
        h: T.Tensor((n_chunks, H, K, V), in_dtype),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        o: T.Tensor((n_tokens, H, V), in_dtype),
    ):
        with T.Kernel(n_chunks, H, V // BV, threads=threads) as (i_c, i_h, i_v):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            # Sentinel chunk slots take the zero-padded path and skip stores.
            is_valid_chunk = (i_n >= 0) and (bos_raw < eos_raw)
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))

            h_shared = T.alloc_shared((K, BV), in_dtype)
            qg_shared = T.alloc_shared((BT, K), in_dtype)
            v_shared = T.alloc_shared((BT, BV), in_dtype)
            v_new_shared = T.alloc_shared((BT, BV), in_dtype)
            A_qk_shared = T.alloc_shared((BT, BT), in_dtype)
            A_qb_shared = T.alloc_shared((BT, BT), in_dtype)
            o_shared = T.alloc_shared((BT, BV), in_dtype)
            o_frag = T.alloc_fragment((BT, BV), acc_dtype)

            # Chunk-slot rows are defined for every slot (no ragged edge);
            # sentinel contents never reach a store.
            T.copy(h[i_c, i_h, 0:K, i_v * BV: i_v * BV + BV], h_shared)

            full_tile = is_valid_chunk and (bos + BT <= eos)
            if full_tile:
                # Bulk vectorized copies for interior chunks.
                T.copy(qg[bos: bos + BT, i_h, 0:K], qg_shared)
                T.copy(v[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], v_shared)
                T.copy(v_new[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV], v_new_shared)
                # Stored A matrices are already causally masked.
                T.copy(A_qk[bos: bos + BT, i_h, 0:BT], A_qk_shared)
                T.copy(A_qb[bos: bos + BT, i_h, 0:BT], A_qb_shared)
            else:
                for r, k_idx in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        qg_shared[r, k_idx] = qg[t, i_h, k_idx]
                    else:
                        qg_shared[r, k_idx] = T.Cast(in_dtype, 0.0)
                for r, vv in T.Parallel(BT, BV):
                    t = bos + r
                    g_v = i_v * BV + vv
                    if (t < eos) and (g_v < V):
                        v_shared[r, vv] = v[t, i_h, g_v]
                        v_new_shared[r, vv] = v_new[t, i_h, g_v]
                    else:
                        v_shared[r, vv] = T.Cast(in_dtype, 0.0)
                        v_new_shared[r, vv] = T.Cast(in_dtype, 0.0)
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if (t < eos) and (r >= c):
                        A_qk_shared[r, c] = A_qk[t, i_h, c]
                        A_qb_shared[r, c] = A_qb[t, i_h, c]
                    else:
                        A_qk_shared[r, c] = T.Cast(in_dtype, 0.0)
                        A_qb_shared[r, c] = T.Cast(in_dtype, 0.0)

            T.gemm(qg_shared, h_shared, o_frag, clear_accum=True)
            T.gemm(A_qk_shared, v_shared, o_frag)
            T.gemm(A_qb_shared, v_new_shared, o_frag)

            # Stage o through shared memory so the accumulator store lowers
            # to vectorized writes (the fused kernel's idiom).
            T.copy(o_frag, o_shared)
            if full_tile and (i_v * BV + BV <= V):
                T.copy(o_shared, o[bos: bos + BT, i_h, i_v * BV: i_v * BV + BV])
            elif is_valid_chunk:
                for r, vv in T.Parallel(BT, BV):
                    t = bos + r
                    g_v = i_v * BV + vv
                    if (t < eos) and (g_v < V):
                        o[t, i_h, g_v] = o_shared[r, vv]

    return chunk_dplr_fwd_o_tl


def chunk_dplr_fwd_o(
    qg: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    A_qk: torch.Tensor,
    A_qb: torch.Tensor,
    h: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_layout: ChunkLayout | None = None,
) -> torch.Tensor:
    B, T_, H, K = qg.shape
    V = v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = chunk_layout if chunk_layout is not None else build_rect_chunk_layout(B, T_, BT, qg.device)
    in_dtype = str(qg.dtype).split(".")[-1]
    token_rows = B * T_

    kernel = _chunk_dplr_fwd_o_kernel(H, K, V, BT, in_dtype, **_chunk_o_fwd_config(V))
    o_f = torch.empty((token_rows, H, V), dtype=v.dtype, device=v.device)
    kernel(
        qg.reshape(token_rows, H, K).contiguous(),
        v.reshape(token_rows, H, V).contiguous(),
        v_new.reshape(token_rows, H, V).contiguous(),
        A_qk.reshape(token_rows, H, BT).contiguous(),
        A_qb.reshape(token_rows, H, BT).contiguous(),
        h.reshape(-1, H, K, V).contiguous(),
        layout.cu_seqlens,
        layout.chunk_indices,
        o_f,
    )
    return o_f.view(B, T_, H, V)
