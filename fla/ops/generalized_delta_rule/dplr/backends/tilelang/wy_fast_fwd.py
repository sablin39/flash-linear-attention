# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""DPLR WY representation forward.

Two stages:
1. `_prepare_wy_repr_fwd_kernel`: invert the strictly lower-triangular A_ab
   per chunk with the exact tensor-core scheme (FLA #1162): each 16x16
   diagonal block is inverted as (I+L)(I+L^2)(I+L^4)(I+L^8) = (I-L)^-1
   (exact since L^16 = 0), off-diagonal blocks follow by block forward
   substitution, and every GEMM round is rounded back to fp16, matching
   FLA's Triton safe-gate path. Yields fp32 A_ab_inv, matching FLA's
   public dtype boundary.
2. `_wu_fwd_kernel`: compute `b_Aak = A_ab_inv @ A_ak` (causal masks applied),
   then `w = A_ab_inv @ ag` and `u = b_Aak @ v`.

Adapted from FLA's prepare_wy_repr_fwd_kernel_safe + wu_fwd_kernel.
"""

import tilelang
import tilelang.language as T
import torch

from .layout import ChunkLayout, build_rect_chunk_layout, build_varlen_chunk_layout
from .schedules import device_cc


def _wu_fwd_config(BT: int, K: int, device: torch.device) -> dict[str, int]:
    # Bulk copies + wide threads pay at BT=64 on cc90 and cc120 alike
    # (kernel-level 1.7-2.1x over the scalar path on both); cc90 also wins
    # at BT=32 with 128 threads, while cc120 measures flat there, so the
    # cc120 gate stays BT>=64. At BT=32/K>=128 on cc90, 256 threads spread
    # the wide ag/v tiles and N=128 GEMMs further (-21% kernel time); at
    # K=64 they over-subscribe the narrow tiles (+3%).
    if BT <= 16:
        return {"threads": 32, "bulk_copy": False}
    cc = device_cc(device)
    if cc == 90:
        return {"threads": 256 if BT >= 64 or K >= 128 else 128, "bulk_copy": True}
    if cc == 120 and BT >= 64:
        return {"threads": 256, "bulk_copy": True}
    return {"threads": 128, "bulk_copy": False}


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _prepare_wy_repr_fwd_kernel(H, BT, in_dtype, threads: int = 32):
    acc_dtype = "float32"
    NB = 16
    NS = BT // NB
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def prepare_wy_repr_fwd_tl(
        A_ab: T.Tensor((n_tokens, H, BT), "float16"),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        A_ab_inv: T.Tensor((n_tokens, H, BT), acc_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))

            # L tiles hold the strictly lower part of A_ab per 16x16 block
            # (off-diagonal blocks are full); T tiles hold the inverse blocks.
            L00 = T.alloc_shared((NB, NB), "float16")
            T00 = T.alloc_shared((NB, NB), "float16")
            I_L = T.alloc_shared((NB, NB), "float16")
            L2 = T.alloc_shared((NB, NB), "float16")
            L4 = T.alloc_shared((NB, NB), "float16")
            L8 = T.alloc_shared((NB, NB), "float16")
            I_L2 = T.alloc_shared((NB, NB), "float16")
            I_L4 = T.alloc_shared((NB, NB), "float16")
            I_L8 = T.alloc_shared((NB, NB), "float16")
            P = T.alloc_shared((NB, NB), "float16")
            fA = T.alloc_fragment((NB, NB), acc_dtype)
            if NS >= 2:
                L10 = T.alloc_shared((NB, NB), "float16")
                L11 = T.alloc_shared((NB, NB), "float16")
                T10 = T.alloc_shared((NB, NB), "float16")
                T11 = T.alloc_shared((NB, NB), "float16")
            if NS == 4:
                L20 = T.alloc_shared((NB, NB), "float16")
                L21 = T.alloc_shared((NB, NB), "float16")
                L22 = T.alloc_shared((NB, NB), "float16")
                L30 = T.alloc_shared((NB, NB), "float16")
                L31 = T.alloc_shared((NB, NB), "float16")
                L32 = T.alloc_shared((NB, NB), "float16")
                L33 = T.alloc_shared((NB, NB), "float16")
                T20 = T.alloc_shared((NB, NB), "float16")
                T21 = T.alloc_shared((NB, NB), "float16")
                T22 = T.alloc_shared((NB, NB), "float16")
                T30 = T.alloc_shared((NB, NB), "float16")
                T31 = T.alloc_shared((NB, NB), "float16")
                T32 = T.alloc_shared((NB, NB), "float16")
                T33 = T.alloc_shared((NB, NB), "float16")
                Q = T.alloc_shared((NB, NB), "float16")

            # Load the strict-lower blocks; rows past eos stay zero, so
            # partial chunks invert to identity there (stores are masked).
            for r, c in T.Parallel(NB, NB):
                t0 = bos + r
                if (r > c) and (t0 < eos):
                    L00[r, c] = A_ab[t0, i_h, c]
                else:
                    L00[r, c] = T.Cast("float16", 0.0)
                if NS >= 2:
                    t1 = bos + NB + r
                    if t1 < eos:
                        L10[r, c] = A_ab[t1, i_h, c]
                        if r > c:
                            L11[r, c] = A_ab[t1, i_h, NB + c]
                        else:
                            L11[r, c] = T.Cast("float16", 0.0)
                    else:
                        L10[r, c] = T.Cast("float16", 0.0)
                        L11[r, c] = T.Cast("float16", 0.0)
                if NS == 4:
                    t2 = bos + 2 * NB + r
                    if t2 < eos:
                        L20[r, c] = A_ab[t2, i_h, c]
                        L21[r, c] = A_ab[t2, i_h, NB + c]
                        if r > c:
                            L22[r, c] = A_ab[t2, i_h, 2 * NB + c]
                        else:
                            L22[r, c] = T.Cast("float16", 0.0)
                    else:
                        L20[r, c] = T.Cast("float16", 0.0)
                        L21[r, c] = T.Cast("float16", 0.0)
                        L22[r, c] = T.Cast("float16", 0.0)
                    t3 = bos + 3 * NB + r
                    if t3 < eos:
                        L30[r, c] = A_ab[t3, i_h, c]
                        L31[r, c] = A_ab[t3, i_h, NB + c]
                        L32[r, c] = A_ab[t3, i_h, 2 * NB + c]
                        if r > c:
                            L33[r, c] = A_ab[t3, i_h, 3 * NB + c]
                        else:
                            L33[r, c] = T.Cast("float16", 0.0)
                    else:
                        L30[r, c] = T.Cast("float16", 0.0)
                        L31[r, c] = T.Cast("float16", 0.0)
                        L32[r, c] = T.Cast("float16", 0.0)
                        L33[r, c] = T.Cast("float16", 0.0)

            # Diagonal blocks: (I-L)^-1 = (I+L)(I+L^2)(I+L^4)(I+L^8), exact
            # because L is strictly lower triangular (L^16 = 0). Every GEMM
            # round is rounded back to fp16, matching FLA's Triton safe path.
            T.gemm(L00, L00, fA, clear_accum=True)
            for r, c in T.Parallel(NB, NB):
                L2[r, c] = T.Cast("float16", fA[r, c])
            T.gemm(L2, L2, fA, clear_accum=True)
            for r, c in T.Parallel(NB, NB):
                L4[r, c] = T.Cast("float16", fA[r, c])
            T.gemm(L4, L4, fA, clear_accum=True)
            for r, c in T.Parallel(NB, NB):
                L8[r, c] = T.Cast("float16", fA[r, c])
            for r, c in T.Parallel(NB, NB):
                I_L[r, c] = L00[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                I_L2[r, c] = L2[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                I_L4[r, c] = L4[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                I_L8[r, c] = L8[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
            T.gemm(I_L, I_L2, fA, clear_accum=True)
            for r, c in T.Parallel(NB, NB):
                P[r, c] = T.Cast("float16", fA[r, c])
            T.gemm(P, I_L4, fA, clear_accum=True)
            for r, c in T.Parallel(NB, NB):
                P[r, c] = T.Cast("float16", fA[r, c])
            T.gemm(P, I_L8, fA, clear_accum=True)
            for r, c in T.Parallel(NB, NB):
                T00[r, c] = T.Cast("float16", fA[r, c])
            if NS >= 2:
                T.gemm(L11, L11, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L2[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L2, L2, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L4[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L4, L4, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L8[r, c] = T.Cast("float16", fA[r, c])
                for r, c in T.Parallel(NB, NB):
                    I_L[r, c] = L11[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L2[r, c] = L2[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L4[r, c] = L4[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L8[r, c] = L8[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                T.gemm(I_L, I_L2, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(P, I_L4, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(P, I_L8, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T11[r, c] = T.Cast("float16", fA[r, c])
                # T10 = T11 @ (L10 @ T00)
                T.gemm(L10, T00, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(T11, P, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T10[r, c] = T.Cast("float16", fA[r, c])
            if NS == 4:
                T.gemm(L22, L22, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L2[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L2, L2, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L4[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L4, L4, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L8[r, c] = T.Cast("float16", fA[r, c])
                for r, c in T.Parallel(NB, NB):
                    I_L[r, c] = L22[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L2[r, c] = L2[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L4[r, c] = L4[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L8[r, c] = L8[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                T.gemm(I_L, I_L2, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(P, I_L4, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(P, I_L8, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T22[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L33, L33, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L2[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L2, L2, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L4[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L4, L4, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    L8[r, c] = T.Cast("float16", fA[r, c])
                for r, c in T.Parallel(NB, NB):
                    I_L[r, c] = L33[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L2[r, c] = L2[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L4[r, c] = L4[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                    I_L8[r, c] = L8[r, c] + T.if_then_else(r == c, T.Cast("float16", 1.0), T.Cast("float16", 0.0))
                T.gemm(I_L, I_L2, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(P, I_L4, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(P, I_L8, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T33[r, c] = T.Cast("float16", fA[r, c])
                # Block forward substitution for the off-diagonal blocks
                # (FLA #1162 formulas).
                # T21 = T22 @ (L21 @ T11)
                T.gemm(L21, T11, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(T22, P, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T21[r, c] = T.Cast("float16", fA[r, c])
                # T32 = T33 @ (L32 @ T22)
                T.gemm(L32, T22, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(T33, P, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T32[r, c] = T.Cast("float16", fA[r, c])
                # P1 = L20 @ T00 + L21 @ T10; P2 = L30 @ T00 + L31 @ T10
                T.gemm(L20, T00, fA, clear_accum=True)
                T.gemm(L21, T10, fA)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L30, T00, fA, clear_accum=True)
                T.gemm(L31, T10, fA)
                for r, c in T.Parallel(NB, NB):
                    Q[r, c] = T.Cast("float16", fA[r, c])
                # T20 = T22 @ P1
                T.gemm(T22, P, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    T20[r, c] = T.Cast("float16", fA[r, c])
                # T30 = T32 @ P1 + T33 @ P2
                T.gemm(T32, P, fA, clear_accum=True)
                T.gemm(T33, Q, fA)
                for r, c in T.Parallel(NB, NB):
                    T30[r, c] = T.Cast("float16", fA[r, c])
                # T31 = T32 @ (L21 @ T11) + T33 @ (L31 @ T11)
                T.gemm(L21, T11, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    P[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(L31, T11, fA, clear_accum=True)
                for r, c in T.Parallel(NB, NB):
                    Q[r, c] = T.Cast("float16", fA[r, c])
                T.gemm(T32, P, fA, clear_accum=True)
                T.gemm(T33, Q, fA)
                for r, c in T.Parallel(NB, NB):
                    T31[r, c] = T.Cast("float16", fA[r, c])

            # Store inverse blocks; upper off-diagonal blocks are exact zeros.
            for r, c in T.Parallel(NB, NB):
                t0 = bos + r
                if t0 < eos:
                    A_ab_inv[t0, i_h, c] = T.Cast(acc_dtype, T00[r, c])
                    if NS >= 2:
                        A_ab_inv[t0, i_h, NB + c] = 0.0
                    if NS == 4:
                        A_ab_inv[t0, i_h, 2 * NB + c] = 0.0
                        A_ab_inv[t0, i_h, 3 * NB + c] = 0.0
                if NS >= 2:
                    t1 = bos + NB + r
                    if t1 < eos:
                        A_ab_inv[t1, i_h, c] = T.Cast(acc_dtype, T10[r, c])
                        A_ab_inv[t1, i_h, NB + c] = T.Cast(acc_dtype, T11[r, c])
                        if NS == 4:
                            A_ab_inv[t1, i_h, 2 * NB + c] = 0.0
                            A_ab_inv[t1, i_h, 3 * NB + c] = 0.0
                if NS == 4:
                    t2 = bos + 2 * NB + r
                    if t2 < eos:
                        A_ab_inv[t2, i_h, c] = T.Cast(acc_dtype, T20[r, c])
                        A_ab_inv[t2, i_h, NB + c] = T.Cast(acc_dtype, T21[r, c])
                        A_ab_inv[t2, i_h, 2 * NB + c] = T.Cast(acc_dtype, T22[r, c])
                        A_ab_inv[t2, i_h, 3 * NB + c] = 0.0
                    t3 = bos + 3 * NB + r
                    if t3 < eos:
                        A_ab_inv[t3, i_h, c] = T.Cast(acc_dtype, T30[r, c])
                        A_ab_inv[t3, i_h, NB + c] = T.Cast(acc_dtype, T31[r, c])
                        A_ab_inv[t3, i_h, 2 * NB + c] = T.Cast(acc_dtype, T32[r, c])
                        A_ab_inv[t3, i_h, 3 * NB + c] = T.Cast(acc_dtype, T33[r, c])

    return prepare_wy_repr_fwd_tl


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
                  tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: False,
                  },
)
def _wu_fwd_kernel(H, K, V, BT, in_dtype, threads: int = 32, bulk_copy: bool = False):
    acc_dtype = "float32"
    n_tokens, n_seq_plus_one, n_chunks = T.dynamic(
        "n_tokens, n_seq_plus_one, n_chunks"
    )

    @T.prim_func
    def wu_fwd_tl(
        ag: T.Tensor((n_tokens, H, K), in_dtype),
        v: T.Tensor((n_tokens, H, V), in_dtype),
        A_ab_inv: T.Tensor((n_tokens, H, BT), acc_dtype),
        A_ak: T.Tensor((n_tokens, H, BT), "float16"),
        cu_seqlens: T.Tensor((n_seq_plus_one,), "int32"),
        chunk_indices: T.Tensor((n_chunks, 2), "int32"),
        w: T.Tensor((n_tokens, H, K), in_dtype),
        u: T.Tensor((n_tokens, H, V), in_dtype),
    ):
        with T.Kernel(n_chunks, H, threads=threads) as (i_c, i_h):
            i_n = chunk_indices[i_c, 0]
            i_t = chunk_indices[i_c, 1]
            safe_i_n = T.max(i_n, 0)
            seq_bos = cu_seqlens[safe_i_n]
            seq_eos = cu_seqlens[safe_i_n + 1]
            bos_raw = seq_bos + i_t * BT
            eos_raw = T.min(bos_raw + BT, seq_eos)
            is_valid_chunk = i_n >= 0
            bos = T.if_then_else(is_valid_chunk, bos_raw, T.int32(0))
            eos = T.if_then_else(is_valid_chunk, eos_raw, T.int32(0))

            # FLA keeps A_ab_inv @ A_ak in fp32/tf32, then downcasts the
            # processed Aak before the u = Aak @ v GEMM. Keep separate
            # fp32 and input-dtype views so w remains aligned with FLA's
            # bf16 A_ab_inv @ ag path.
            A_inv_acc_shared = T.alloc_shared((BT, BT), acc_dtype)
            A_ak_acc_shared = T.alloc_shared((BT, BT), acc_dtype)
            A_inv_shared = T.alloc_shared((BT, BT), in_dtype)
            Aak_processed_frag = T.alloc_fragment((BT, BT), acc_dtype)
            Aak_processed_shared = T.alloc_shared((BT, BT), in_dtype)
            ag_shared = T.alloc_shared((BT, K), in_dtype)
            v_shared = T.alloc_shared((BT, V), in_dtype)
            w_frag = T.alloc_fragment((BT, K), acc_dtype)
            u_frag = T.alloc_fragment((BT, V), acc_dtype)

            # Load A_ab_inv (inclusive lower-tri after diagonal-add) and A_ak.
            # Interior chunks bulk-copy and mask in shared; boundary chunks
            # keep the scalar predicated path (same gating as chunk_A_bwd).
            full_tile = (is_valid_chunk and (bos + BT <= eos)) if bulk_copy else False
            if full_tile:
                T.copy(A_ab_inv[bos: bos + BT, i_h, 0:BT], A_inv_acc_shared)
                T.copy(A_ak[bos: bos + BT, i_h, 0:BT], A_ak_acc_shared)
                # The bf16 view is the same tile after masking, so derive it
                # from the fp32 copy in shared instead of reading global twice.
                for r, c in T.Parallel(BT, BT):
                    if r < c:
                        A_inv_acc_shared[r, c] = 0.0
                    if r <= c:
                        A_ak_acc_shared[r, c] = 0.0
                    A_inv_shared[r, c] = T.Cast(in_dtype, A_inv_acc_shared[r, c])
            else:
                for r, c in T.Parallel(BT, BT):
                    t = bos + r
                    if (t < eos) and (r >= c):
                        A_inv_acc_shared[r, c] = A_ab_inv[t, i_h, c]
                    else:
                        A_inv_acc_shared[r, c] = 0.0
                    A_inv_shared[r, c] = T.Cast(in_dtype, A_inv_acc_shared[r, c])
                    if (t < eos) and (r > c):
                        A_ak_acc_shared[r, c] = A_ak[t, i_h, c]
                    else:
                        A_ak_acc_shared[r, c] = 0.0

            T.gemm(A_inv_acc_shared, A_ak_acc_shared, Aak_processed_frag, clear_accum=True)
            for r, c in T.Parallel(BT, BT):
                Aak_processed_shared[r, c] = T.Cast(in_dtype, Aak_processed_frag[r, c])

            if full_tile:
                T.copy(ag[bos: bos + BT, i_h, 0:K], ag_shared)
            else:
                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        ag_shared[r, c] = ag[t, i_h, c]
                    else:
                        ag_shared[r, c] = T.Cast(in_dtype, 0.0)
            T.gemm(A_inv_shared, ag_shared, w_frag, clear_accum=True)
            if full_tile:
                for r, c in T.Parallel(BT, K):
                    w[bos + r, i_h, c] = T.Cast(in_dtype, w_frag[r, c])
            else:
                for r, c in T.Parallel(BT, K):
                    t = bos + r
                    if t < eos:
                        w[t, i_h, c] = T.Cast(in_dtype, w_frag[r, c])

            if full_tile:
                T.copy(v[bos: bos + BT, i_h, 0:V], v_shared)
            else:
                for r, c in T.Parallel(BT, V):
                    t = bos + r
                    if t < eos:
                        v_shared[r, c] = v[t, i_h, c]
                    else:
                        v_shared[r, c] = T.Cast(in_dtype, 0.0)
            T.gemm(Aak_processed_shared, v_shared, u_frag, clear_accum=True)
            if full_tile:
                for r, c in T.Parallel(BT, V):
                    u[bos + r, i_h, c] = T.Cast(in_dtype, u_frag[r, c])
            else:
                for r, c in T.Parallel(BT, V):
                    t = bos + r
                    if t < eos:
                        u[t, i_h, c] = T.Cast(in_dtype, u_frag[r, c])

    return wu_fwd_tl


def prepare_wy_repr_fwd(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
    chunk_layout: ChunkLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T_, H, K = ag.shape
    V = v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert B == 1
        layout = chunk_layout if chunk_layout is not None else build_varlen_chunk_layout(cu_seqlens, BT, T_)
    else:
        layout = build_rect_chunk_layout(B, T_, BT, ag.device)
    N_tokens = B * T_
    in_dtype = str(ag.dtype).split(".")[-1]

    A_ab_f = A_ab.reshape(N_tokens, H, BT).contiguous()
    A_ak_f = A_ak.reshape(N_tokens, H, BT).contiguous()

    inv_kernel = _prepare_wy_repr_fwd_kernel(H, BT, in_dtype)
    A_ab_inv_f = torch.empty((N_tokens, H, BT), dtype=torch.float32, device=ag.device)
    inv_kernel(A_ab_f, layout.cu_seqlens, layout.chunk_indices, A_ab_inv_f)

    ag_f = ag.reshape(N_tokens, H, K).contiguous()
    v_f = v.reshape(N_tokens, H, V).contiguous()

    wu_kernel = _wu_fwd_kernel(
        H, K, V, BT, in_dtype,
        **_wu_fwd_config(BT, K, ag.device),
    )
    w_f = torch.empty((N_tokens, H, K), dtype=ag.dtype, device=ag.device)
    u_f = torch.empty((N_tokens, H, V), dtype=v.dtype, device=v.device)
    wu_kernel(ag_f, v_f, A_ab_inv_f, A_ak_f, layout.cu_seqlens, layout.chunk_indices, w_f, u_f)

    w = w_f.view(B, T_, H, K)
    u = u_f.view(B, T_, H, V)
    A_ab_inv = A_ab_inv_f.view(B, T_, H, BT)
    return w, u, A_ab_inv
