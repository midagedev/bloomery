#!/usr/bin/env python3
"""ernie55_ref.py: a layer-streaming CPU f32 reference for ERNIE 5.5 Flash Preview, and its frozen reference sets.

What it is.  The forward of `Ernie5_5MoeForCausalLM` written in numpy, one layer's weights at a time (safetensors read
with os.pread, FP8 e4m3 x 128x128 block scale and MXFP4 x e8m0 dequantized to f32 per row chunk inside the matmul, so
the resident set stays a few GB whatever the file size).  It is an oracle for the engine port (no card, no ggml): every
formula below names its source.

Sources (all read, none run):
  PR    llama.cpp PR #31925, ernie5-5.cpp + the deepseek4.cpp deltas + conversion/ernie5.py   (pr31925.diff:LINE)
  VLLM  vllm-ernie  src/vllm_ernie/models/ernie5_5/{nvidia/model.py,attention.py,moe.py,mhc.py}, tests/.../test_mhc.py
  DS4   llama.cpp src/models/deepseek4.cpp (the V4 base the PR extends)
  GGML  ggml's rope (norm = interleaved pairs, neox = halves, yarn = corr dims + ramp)

Usage (on ws, system python has numpy 1.26 with a netlib BLAS: pass --pylib to a numpy 2.x OpenBLAS dir for the real run):
  python3 -I tools/ref/ernie55_ref.py --self-test
  python3 tools/ref/ernie55_ref.py tokenize --model DIR --out DIR --pylib PYLIB
  python3 tools/ref/ernie55_ref.py run --model DIR --out SETS_DIR --prompt short=IDS --prompt long=IDS --steps 32
  python3 tools/ref/ernie55_ref.py sanity --model DIR --set SETS_DIR/short --set SETS_DIR/long

Set format (one dir per prompt, the ik dump_ref grammar `RefManifest::read` parses; crates/refset/src/ik.rs):
  MANIFEST.tsv  header (# model, # arch ernie5_5, # build, # tokens, # tokens_file ...), `# kind` column line and `tensor`
                rows (name occurrence type ne0..ne3 bytes sum op contig logical src0 src1), the `# int` column line and
                `int` rows, then the trailer `# complete <tensor rows> 0`.  Files `<stem>.<occ>.f32` (+ `.i32` for ids).
  greedy.tsv    the 7-column `--gen` argmax format (`read_greedy`): id n_tokens argmax top5_ids top5_logits gen_ids margins.
  rows:  l_out-<il>      f32 [4096, 4 streams, 4 positions]  residual after layer il, the last 4 prompt positions
         ffn_moe_topk-<il>     i32 [10, 4]  routed expert ids of those positions (descending biased score)
         ffn_moe_weights-<il>  f32 [10, 4]  their final weights (renormalized x learned per-expert scale)
         indexer_top_k-<il>    i32 [2048]   (long prompt, the 6 DSA layers) the last position's selected non-window
                                            positions, ascending
         result_norm  f32 [4096]; result_output_raw / result_output  f32 [201216]  the last position, before / after MultiMax
"""
import os
import sys


def _early_pylib():
    # A numpy 2.x OpenBLAS directory is prepended before numpy is imported.  The self-test never needs it.
    for i, a in enumerate(sys.argv):
        if a == "--pylib" and i + 1 < len(sys.argv):
            sys.path.insert(0, sys.argv[i + 1])
        elif a.startswith("--pylib="):
            sys.path.insert(0, a.split("=", 1)[1])


_early_pylib()
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per pool thread; the pool is the parallelism

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

np.seterr(over="ignore", under="ignore")
F32 = np.float32
NEG_INF = F32(-np.inf)


# ---------------------------------------------------------------------------------------------------------------------
# Dequantizers
# ---------------------------------------------------------------------------------------------------------------------
def _fp8_e4m3_value(b):
    """One e4m3fn byte (sign 1, exp 4 bias 7, mantissa 3; exp 15 + mantissa 7 is NaN, no inf)."""
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 7
    if e == 15 and m == 7:
        return float("nan")
    if e == 0:
        return s * (m / 8.0) * 2.0 ** -6
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)


FP8_LUT = np.array([_fp8_e4m3_value(b) for b in range(256)], dtype=F32)

# MXFP4 (OCP e2m1): sign bit 8, magnitudes [0, .5, 1, 1.5, 2, 3, 4, 6]; crates/hf/src/modelopt.rs fixes the nibble order:
# the low nibble is the even element of the pair.
_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)
_E2M1 = np.concatenate([_E2M1, -_E2M1])
FP4_LUT2 = np.stack([_E2M1[np.arange(256) & 15], _E2M1[np.arange(256) >> 4]], axis=1).astype(F32)  # byte -> (even, odd)


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(F32)


def dequant_fp8(codes, scale, blk=128):
    """FP8 e4m3 rows times their block scale (`weight_scale_inv`, multiplied: vllm quant_config.dequantize_block_fp8).

    codes uint8 [R, K] whose first row starts a block row; scale f32 [ceil(R/blk), ceil(K/blk)].  A partial block
    repeats its scale and is sliced (N=576 of kv_a_proj_with_mqa ends in a half block)."""
    R, K = codes.shape
    w = FP8_LUT[codes]
    scx = np.repeat(scale, blk, axis=1)[:, :K]
    nfull = R // blk
    if nfull:
        w[: nfull * blk].reshape(nfull, blk, K)[...] *= scx[:nfull, None, :]
    if R % blk:
        w[nfull * blk :] *= scx[nfull][None, :]
    return w


def e8m0_to_f32(e):
    """2^(e-127); e = 255 is NaN, e = 0 is the denormal 2^-127."""
    e = np.asarray(e, dtype=np.uint8)
    s = (e.astype(np.uint32) << 23).view(F32).copy()
    s[e == 0] = F32(2.0 ** -127)
    s[e == 255] = F32(np.nan)  # bit pattern 0x7f800000 would be inf
    return s


def dequant_mxfp4(packed, e8m0, block=32):
    """MXFP4 rows: packed uint8 [R, K/2] (low nibble first), e8m0 uint8 [R, K/block] scale per block of 32."""
    R, K2 = packed.shape
    w = FP4_LUT2[packed].reshape(R, K2 * 2)
    s = e8m0_to_f32(e8m0)
    w.reshape(R, K2 * 2 // block, block)[...] *= s[:, :, None]
    return w


# ---------------------------------------------------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------------------------------------------------
def sigmoid(x):
    return F32(1.0) / (F32(1.0) + np.exp(-x))


def _mean_sq(x):
    return np.einsum("...i,...i->...", x, x, dtype=np.float64) / x.shape[-1]


def rms_norm(x, w=None, eps=1e-6):
    """x * rsqrt(mean(x^2) + eps) [* w]   (ggml rms_norm; PR build_norm LLM_NORM_RMS)."""
    y = x * (1.0 / np.sqrt(_mean_sq(x) + eps)).astype(F32)[..., None]
    return y * w if w is not None else y


def layer_norm(x, w, b, eps=1e-6):
    """LayerNorm with weight and bias: the indexer's k_norm (PR build_mla_qk: LLM_NORM, f_norm_eps = 1e-6)."""
    mu = x.mean(axis=-1, keepdims=True, dtype=np.float64)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True, dtype=np.float64)
    return ((x - mu) / np.sqrt(var + eps)).astype(F32) * w + b


def situ(g, u, beta, lb):
    """SiTU-GLU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)   (PR build_ffn LLM_FFN_SITU, diff 703-717)."""
    a = np.tanh(g * F32(1.0 / beta)) * F32(beta)
    v = np.tanh(u * F32(1.0 / lb)) * F32(lb)
    return a * sigmoid(g) * v


def multimax(logits, r, t):
    """l + t0*relu(r0-l) + t1*relu(l-r1) + t2*relu(r2-l)^2 + t3*relu(l-r3)^2
    (PR build_multimax, diff 1703-1720; VLLM nvidia/model.py:129-150 Ernie5_5ParallelLMHead.multimax)."""
    relu = lambda z: np.maximum(z, F32(0.0))  # noqa: E731
    out = logits + t[0] * relu(r[0] - logits)
    out = out + t[1] * relu(logits - r[1])
    out = out + t[2] * np.square(relu(r[2] - logits))
    out = out + t[3] * np.square(relu(logits - r[3]))
    return out.astype(F32)


# ---------------------------------------------------------------------------------------------------------------------
# mHC (4 streams)
# ---------------------------------------------------------------------------------------------------------------------
def hc_split(mixes, scale, base, hc_eps, iters, hc=4):
    """mixes [T, 6*hc] -> pre [T,hc], post [T,hc], comb [T,hc,hc] (comb[i][j]: source stream i, destination j).

    pre = sigmoid(mix[:hc]*s0 + base[:hc]) + eps; post = 2*sigmoid(mix[hc:2hc]*s1 + base[hc:2hc]);
    comb = softmax_row(mix[2hc:]*s2 + base[2hc:]) + eps, then one column normalization, then (iters-1) x (row, column)
    normalizations, each divisor + eps.   Sources: VLLM tests/models/ernie5_5/test_mhc.py:18-33 (_mhc_pre_ref);
    DS4 deepseek4.cpp:308-345 build_hc_sinkhorn, :357-411 build_hc_mixes, PR diff 1166-1200 build_hc_split_mixes."""
    pre = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + F32(hc_eps)
    post = F32(2.0) * sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + F32(hc_eps)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + F32(hc_eps))
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + F32(hc_eps))
        comb = comb / (comb.sum(axis=-2, keepdims=True) + F32(hc_eps))
    return pre, post, comb


def hc_pre(x, fn, base, scale, rms_eps, hc_eps, iters):
    """x [T,hc,H], fn f32 [hc*H, 6*hc] (checkpoint `mapping_proj.weight`, stored [in,out]).

    The ERNIE factor is 1/(sqrt(mean(x^2)) + eps), not rsqrt(mean+eps) (PR build_hc_pre_ernie, diff 1424-1446;
    VLLM test_mhc.py:22).  layer input = sum_i pre[i]*x[i]."""
    T, hc, H = x.shape
    flat = x.reshape(T, hc * H)
    rms = (np.sqrt(_mean_sq(flat)) + rms_eps).astype(F32)[:, None]
    mixes = (flat @ fn) / rms
    pre, post, comb = hc_split(mixes, scale, base, hc_eps, iters, hc)
    layer_in = pre[:, 0, None] * x[:, 0]
    for i in range(1, hc):
        layer_in += pre[:, i, None] * x[:, i]
    return layer_in, post, comb


def hc_post(xo, res, post, comb):
    """new[j] = post[j]*xo + sum_i comb[i][j]*res[i]   (PR graph hc_post via DS4 deepseek4.cpp:413-448 build_hc_post:
    cur[dst] = x*post[dst] + sum_src res[src]*comb[src][dst]).  xo [T,H], res [T,hc,H]."""
    T, hc, H = res.shape
    new = np.empty_like(res)
    for j in range(hc):
        acc = post[:, j, None] * xo
        for i in range(hc):
            acc = acc + comb[:, i, j, None] * res[:, i]
        new[:, j] = acc
    return new


def hc_head(x, fn, base, scale, rms_eps, hc_eps):
    """The head keeps the V4 factor rsqrt(mean+eps): mixes = rms_norm(flat) @ fn (fn stored [in,out]),
    pre = sigmoid(mixes*scale + base) + eps, out = sum_i pre[i]*x[i]   (DS4 deepseek4.cpp:450-470 build_hc_head;
    the PR graph calls it for the final head, diff 1790)."""
    T, hc, H = x.shape
    flat = x.reshape(T, hc * H)
    mixes = rms_norm(flat, None, rms_eps) @ fn
    pre = sigmoid(mixes * scale[0] + base) + F32(hc_eps)
    out = pre[:, 0, None] * x[:, 0]
    for i in range(1, hc):
        out += pre[:, i, None] * x[:, i]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# RoPE (ggml semantics)
# ---------------------------------------------------------------------------------------------------------------------
def plain_inv_freq(dims, base):
    return base ** (-(2.0 * np.arange(dims // 2)) / dims)


def yarn_inv_freq(dims, base, factor, orig_ctx, beta_fast, beta_slow, ext_factor=1.0):
    """ggml rope_yarn: per pair i, theta = interp*(1-mix) + extrap*mix, interp = extrap/factor,
    mix = ramp(i)*ext_factor, ramp = 1 - clamp((i - lo)/max(0.001, hi-lo), 0, 1), lo/hi the corr dims of beta_fast/slow.
    The magnitude factor (1 + 0.1 ln factor) is cancelled by the engine (models.h:1277 dsv4_rope_attn_factor): none here."""
    def corr_dim(nrot):
        return dims * math.log(orig_ctx / (nrot * 2.0 * math.pi)) / (2.0 * math.log(base))

    lo = max(0, math.floor(corr_dim(beta_fast)))
    hi = min(dims - 1, math.ceil(corr_dim(beta_slow)))
    extrap = plain_inv_freq(dims, base)
    interp = extrap / factor
    ramp = 1.0 - np.clip((np.arange(dims // 2) - lo) / max(0.001, hi - lo), 0.0, 1.0)
    mix = ramp * ext_factor
    return interp * (1.0 - mix) + extrap * mix


def rope_cs(pos, inv):
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(F32), np.sin(ang).astype(F32)


def rope(x, cs, offset, dims, mode, inverse=False):
    """Rotate x[..., offset:offset+dims] (token axis 0).  mode 'norm': pairs (2i, 2i+1) (ggml NORM, interleaved);
    'neox': pairs (i, i+dims/2).  inverse rotates by -angle (ggml_rope_ext_back)."""
    c, s = cs
    shp = (x.shape[0],) + (1,) * (x.ndim - 2) + (dims // 2,)
    c = c.reshape(shp)
    s = (-s if inverse else s).reshape(shp)
    y = x.copy()
    xs = x[..., offset : offset + dims]
    if mode == "norm":
        a, b = xs[..., 0::2], xs[..., 1::2]
        y[..., offset : offset + dims : 2] = a * c - b * s
        y[..., offset + 1 : offset + dims : 2] = a * s + b * c
    else:
        h = dims // 2
        a, b = xs[..., :h], xs[..., h:]
        y[..., offset : offset + h] = a * c - b * s
        y[..., offset + h : offset + dims] = a * s + b * c
    return y


# ---------------------------------------------------------------------------------------------------------------------
# Attention cores
# ---------------------------------------------------------------------------------------------------------------------
def attend_sink(Q, K, V, valid, sink, scale):
    """softmax over [scores..., sink_h] per head, the sink contributes no value:
    out = sum_s p_s V_s / (sum_s p_s + exp(sink - m)).  Q [t,NH,Dq], K [S,Dq], V [S,Dv], valid [t,S], sink [NH].
    (ggml soft_max with sinks; PR build_attn_mha(... layer.attn_sinks ...), diff 1597.)"""
    t, NH, Dq = Q.shape
    S = K.shape[0]
    s = (Q.reshape(t * NH, Dq) @ K.T).reshape(t, NH, S) * F32(scale)
    s = np.where(valid[:, None, :], s, NEG_INF)
    m = np.maximum(s.max(axis=-1), sink[None, :])
    p = np.exp(s - m[..., None])
    den = p.sum(axis=-1) + np.exp(sink[None, :] - m)
    out = (p.reshape(t * NH, S) @ V).reshape(t, NH, V.shape[1])
    return out / den[..., None]


def vha_matrix(U, V):
    """VHA head mix m = I + V U^T, o[h] <- sum_g m[h,g] o[g]   (PR conversion diff 252-262; VLLM attention.py:136-175)."""
    return np.eye(U.shape[0], dtype=F32) + V @ U.T


def head_mix(out, m):
    """out [t,NH,D] -> sum_g m[h,g] out[g] (PR build_head_mix, diff 1134-1156)."""
    return np.einsum("hg,tgd->thd", m, out, optimize=True).astype(F32)


# --- HCA (compress ratio 128, window 128) ----------------------------------------------------------------------------
class HcaState:
    """What a sequence carries per HCA layer: the raw KV of the last `window` positions, the compressor's pending rows of
    the unfinished block, and the finished compressed blocks."""

    def __init__(self, D):
        self.raw = np.zeros((0, D), F32)
        self.raw_pos0 = 0
        self.pend_kv = np.zeros((0, D), F32)
        self.pend_sc = np.zeros((0, D), F32)
        self.comps = np.zeros((0, D), F32)


def hca_compress(kv_rows, sc_rows, ratio):
    """Per channel softmax over the `ratio` rows of a block, weights times kv, summed
    (DS4 deepseek4.cpp:472-515 build_hca_compressed_kv_from_state).  [nb*ratio, D] -> [nb, D]."""
    D = kv_rows.shape[1]
    kv = kv_rows.reshape(-1, ratio, D)
    sc = sc_rows.reshape(-1, ratio, D)
    w = np.exp(sc - sc.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return (w * kv).sum(axis=1)


def hca_update(st, kv, ckv, csc, p0, ratio, window, comp_post):
    """Append the t new positions p0..p0+t-1: raw kv rows (normed + roped), compressor rows (wkv.x, wgate.x + ape[pos%ratio]).
    A block finishes at position ratio*j + ratio - 1; comp_post(comp, starts) = RMS norm then RoPE at the block's first
    position (DS4 deepseek4.cpp:504-515 comp = rms_norm then rope at comp_pos; llama-kv-cache-dsv4.cpp:544-546 comp_pos = pos+1-ratio)."""
    t = kv.shape[0]
    if st.raw.shape[0]:
        assert st.raw_pos0 + st.raw.shape[0] == p0, "HCA state not contiguous"
    raw = np.concatenate([st.raw, kv])
    pos0 = st.raw_pos0 if st.raw.shape[0] else p0
    assert st.pend_kv.shape[0] == p0 % ratio, "pending compressor rows != p0 % ratio"
    allkv = np.concatenate([st.pend_kv, ckv])
    allsc = np.concatenate([st.pend_sc, csc])
    nb = allkv.shape[0] // ratio
    if nb:
        j0 = p0 // ratio
        comp = hca_compress(allkv[: nb * ratio], allsc[: nb * ratio], ratio)
        starts = (j0 + np.arange(nb)) * ratio
        st.comps = np.concatenate([st.comps, comp_post(comp, starts)])
    st.pend_kv = allkv[nb * ratio :]
    st.pend_sc = allsc[nb * ratio :]
    keep = min(raw.shape[0], window)  # the next query at p0+t sees at most window-1 old cells; keep window
    st.raw = raw[raw.shape[0] - keep :]
    st.raw_pos0 = pos0 + (raw.shape[0] - keep)
    return raw, pos0  # all raw rows incl. the cache's (attention reads from these)


def hca_attend_chunk(q, qpos, raw, raw_pos0, comps, ratio, window, sink, scale):
    """Queries qpos (consecutive) against the raw window cells and the visible compressed blocks.  q [t,NH,D] (roped).
    Raw cell p is visible iff 0 <= qpos-p < window; block j iff j < (qpos+1)//ratio
    (llama.cpp src/llama-graph.cpp:932 dsv4_set_kq_mask; src/llama-kv-cache-dsv4.cpp:520 n_visible = (pos+1)/ratio).  K = V (one KV head)."""
    lo = max(raw_pos0, int(qpos[0]) - window + 1)
    hi = int(qpos[-1])
    rawk = raw[lo - raw_pos0 : hi - raw_pos0 + 1]
    kpos = np.arange(lo, hi + 1)
    nb = min(comps.shape[0], (int(qpos[-1]) + 1) // ratio)
    K = np.concatenate([rawk, comps[:nb]]) if nb else rawk
    qv = qpos[:, None]
    valid_raw = (kpos[None, :] <= qv) & (qv - kpos[None, :] < window)
    valid_blk = np.arange(nb)[None, :] < ((qv + 1) // ratio)
    valid = np.concatenate([valid_raw, valid_blk], axis=1)
    return attend_sink(q, K, K, valid, sink, scale)


# --- MLA + DSA indexer ------------------------------------------------------------------------------------------------
def indexer_scores(idx_q, idx_w, idx_k):
    """score[t,s] = sum_h relu(q[t,h] . k[s]) * w[t,h]   (PR build_mla_top_k non-fused path, diff 1556-1566)."""
    t, NH, D = idx_q.shape
    kq = np.maximum((idx_q.reshape(t * NH, D) @ idx_k.T).reshape(t, NH, -1), F32(0.0))
    return (kq * idx_w[:, :, None]).sum(axis=1)


def indexer_select(scores, qpos, topk, window):
    """Boolean [t,N] of the keys each query attends: the window cells (score +inf in the PR) and the top `topk` of the
    positions older than the window; all of them while there are <= topk (PR build_mla_top_k, diff 1568-1574:
    n_top = min(n_kv, topk + window), window cells +inf).  Non-trivial once qpos - window + 1 > topk, i.e. qpos >= 2176."""
    t, N = scores.shape
    sel = np.zeros((t, N), dtype=bool)
    for i in range(t):
        p = int(qpos[i])
        wlo = max(0, p - window + 1)
        sel[i, wlo : p + 1] = True
        c = wlo  # candidates 0..wlo-1
        if c <= topk:
            sel[i, :c] = True
        else:
            idx = np.argpartition(-scores[i, :c], topk - 1)[:topk]
            sel[i, idx] = True
    return sel


def mla_attend_chunk(Qc, Kc, kv_lora, qpos, sel, sink, scale):
    """Absorbed MLA: Qc [t,NH,kv_lora+rope] = [q_nope absorbed through k_b | q_pe], Kc [N, kv_lora+rope] = [kv | k_pe],
    V = the latent Kc[:, :kv_lora]; key index = position.  (PR build_mla_out, diff 1579-1613; K/V views of one cache row.)
    Returns the latent attention output [t,NH,kv_lora] (the caller applies wv_b)."""
    N = Kc.shape[0]
    valid = np.arange(N)[None, :] <= qpos[:, None]
    if sel is not None:
        valid = valid & sel
    return attend_sink(Qc, Kc, Kc[:, :kv_lora], valid, sink, scale)


# --- routing -----------------------------------------------------------------------------------------------------------
def route(l0, l1, bias, escale, topk):
    """Two sigmoid routers summed; the bias-corrected scores pick, the plain scores weight, renormalized over the picked
    and times the learned per-expert scale:  w = p[id] / max(sum_sel p, 1e-20) * escale[id].
    PR build_latent_moe (diff 1643-1672); VLLM nvidia/moe.py:170-190 _ernie5_5_routing_torch (the picked are unordered
    there; here descending biased score, ties to the lower id)."""
    probs = sigmoid(l0) + sigmoid(l1)
    ids = np.argsort(-(probs + bias[None, :]), axis=1, kind="stable")[:, :topk]
    psel = np.take_along_axis(probs, ids, axis=1)
    w = psel / np.maximum(psel.sum(axis=1, keepdims=True), F32(1e-20)) * escale[ids]
    return ids.astype(np.int64), w.astype(F32)


# ---------------------------------------------------------------------------------------------------------------------
# safetensors store and the matmul engine
# ---------------------------------------------------------------------------------------------------------------------
_ST_ITEMSIZE = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "F8_E8M0": 1, "U8": 1}


def _pread_full(fd, n, off):
    b = os.pread(fd, n, off)
    if len(b) == n:
        return b
    parts, got = [b], len(b)
    while got < n:
        c = os.pread(fd, n - got, off + got)
        if not c:
            raise EOFError(f"short read at {off + got} of {n - got} more bytes")
        parts.append(c)
        got += len(c)
    return b"".join(parts)


class Store:
    """A safetensors checkpoint read with os.pread (the page cache stays out of RSS)."""

    def __init__(self, root, index=None):
        self.root = root
        if index is None:
            with open(os.path.join(root, "model.safetensors.index.json")) as f:
                index = json.load(f)["weight_map"]
        self.index = index
        self._fd, self._hdr, self._lock = {}, {}, threading.Lock()

    def _file(self, fname):
        with self._lock:
            if fname not in self._fd:
                fd = os.open(os.path.join(self.root, fname), os.O_RDONLY)
                hlen = int.from_bytes(_pread_full(fd, 8, 0), "little")
                hdr = json.loads(_pread_full(fd, hlen, 8))
                self._fd[fname], self._hdr[fname] = fd, (hdr, 8 + hlen)
            return self._fd[fname], self._hdr[fname]

    def has(self, name):
        return name in self.index

    def meta(self, name):
        if name not in self.index:
            raise KeyError(f"tensor {name!r} is not in the checkpoint index")
        fd, (hdr, base) = self._file(self.index[name])
        m = hdr[name]
        return fd, m["dtype"], m["shape"], base + m["data_offsets"][0]

    def read(self, name, byte0=0, nbytes=None):
        fd, dt, shape, off = self.meta(name)
        total = _ST_ITEMSIZE[dt] * int(np.prod(shape, dtype=np.int64))
        if nbytes is None:
            nbytes = total - byte0
        assert 0 <= byte0 and byte0 + nbytes <= total, f"{name}: read {byte0}+{nbytes} past {total}"
        return _pread_full(fd, nbytes, off + byte0)

    def close(self):
        for fd in self._fd.values():
            os.close(fd)
        self._fd.clear()


class Eng:
    """Dequantizing matmul over a Store, parallel over row chunks of the weight."""

    def __init__(self, store, nthreads=32, blk=128):
        self.store, self.blk = store, blk
        self.pool = ThreadPoolExecutor(nthreads)
        self.nthreads = nthreads
        self._scale = {}
        self.bytes_read = 0

    def layer_begin(self):
        self._scale = {}

    def f32(self, name):
        """A whole small tensor as f32 (BF16 or F32 in the file)."""
        _, dt, shape, _ = self.store.meta(name)
        raw = self.store.read(name)
        self.bytes_read += len(raw)
        if dt == "BF16":
            return bf16_to_f32(np.frombuffer(raw, np.uint16)).reshape(shape)
        if dt == "F32":
            return np.frombuffer(raw, F32).reshape(shape).copy()
        raise ValueError(f"{name}: dtype {dt} is not a plain tensor")

    def shape(self, name):
        _, dt, shape, _ = self.store.meta(name)
        return (shape[0], shape[1] * 2) if dt == "I8" else tuple(shape)

    def w_rows(self, name, r0, r1):
        """Rows r0:r1 of a weight as f32 [r1-r0, K]."""
        _, dt, shape, _ = self.store.meta(name)
        rowelems = int(np.prod(shape[1:], dtype=np.int64))
        rb = rowelems * _ST_ITEMSIZE[dt]
        raw = self.store.read(name, r0 * rb, (r1 - r0) * rb)
        self.bytes_read += len(raw)
        if dt == "BF16":
            return bf16_to_f32(np.frombuffer(raw, np.uint16)).reshape(r1 - r0, rowelems)
        if dt == "F32":
            return np.frombuffer(raw, F32).reshape(r1 - r0, rowelems)
        if dt == "F8_E4M3":
            assert r0 % self.blk == 0, f"{name}: chunk start {r0} is not a block row"
            sc = self._scale.get(name)
            if sc is None:
                sc = self.f32(name + "_scale_inv")
                self._scale[name] = sc
            codes = np.frombuffer(raw, np.uint8).reshape(r1 - r0, rowelems)
            return dequant_fp8(codes, sc[r0 // self.blk : (r1 + self.blk - 1) // self.blk], self.blk)
        if dt == "I8":
            sname = name + "_scale_inv"
            _, sdt, sshape, _ = self.store.meta(sname)
            assert sdt == "F8_E8M0", f"{sname}: {sdt}"
            srb = sshape[1]
            sraw = self.store.read(sname, r0 * srb, (r1 - r0) * srb)
            self.bytes_read += len(sraw)
            return dequant_mxfp4(
                np.frombuffer(raw, np.uint8).reshape(r1 - r0, rowelems),
                np.frombuffer(sraw, np.uint8).reshape(r1 - r0, srb),
            )
        raise ValueError(f"{name}: dtype {dt} is not a weight")

    def _rows_per_task(self, N):
        per = -(-N // (2 * self.nthreads))
        per = -(-per // self.blk) * self.blk
        return int(min(max(per, self.blk), 2048))

    def linear(self, name, X, group_rows=None):
        """Y = X @ W.T for the weight `name` [N,K].  group_rows: the weight's rows split into groups of that many, group g
        reading X[:, g*K:(g+1)*K] (the grouped low-rank o projection, PR wo_a with o_groups = 8)."""
        N, K = self.shape(name)
        T = X.shape[0]
        per = self._rows_per_task(N)
        if group_rows is not None:
            per = min(per, group_rows)
            while group_rows % per:
                per -= self.blk
        spans = [(r, min(r + per, N)) for r in range(0, N, per)]
        out = np.empty((T, N), F32)

        def task(span):
            r0, r1 = span
            W = self.w_rows(name, r0, r1)
            Xg = X if group_rows is None else X[:, (r0 // group_rows) * K : (r0 // group_rows + 1) * K]
            out[:, r0:r1] = Xg @ W.T

        for _ in self.pool.map(task, spans):
            pass
        return out

    def pmap(self, fn, items):
        return list(self.pool.map(fn, items))

    def par_rows(self, fn, n, chunk=256):
        """fn(r0, r1) over row spans in parallel; results in order."""
        return self.pmap(lambda s: fn(*s), [(r, min(r + chunk, n)) for r in range(0, n, chunk)])


# ---------------------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------------------
class Cfg:
    """config.json values the forward uses (asserted against the facts the code hardwires)."""

    def __init__(self, d):
        self.H = d["hidden_size"]
        self.L = d["num_hidden_layers"]
        self.NH = d["num_attention_heads"]
        self.HD = d["head_dim"]
        self.vocab = d["vocab_size"]
        self.eps = d["rms_norm_eps"]
        self.hc = d["hc_mult"]
        self.hc_eps = d["hc_eps"]
        self.iters = d["hc_sinkhorn_iters"]
        self.ratios = d["compress_ratios"]
        self.window = d["sliding_window"]
        self.rope_dim = d["qk_rope_head_dim"]
        self.o_groups = d["o_groups"]
        self.o_lora = d["o_lora_rank"]
        self.nope = d["hybrid_mla_qk_nope_head_dim"]
        self.v_mla = d["hybrid_mla_v_head_dim"]
        self.kv_lora = d["hybrid_mla_kv_lora_rank"]
        self.idx_nh = d["index_n_heads"]
        self.idx_hd = d["index_head_dim"]
        self.topk = d["index_topk"]
        self.n_exp = d["n_routed_experts"]
        self.k_exp = d["num_experts_per_tok"]
        self.dense_lead = d["first_k_dense_replace"]
        self.situ_beta = d["activation_situ_beta"]
        self.situ_lb = d["activation_situ_linear_beta"]
        self.theta = float(d["rope_theta"])
        self.hca_base = float(d["compress_rope_theta"])
        ys = d["rope_scaling"]
        self.yarn = (ys["factor"], ys["original_max_position_embeddings"], ys["beta_fast"], ys["beta_slow"])
        assert d["hc_mult"] == 4 and self.NH == 64 and self.HD == 512 and self.rope_dim == 64
        assert d["hybrid_mla_num_attention_heads"] == 64 and self.o_groups == 8
        assert all(r in (128, -2) for r in self.ratios[: self.L]) and len(self.ratios) >= self.L
        assert d["hidden_act"] == "situ" and d["scoring_func"] == "sigmoid" and d["rope_interleave"] is True
        assert d["use_vha_attention"] and d["use_gated_attn"] and d["gated_attn_act"] == "sigmoid"
        assert d["multimax"] == "lm_head" and d["moe_split_feature_routing"] and d["norm_topk_prob"]

    def is_dsa(self, il):
        return self.ratios[il] == -2


class Seq:
    """One sequence's caches."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.n = 0  # positions processed
        self.hca = {}
        self.mla = {}


class Model:
    def __init__(self, eng, cfg, log=None):
        self.E, self.c, self.log = eng, cfg, log or (lambda s: None)
        c = cfg
        self.inv_hca = yarn_inv_freq(c.rope_dim, c.hca_base, *c.yarn)  # HCA layers: YaRN at compress_rope_theta
        self.inv_mla = plain_inv_freq(c.rope_dim, c.theta)  # MLA and indexer: plain RoPE at rope_theta (PR diff 1469-1471)

    def W(self, il, key):
        return self.E.f32(f"model.layers.{il}.{key}")

    # --- mHC pieces ---------------------------------------------------------------------------------------------------
    def _hc_params(self, il, kind):
        p = f"model.layers.{il}.{kind}_hyper_connection."
        fn = self.E.f32(p + "mapping_proj.weight")  # [4H, 24] stored [in, out] (PR conversion diff 236: .t())
        base = self.E.f32(p + "bias")
        scale = np.concatenate([self.E.f32(p + f"alpha_{n}").reshape(1) for n in ("pre", "post", "res")])
        return fn, base, scale

    def _hc_pre(self, x, il, kind):
        c = self.c
        fn, base, scale = self._hc_params(il, kind)
        parts = self.E.par_rows(lambda a, b: hc_pre(x[a:b], fn, base, scale, c.eps, c.hc_eps, c.iters), x.shape[0])
        return tuple(np.concatenate(p) for p in zip(*parts))

    def _hc_post(self, xo, res, post, comb):
        parts = self.E.par_rows(lambda a, b: hc_post(xo[a:b], res[a:b], post[a:b], comb[a:b]), xo.shape[0])
        return np.concatenate(parts)

    # --- HCA attention ------------------------------------------------------------------------------------------------
    def _hca_attention(self, il, cur, segs):
        c, E = self.c, self.E
        P = f"model.layers.{il}.self_attn."
        T = cur.shape[0]
        # q latent: down proj, RMS norm with weight (DS4 deepseek4.cpp:905-933 build_attention_impl)
        qr = rms_norm(E.linear(P + "linear_q_down_proj.weight", cur), self.W(il, "self_attn.q_layernorm.weight"), c.eps)
        qraw = E.linear(P + "linear_q_up_proj.weight", qr)  # [T, NH*HD]
        kvraw = rms_norm(E.linear(P + "linear_kv_proj.weight", cur), self.W(il, "self_attn.kv_layernorm.weight"), c.eps)
        ca = P + "core_attention.compressor."
        ckv = E.linear(ca + "linear_wkv.weight", cur)
        csc = E.linear(ca + "linear_wgate.weight", cur)
        ape = self.W(il, "self_attn.core_attention.compressor.ape")  # [ratio, HD]
        cnorm = self.W(il, "self_attn.core_attention.compressor.norm.weight")
        sink = self.W(il, "self_attn.core_attention.attn_sink")
        m = vha_matrix(self.W(il, "self_attn.vha_postmix_U"), self.W(il, "self_attn.vha_postmix_V"))
        gate = sigmoid(E.linear(P + "gate_proj.weight", qr))  # sigmoid gate from the normalized q latent (PR diff 1255-1262)
        ratio, off, rd = c.ratios[il], c.HD - c.rope_dim, c.rope_dim
        scale = 1.0 / math.sqrt(c.HD)
        out = np.empty((T, c.NH, c.HD), F32)
        for seq, t0, t1 in segs:
            t = t1 - t0
            qpos = seq.n + np.arange(t)
            cs = rope_cs(qpos, self.inv_hca)
            q = qraw[t0:t1].reshape(t, c.NH, c.HD)
            q = rope(rms_norm(q, None, c.eps), cs, off, rd, "norm")  # weightless per-head RMS norm, then YaRN RoPE on the last 64
            kv = rope(kvraw[t0:t1], cs, off, rd, "norm")
            sc = csc[t0:t1] + ape[qpos % ratio]  # score = wgate.x + ape[pos % ratio]

            def comp_post(comp, starts):
                return rope(rms_norm(comp, cnorm, c.eps), rope_cs(starts, self.inv_hca), off, rd, "norm")

            st = seq.hca.setdefault(il, HcaState(c.HD))
            raw, pos0 = hca_update(st, kv, ckv[t0:t1], sc, seq.n, ratio, c.window, comp_post)
            comps = st.comps
            spans = [(a, min(a + 128, t)) for a in range(0, t, 128)]

            def chunk(sp):
                a, b = sp
                o = hca_attend_chunk(q[a:b], qpos[a:b], raw, pos0, comps, ratio, c.window, sink, scale)
                o = rope(o, rope_cs(qpos[a:b], self.inv_hca), off, rd, "norm", inverse=True)  # attn_derope (DS4 deepseek4.cpp:1205-1211)
                return head_mix(o, m)  # VHA after the inverse RoPE (PR diff 1181-1185)

            for (a, b), o in zip(spans, E.pmap(chunk, spans)):
                out[t0 + a : t0 + b] = o
        # grouped low-rank output: 8 groups of 8 heads x 512 -> 1024 each, sigmoid gate, then wo_b
        oa = E.linear(P + "linear_o_group_proj", out.reshape(T, c.NH * c.HD), group_rows=c.o_lora)
        oa *= gate
        return E.linear(P + "o_proj.weight", oa)

    # --- MLA + DSA attention ------------------------------------------------------------------------------------------
    def _mla_attention(self, il, cur, segs, rec):
        c, E = self.c, self.E
        P = f"model.layers.{il}.self_attn."
        T = cur.shape[0]
        qr = rms_norm(E.linear(P + "q_a_proj.weight", cur), self.W(il, "self_attn.q_a_layernorm.weight"), c.eps)
        gate = sigmoid(E.linear(P + "gate_proj.weight", qr))  # [T, NH*v] (PR build_mla_qk diff 1459)
        qfull = E.linear(P + "q_b_proj.weight", qr).reshape(T, c.NH, c.nope + c.rope_dim)
        kvpe = E.linear(P + "kv_a_proj_with_mqa.weight", cur)  # [T, kv_lora + rope]
        kvc = rms_norm(kvpe[:, : c.kv_lora], self.W(il, "self_attn.kv_a_layernorm.weight"), c.eps)
        kb_w = E.w_rows(P + "kv_b_proj.weight", 0, c.NH * (c.nope + c.v_mla)).reshape(c.NH, c.nope + c.v_mla, c.kv_lora)
        k_b = kb_w[:, : c.nope]  # [NH, nope, kv_lora]: q_abs[h] = k_b[h]^T q_nope[h]   (PR conversion diff 263-274)
        v_b = kb_w[:, c.nope :]  # [NH, v, kv_lora]: out[h] = v_b[h] latent[h]
        sink = self.W(il, "self_attn.core_attention.softmax_offset")
        m = vha_matrix(self.W(il, "self_attn.vha_postmix_U"), self.W(il, "self_attn.vha_postmix_V"))
        ip = P + "core_attention.indexer."
        idx_w_all = E.linear(ip + "weights_proj.weight", cur) * F32(1.0 / math.sqrt(c.idx_hd * c.idx_nh))  # PR diff 1544
        idx_k_raw = E.linear(ip + "wk.weight", cur)
        idx_q_raw = E.linear(ip + "wq_b.weight", qr)  # [T, idx_nh*idx_hd], from the normalized q latent
        kn_w, kn_b = self.W(il, "self_attn.core_attention.indexer.k_norm.weight"), self.W(il, "self_attn.core_attention.indexer.k_norm.bias")
        scale = 1.0 / math.sqrt(c.nope + c.rope_dim)
        out = np.empty((T, c.NH, c.v_mla), F32)
        for seq, t0, t1 in segs:
            t = t1 - t0
            qpos = seq.n + np.arange(t)
            cs = rope_cs(qpos, self.inv_mla)
            q = qfull[t0:t1]
            q_pe = rope(q[..., c.nope :], cs, 0, c.rope_dim, "norm")  # plain interleaved RoPE at rope_theta
            k_pe = rope(kvpe[t0:t1, c.kv_lora :], cs, 0, c.rope_dim, "norm")
            # indexer key: LayerNorm then NEOX RoPE on its first 64 dims (PR diff 1486-1492); the Hadamard rotation of the
            # reference leaves q.k unchanged and is left out (PR comment).
            ik = rope(layer_norm(idx_k_raw[t0:t1], kn_w, kn_b, c.eps), cs, 0, c.rope_dim, "neox")
            st = seq.mla.setdefault(il, {"kv": np.zeros((0, c.kv_lora + c.rope_dim), F32), "ik": np.zeros((0, c.idx_hd), F32)})
            assert st["kv"].shape[0] == seq.n, "MLA cache not contiguous"
            st["kv"] = np.concatenate([st["kv"], np.concatenate([kvc[t0:t1], k_pe], axis=1)])
            st["ik"] = np.concatenate([st["ik"], ik])
            Kc, IK = st["kv"], st["ik"]
            iq = rope(idx_q_raw[t0:t1].reshape(t, c.idx_nh, c.idx_hd), cs, 0, c.rope_dim, "neox")
            iw = idx_w_all[t0:t1]
            spans = [(a, min(a + 32, t)) for a in range(0, t, 32)]

            def chunk(sp):
                a, b = sp
                q_abs = np.einsum("thd,hdk->thk", q[a:b, :, : c.nope], k_b, optimize=True).astype(F32)
                Qc = np.concatenate([q_abs, q_pe[a:b]], axis=2)
                N = int(qpos[b - 1]) + 1
                sel = None
                if N - c.window > c.topk:  # an older-than-window set larger than top_k: the indexer selects
                    sc = indexer_scores(iq[a:b], iw[a:b], IK[:N])
                    sel = indexer_select(sc, qpos[a:b], c.topk, c.window)
                lat = mla_attend_chunk(Qc, Kc[:N], c.kv_lora, qpos[a:b], sel, sink, scale)
                o = np.einsum("thk,hvk->thv", lat, v_b, optimize=True).astype(F32)
                return head_mix(o, m), sel

            res = E.pmap(chunk, spans)
            for (a, b), (o, sel) in zip(spans, res):
                out[t0 + a : t0 + b] = o
            sel = res[-1][1]
            if rec is not None and sel is not None:  # the last position's selected non-window positions
                wlo = max(0, int(qpos[-1]) - c.window + 1)
                rec["idx_topk"][(id(seq), il)] = np.nonzero(sel[-1, :wlo])[0].astype(np.int32)
        og = (out.reshape(T, c.NH * c.v_mla)) * gate
        return E.linear(P + "o_proj.weight", og)

    # --- FFN ----------------------------------------------------------------------------------------------------------
    def _dense_ffn(self, il, cur):
        c, E = self.c, self.E
        P = f"model.layers.{il}.mlp."
        g = E.linear(P + "gate_proj.weight", cur)
        u = E.linear(P + "up_proj.weight", cur)
        return E.linear(P + "down_proj.weight", situ(g, u, c.situ_beta, c.situ_lb))

    def _moe_ffn(self, il, cur, rec_slots):
        c, E = self.c, self.E
        P = f"model.layers.{il}.block_sparse_moe."
        T = cur.shape[0]
        l0 = E.linear(P + "gate.weight", cur)
        l1 = E.linear(P + "gate.weight_1", cur)
        ids, w = route(l0, l1, E.f32(P + "e_score_correction_bias"), E.f32(P + "gate.routed_scaling_factor_param"), c.k_exp)
        for slot, rows in rec_slots:
            self._moe_rec[(slot, il)] = (ids[rows].astype(np.int32), w[rows])
        routed = E.linear(P + "fc1_latent_proj.weight", cur)  # latent 2048
        flat = ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        cnt = np.bincount(flat, minlength=c.n_exp)
        starts = np.concatenate([[0], np.cumsum(cnt)])
        work = [e for e in range(c.n_exp) if cnt[e]]
        moe = np.zeros((T, routed.shape[1]), F32)

        def expert(e):
            sl = order[starts[e] : starts[e + 1]]
            tok, k = sl // c.k_exp, sl % c.k_exp
            X = routed[tok]
            pe = P + f"experts.{e}."
            g = X @ E.w_rows(pe + "w1.weight", 0, E.shape(pe + "w1.weight")[0]).T  # w1 = gate (PR conversion diff 289)
            u = X @ E.w_rows(pe + "w3.weight", 0, E.shape(pe + "w3.weight")[0]).T  # w3 = up
            y = situ(g, u, c.situ_beta, c.situ_lb) @ E.w_rows(pe + "w2.weight", 0, E.shape(pe + "w2.weight")[0]).T  # w2 = down
            return tok, w[tok, k][:, None] * y

        for tok, y in E.pool.map(expert, work):
            moe[tok] += y
        moe = rms_norm(moe, E.f32(P + "latent_norm.weight"), c.eps)  # PR diff 1690
        moe_out = E.linear(P + "fc2_latent_proj.weight", moe)
        sh = P + "shared_experts."  # shared expert on the full-width input, SiTU (PR diff 1693-1698)
        g = E.linear(sh + "w1.weight", cur)
        u = E.linear(sh + "w3.weight", cur)
        shexp = E.linear(sh + "w2.weight", situ(g, u, c.situ_beta, c.situ_lb))
        return moe_out + shexp

    # --- one layer over all tokens of the call ------------------------------------------------------------------------
    def layer(self, il, H, segs, rec=None):
        """H [T, 4, hidden] -> next H.  Graph order: PR llama_model_ernie5_5::graph::graph (diff 1722-1805)."""
        c, E = self.c, self.E
        E.layer_begin()
        x = H
        lin, post, comb = self._hc_pre(x, il, "self_attention")
        cur = rms_norm(lin, self.W(il, "input_layernorm.weight"), c.eps)
        attn = self._mla_attention(il, cur, segs, rec) if c.is_dsa(il) else self._hca_attention(il, cur, segs)
        x = self._hc_post(attn, x, post, comb)
        lin, post, comb = self._hc_pre(x, il, "mlp")
        cur = rms_norm(lin, self.W(il, "post_attention_layernorm.weight"), c.eps)
        if il < c.dense_lead:
            ffn = self._dense_ffn(il, cur)
        else:
            slots = rec["moe_rows"] if rec is not None else []
            ffn = self._moe_ffn(il, cur, slots)
        return self._hc_post(ffn, x, post, comb)

    def embed(self, ids):
        E = self.E
        rb = self.c.H * 2
        rows = [np.frombuffer(E.store.read("model.embed_tokens.weight", int(i) * rb, rb), np.uint16) for i in ids]
        return bf16_to_f32(np.stack(rows))

    def forward(self, seqs, new_ids, capture=False, nlayers=None, tag=""):
        """Run `new_ids[i]` (a list) through the stack for seqs[i], advancing each Seq.  Returns the final H rows.
        capture: record l_out of the last 4 rows per sequence, routing of those rows, the DSA top-k of the last row."""
        c = self.c
        toks, segs, t0 = [], [], 0
        for seq, ids in zip(seqs, new_ids):
            segs.append((seq, t0, t0 + len(ids)))
            toks += list(ids)
            t0 += len(ids)
        H = np.repeat(self.embed(toks)[:, None, :], c.hc, axis=1)  # hc_init: every stream starts as the embedding
        rec = None
        if capture:
            rec = {"moe_rows": [], "idx_topk": {}}
            self._moe_rec = {}
            for i, (seq, a, b) in enumerate(segs):
                rec["moe_rows"].append((i, np.arange(max(a, b - 4), b)))
            self.lout = {}
        n = c.L if nlayers is None else nlayers
        for il in range(n):
            t = time.time()
            H = self.layer(il, H, segs, rec)
            if capture:
                for i, (seq, a, b) in enumerate(segs):
                    self.lout[(i, il)] = H[max(a, b - 4) : b].copy()
            self.log(f"{tag} L{il:02d} {time.time() - t:6.1f}s rss={rss_gb():.1f}G read={self.E.bytes_read / 1e9:.1f}G")
        for seq, ids in zip(seqs, new_ids):
            seq.n += len(ids)
        self.last_rec = rec
        return H, segs

    def logits_last(self, H, segs):
        """hc head + final norm + lm_head on each segment's last row: raw and MultiMax logits [S, V] and the normed [S, H]."""
        c, E = self.c, self.E
        rows = np.stack([H[b - 1] for _, _, b in segs])  # [S, 4, H]
        hfn = E.f32("model.mhc_contract.hc_head_fn")
        hb = E.f32("model.mhc_contract.hc_head_base")
        hs = E.f32("model.mhc_contract.hc_head_scale")
        cur = hc_head(rows, hfn, hb, hs, c.eps, c.hc_eps)
        normed = rms_norm(cur, E.f32("model.norm.weight"), c.eps)
        raw = E.linear("lm_head.weight", normed)
        r = E.f32("model.lm_head.multimax_ranges")
        t = E.f32("model.lm_head.multimax_ts")
        return raw, multimax(raw, r, t), normed, (r, t)


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


# ---------------------------------------------------------------------------------------------------------------------
# The set writer (ik dump_ref grammar)
# ---------------------------------------------------------------------------------------------------------------------
def dump_stem(name):  # crates/refset/src/ik.rs dump_stem
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


class SetWriter:
    def __init__(self, d, header):
        self.dir, self.header = d, header
        os.makedirs(d, exist_ok=True)
        self.rows, self.ints, self.written = [], [], 0

    def tensor(self, name, arr, ne, op, occ=0):
        a = np.ascontiguousarray(arr, dtype=F32)
        ne = list(ne) + [1] * (4 - len(ne))
        assert int(np.prod(ne)) == a.size, f"{name}: ne {ne} != {a.size} elements"
        assert np.isfinite(a).all(), f"{name}: non-finite values"
        fn = f"{dump_stem(name)}.{occ}.f32"
        a.astype("<f4").tofile(os.path.join(self.dir, fn))
        self.rows.append(f"tensor\t{name}\t{occ}\tf32\t{ne[0]}\t{ne[1]}\t{ne[2]}\t{ne[3]}\t{a.size * 4}\t{float(a.astype(np.float64).sum()):.6f}\t{op}\t1\t0\t-\t-")

    def int_tensor(self, name, arr, ne, op, occ=0):
        a = np.ascontiguousarray(arr, dtype="<i4")
        ne = list(ne) + [1] * (4 - len(ne))
        assert int(np.prod(ne)) == a.size
        stem = f"{dump_stem(name)}.{occ}"
        a.astype("<f4").tofile(os.path.join(self.dir, stem + ".f32"))
        a.tofile(os.path.join(self.dir, stem + ".i32"))
        s = float(a.astype(np.float64).sum())
        self.rows.append(f"tensor\t{name}\t{occ}\ti32\t{ne[0]}\t{ne[1]}\t{ne[2]}\t{ne[3]}\t{a.size * 4}\t{s:.6f}\t{op}\t1\t0\t-\t-")
        self.ints.append(f"int\t{name}\t{occ}\ttensor\ti32\ti32\tflat\t{a.size}\t{a.size * 4}\t{int(a.astype(np.int64).sum())}\t{int(np.abs(a).max()) if a.size else 0}\t{stem}.i32")

    def finish(self):
        lines = list(self.header)
        lines.append("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1")
        lines += self.rows
        lines.append("# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile")
        lines += self.ints
        lines.append(f"# complete\t{len(self.rows)}\t0")
        with open(os.path.join(self.dir, "MANIFEST.tsv"), "w") as f:
            f.write("\n".join(lines) + "\n")
        return len(self.rows)


def write_greedy(path, model_dir, rows, gen):
    """The 7-column --gen argmax format read_greedy takes (crates/gpu-gates/src/prompts.rs:122)."""
    with open(path, "w") as f:
        f.write(f"# argmax_ref\tbloomery-ernie55-ref\tmodel={model_dir}\tgen={gen}\n")
        f.write("#id\tn_tokens\targmax\ttop5_ids\ttop5_logits\tgen_ids\tgen_margins\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------
def top5(v):
    i = np.argsort(-v, kind="stable")[:5]
    return i.tolist(), v[i].tolist()


def run(args):
    t_start = time.time()
    model = args.model
    with open(os.path.join(model, "config.json")) as f:
        cfg = Cfg(json.load(f))
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    store = Store(model)
    eng = Eng(store, args.threads)
    M = Model(eng, cfg, log)
    prompts = []
    for spec in args.prompt:
        name, path = spec.split("=", 1)
        ids = [int(x) for x in open(path).read().replace("\n", ",").split(",") if x.strip()]
        prompts.append((name, path, ids))
    seqs = [Seq(ids) for _, _, ids in prompts]
    nl = args.layers
    log(f"prompts: {[(n, len(i)) for n, _, i in prompts]} steps={args.steps} threads={args.threads} layers={nl or cfg.L}")
    t = time.time()
    H, segs = M.forward(seqs, [s.ids for s in seqs], capture=True, nlayers=nl, tag="prefill")
    log(f"prefill {time.time() - t:.1f}s")
    idx_topk = M.last_rec["idx_topk"]
    raw, mm, normed, (mr, mt) = M.logits_last(H, segs)
    gen = [[int(np.argmax(mm[i]))] for i in range(len(seqs))]
    margins = [[float(np.sort(mm[i])[-1] - np.sort(mm[i])[-2])] for i in range(len(seqs))]
    first = [top5(mm[i]) for i in range(len(seqs))]
    pre_raw, pre_mm, pre_norm = raw.copy(), mm.copy(), normed.copy()
    for step in range(1, args.steps):
        t = time.time()
        H, segs = M.forward(seqs, [[g[-1]] for g in gen], nlayers=nl, tag=f"step{step:02d}")
        raw, mm, _, _ = M.logits_last(H, segs)
        for i in range(len(seqs)):
            gen[i].append(int(np.argmax(mm[i])))
            s = np.sort(mm[i])
            margins[i].append(float(s[-1] - s[-2]))
        log(f"step {step} {time.time() - t:.1f}s -> {[g[-1] for g in gen]}")
    # write the sets
    wall_run = time.time() - t_start
    for i, (name, path, ids) in enumerate(prompts):
        d = os.path.join(args.out, name)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        header = [
            "# ernie55_ref -- layer-streaming CPU f32 reference (tools/ref/ernie55_ref.py), raw f32 little-endian",
            f"# model\t{model}",
            "# arch\ternie5_5",
            "# build\ternie55_ref.py-1",
            f"# tokens\t{','.join(str(x) for x in ids)}",
            f"# tokens_file\t{path}",
            f"# tokens_file_sha256\t{sha}",
            f"# tokens_count\t{len(ids)}",
            f"# gen\t{args.steps}",
        ]
        sw = SetWriter(d, header)
        n = len(ids)
        r0 = max(0, n - 4)
        k = n - r0
        for il in range(nl or cfg.L):
            lo = M.lout[(i, il)]  # [k, 4, H]
            sw.tensor(f"l_out-{il}", lo.transpose(0, 1, 2).reshape(-1), [cfg.H, cfg.hc, k], "HC_POST")
            if il >= cfg.dense_lead:
                tid, tw = M._moe_rec[(i, il)]
                sw.int_tensor(f"ffn_moe_topk-{il}", tid.reshape(-1), [cfg.k_exp, k], "ARGSORT")
                sw.tensor(f"ffn_moe_weights-{il}", tw.reshape(-1), [cfg.k_exp, k], "ROUTE")
            key = (id(seqs[i]), il)
            if key in idx_topk:
                ids_k = np.sort(idx_topk[key])
                assert len(ids_k) == cfg.topk, f"indexer selected {len(ids_k)} non-window positions, want {cfg.topk}"
                sw.int_tensor(f"indexer_top_k-{il}", ids_k, [len(ids_k)], "TOP_K")
        sw.tensor("result_norm", pre_norm[i], [cfg.H], "RMS_NORM")
        sw.tensor("result_output_raw", pre_raw[i], [cfg.vocab], "MUL_MAT")
        sw.tensor("result_output", pre_mm[i], [cfg.vocab], "MULTIMAX")
        wrote = sw.finish()
        t5i, t5v = first[i]
        write_greedy(
            os.path.join(d, "greedy.tsv"),
            model,
            [(0, n, t5i[0], ",".join(map(str, t5i)), ",".join(f"{v:.6f}" for v in t5v), ",".join(map(str, gen[i])), ",".join(f"{v:.6f}" for v in margins[i]))],
            args.steps,
        )
        log(f"set {name}: {wrote} tensor rows -> {d}")
        try:
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
            text = tk.decode(gen[i], skip_special_tokens=False)
            print(f"GREEDY[{name}] ids={gen[i]}")
            print(f"GREEDY[{name}] text={text!r}")
        except Exception as e:  # tokenizers is optional here
            print(f"GREEDY[{name}] ids={gen[i]} (no decode: {e})")
        # sanity: argmax(raw) vs argmax(mm); routing sums
        print(f"SANITY[{name}] argmax raw={int(np.argmax(pre_raw[i]))} multimax={int(np.argmax(pre_mm[i]))} multimax ranges={mr.tolist()} ts={mt.tolist()}")
    print(f"wall_run={wall_run:.1f}s maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB read={eng.bytes_read / 1e9:.1f}GB")
    eng.pool.shutdown()
    store.close()


# ---------------------------------------------------------------------------------------------------------------------
# tokenize: render the chat with the model's own renderer and encode with tokenizer.json
# ---------------------------------------------------------------------------------------------------------------------
def cmd_tokenize(args):
    sys.path.insert(0, args.model)
    import encoding_ernie as E  # the model folder's own renderer
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    os.makedirs(args.out, exist_ok=True)

    def render(user):
        r = E.render_chat([{"role": "user", "content": user}], enable_thinking=False)
        text = E.segments_to_text(r.segments)
        # the rendered text already opens with <|ERNIE_MASK|>: encode it as is, no second special token
        return text, tk.encode(text, add_special_tokens=False).ids

    short_user = args.short
    text, ids = render(short_user)
    open(os.path.join(args.out, "short.txt"), "w").write(text)
    open(os.path.join(args.out, "short.ids"), "w").write(",".join(map(str, ids)) + "\n")
    print(f"short: {len(ids)} tokens")
    lic = open(os.path.join(args.model, "LICENSE")).read()
    body, n = "", 0
    for para in lic.split("\n\n"):
        cand = body + para + "\n\n"
        t, i = render("Summarize the following license text in two sentences.\n\n" + cand)
        if len(i) > args.long_max:
            break
        body, n = cand, len(i)
        text, ids = t, i
    open(os.path.join(args.out, "long.txt"), "w").write(text)
    open(os.path.join(args.out, "long.ids"), "w").write(",".join(map(str, ids)) + "\n")
    print(f"long: {len(ids)} tokens")


# ---------------------------------------------------------------------------------------------------------------------
# sanity: invariants of a written set that need no oracle
# ---------------------------------------------------------------------------------------------------------------------
def cmd_sanity(args):
    """Routed weights renormalize (sum_k w_k / escale[id_k] == 1, ids distinct and in range), MultiMax of the stored raw
    logits equals the stored result and changes a logit only where one of its four ranges is crossed, greedy.tsv holds
    the argmax (read_greedy's contract: gen_ids[0] == argmax)."""
    with open(os.path.join(args.model, "config.json")) as f:
        cfg = Cfg(json.load(f))
    store = Store(args.model)
    eng = Eng(store, 4)
    rg, ts = eng.f32("model.lm_head.multimax_ranges"), eng.f32("model.lm_head.multimax_ts")
    bad = 0
    # the Python side's one set reader (tools/bloomery/manifest.py, by column name): the sets must parse through it
    import importlib.util

    sys.dont_write_bytecode = True  # a check never leaves a __pycache__ in the tree it reads
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bloomery", "manifest.py")
    spec = importlib.util.spec_from_file_location("bloomery_manifest", mp)
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    for d in args.set:
        man = mf.read(d)
        trows, irows = man.rows("tensor"), man.rows("int")
        assert man.complete is not None and int(man.complete.split("\t")[0]) == len(trows), f"{d}: trailer != tensor rows"
        for r in trows:
            fn = mf.dump_file_name(r["name"], r.int("occurrence"), "tensor", "flat", "f32")
            assert os.path.getsize(os.path.join(d, fn)) == r.int("bytes"), f"{d}: {fn} size != bytes"
        for r in irows:
            assert os.path.getsize(os.path.join(d, r["file"])) == r.int("bytes"), f"{d}: {r['file']} size != bytes"
        print(f"SANITY {d}: manifest.py reads {len(trows)} tensor rows, {len(irows)} int rows, trailer {man.complete!r}, header keys {sorted(man.header)}")
        rd = lambda n, dt: np.fromfile(os.path.join(d, n), dtype=dt)  # noqa: E731
        worst, rows = 0.0, 0
        for il in range(cfg.dense_lead, cfg.L):
            if not os.path.exists(os.path.join(d, f"ffn_moe_topk-{il}.0.i32")):
                continue
            ids = rd(f"ffn_moe_topk-{il}.0.i32", "<i4").reshape(-1, cfg.k_exp)
            w = rd(f"ffn_moe_weights-{il}.0.f32", "<f4").reshape(-1, cfg.k_exp)
            esc = eng.f32(f"model.layers.{il}.block_sparse_moe.gate.routed_scaling_factor_param")
            assert ids.min() >= 0 and ids.max() < cfg.n_exp, f"{d} L{il}: expert id out of range"
            assert all(len(set(r.tolist())) == cfg.k_exp for r in ids), f"{d} L{il}: duplicate expert ids in a row"
            assert (w > 0).all(), f"{d} L{il}: non-positive routed weight"
            err = np.abs((w.astype(np.float64) / esc[ids]).sum(axis=1) - 1.0).max()
            worst, rows = max(worst, float(err)), rows + ids.shape[0]
        print(f"SANITY {d}: routed rows={rows} max|sum_k w_k/escale[id_k] - 1|={worst:.2e}")
        bad += worst > 1e-5
        raw, mm = rd("result_output_raw.0.f32", "<f4"), rd("result_output.0.f32", "<f4")
        rr, tt = rg.astype(np.float64), ts.astype(np.float64)
        x = raw.astype(np.float64)
        terms = [tt[0] * np.maximum(rr[0] - x, 0), tt[1] * np.maximum(x - rr[1], 0), tt[2] * np.maximum(rr[2] - x, 0) ** 2, tt[3] * np.maximum(x - rr[3], 0) ** 2]
        active = [t != 0 for t in terms]
        none = ~(active[0] | active[1] | active[2] | active[3])
        dev = np.abs(mm.astype(np.float64) - (x + sum(terms))).max()
        print(
            f"SANITY {d}: multimax ranges={rg.tolist()} ts={ts.tolist()} max|mm-(l+terms)|={dev:.2e} "
            f"logits with a term active: below r0={int(active[0].sum())} above r1={int(active[1].sum())} below r2={int(active[2].sum())} "
            f"above r3={int(active[3].sum())}; none active={int(none.sum())} (changed there: {int((mm[none] != raw[none]).sum())}); "
            f"argmax raw={int(raw.argmax())} mm={int(mm.argmax())}"
        )
        bad += dev > 1e-3 or int((mm[none] != raw[none]).sum()) != 0
        gl = [ln.split("\t") for ln in open(os.path.join(d, "greedy.tsv")).read().splitlines() if ln and not ln.startswith("#")]
        assert all(len(r) == 7 and r[5].split(",")[0] == r[2] for r in gl), f"{d}: greedy.tsv breaks the --gen contract"
        assert int(gl[0][2]) == int(mm.argmax()), f"{d}: greedy argmax != argmax of result_output"
    eng.pool.shutdown()
    store.close()
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------------------------------------------------------------
# self-test: each op against a hand-written loop on synthetic tensors (no model file)
# ---------------------------------------------------------------------------------------------------------------------
def _rng(seed):
    return np.random.RandomState(seed)


def _close(a, b, tol=1e-5, what=""):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    err = np.abs(a - b).max() / (np.abs(b).max() + 1e-12)
    assert err < tol, f"{what}: relative error {err:.3e} >= {tol}"


def _t_fp8():
    for b in range(256):
        v = _fp8_e4m3_value(b)
        assert (math.isnan(v) and math.isnan(FP8_LUT[b])) or float(FP8_LUT[b]) == v
    assert FP8_LUT[0x38] == 1.0 and FP8_LUT[0x7E] == 448.0 and FP8_LUT[0x01] == 2.0 ** -9
    r = _rng(1)
    N, K, blk = 11, 10, 4  # partial blocks in both dims
    codes = r.randint(0, 0x70, size=(N, K)).astype(np.uint8)
    scale = (r.rand(-(-N // blk), -(-K // blk)) + 0.5).astype(F32)
    want = np.array([[float(FP8_LUT[codes[i, j]]) * float(scale[i // blk, j // blk]) for j in range(K)] for i in range(N)])
    _close(dequant_fp8(codes, scale, blk), want, 1e-6, "fp8 block scale")
    _close(dequant_fp8(codes[4:], scale[1:], blk), want[4:], 1e-6, "fp8 chunk at a block row")


def _t_fp4():
    r = _rng(2)
    N, K = 3, 64
    packed = r.randint(0, 256, size=(N, K // 2)).astype(np.uint8)
    e = r.randint(120, 134, size=(N, K // 32)).astype(np.uint8)
    tab = [0, .5, 1, 1.5, 2, 3, 4, 6]
    nib = lambda n: (-1 if n & 8 else 1) * tab[n & 7]  # noqa: E731
    want = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            byte = int(packed[i, j // 2])
            n = (byte >> 4) if j % 2 else (byte & 15)  # the low nibble is the even element
            want[i, j] = nib(n) * 2.0 ** (int(e[i, j // 32]) - 127)
    _close(dequant_mxfp4(packed, e), want, 1e-7, "mxfp4")
    assert e8m0_to_f32(np.array([0, 127, 255], np.uint8))[1] == 1.0 and np.isnan(e8m0_to_f32(np.array([255], np.uint8))[0])


def _t_bf16():
    x = np.array([1.0, -2.5, 3.14159, 0.0, 1e-3], F32)
    u = (x.view(np.uint32) >> 16).astype(np.uint16)
    want = [float(np.uint32(int(v) << 16).view(F32)) for v in u]
    _close(bf16_to_f32(u), want, 1e-12, "bf16")


def _t_norms():
    r = _rng(3)
    x = r.randn(3, 7).astype(F32)
    w = r.randn(7).astype(F32)
    b = r.randn(7).astype(F32)
    for i in range(3):
        ms = sum(float(v) ** 2 for v in x[i]) / 7
        _close(rms_norm(x, w, 1e-6)[i], [float(x[i, j]) / math.sqrt(ms + 1e-6) * float(w[j]) for j in range(7)], 1e-6, "rms")
        mu = sum(float(v) for v in x[i]) / 7
        var = sum((float(v) - mu) ** 2 for v in x[i]) / 7
        _close(layer_norm(x, w, b, 1e-6)[i], [(float(x[i, j]) - mu) / math.sqrt(var + 1e-6) * float(w[j]) + float(b[j]) for j in range(7)], 1e-5, "ln")
    g, u = r.randn(5).astype(F32) * 3, r.randn(5).astype(F32) * 30
    want = [4 * math.tanh(float(a) / 4) * (1 / (1 + math.exp(-float(a)))) * 25 * math.tanh(float(b) / 25) for a, b in zip(g, u)]
    _close(situ(g, u, 4.0, 25.0), want, 1e-6, "situ")


def _t_hc():
    r = _rng(4)
    T, hc, H, it = 2, 4, 3, 4
    x = r.randn(T, hc, H).astype(F32)
    fn = (r.randn(hc * H, 6 * hc) * 0.2).astype(F32)
    base = r.randn(6 * hc).astype(F32) * 0.3
    scale = np.array([0.7, 0.9, 1.3], F32)
    sig = lambda v: 1 / (1 + math.exp(-v))  # noqa: E731
    reps = 0.5  # large, as VLLM test_mhc.py does: it separates 1/(sqrt(mean)+eps) from rsqrt(mean+eps)
    lin, post, comb = hc_pre(x, fn, base, scale, reps, 1e-6, it)
    for t in range(T):
        flat = [float(v) for v in x[t].reshape(-1)]
        rms = math.sqrt(sum(v * v for v in flat) / len(flat)) + reps  # the ERNIE factor: sqrt, then + eps
        mix = [sum(flat[k] * float(fn[k, m]) for k in range(hc * H)) / rms for m in range(6 * hc)]
        pre = [sig(mix[i] * 0.7 + float(base[i])) + 1e-6 for i in range(hc)]
        po = [2 * sig(mix[hc + i] * 0.9 + float(base[hc + i])) for i in range(hc)]
        cb = [[mix[2 * hc + 4 * i + j] * 1.3 + float(base[2 * hc + 4 * i + j]) for j in range(hc)] for i in range(hc)]
        for i in range(hc):  # row softmax + eps
            m = max(cb[i])
            e = [math.exp(v - m) for v in cb[i]]
            cb[i] = [v / sum(e) + 1e-6 for v in e]
        for j in range(hc):  # column normalize
            s = sum(cb[i][j] for i in range(hc)) + 1e-6
            for i in range(hc):
                cb[i][j] /= s
        for _ in range(it - 1):
            for i in range(hc):
                s = sum(cb[i]) + 1e-6
                cb[i] = [v / s for v in cb[i]]
            for j in range(hc):
                s = sum(cb[i][j] for i in range(hc)) + 1e-6
                for i in range(hc):
                    cb[i][j] /= s
        _close(post[t], po, 1e-6, "hc post mix")
        _close(comb[t], cb, 1e-5, "hc comb")
        _close(lin[t], [sum(pre[i] * float(x[t, i, h]) for i in range(hc)) for h in range(H)], 1e-5, "hc layer input")
        xo = r.randn(1, H).astype(F32)
        new = hc_post(xo, x[t : t + 1], post[t : t + 1], comb[t : t + 1])[0]
        want = [[po[j] * float(xo[0, h]) + sum(cb[i][j] * float(x[t, i, h]) for i in range(hc)) for h in range(H)] for j in range(hc)]
        _close(new, want, 1e-5, "hc post")
    hf = (r.randn(hc * H, hc) * 0.2).astype(F32)
    hb = r.randn(hc).astype(F32)
    hs = np.array([0.8], F32)
    out = hc_head(x, hf, hb, hs, reps, 1e-6)
    for t in range(T):
        flat = [float(v) for v in x[t].reshape(-1)]
        inv = 1 / math.sqrt(sum(v * v for v in flat) / len(flat) + reps)  # the head keeps rsqrt(mean + eps)
        mix = [sum(flat[k] * inv * float(hf[k, m]) for k in range(hc * H)) for m in range(hc)]
        pre = [sig(mix[i] * 0.8 + float(hb[i])) + 1e-6 for i in range(hc)]
        _close(out[t], [sum(pre[i] * float(x[t, i, h]) for i in range(hc)) for h in range(H)], 1e-5, "hc head")


def _t_rope():
    r = _rng(5)
    D, off, dims, base = 12, 2, 8, 10000.0
    x = r.randn(3, 2, D).astype(F32)
    pos = np.array([0, 5, 130])
    inv = plain_inv_freq(dims, base)
    cs = rope_cs(pos, inv)
    yn = rope(x, cs, off, dims, "norm")
    yx = rope(x, cs, off, dims, "neox")
    for t in range(3):
        for hd in range(2):
            for i in range(dims // 2):
                th = pos[t] * base ** (-2.0 * i / dims)
                c, s = math.cos(th), math.sin(th)
                a, b = float(x[t, hd, off + 2 * i]), float(x[t, hd, off + 2 * i + 1])
                assert abs(yn[t, hd, off + 2 * i] - (a * c - b * s)) < 1e-4 and abs(yn[t, hd, off + 2 * i + 1] - (a * s + b * c)) < 1e-4
                a, b = float(x[t, hd, off + i]), float(x[t, hd, off + dims // 2 + i])
                assert abs(yx[t, hd, off + i] - (a * c - b * s)) < 1e-4 and abs(yx[t, hd, off + dims // 2 + i] - (a * s + b * c)) < 1e-4
            assert yn[t, hd, 0] == x[t, hd, 0] and yn[t, hd, D - 1] == x[t, hd, D - 1]  # outside the rotated span: untouched
    _close(rope(yn, cs, off, dims, "norm", inverse=True), x, 1e-5, "inverse rope")
    # yarn: ggml rope_yarn per pair
    dims, base, factor, orig, bf, bs = 64, 160000.0, 16.0, 65536, 32.0, 1.0
    inv = yarn_inv_freq(dims, base, factor, orig, bf, bs)
    cd = lambda nr: dims * math.log(orig / (nr * 2 * math.pi)) / (2 * math.log(base))  # noqa: E731
    lo, hi = max(0, math.floor(cd(bf))), min(dims - 1, math.ceil(cd(bs)))
    for i in range(dims // 2):
        ext = base ** (-2.0 * i / dims)
        y = (i - lo) / max(0.001, hi - lo)
        mix = (1 - min(1, max(0, y))) * 1.0
        want = ext / factor * (1 - mix) + ext * mix
        assert abs(inv[i] - want) <= 1e-12 * want, (i, inv[i], want)
    assert inv[0] == 1.0 and abs(inv[-1] - base ** (-62 / 64) / factor) < 1e-12  # high-frequency pairs extrapolate, low interpolate


def _t_attn():
    r = _rng(6)
    t, NH, Dq, S = 3, 2, 5, 6
    Q, K = r.randn(t, NH, Dq).astype(F32), r.randn(S, Dq).astype(F32)
    V = r.randn(S, 4).astype(F32)
    sink = r.randn(NH).astype(F32)
    valid = r.rand(t, S) > 0.3
    valid[:, 0] = True
    got = attend_sink(Q, K, V, valid, sink, 0.4)
    for i in range(t):
        for h in range(NH):
            s = [float(Q[i, h] @ K[j]) * 0.4 if valid[i, j] else -math.inf for j in range(S)]
            m = max(max(s), float(sink[h]))
            p = [math.exp(v - m) for v in s]
            den = sum(p) + math.exp(float(sink[h]) - m)
            _close(got[i, h], [sum(p[j] * float(V[j, d]) for j in range(S)) / den for d in range(4)], 1e-5, "sink attention")


def _t_hca():
    """The stateful HCA path (compressor + window + visibility) against a brute force, one shot and incrementally."""
    r = _rng(7)
    D, NH, ratio, window, n = 6, 2, 4, 5, 23
    kv = r.randn(n, D).astype(F32)
    ckv = r.randn(n, D).astype(F32)
    sc = r.randn(n, D).astype(F32)
    q = r.randn(n, NH, D).astype(F32)
    sink = r.randn(NH).astype(F32)
    post = lambda comp, starts: comp * 1.0 + starts[:, None].astype(F32) * 0.01  # noqa: E731 (stands for norm + rope)

    def brute(p):
        blocks = []
        for j in range((p + 1) // ratio):
            rows = range(j * ratio, (j + 1) * ratio)
            comp = []
            for d in range(D):
                e = [math.exp(float(sc[k, d]) - max(float(sc[kk, d]) for kk in rows)) for k in rows]
                comp.append(sum(e[a] * float(ckv[k, d]) for a, k in enumerate(rows)) / sum(e))
            blocks.append(np.array(comp, F32) + j * ratio * 0.01)
        keys = [kv[k] for k in range(max(0, p - window + 1), p + 1)] + blocks
        out = []
        for h in range(NH):
            s = [float(q[p, h] @ k) * 0.5 for k in keys]
            m = max(max(s), float(sink[h]))
            e = [math.exp(v - m) for v in s]
            den = sum(e) + math.exp(float(sink[h]) - m)
            out.append([sum(e[a] * float(keys[a][d]) for a in range(len(keys))) / den for d in range(D)])
        return out

    for cuts in ([n], [9, 1, 1, 1, 7, 4]):
        st, p0, got = HcaState(D), 0, []
        for c in cuts:
            raw, pos0 = hca_update(st, kv[p0 : p0 + c], ckv[p0 : p0 + c], sc[p0 : p0 + c], p0, ratio, window, post)
            qpos = np.arange(p0, p0 + c)
            got.append(hca_attend_chunk(q[p0 : p0 + c], qpos, raw, pos0, st.comps, ratio, window, sink, 0.5))
            p0 += c
        got = np.concatenate(got)
        for p in range(n):
            _close(got[p], brute(p), 1e-5, f"hca position {p} cuts {cuts}")


def _t_vha_mla():
    r = _rng(8)
    NH, rk, D = 4, 2, 3
    U, V = r.randn(NH, rk).astype(F32), r.randn(NH, rk).astype(F32)
    m = vha_matrix(U, V)
    o = r.randn(2, NH, D).astype(F32)
    got = head_mix(o, m)
    for t in range(2):
        for h in range(NH):
            for d in range(D):
                want = float(o[t, h, d]) + sum(float(V[h, k]) * sum(float(U[g, k]) * float(o[t, g, d]) for g in range(NH)) for k in range(rk))
                assert abs(got[t, h, d] - want) < 1e-5
    # absorbed MLA == explicit per-head K/V expansion
    nope, rope_d, vd, kl, N = 3, 2, 4, 5, 6
    kvc = r.randn(N, kl).astype(F32)
    kpe = r.randn(N, rope_d).astype(F32)
    kvb = r.randn(NH, nope + vd, kl).astype(F32)
    qn, qp = r.randn(1, NH, nope).astype(F32), r.randn(1, NH, rope_d).astype(F32)
    sink = r.randn(NH).astype(F32)
    k_b, v_b = kvb[:, :nope], kvb[:, nope:]
    q_abs = np.einsum("thd,hdk->thk", qn, k_b)
    lat = mla_attend_chunk(np.concatenate([q_abs, qp], axis=2).astype(F32), np.concatenate([kvc, kpe], axis=1), kl, np.array([N - 1]), None, sink, 0.3)
    got = np.einsum("thk,hvk->thv", lat, v_b)
    for h in range(NH):
        K = [kvb[h, :nope] @ kvc[j] for j in range(N)]  # explicit k_nope = kv_b[h, :nope] @ c
        Vx = [kvb[h, nope:] @ kvc[j] for j in range(N)]
        s = [(float(qn[0, h] @ K[j]) + float(qp[0, h] @ kpe[j])) * 0.3 for j in range(N)]
        mx = max(max(s), float(sink[h]))
        e = [math.exp(v - mx) for v in s]
        den = sum(e) + math.exp(float(sink[h]) - mx)
        _close(got[0, h], [sum(e[j] * float(Vx[j][d]) for j in range(N)) / den for d in range(vd)], 1e-4, "absorbed MLA")


def _t_indexer():
    r = _rng(9)
    t, NH, D, N, topk, window = 3, 3, 4, 12, 4, 2
    q, w, k = r.randn(t, NH, D).astype(F32), r.randn(t, NH).astype(F32), r.randn(N, D).astype(F32)
    sc = indexer_scores(q, w, k)
    for i in range(t):
        for s in range(N):
            want = sum(max(0.0, float(q[i, h] @ k[s])) * float(w[i, h]) for h in range(NH))
            assert abs(sc[i, s] - want) < 1e-4
    qpos = np.array([6, 9, 11])
    sel = indexer_select(sc, qpos, topk, window)
    for i, p in enumerate(qpos):
        wlo = p - window + 1
        cand = sorted(range(wlo), key=lambda s: (-sc[i, s], s))[:topk]
        want = set(cand) | set(range(wlo, p + 1))
        assert set(np.nonzero(sel[i])[0]) == want, (p, sorted(want), np.nonzero(sel[i])[0])
    sel2 = indexer_select(sc, np.array([3, 4, 5]), topk, window)  # <= topk older cells: everything up to qpos
    for i, p in enumerate([3, 4, 5]):
        assert set(np.nonzero(sel2[i])[0]) == set(range(p + 1))


def _t_route_multimax():
    r = _rng(10)
    T, E, K = 3, 9, 4
    l0, l1 = r.randn(T, E).astype(F32), r.randn(T, E).astype(F32)
    bias = (r.randn(E) * 0.5).astype(F32)
    es = (r.rand(E) + 1).astype(F32)
    ids, w = route(l0, l1, bias, es, K)
    sg = lambda v: 1 / (1 + math.exp(-v))  # noqa: E731
    for t in range(T):
        p = [sg(float(l0[t, e])) + sg(float(l1[t, e])) for e in range(E)]
        pick = sorted(range(E), key=lambda e: (-(p[e] + float(bias[e])), e))[:K]
        assert list(ids[t]) == pick
        den = max(sum(p[e] for e in pick), 1e-20)
        _close(w[t], [p[e] / den * float(es[e]) for e in pick], 1e-6, "routed weights")
        assert abs(w[t].sum() - sum(p[e] / den * float(es[e]) for e in pick)) < 1e-5  # sum = sum of scaled shares
    rg, ts = [15.625, 15.9375, 8.75, 0.3671875], [-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875]
    x = np.array([-5.0, 0.0, 0.3, 1.0, 8.0, 9.0, 15.7, 20.0, 30.0], F32)
    got = multimax(x, np.array(rg, F32), np.array(ts, F32))
    for a, b in zip(x, got):
        a = float(a)
        want = a + ts[0] * max(rg[0] - a, 0) + ts[1] * max(a - rg[1], 0) + ts[2] * max(rg[2] - a, 0) ** 2 + ts[3] * max(a - rg[3], 0) ** 2
        assert abs(want - float(b)) < 1e-4, (a, want, b)


def _t_store_engine():
    """A synthetic safetensors file through Store + Eng.linear: FP8 (partial block), MXFP4, BF16, grouped rows."""
    r = _rng(11)
    with tempfile.TemporaryDirectory() as d:
        N, K = 260, 256
        fp8 = r.randint(0, 0x70, size=(N, K)).astype(np.uint8)
        fsc = (r.rand(3, 2) + 0.5).astype(F32)
        mx = r.randint(0, 256, size=(40, 32)).astype(np.uint8)  # [N=40, K=64] packed
        mxs = r.randint(122, 130, size=(40, 2)).astype(np.uint8)
        bf = ((r.randn(7, 16).astype(F32)).view(np.uint32) >> 16).astype(np.uint16)
        og = r.randint(0, 0x70, size=(256, 128)).astype(np.uint8)  # 2 groups of 128 rows, K=128
        osc = (r.rand(2, 1) + 0.5).astype(F32)
        tensors = {"w": ("F8_E4M3", [N, K], fp8.tobytes()), "w_scale_inv": ("F32", [3, 2], fsc.tobytes()),
                   "x.w": ("I8", [40, 32], mx.tobytes()), "x.w_scale_inv": ("F8_E8M0", [40, 2], mxs.tobytes()),
                   "b": ("BF16", [7, 16], bf.tobytes()), "og": ("F8_E4M3", [256, 128], og.tobytes()), "og_scale_inv": ("F32", [2, 1], osc.tobytes())}
        hdr, off = {}, 0
        for k, (dt, sh, raw) in tensors.items():
            hdr[k] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(raw)]}
            off += len(raw)
        hb = json.dumps(hdr).encode()
        with open(os.path.join(d, "a.safetensors"), "wb") as f:
            f.write(len(hb).to_bytes(8, "little") + hb + b"".join(v[2] for v in tensors.values()))
        eng = Eng(Store(d, {k: "a.safetensors" for k in tensors}), nthreads=3)
        X = r.randn(5, K).astype(F32)
        wfp8 = dequant_fp8(fp8, fsc, 128)
        _close(eng.linear("w", X), X @ wfp8.T, 1e-5, "fp8 linear over chunks")
        X2 = r.randn(5, 64).astype(F32)
        _close(eng.linear("x.w", X2), X2 @ dequant_mxfp4(mx, mxs).T, 1e-5, "mxfp4 linear")
        X3 = r.randn(5, 16).astype(F32)
        _close(eng.linear("b", X3), X3 @ bf16_to_f32(bf).T, 1e-5, "bf16 linear")
        Xg = r.randn(5, 256).astype(F32)  # two groups of K=128 inputs
        wog = dequant_fp8(og, osc, 128)
        want = np.concatenate([Xg[:, :128] @ wog[:128].T, Xg[:, 128:] @ wog[128:].T], axis=1)
        _close(eng.linear("og", Xg, group_rows=128), want, 1e-5, "grouped linear")
        assert eng.shape("x.w") == (40, 64)
        eng.pool.shutdown()
        eng.store.close()


def _t_set_writer():
    r = _rng(12)
    with tempfile.TemporaryDirectory() as d:
        sw = SetWriter(d, ["# ernie55_ref test", "# model\t/x", "# arch\ternie5_5", "# tokens\t1,2,3"])
        sw.tensor("l_out-0", r.randn(2 * 3 * 4), [2, 3, 4], "HC_POST")
        sw.int_tensor("ffn_moe_topk-1", np.arange(6), [3, 2], "ARGSORT")
        n = sw.finish()
        rows = [ln.split("\t") for ln in open(os.path.join(d, "MANIFEST.tsv")).read().splitlines()]
        kind_cols = next(x for x in rows if x[0] == "# kind")[1:]
        assert kind_cols[:8] == ["name", "occurrence", "type", "ne0", "ne1", "ne2", "ne3", "bytes"]
        tensors = [x for x in rows if x[0] == "tensor"]
        assert len(tensors) == n == 2 and all(len(x) == len(kind_cols) + 1 for x in tensors)
        assert rows[-1] == ["# complete", "2", "0"]
        for x in tensors:
            assert os.path.getsize(os.path.join(d, f"{x[1]}.{x[2]}.f32")) == int(x[8]) == 4 * int(np.prod([int(v) for v in x[4:8]]))
        ints = [x for x in rows if x[0] == "int"]
        assert len(ints) == 1 and os.path.getsize(os.path.join(d, ints[0][-1])) == int(ints[0][8])
        write_greedy(os.path.join(d, "greedy.tsv"), "/x", [(0, 3, 7, "7,1,2,3,4", "1,0.5,0.4,0.3,0.2", "7,8", "0.5,0.2")], 2)
        line = [ln for ln in open(os.path.join(d, "greedy.tsv")).read().splitlines() if not ln.startswith("#")][0].split("\t")
        assert len(line) == 7 and line[5].split(",")[0] == line[2]  # read_greedy: gen_ids[0] == argmax


def selftest():
    tests = [_t_fp8, _t_fp4, _t_bf16, _t_norms, _t_hc, _t_rope, _t_attn, _t_hca, _t_vha_mla, _t_indexer, _t_route_multimax, _t_store_engine, _t_set_writer]
    for t in tests:
        t0 = time.time()
        t()
        print(f"ok {t.__name__[3:]} ({time.time() - t0:.2f}s)")
    print(f"ernie55_ref self-test: {len(tests)} ok (numpy {np.__version__})")


def main():
    if "--self-test" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sanity")
    ps.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
    ps.add_argument("--set", action="append", required=True)
    ps.add_argument("--pylib", default=None)
    for name in ("run", "tokenize"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
        p.add_argument("--out", required=True)
        p.add_argument("--pylib", default=None)
        if name == "run":
            p.add_argument("--prompt", action="append", required=True, help="NAME=IDS_FILE (comma separated ids)")
            p.add_argument("--steps", type=int, default=32)
            p.add_argument("--threads", type=int, default=32)
            p.add_argument("--layers", type=int, default=None, help="debug: run only the first N layers")
        else:
            p.add_argument("--short", default="Tell me a short story about a robot who learns to paint.")
            p.add_argument("--long-max", type=int, default=2400)
    args = ap.parse_args()
    {"run": run, "tokenize": cmd_tokenize, "sanity": cmd_sanity}[args.cmd](args)


if __name__ == "__main__":
    main()
