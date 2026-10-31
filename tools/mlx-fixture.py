#!/usr/bin/env python3
"""mlx-fixture.py — the qwen35moe or qwen3moe GGUF fixture as an MLX-format checkpoint folder (bloomery #593 R3).

    tools/mlx-fixture.py make <fixture.gguf> <out dir> --config <real config.json> [--f16]
    tools/mlx-fixture.py check <out dir> <fixture.gguf>
    tools/mlx-fixture.py --self-test

The family is the fixture's `general.architecture`: `qwen35moe` (Qwen3.6, mlx-community's names below) or
`qwen3moe` (Qwen3-30B-A3B: `model.layers.N.…`, `lm_head`, every layer attention over the stacked `switch_mlp`
experts, no delta rule and no shared expert, the norm gammas as the GGUF holds them, the router 8 bits and every other
linear 4 — a `quantization` block with no `mode`, as the real file's).

`make` reads the fixture's tensors through the repo's one GGUF header parser
(tools/ref/gguf-ranges.py, loaded the way tools/ref/gguf-respell.py loads it), dequantizes each
Q4_K, Q6_K and F32 tensor by the rule crates/gguf/src/quant.rs ports from ggml (dequant_q4_k,
dequant_q6_k), moves it to the Qwen3.6 checkpoint's MLX names — the inverse of the map's per-tensor
steps (crates/model-hf/src/arch/qwen35moe.rs, Family::Qwen36): the V heads of the gated delta-rule
tensors from tiled to grouped by K head (VRows on in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and
the conv taps, VCols on out_proj, VCols{head:1} on A_log and dt_bias; the same order mlx-lm's
gated_delta.py reads), `ln(-x)` on A_log, the conv taps relabelled [channels, taps, 1]; the norm
gammas are written as the GGUF holds them (the GGUF's `x + 1` is what mlx-lm's sanitize stores,
qwen3_5.py norm_keys) — and quantizes every linear with mx.quantize(..., group_size=64,
bits=4|8, mode="affine") over its bf16 cast: 8 bits for the
router (mlp.gate) and the shared-expert gate (mlp.shared_expert_gate), the per-module entries the
config carries, 4 bits for every other linear; the norms, A_log, dt_bias and conv1d stay BF16. A
quantized linear `X.weight` writes its companions as `X.scales` and `X.biases`, the real file's
names. It writes safetensors shards (~1 GiB each, cut greedily in sorted-name order),
model.safetensors.index.json and a config.json cut from the real one (num_hidden_layers,
layer_types and the quantization module entries cut to the fixture's mirrored layers), copies the
tokenizer files that sit beside the real config.json (tokenizer.json, which `prepare` reads, is
refused by name when absent; tokenizer_config.json, chat_template.jinja and generation_config.json
are copied when present), then runs `check`.

`check` re-reads the folder's shard headers, decodes every quantized tensor with the affine rule
bloomery's MLX_A4/MLX_A8 rows own (value i the 4 bits at bit 4·(i mod 8) of word i/8, low nibble
first, or byte i; w = s·q + b in f32, the scale and bias widened from the pair the companions'
own dtype names) and holds it to mx.dequantize bit for bit: MLX returns the f32 value rounded
once to that pair's dtype (measured at bf16, mlx 0.32.1), so every value must survive that
rounding untouched. It also holds every value within the quantizer's own error of the GGUF fixture's
dequantized values: |err| ≤ |w|·2^-7 + |group scale|·1.25 (mx's affine fit is an optimizer
whose group extremes can sit past the top code — measured worst 1.047·scale over this fixture,
mlx 0.32.1 — and a near-constant group keeps only the bf16 grid; a scale can be negative, a
descending fit), prints the worst |err| and err/scale per tensor kind, and
prints each file's sha256.

`--self-test` runs the dequant ports against hand-computed blocks, the affine decode against the
pinned mx.quantize vector of crates/gguf's MLX test, and make+check end to end over a tiny
synthetic GGUF with a stub quantizer (exact integer grids, error 0) — no mlx and no fixture needed.

`--f16` writes the scale and bias companions as F16 instead of BF16 (the quantizer's cast and
every check follow the pair), the split the folder map reads as MLX_A4H/MLX_A8H.

Exit: 0; 2 a file missing or unreadable; 64 usage; 65 a malformed input, a transform refusal or a
failed check.
"""

import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile

USAGE = ("usage: tools/mlx-fixture.py make <fixture.gguf> <out dir> --config <real config.json> "
         "[--f16] | check <out dir> <fixture.gguf> | --self-test")

# The real folder's tokenizer files make copies of, beside its config.json; the first is required.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                   "generation_config.json")

EXIT_MISSING = 2
EXIT_USAGE = 64
EXIT_REFUSED = 65

F32, Q4_K, Q6_K = 0, 12, 14
N_V, N_K = 32, 16  # the linear layer's value and key head counts (ssm.inner_size/state, group count)
GROUP = 64
SHARD_LIMIT = 1_000_000_000

_spec = importlib.util.spec_from_file_location(
    "gguf_ranges", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "gguf-ranges.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)  # the one header parser: header(), nbytes(), write_gguf()
Refusal = R.Refusal

import numpy as np  # noqa: E402  (after the exit-code table, like the repo's tools)


# --------------------------------------------------------------- the GGUF side



def f16_pair(col2):
    """An (n, 2) little-endian byte column as its f16 values widened to f32."""
    return np.ascontiguousarray(col2).view("<u2").view("<f2").astype(np.float32)


def _scale_min(j, q):
    """get_scale_min_k4 (quant.rs:587) over an (n, 12) int32 view of the 6-bit scale bytes."""
    if j < 4:
        return q[:, j] & 63, q[:, j + 4] & 63
    d = (q[:, j + 4] & 0x0F) | ((q[:, j - 4] >> 6) << 4)
    m = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return d, m


def dequant_q4_k(buf):
    """quant.rs dequant_q4_k (ggml-quants.c:2807): d f16 @0, dmin f16 @2, 6-bit scales @4,
    128 code bytes @16 — value 64j+l the low nibble of qs[32j+l], 64j+32+l its high
    nibble, `d·sc·q − dmin·mi` (every product exact in f32, the subtraction the one
    rounding, fused or not)."""
    blocks = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 144)
    n = blocks.shape[0]
    d = f16_pair(blocks[:, 0:2])[:, 0]
    dmin = f16_pair(blocks[:, 2:4])[:, 0]
    sc = blocks[:, 4:16].astype(np.int32)
    qs = blocks[:, 16:144]
    out = np.empty((n, 256), dtype=np.float32)
    for j in range(4):
        s0, m0 = _scale_min(2 * j, sc)
        s1, m1 = _scale_min(2 * j + 1, sc)
        d1 = (d * s0.astype(np.float32))[:, None]
        m1v = (dmin * m0.astype(np.float32))[:, None]
        d2 = (d * s1.astype(np.float32))[:, None]
        m2v = (dmin * m1.astype(np.float32))[:, None]
        q = qs[:, 32 * j:32 * j + 32]
        out[:, 64 * j:64 * j + 32] = d1 * (q & 0x0F) - m1v
        out[:, 64 * j + 32:64 * j + 64] = d2 * (q >> 4) - m2v
    return out.reshape(-1)


def dequant_q6_k(buf):
    """quant.rs dequant_q6_k (ggml-quants.c:3243): ql[128] @0, qh[64] @128, int8
    scales[16] @192, d f16 @208; value 128c+32a+l reads ql byte 64c+32a+l nibble a
    and qh's 2 high bits, `d·sc·(q−32)` left-associated."""
    blocks = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 210)
    n = blocks.shape[0]
    d = f16_pair(blocks[:, 208:210])[:, 0]
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192].astype(np.int32)
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)
    out = np.empty((n, 256), dtype=np.float32)
    for half in range(2):
        qh_ = qh[:, 32 * half:32 * half + 32]
        for l in range(32):
            is_ = l // 16
            q1 = (ql[:, 64 * half + l] & 0x0F).astype(np.int32) | ((qh_[:, l] & 3) << 4)
            q2 = (ql[:, 64 * half + 32 + l] & 0x0F).astype(np.int32) | ((qh_[:, l] >> 2 & 3) << 4)
            q3 = (ql[:, 64 * half + l] >> 4).astype(np.int32) | ((qh_[:, l] >> 4 & 3) << 4)
            q4 = (ql[:, 64 * half + 32 + l] >> 4).astype(np.int32) | ((qh_[:, l] >> 6 & 3) << 4)
            base = 128 * half + l
            out[:, base] = (d * sc[:, is_]) * (q1 - 32)
            out[:, base + 32] = (d * sc[:, is_ + 2]) * (q2 - 32)
            out[:, base + 64] = (d * sc[:, is_ + 4]) * (q3 - 32)
            out[:, base + 96] = (d * sc[:, is_ + 6]) * (q4 - 32)
    return out.reshape(-1)


def gguf_f32(blob, ty):
    """One GGUF tensor's bytes as f32 values in file order (ne[0] contiguous)."""
    if ty == F32:
        return np.frombuffer(blob, dtype="<f4").astype(np.float32).copy()
    if ty == Q4_K:
        return dequant_q4_k(blob)
    if ty == Q6_K:
        return dequant_q6_k(blob)
    raise Refusal(EXIT_REFUSED, f"tensor type id {ty} is not one this maker dequantizes (F32, Q4_K, Q6_K)")


# ------------------------------------------------------------------ the naming

class Family:
    """What differs between the families this maker writes: the checkpoint's name prefix of the text model
    and of the head."""

    def __init__(self, arch, prefix, lm_head):
        self.arch, self.p, self.lm_head = arch, prefix, lm_head


QWEN35 = Family("qwen35moe", "language_model.model.", "language_model.lm_head.weight")
QWEN3 = Family("qwen3moe", "model.", "lm_head.weight")
FAMILIES = {f.arch: f for f in (QWEN35, QWEN3)}


def family_of(meta):
    """The family of a fixture, by its header's `general.architecture`."""
    arch = meta.get("general.architecture")
    arch = arch.decode() if isinstance(arch, bytes) else arch
    if arch not in FAMILIES:
        raise Refusal(EXIT_REFUSED, f"general.architecture {arch!r} is not a family this maker "
                                    f"writes ({sorted(FAMILIES)})")
    return FAMILIES[arch]


def mlx_names(family, layers):
    """The output tensors, in a fixed order: (mlx name, gguf name, kind, bits).

    `layers` is [(blk index, 'linear'|'full')]. Kinds: `quant` a linear transposed and
    quantized at `bits`; `bf16` a plain BF16 tensor; `alog`, `dt`, `conv` the
    step-bearing BF16 tensors.
    """
    P = family.p
    out = [
        (f"{P}embed_tokens.weight", "token_embd.weight", "quant", 4),
        (family.lm_head, "output.weight", "quant", 4),
        (f"{P}norm.weight", "output_norm.weight", "bf16", 0),
    ]
    for blk, kind in layers:
        L = f"{P}layers.{blk}."
        G = f"blk.{blk}."
        routed = [
            (f"{L}mlp.gate.weight", f"{G}ffn_gate_inp.weight", "quant", 8),
            (f"{L}mlp.switch_mlp.gate_proj.weight", f"{G}ffn_gate_exps.weight", "quant", 4),
            (f"{L}mlp.switch_mlp.up_proj.weight", f"{G}ffn_up_exps.weight", "quant", 4),
            (f"{L}mlp.switch_mlp.down_proj.weight", f"{G}ffn_down_exps.weight", "quant", 4),
        ]
        norms = [
            (f"{L}input_layernorm.weight", f"{G}attn_norm.weight", "bf16", 0),
            (f"{L}post_attention_layernorm.weight",
             f"{G}{'ffn_norm' if family is QWEN3 else 'post_attention_norm'}.weight", "bf16", 0),
        ]
        shared = [
            (f"{L}mlp.shared_expert_gate.weight", f"{G}ffn_gate_inp_shexp.weight", "quant", 8),
            (f"{L}mlp.shared_expert.gate_proj.weight", f"{G}ffn_gate_shexp.weight", "quant", 4),
            (f"{L}mlp.shared_expert.up_proj.weight", f"{G}ffn_up_shexp.weight", "quant", 4),
            (f"{L}mlp.shared_expert.down_proj.weight", f"{G}ffn_down_shexp.weight", "quant", 4),
        ]
        linear = [
            (f"{L}linear_attn.in_proj_qkv.weight", f"{G}attn_qkv.weight", "quant", 4),
            (f"{L}linear_attn.in_proj_z.weight", f"{G}attn_gate.weight", "quant", 4),
            (f"{L}linear_attn.in_proj_a.weight", f"{G}ssm_alpha.weight", "quant", 4),
            (f"{L}linear_attn.in_proj_b.weight", f"{G}ssm_beta.weight", "quant", 4),
            (f"{L}linear_attn.out_proj.weight", f"{G}ssm_out.weight", "quant", 4),
            (f"{L}linear_attn.conv1d.weight", f"{G}ssm_conv1d.weight", "conv", 0),
            (f"{L}linear_attn.A_log", f"{G}ssm_a", "alog", 0),
            (f"{L}linear_attn.dt_bias", f"{G}ssm_dt.bias", "dt", 0),
            (f"{L}linear_attn.norm.weight", f"{G}ssm_norm.weight", "bf16", 0),
        ]
        full = [
            (f"{L}self_attn.q_proj.weight", f"{G}attn_q.weight", "quant", 4),
            (f"{L}self_attn.k_proj.weight", f"{G}attn_k.weight", "quant", 4),
            (f"{L}self_attn.v_proj.weight", f"{G}attn_v.weight", "quant", 4),
            (f"{L}self_attn.o_proj.weight", f"{G}attn_output.weight", "quant", 4),
            (f"{L}self_attn.q_norm.weight", f"{G}attn_q_norm.weight", "bf16", 0),
            (f"{L}self_attn.k_norm.weight", f"{G}attn_k_norm.weight", "bf16", 0),
        ]
        if family is QWEN3:
            out.extend(norms + routed + full)
        else:
            out.extend(norms + routed[:1] + shared[:1] + routed[1:] + shared[1:]
                       + (linear if kind == "linear" else full))
    return out


def untile():
    """The inverse of the map's V-head tile (qwen35moe.rs Step::VRows/VCols: grouped head h goes
    to GGUF place tiled(h) = (h % per)·k + h/per, over 32 value heads of 16 key heads):
    MLX[h] = GGUF[tiled(h)]."""
    per = N_V // N_K
    return [(h % per) * N_K + h // per for h in range(N_V)]


# The V-head moves by GGUF tensor: `tail` the rows after the q and k rows, `all` every row, `cols`
# the columns of each row.
V_MOVES = {
    "attn_qkv.weight": "tail",
    "ssm_conv1d.weight": "tail",
    "attn_gate.weight": "all",
    "ssm_alpha.weight": "all",
    "ssm_beta.weight": "all",
    "ssm_out.weight": "cols",
}


def value_head(info, layers):
    """The V head's size: the first delta-rule layer's z projection rows over the V heads (0 when
    the fixture has no delta-rule layer)."""
    for blk, kind in layers:
        if kind == "linear":
            rows = info[f"blk.{blk}.attn_gate.weight"][1][1]
            if rows % N_V:
                raise Refusal(EXIT_REFUSED, f"blk.{blk}.attn_gate.weight: {rows} rows are not "
                                            f"{N_V} V heads")
            return rows // N_V
    return 0


def grouped(values, dims, how, head_v):
    """A GGUF tensor's f32 values (ggml `dims`, 2-D) with the V heads from tiled to grouped by K
    head (`how` is the V_MOVES entry): MLX head h is GGUF head tiled(h)."""
    perm = untile()
    a = np.asarray(values, dtype=np.float32).reshape(dims[::-1])
    if how == "cols":
        r, k = a.shape
        if k % N_V:
            raise Refusal(EXIT_REFUSED, f"{k} columns are not {N_V} V heads")
        return a.reshape(r, N_V, k // N_V)[:, perm, :].reshape(r, k).copy()
    rows = a.shape[0]
    head = head_v if how == "tail" else rows // N_V
    start = rows - N_V * head
    if head == 0 or start < 0 or (how == "all" and rows % N_V):
        raise Refusal(EXIT_REFUSED, f"{rows} rows are not {N_V} V heads of {head}"
                                    f"{' after the q and k rows' if how == 'tail' else ''}")
    tail = a[start:].reshape(N_V, head, -1)[perm].reshape(rows - start, -1)
    return np.concatenate([a[:start], tail]).copy()


def mlx_shape(kind, dims):
    """The MLX-side logical shape of a GGUF tensor with ggml dims `dims` (ne[0] first)."""
    if kind == "conv":
        return [dims[1], dims[0], 1]
    if kind in ("bf16", "dt", "alog"):
        return list(dims)
    if len(dims) == 1:
        return [1, dims[0]]  # the shared-expert gate: one row, as the real file holds it
    return list(dims[::-1])  # GGUF [K, R, E] memory is MLX [E, R, K] row-major


def to_mlx(kind, values, dims, gguf_name, head_v):
    """One tensor's f32 values (file order, ne[0] contiguous) as the MLX-side f32 array."""
    how = V_MOVES.get(gguf_name.split(".", 2)[-1])
    if kind == "quant":
        w = grouped(values, dims, how, head_v) if how else np.asarray(values)
        shape = [1, dims[0]] if len(dims) == 1 else dims[::-1]
        return w.reshape(shape).copy()
    if kind == "bf16":
        return np.asarray(values, dtype=np.float32).copy()
    if kind == "dt":
        return np.asarray(values, dtype=np.float32)[untile()].copy()
    if kind == "alog":
        neg = -np.asarray(values, dtype=np.float32)[untile()]
        if not np.all(neg > 0):
            raise Refusal(EXIT_REFUSED, "A_log: an ssm_a value is not the -exp of a real A_log")
        return np.log(neg).astype(np.float32)
    if kind == "conv":
        # GGUF [taps, channels] and MLX [channels, taps, 1] hold the same order.
        return grouped(values, dims, how, head_v).reshape(dims[1], dims[0], 1).copy()
    raise Refusal(EXIT_USAGE, f"unknown kind {kind}")


# ------------------------------------------------------------ bf16 and the rule

def to_bf16_bits(f):
    """f32 -> bf16 bits, round to nearest even — what mx.astype(bfloat16) does."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).copy()
    low = u & 0xFFFF
    hi = u >> 16
    up = (low > 0x8000) | ((low == 0x8000) & ((hi & 1) == 1))
    return (hi + up.astype(np.uint32)).astype(np.uint16)


def bf16_bits_to_f32(bits):
    u = (np.asarray(bits, dtype=np.uint32) << 16).astype("<u4")
    return np.frombuffer(u.tobytes(), dtype="<f4").reshape(np.asarray(bits).shape).copy()


# The two scale-and-bias pairs a folder stores: the safetensors dtype of the companions, and
# `pair_bits_to_f32`'s and `to_pair_bits`' rule for one of them.
PAIRS = {"bf16": "BF16", "f16": "F16"}


def pair_bits_to_f32(bits, pair):
    """A u16 array of scale or bias words widened to f32: bf16 bits shifted, f16 bits decoded."""
    if pair == "f16":
        return np.ascontiguousarray(bits, dtype=np.uint16).view("<f2").astype(np.float32)
    return bf16_bits_to_f32(bits)


def to_pair_bits(f, pair):
    """f32 -> the pair's bits, round to nearest even (bf16 by hand, f16 by numpy's cast)."""
    if pair == "f16":
        return np.ascontiguousarray(f, dtype=np.float32).astype(np.float16).view(np.uint16)
    return to_bf16_bits(f)


def decode_affine(q_u32, s_bits, b_bits, bits, pair="bf16"):
    """The MLX affine rule (crates/gguf dequant_mlx_a4/a8 and their F16 pairs): value i of
    group g is the code at bit (i % (32//bits))·bits of word g·(64//(32//bits)) + i//(32//bits),
    `s·q + b` in f32 with the widened pair (bf16 for MLX_A4/MLX_A8, f16 for MLX_A4H/MLX_A8H);
    `q_u32` is (…, words)."""
    q = np.asarray(q_u32, dtype=np.uint64)
    lead = q.shape[:-1]
    q2 = q.reshape(-1, q.shape[-1])
    s = pair_bits_to_f32(s_bits, pair).reshape(-1, s_bits.shape[-1])
    b = pair_bits_to_f32(b_bits, pair).reshape(-1, b_bits.shape[-1])
    per = 32 // bits
    mask = (1 << bits) - 1
    groups = s.shape[1]
    out = np.empty((q2.shape[0], groups * GROUP), dtype=np.float32)
    for g in range(groups):
        base = g * (GROUP // per)
        for i in range(GROUP):
            code = ((q2[:, base + i // per] >> ((i % per) * bits)) & mask).astype(np.uint32)
            out[:, g * GROUP + i] = s[:, g] * code + b[:, g]
    return out.reshape(lead + (groups * GROUP,))


# ------------------------------------------------------------- the quantizers

class MlxQuantizer:
    """mx.quantize/mx.dequantize, group 64, affine (pip mlx; 0.32.1 measured)."""

    def __init__(self):
        try:
            import mlx.core as mx
        except ImportError as e:
            raise Refusal(EXIT_MISSING, f"mlx is not importable: {e}") from e
        self.mx = mx

    def quantize(self, w, bits, pair="bf16"):
        """w f32 -> (words u32, scale bits, bias bits) of one tensor, MLX's own dtypes, the
        quantizer's cast and so its companions in the pair's dtype."""
        mx = self.mx
        wide = mx.bfloat16 if pair == "bf16" else mx.float16
        wq, s, b = self.mx.quantize(self.mx.array(w).astype(wide),
                                    group_size=GROUP, bits=bits, mode="affine")
        return (np.array(wq, copy=False),
                np.array(s.view(self.mx.uint16), copy=False),
                np.array(b.view(self.mx.uint16), copy=False))

    def dequantize(self, q, s_bits, b_bits, bits, pair="bf16"):
        """mx.dequantize widened to f32: the f32 value rounded once to the pair's dtype."""
        mx = self.mx
        wide = mx.bfloat16 if pair == "bf16" else mx.float16
        qm = mx.array(np.ascontiguousarray(q, dtype=np.uint32))
        sm = mx.array(np.ascontiguousarray(s_bits, dtype=np.uint16)).view(wide)
        bm = mx.array(np.ascontiguousarray(b_bits, dtype=np.uint16)).view(wide)
        return np.array(mx.dequantize(qm, sm, bm, group_size=GROUP, bits=bits)
                        .astype(mx.float32), copy=False)


class StubQuantizer:
    """Exact integer grids, no mlx: the codes are the values themselves at scale 1,
    bias 0, so decode is exact and the quantizer's error is 0 (the self-test's
    weights are integers)."""

    def quantize(self, w, bits, pair="bf16"):
        codes = w.astype(np.uint32).reshape(-1, w.shape[-1])
        top = (1 << bits) - 1
        if codes.max(initial=0) > top:
            raise Refusal(EXIT_REFUSED, f"stub: a value past {bits}-bit code {top}")
        per = 32 // bits
        k = w.shape[-1]
        words = np.zeros((codes.shape[0], k // per), dtype=np.uint32)
        for i in range(k):
            words[:, i // per] |= codes[:, i].astype(np.uint64) << ((i % per) * bits)
        words = words.astype(np.uint32)
        groups = k // GROUP
        one = 0x3F80 if pair == "bf16" else 0x3C00  # 1.0 in the pair's dtype
        s = np.full((codes.shape[0], groups), one, dtype=np.uint16)
        b = np.zeros((codes.shape[0], groups), dtype=np.uint16)
        shape = w.shape[:-1]
        return (words.reshape(shape + (words.shape[1],)),
                s.reshape(shape + (groups,)),
                b.reshape(shape + (groups,)))

    def dequantize(self, q, s_bits, b_bits, bits, pair="bf16"):
        return decode_affine(q, s_bits, b_bits, bits, pair)


# ------------------------------------------------------------ safetensors files

def write_shard(path, tensors):
    """tensors: name -> (dtype 'U32'|'BF16', shape, bytes), written in sorted-name order
    with the framing the safetensors format fixes (8-byte LE length, JSON, data)."""
    header = {}
    at = 0
    blobs = []
    for name in sorted(tensors):
        dtype, shape, blob = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(blob)]}
        at += len(blob)
        blobs.append(blob)
    js = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)) + js)
        for blob in blobs:
            f.write(blob)
    return at


class ShardReader:
    """One shard's tensor table: name -> (dtype, shape, read(at..end))."""

    def __init__(self, path):
        self.f = open(path, "rb")
        (n,) = struct.unpack("<Q", self.f.read(8))
        self.json = self.f.read(n)
        self.base = 8 + n
        self.entries = {k: (v["dtype"], v["shape"], v["data_offsets"])
                        for k, v in json.loads(self.json).items() if k != "__metadata__"}

    def read(self, name):
        dtype, shape, (a, b) = self.entries[name]
        self.f.seek(self.base + a)
        blob = self.f.read(b - a)
        if len(blob) != b - a:
            raise Refusal(EXIT_REFUSED, f"{name}: short read ({len(blob)} of {b - a} bytes)")
        return dtype, shape, blob

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ make/check

def read_gguf(path):
    """(metadata, {gguf name: (type, dims, absolute data offset)}, [(blk, kind)])."""
    meta, tensors, start, _size, _align = R.header(path)
    info = {name: (ty, dims, start + off) for name, ty, dims, off in tensors}
    blks = sorted({int(n.split(".")[1]) for n in info if n.startswith("blk.")})
    layers = [(blk, "full" if f"blk.{blk}.attn_q.weight" in info else "linear") for blk in blks]
    return meta, info, layers


class GgufTensor:
    """Pread-backed reader of one open fixture file."""

    def __init__(self, path, info):
        self.f = open(path, "rb")
        self.info = info

    def __call__(self, name):
        ty, dims, at = self.info[name]
        n = R.nbytes("<fixture>", name, ty, dims)
        blob = os.pread(self.f.fileno(), n, at)
        if len(blob) != n:
            raise Refusal(EXIT_REFUSED, f"{name}: short read ({len(blob)} of {n} bytes)")
        return gguf_f32(blob, ty), dims

    def close(self):
        self.f.close()


def tensor_bytes(kind, bits, shape):
    """A tensor's output bytes: the U32 words and the pair beside a quantized weight,
    the bf16 values of a plain one."""
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if kind == "quant":
        k = shape[-1]
        return [n * 4 // (32 // bits), (n // k) * (k // GROUP) * 2, (n // k) * (k // GROUP) * 2]
    return [n * 2]


def make(fixture, out, config_path, quantizer, quiet=False, pair="bf16"):
    if os.path.exists(out):
        raise Refusal(EXIT_REFUSED, f"{out} already exists (remove it first)")
    tokenizer_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isfile(os.path.join(tokenizer_dir, TOKENIZER_FILES[0])):
        raise Refusal(EXIT_REFUSED, f"{tokenizer_dir} has no {TOKENIZER_FILES[0]} beside the config")
    meta, info, layers = read_gguf(fixture)
    family = family_of(meta)
    head_v = value_head(info, layers)
    entries = mlx_names(family, layers)
    missing = [g for _, g, _, _ in entries if g not in info]
    if missing:
        raise Refusal(EXIT_REFUSED, f"the fixture has no tensor {missing[0]}")
    gguf = GgufTensor(fixture, info)
    plan = {}
    for mlx_name, gguf_name, kind, bits in entries:
        _values, dims = gguf(gguf_name)
        shape = mlx_shape(kind, dims)
        plan[mlx_name] = (gguf_name, kind, bits, shape)
    order = sorted(plan)
    shards, cur, cur_bytes = [], [], 0
    for name in order:
        gguf_name, kind, bits, shape = plan[name]
        total = sum(tensor_bytes(kind, bits, shape))
        if cur and cur_bytes + total > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += total
    if cur:
        shards.append(cur)
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    weight_map = {}
    os.makedirs(out)
    total_size = 0
    for i, shard in enumerate(shards):
        tensors = {}
        for name in shard:
            gguf_name, kind, bits, shape = plan[name]
            values, dims = gguf(gguf_name)
            w = to_mlx(kind, values, dims, gguf_name, head_v)
            if list(w.shape) != shape:
                raise Refusal(EXIT_REFUSED, f"{name}: shape {w.shape}, planned {shape}")
            if kind == "quant":
                q, s, b = quantizer.quantize(w, bits, pair)
                if q.dtype != np.uint32 or q.shape[-1] != shape[-1] * bits // 32:
                    raise Refusal(EXIT_REFUSED, f"{name}: quantized {q.shape}, not {shape} words")
                stem = name[: -len(".weight")]
                tensors[name] = ("U32", list(q.shape), q.astype("<u4").tobytes())
                tensors[f"{stem}.scales"] = (PAIRS[pair], list(s.shape), s.astype("<u2").tobytes())
                tensors[f"{stem}.biases"] = (PAIRS[pair], list(b.shape), b.astype("<u2").tobytes())
            else:
                tensors[name] = ("BF16", shape, to_bf16_bits(w.reshape(-1)).astype("<u2").tobytes())
            del values, w
        path = os.path.join(out, names[i])
        total_size += write_shard(path, tensors)
        for stored in tensors:
            weight_map[stored] = names[i]
        if not quiet:
            print(f"shard {names[i]}: {len(shard)} tensors, "
                  f"{sum(len(v[2]) for v in tensors.values())} bytes")
    gguf.close()
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": {k: weight_map[k] for k in sorted(weight_map)}},
                  fh, indent=2)
        fh.write("\n")
    cut_config(family, config_path, layers, out)
    copy_tokenizer(tokenizer_dir, out)
    return layers


def copy_tokenizer(src, out):
    """The tokenizer files of TOKENIZER_FILES that sit in `src`, copied byte for byte into `out`."""
    for name in TOKENIZER_FILES:
        path = os.path.join(src, name)
        if os.path.isfile(path):
            with open(path, "rb") as fh, open(os.path.join(out, name), "wb") as dst:
                dst.write(fh.read())


def cut_config(family, config_path, layers, out):
    """The real config.json cut to the fixture's mirrored layers: num_hidden_layers (and for qwen35moe
    layer_types) and the quantization module entries; everything else verbatim, the
    `quantization_config` copy dropped (the maker's config carries the block alone)."""
    with open(config_path) as fh:
        config = json.load(fh)
    if family is QWEN35:
        text = config["text_config"]
        text["num_hidden_layers"] = len(layers)
        text["layer_types"] = ["full_attention" if kind == "full" else "linear_attention"
                               for _, kind in layers]
    else:
        config["num_hidden_layers"] = len(layers)
    blks = {blk for blk, _ in layers}
    at = f"{family.p}layers."
    kept = {k: v for k, v in config["quantization"].items()
            if not k.startswith(at) or int(k[len(at):].split(".")[0]) in blks}
    # every module the map quantizes at 8 bits carries its entry, whatever the
    # source config held (the folder's tensors and its config agree by construction)
    for mlx_name, _, kind, bits in mlx_names(family, layers):
        if kind == "quant" and bits == 8:
            kept[mlx_name[: -len(".weight")]] = {"group_size": GROUP, "bits": 8}
    config["quantization"] = kept
    config.pop("quantization_config", None)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=4)
        fh.write("\n")


def mlx_kind_key(family, name):
    """The per-kind error group: the tensor's role inside its layer, or its top-level name."""
    at = f"{family.p}layers."
    if name.startswith(at):
        return name[len(at):].split(".", 1)[1]
    if name.startswith(family.p):
        return name[len(family.p):].split(".")[0]  # embed_tokens, norm
    return name.split(".")[-2]  # lm_head


def check(out, fixture, quantizer, quiet=False):
    with open(os.path.join(out, "config.json")) as fh:
        qconf = json.load(fh)["quantization"]
    # an absent mode is the affine one, as the real qwen3moe file's block spells it
    if (qconf.get("group_size"), qconf.get("bits")) != (GROUP, 4) or qconf.get("mode", "affine") != "affine":
        raise Refusal(EXIT_REFUSED,
                      f"config quantization is not affine group {GROUP} 4-bit: {qconf}")
    with open(os.path.join(out, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    meta, info, layers = read_gguf(fixture)
    family = family_of(meta)
    head_v = value_head(info, layers)
    entries = {m: (g, kind, bits) for m, g, kind, bits in mlx_names(family, layers)}
    expected = set(entries)
    for name, (_, kind, _) in entries.items():
        if kind == "quant":
            stem = name.removesuffix(".weight")
            expected.update((f"{stem}.scales", f"{stem}.biases"))
    if set(weight_map) != expected:
        missing, extra = sorted(expected - set(weight_map)), sorted(set(weight_map) - expected)
        raise Refusal(EXIT_REFUSED, f"tensor index: {len(missing)} missing {missing[:3]}, "
                                    f"{len(extra)} not in the map {extra[:3]}")
    gguf = GgufTensor(fixture, info)
    worst = {}
    checked = 0
    for sf in sorted(set(weight_map.values())):
        shard = ShardReader(os.path.join(out, sf))
        for name in sorted(n for n, f in weight_map.items() if f == sf):
            if name.endswith(".scales") or name.endswith(".biases"):
                continue  # read beside their weight below
            if name not in entries:
                raise Refusal(EXIT_REFUSED, f"{name}: not a tensor the map names")
            gguf_name, kind, bits = entries[name]
            values, dims = gguf(gguf_name)
            want = to_mlx(kind, values, dims, gguf_name, head_v)
            if kind == "quant":
                override = qconf.get(name[: -len(".weight")], {})
                if (override.get("bits", 4), override.get("group_size", GROUP)) != (bits, GROUP):
                    raise Refusal(EXIT_REFUSED,
                                  f"{name}: config says {override}, the folder holds {bits}-bit")
                dty, qshape, blob = shard.read(name)
                if dty != "U32":
                    raise Refusal(EXIT_REFUSED, f"{name}: dtype {dty}, not U32")
                words = np.frombuffer(blob, dtype="<u4").reshape(qshape)
                stem = name[: -len(".weight")]
                s_dty, s_shape, s_blob = shard.read(f"{stem}.scales")
                b_dty, b_shape, b_blob = shard.read(f"{stem}.biases")
                by_dtype = {v: k for k, v in PAIRS.items()}
                if s_dty != b_dty or s_dty not in by_dtype:
                    raise Refusal(EXIT_REFUSED,
                                  f"{name}: a companion is {s_dty}/{b_dty}, not one pair "
                                  f"({', '.join(PAIRS.values())})")
                pair = by_dtype[s_dty]
                s_bits = np.frombuffer(s_blob, dtype="<u2").reshape(s_shape)
                b_bits = np.frombuffer(b_blob, dtype="<u2").reshape(b_shape)
                mine = decode_affine(words, s_bits, b_bits, bits, pair)
                mxs = quantizer.dequantize(words, s_bits, b_bits, bits, pair)
                if not np.array_equal(to_pair_bits(mine.reshape(-1), pair),
                                      to_pair_bits(mxs.reshape(-1), pair)):
                    bad = int((to_pair_bits(mine.reshape(-1), pair)
                               != to_pair_bits(mxs.reshape(-1), pair)).sum())
                    raise Refusal(EXIT_REFUSED,
                                  f"{name}: the f32 decode rounded once to {PAIRS[pair]} differs "
                                  f"from mx.dequantize at {bad} of {mine.size} values")
                s = np.abs(np.repeat(pair_bits_to_f32(s_bits, pair), GROUP, axis=-1).reshape(-1))
                err = np.abs(mxs.reshape(-1) - want.reshape(-1))
                # mx's affine fit is an optimizer, not a min-max range: a group's
                # extremes can sit past the top code by its choice (measured worst
                # 1.047·scale over this fixture, mlx 0.32.1), and a group whose scale
                # is far below its values' magnitude keeps only the pair's grid (two
                # ulps of |w|: the input cast and the reconstructed bias — a measured
                # 2^-7 term at bf16, the derived 2^-10 one at f16, whose ulp is 2^-11
                # of |w|).
                ulp = np.float32(2.0 ** -7 if pair == "bf16" else 2.0 ** -10)
                bound = (np.abs(want.reshape(-1)) * ulp + s * np.float32(1.25))
                if not np.all(err <= bound):
                    raise Refusal(EXIT_REFUSED,
                                  f"{name}: |err| past the quantizer's band (worst err/bound "
                                  f"{float((err / np.maximum(bound, np.float32(1e-30))).max()):.4f})")
                key = mlx_kind_key(family, name)
                w = worst.setdefault(key, [0, 0.0, 0])
                w[0] = max(w[0], float(err.max()))
                w[1] = max(w[1], float((err / np.maximum(s, np.float32(1e-30))).max()))
                w[2] += 1
            else:
                dty, shape, blob = shard.read(name)
                if dty != "BF16":
                    raise Refusal(EXIT_REFUSED, f"{name}: dtype {dty}, not BF16")
                if not np.array_equal(np.frombuffer(blob, dtype="<u2"),
                                      to_bf16_bits(want.reshape(-1))):
                    raise Refusal(EXIT_REFUSED, f"{name}: the BF16 bytes are not the GGUF values")
            checked += 1
        shard.close()
    gguf.close()
    print(f"check: {checked} tensors; every quantized value within the quantizer's band and "
          "the f32 decode = mx.dequantize's pair values bit for bit")
    for key in sorted(worst):
        err, ratio, n = worst[key]
        print(f"  {key}: {n} tensors, worst |err| {err:.6g}, worst err/scale {ratio:.4f}")
    tokenizer = [n for n in TOKENIZER_FILES if os.path.isfile(os.path.join(out, n))]
    for sf in sorted(set(weight_map.values())) + ["model.safetensors.index.json", "config.json"] + tokenizer:
        with open(os.path.join(out, sf), "rb") as fh:
            print(f"sha256 {sf} {hashlib.sha256(fh.read()).hexdigest()}")


# ------------------------------------------------------------------ the self-test

def self_test():
    import shutil

    P = QWEN35.p

    def expect(name, cond, detail=""):
        if not cond:
            print(f"mlx-fixture self-test: {name} {detail}", file=sys.stderr)
            raise SystemExit(65)

    # Q4_K: one hand-built block. d = 1.0 (0x3C00), dmin = 0.25 (0x3400); the first
    # 64 values read scale pair 0 (scales[0] = 5, scales[4] = 3) and the next 32 pair 1
    # (scales[1] = 4, scales[5] = 2); qs[0] = 0x21:
    # value 0 = 1·5·1 − 0.25·3 = 4.25, value 32 = 1·4·2 − 0.25·2 = 7.5.
    blk = bytearray(144)
    blk[0:2] = struct.pack("<H", 0x3C00)
    blk[2:4] = struct.pack("<H", 0x3400)
    blk[4] = 5
    blk[5] = 4
    blk[8] = 3
    blk[9] = 2
    blk[16] = 0x21
    v = dequant_q4_k(bytes(blk))
    expect("q4_k value 0", v[0] == np.float32(4.25), f"{v[0]}")
    expect("q4_k value 32", v[32] == np.float32(7.5), f"{v[32]}")

    # Q6_K: ql[0] = 0x12, ql[32] = 0x01, qh zero, d = 0.5 (0x3800) at @208; value l
    # (l < 16) reads scale is_+0, value 32+l is_+2, value 64+l is_+4, value 96+l is_+6
    # with is_ = l/16 = 0, so the int8 scale bytes @192 are [2, 4, 8, 1] at 0, 2, 4, 6:
    # value 0 = 0.5*2*(2 - 32) = -30, value 32 = 0.5*4*(1 - 32) = -62,
    # value 64 = 0.5*8*(1 - 32) = -124, value 96 = 0.5*1*(0 - 32) = -16.
    blk = bytearray(210)
    blk[0] = 0x12
    blk[32] = 0x01
    blk[192], blk[194], blk[196], blk[198] = 2, 4, 8, 1
    blk[208:210] = struct.pack("<H", 0x3800)
    v = dequant_q6_k(bytes(blk))
    for at, want in ((0, -30.0), (32, -62.0), (64, -124.0), (96, -16.0)):
        expect(f"q6_k value {at}", v[at] == np.float32(want), f"{v[at]}")

    # The affine decode against crates/gguf's pinned mx.quantize vector (mlx 0.32.1):
    # the first group's words 0xd450dfe9, 0x3a4bf290, scale bf16 0xbeb4 (-0.3515625),
    # bias 0x4034 (2.8125); value 0 = -0.3515625 (0xbeb40000), value 1 = -2.109375
    # (0xc0070000), value 3 = -1.8828125 (0xbfe10000).
    words = np.array([[0xd450dfe9, 0x3a4bf290, 0, 0, 0, 0, 0, 0]], dtype=np.uint32)
    s = np.array([[0xbeb4]], dtype=np.uint16)
    b = np.array([[0x4034]], dtype=np.uint16)
    got = decode_affine(words, s, b, 4)
    for i, want in ((0, 0xbeb40000), (1, 0xc0070000), (3, 0xbfe10000)):
        expect(f"affine value {i}", int(got.reshape(-1)[i].view(np.uint32)) == want,
              f"{got.reshape(-1)[i]}")

    # The F16 pair of the same rule: 1.0 is 0x3C00, a value between f16 grid points
    # rounds to the nearer, and the widen is the identity on grid values.
    expect("f16 one", int(to_pair_bits(np.array([1.0], dtype=np.float32), "f16")[0]) == 0x3C00)
    near = np.array([1.0 + 2.0 ** -12], dtype=np.float32)
    expect("f16 rounds to the grid",
           int(to_pair_bits(near, "f16")[0]) == 0x3C00)
    back = pair_bits_to_f32(np.array([[0x3C00]], dtype=np.uint16), "f16")
    expect("f16 widens", float(back[0, 0]) == 1.0, f"{back[0, 0]}")
    words16 = np.array([[0x00000002, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint32)
    s16 = np.array([[0x3C00]], dtype=np.uint16)
    b16 = np.array([[0xB400]], dtype=np.uint16)  # -0.25
    got16 = decode_affine(words16, s16, b16, 4, "f16")
    expect("affine f16 value 0", float(got16.reshape(-1)[0]) == 1.75,
           f"{got16.reshape(-1)[0]}")
    expect("affine f16 value 1", float(got16.reshape(-1)[1]) == -0.25,
           f"{got16.reshape(-1)[1]}")

    # make + check over a tiny synthetic fixture with the stub quantizer: two layers
    # (linear, full), every transform exercised, integers only (error 0).
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "tiny.gguf")
        kvs = [R._kv_string("general.architecture", "qwen35moe")]
        rng = np.random.default_rng(3)
        f32 = lambda shape: rng.integers(1, 9, size=shape).astype(np.float32)
        tensors = [
            ("token_embd.weight", F32, [64, 8], f32((8, 64)).astype("<f4").tobytes()),
            ("output.weight", F32, [64, 8], f32((8, 64)).astype("<f4").tobytes()),
            ("output_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.0.attn_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.0.post_attention_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            # 32 V heads of 2 rows over 16 K heads of 1: q and k rows 32, V rows 64
            ("blk.0.attn_qkv.weight", F32, [64, 96], f32((96, 64)).astype("<f4").tobytes()),
            ("blk.0.attn_gate.weight", F32, [64, 64], f32((64, 64)).astype("<f4").tobytes()),
            ("blk.0.ssm_alpha.weight", F32, [64, 32], f32((32, 64)).astype("<f4").tobytes()),
            ("blk.0.ssm_beta.weight", F32, [64, 32], f32((32, 64)).astype("<f4").tobytes()),
            ("blk.0.ssm_out.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.0.ssm_conv1d.weight", F32, [4, 96], f32((96, 4)).astype("<f4").tobytes()),
            # ssm_a: -exp(A_log), a permutation that covers the untile
            ("blk.0.ssm_a", F32, [32],
             (-np.exp(np.arange(1, 33, dtype=np.float32))).astype("<f4").tobytes()),
            ("blk.0.ssm_dt.bias", F32, [32], f32((32,)).astype("<f4").tobytes()),
            ("blk.0.ssm_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.1.attn_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.1.post_attention_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.1.attn_q.weight", F32, [64, 12], f32((12, 64)).astype("<f4").tobytes()),
            ("blk.1.attn_k.weight", F32, [64, 4], f32((4, 64)).astype("<f4").tobytes()),
            ("blk.1.attn_v.weight", F32, [64, 4], f32((4, 64)).astype("<f4").tobytes()),
            ("blk.1.attn_output.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.1.attn_q_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.1.attn_k_norm.weight", F32, [16], f32((16,)).astype("<f4").tobytes()),
            ("blk.1.ffn_gate_inp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_gate_inp_shexp.weight", F32, [64, 1], f32((1, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_gate_exps.weight", F32, [64, 6, 3], f32((3, 6, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_up_exps.weight", F32, [64, 6, 3], f32((3, 6, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_down_exps.weight", F32, [64, 6, 3], f32((3, 6, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_gate_shexp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_up_shexp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.1.ffn_down_shexp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
        ]
        # layer 0 needs its MoE tensors too
        tensors += [
            ("blk.0.ffn_gate_inp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_gate_inp_shexp.weight", F32, [64, 1], f32((1, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_gate_exps.weight", F32, [64, 6, 3], f32((3, 6, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_up_exps.weight", F32, [64, 6, 3], f32((3, 6, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_down_exps.weight", F32, [64, 6, 3], f32((3, 6, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_gate_shexp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_up_shexp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
            ("blk.0.ffn_down_shexp.weight", F32, [64, 6], f32((6, 64)).astype("<f4").tobytes()),
        ]
        R.write_gguf(fixture, kvs, tensors)
        config_path = os.path.join(tmp, "config.json")
        with open(config_path, "w") as fh:
            json.dump({"model_type": "qwen3_5_moe",
                       "quantization": {"group_size": GROUP, "bits": 4, "mode": "affine"},
                       "quantization_config": {"group_size": GROUP, "bits": 4, "mode": "affine"},
                       "text_config": {"num_hidden_layers": 40,
                                       "layer_types": ["linear_attention"] * 40}}, fh)
        out = os.path.join(tmp, "out")
        try:
            make(fixture, out, config_path, StubQuantizer(), quiet=True)
            expect("no tokenizer.json refused", False, "made")
        except Refusal as e:
            expect("no tokenizer.json refused", "tokenizer.json" in str(e), str(e))
        expect("refused before writing", not os.path.exists(out))
        with open(os.path.join(tmp, "tokenizer.json"), "w") as fh:
            fh.write('{"version": "1.0"}')
        layers = make(fixture, out, config_path, StubQuantizer(), quiet=True)
        expect("tokenizer copied", open(os.path.join(out, "tokenizer.json")).read() == '{"version": "1.0"}')
        expect("layers", layers == [(0, "linear"), (1, "full")], str(layers))
        with open(os.path.join(out, "config.json")) as fh:
            cut = json.load(fh)
        expect("cut layers", cut["text_config"]["num_hidden_layers"] == 2)
        expect("cut layer_types",
              cut["text_config"]["layer_types"] == ["linear_attention", "full_attention"])
        expect("cut quantization",
              sorted(cut["quantization"]) == [
                  "bits", "group_size",
                  f"{P}layers.0.mlp.gate", f"{P}layers.0.mlp.shared_expert_gate",
                  f"{P}layers.1.mlp.gate", f"{P}layers.1.mlp.shared_expert_gate", "mode"],
              str(sorted(cut["quantization"])))
        expect("copy dropped", "quantization_config" not in cut)
        with open(os.path.join(out, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        # 27 quantized linears (13 in the linear layer, 12 in the full one, the embed
        # and the head) a weight, scales and biases, and 11 plain BF16 tensors
        # (6 + 4 beside the layers, the final norm)
        expect("index holds every tensor", len(weight_map) == 27 * 3 + 11, str(len(weight_map)))
        # the companions are `X.scales` and `X.biases`, the real file's names
        for stem in (f"{P}layers.0.mlp.gate", f"{P}layers.1.self_attn.q_proj", f"{P}embed_tokens"):
            expect("companion names", all(f"{stem}.{x}" in weight_map
                                          for x in ("weight", "scales", "biases")), stem)
        expect("no `.weight.scales`", not any(".weight.scales" in k or ".weight.biases" in k
                                              for k in weight_map))
        reader = ShardReader(os.path.join(out, weight_map[f"{P}layers.0.linear_attn.A_log"]))
        _dt, shape, blob = reader.read(f"{P}layers.0.linear_attn.A_log")
        alog = bf16_bits_to_f32(np.frombuffer(blob, dtype="<u2"))
        # the inverse of the map's steps: A_log[h] = ln(-ssm_a[tiled(h)]); the tiny
        # fixture's ssm_a is -exp(1..32), so A_log is 1..32 permuted by the untile
        want = np.arange(1, 33, dtype=np.float32)[untile()]
        expect("A_log untile and -exp inverse",
              np.array_equal(to_bf16_bits(alog), to_bf16_bits(want.astype(np.float32))),
              f"{alog[:4]} vs {want[:4]}")
        reader.close()
        # The V heads, by hand: MLX head h is GGUF head (h % 2)·16 + h // 2, so heads 0..3 are
        # GGUF heads 0, 16, 1, 17.
        _meta, info, _layers = read_gguf(fixture)
        src = GgufTensor(fixture, info)

        def mlx_plane(stem, bits=4):
            rd = ShardReader(os.path.join(out, weight_map[f"{stem}.weight"]))
            _d, shape, blob = rd.read(f"{stem}.weight")
            words = np.frombuffer(blob, dtype="<u4").reshape(shape)
            _d, sh, blob = rd.read(f"{stem}.scales")
            scales = np.frombuffer(blob, dtype="<u2").reshape(sh)
            _d, sh, blob = rd.read(f"{stem}.biases")
            biases = np.frombuffer(blob, dtype="<u2").reshape(sh)
            rd.close()
            return decode_affine(words, scales, biases, bits)

        tiled = lambda h: (h % 2) * 16 + h // 2
        la = f"{P}layers.0.linear_attn."
        z = mlx_plane(la + "in_proj_z")
        gz = src("blk.0.attn_gate.weight")[0].reshape(64, 64)
        for h in range(32):
            expect(f"z head {h}", np.array_equal(z[2 * h:2 * h + 2],
                                                 gz[2 * tiled(h):2 * tiled(h) + 2]))
        qkv = mlx_plane(la + "in_proj_qkv")
        gq = src("blk.0.attn_qkv.weight")[0].reshape(96, 64)
        expect("qkv q and k rows", np.array_equal(qkv[:32], gq[:32]))
        for h in range(32):
            expect(f"qkv head {h}", np.array_equal(qkv[32 + 2 * h:34 + 2 * h],
                                                   gq[32 + 2 * tiled(h):34 + 2 * tiled(h)]))
        a = mlx_plane(la + "in_proj_a")
        ga = src("blk.0.ssm_alpha.weight")[0].reshape(32, 64)
        expect("alpha rows", all(np.array_equal(a[h], ga[tiled(h)]) for h in range(32)))
        o = mlx_plane(la + "out_proj")
        go = src("blk.0.ssm_out.weight")[0].reshape(6, 64)
        expect("out_proj columns", all(np.array_equal(o[:, 2 * h:2 * h + 2],
                                                      go[:, 2 * tiled(h):2 * tiled(h) + 2])
                                       for h in range(32)))
        rd = ShardReader(os.path.join(out, weight_map[la + "conv1d.weight"]))
        _d, _s, blob = rd.read(la + "conv1d.weight")
        rd.close()
        conv = bf16_bits_to_f32(np.frombuffer(blob, dtype="<u2")).reshape(96, 4)
        gc = src("blk.0.ssm_conv1d.weight")[0].reshape(96, 4)
        expect("conv q and k rows", np.array_equal(conv[:32], gc[:32]))
        expect("conv rows", all(np.array_equal(conv[32 + 2 * h:34 + 2 * h],
                                               gc[32 + 2 * tiled(h):34 + 2 * tiled(h)])
                                for h in range(32)))
        # the norm gammas as the GGUF holds them
        rd = ShardReader(os.path.join(out, weight_map[f"{P}layers.1.self_attn.q_norm.weight"]))
        _d, _s, blob = rd.read(f"{P}layers.1.self_attn.q_norm.weight")
        rd.close()
        qn = bf16_bits_to_f32(np.frombuffer(blob, dtype="<u2"))
        expect("q_norm unchanged", np.array_equal(qn, src("blk.1.attn_q_norm.weight")[0]))
        src.close()
        # the one-row-a-head move on 0..31: MLX heads 0..3 are GGUF 0, 16, 1, 17
        moved = grouped(np.arange(32, dtype=np.float32), [1, 32], "all", 0).reshape(-1)
        expect("grouped by hand", list(moved[:4]) == [0.0, 16.0, 1.0, 17.0], str(moved[:4]))
        check(out, fixture, StubQuantizer(), quiet=True)
        # the same tiny fixture at F16 pairs: companions stored F16, the decode and
        # the error band read the pair back, and the whole check passes the same way
        out16 = os.path.join(tmp, "out16")
        make(fixture, out16, config_path, StubQuantizer(), quiet=True, pair="f16")
        with open(os.path.join(out16, "model.safetensors.index.json")) as fh:
            wm16 = json.load(fh)["weight_map"]
        stem16 = f"{P}layers.1.self_attn.q_proj"
        rd16 = ShardReader(os.path.join(out16, wm16[f"{stem16}.weight"]))
        s_dty, _s_shape, _s_blob = rd16.read(f"{stem16}.scales")
        b_dty, _b_shape, _b_blob = rd16.read(f"{stem16}.biases")
        expect("f16 companions", s_dty == "F16" and b_dty == "F16", f"{s_dty}/{b_dty}")
        rd16.close()
        check(out16, fixture, StubQuantizer(), quiet=True)
        shutil.rmtree(out16)
        # an index that drops a tensor (here every one) is refused before any tensor is read
        index = os.path.join(out, "model.safetensors.index.json")
        with open(index) as fh:
            full = json.load(fh)
        for drop in (list(full["weight_map"])[:1], list(full["weight_map"])):
            with open(index, "w") as fh:
                json.dump({**full, "weight_map": {k: v for k, v in full["weight_map"].items() if k not in drop}}, fh)
            try:
                check(out, fixture, StubQuantizer(), quiet=True)
                expect(f"an index without {len(drop)} tensor(s) is refused", False, "check passed")
            except Refusal as e:
                expect(f"an index without {len(drop)} tensor(s) is refused",
                       e.code == EXIT_REFUSED and f"{len(drop)} missing" in str(e), str(e))
        shutil.rmtree(out)

    # The qwen3moe family over a tiny fixture of two layers (3 experts of 64 values, 2 heads of 32
    # over one key head), the stub quantizer again: the names, the router at 8 bits, a config with
    # no `mode`, and a family this maker does not write.
    with tempfile.TemporaryDirectory() as tmp:
        fixture = os.path.join(tmp, "tiny3.gguf")
        rng = np.random.default_rng(5)
        f32 = lambda shape: rng.integers(1, 9, size=shape).astype(np.float32)

        def tensor(name, dims):
            return (name, F32, dims, f32(tuple(dims[::-1])).astype("<f4").tobytes())

        tensors = [tensor("token_embd.weight", [64, 8]), tensor("output.weight", [64, 8]),
                   tensor("output_norm.weight", [64])]
        for blk in range(2):
            g = f"blk.{blk}."
            tensors += [tensor(g + "attn_norm.weight", [64]), tensor(g + "ffn_norm.weight", [64]),
                        tensor(g + "attn_q.weight", [64, 64]), tensor(g + "attn_k.weight", [64, 32]),
                        tensor(g + "attn_v.weight", [64, 32]), tensor(g + "attn_output.weight", [64, 64]),
                        tensor(g + "attn_q_norm.weight", [32]), tensor(g + "attn_k_norm.weight", [32]),
                        tensor(g + "ffn_gate_inp.weight", [64, 3]),
                        tensor(g + "ffn_gate_exps.weight", [64, 8, 3]),
                        tensor(g + "ffn_up_exps.weight", [64, 8, 3]),
                        tensor(g + "ffn_down_exps.weight", [64, 64, 3])]
        R.write_gguf(fixture, [R._kv_string("general.architecture", "qwen3moe")], tensors)
        config_path = os.path.join(tmp, "config.json")
        block = {"group_size": GROUP, "bits": 4,
                 **{f"model.layers.{l}.mlp.gate": {"group_size": GROUP, "bits": 8} for l in range(48)}}
        with open(config_path, "w") as fh:
            json.dump({"model_type": "qwen3_moe", "num_hidden_layers": 48,
                       "quantization": block, "quantization_config": block}, fh)
        with open(os.path.join(tmp, "tokenizer.json"), "w") as fh:
            fh.write('{"version": "1.0"}')
        out = os.path.join(tmp, "out3")
        layers = make(fixture, out, config_path, StubQuantizer(), quiet=True)
        expect("qwen3moe layers", layers == [(0, "full"), (1, "full")], str(layers))
        with open(os.path.join(out, "config.json")) as fh:
            cut = json.load(fh)
        expect("qwen3moe cut layers", cut["num_hidden_layers"] == 2, str(cut["num_hidden_layers"]))
        expect("qwen3moe cut quantization",
               sorted(cut["quantization"]) == ["bits", "group_size", "model.layers.0.mlp.gate",
                                               "model.layers.1.mlp.gate"], str(sorted(cut["quantization"])))
        expect("qwen3moe copy dropped", "quantization_config" not in cut)
        expect("qwen3moe mode stays absent", "mode" not in cut["quantization"])
        with open(os.path.join(out, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        # 2 + 12 quantized linears per layer... embed, head, and per layer: q k v o, the router, three
        # stacks = 8, a weight, scales and biases each; the BF16 norms: the final one and 4 a layer
        expect("qwen3moe index holds every tensor", len(weight_map) == (2 + 2 * 8) * 3 + 1 + 2 * 4,
               str(len(weight_map)))
        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                     "model.layers.1.self_attn.q_proj.weight", "model.layers.1.mlp.gate.scales",
                     "model.layers.0.mlp.switch_mlp.down_proj.biases",
                     "model.layers.0.post_attention_layernorm.weight"):
            expect("qwen3moe names", name in weight_map, name)
        expect("qwen3moe has no delta or shared tensors",
               not any("linear_attn" in k or "shared_expert" in k for k in weight_map))
        reader = ShardReader(os.path.join(out, weight_map["model.layers.1.mlp.switch_mlp.up_proj.weight"]))
        _dt, shape, _blob = reader.read("model.layers.1.mlp.switch_mlp.up_proj.weight")
        reader.close()
        expect("qwen3moe stack is [experts, rows, packed K]", shape == [3, 8, 8], str(shape))
        check(out, fixture, StubQuantizer(), quiet=True)
        # the router read at 4 bits, against the config's 8, is refused
        with open(os.path.join(out, "config.json")) as fh:
            wrong = json.load(fh)
        wrong["quantization"].pop("model.layers.0.mlp.gate")
        with open(os.path.join(out, "config.json"), "w") as fh:
            json.dump(wrong, fh)
        try:
            check(out, fixture, StubQuantizer(), quiet=True)
            expect("a router without its 8-bit entry is refused", False, "check passed")
        except Refusal as e:
            expect("a router without its 8-bit entry is refused",
                   e.code == EXIT_REFUSED and "mlp.gate.weight: config says" in str(e), str(e))
        other = os.path.join(tmp, "other.gguf")
        R.write_gguf(other, [R._kv_string("general.architecture", "llama")], tensors[:1])
        try:
            make(other, os.path.join(tmp, "out_other"), config_path, StubQuantizer(), quiet=True)
            expect("another family is refused", False, "made")
        except Refusal as e:
            expect("another family is refused",
                   e.code == EXIT_REFUSED and "'llama' is not a family" in str(e), str(e))
    print("mlx-fixture self-test: ok")


def main(argv):
    if argv[0:1] == ["--self-test"]:
        self_test()
        return 0
    if (argv[0:1] == ["make"] and len(argv) in (5, 6) and argv[3] == "--config"
            and (len(argv) == 5 or argv[5] == "--f16")):
        fixture, out, config_path = argv[1], argv[2], argv[4]
        for p in (fixture, config_path):
            if not os.path.isfile(p):
                print(f"mlx-fixture: {p}: no such file", file=sys.stderr)
                return EXIT_MISSING
        pair = "f16" if len(argv) == 6 else "bf16"
        quantizer = MlxQuantizer()
        layers = make(fixture, out, config_path, quantizer, pair=pair)
        label = " (f16 pairs)" if pair == "f16" else ""
        print(f"made {out}{label}: layers {[k for _, k in layers]}")
        check(out, fixture, quantizer)
        return 0
    if len(argv) == 3 and argv[0] == "check":
        out, fixture = argv[1], argv[2]
        for p in (out, fixture):
            if not os.path.exists(p):
                print(f"mlx-fixture: {p}: no such path", file=sys.stderr)
                return EXIT_MISSING
        check(out, fixture, MlxQuantizer())
        return 0
    print(USAGE, file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refusal as e:
        print(f"mlx-fixture: {e}", file=sys.stderr)
        sys.exit(e.code)
