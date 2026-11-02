"""The MiniMax-M3.1-Flash oracle without llama.cpp (refset family tf-minimaxm31,
crates/refset/src/arch/minimaxm31/mod.rs): transformers' own modeling code (MiniMaxM3VLDecoderLayer,
MiniMaxM3VLRotaryEmbedding, MiniMaxM3VLRMSNorm, attention `eager`, f32 throughout) run over the vendor's safetensors
folder, written as the node-dump sets the MiniMax-M3 gates already read.

Weights. Every matrix is dequantized here, by its own rule, into f32 and loaded into the vendor module of one decoder
layer at a time:
  MXFP8 (attention, indexer, dense FFN, shared expert)   e4m3 x 2^(scale_inv - 127), one E8M0 byte per 32 values of a row
  NVFP4 (routed experts)                                 e2m1 x (e4m3 scale / weight_global_scale), one scale per 16
                                                         values, byte i of a row holding values 2i (low) and 2i + 1 (high)
  BF16 / F32 (norms, router, bias, embeddings, head)     widened as stored (the norms stay as stored: the module adds 1)
input_global_scale is read and validated, never applied: no activation is quantized here. The dequantization is checked
against compressed-tensors' own dequantize (NVFP4) and transformers' Fp8Dequantize (MXFP8) on tensors of the folder, bit
for bit, before a run starts and once more on one matrix of every layer; `--dequant-selftest` runs only that check.

Flow. Prefill is layer-major: every prompt's hidden rows advance through layer L before layer L + 1 is read, so each
layer's bytes are read once for all prompts (the next layer's span is read ahead with posix_fadvise). The experts of a
layer are dequantized into stacks that live across layers; the vendor's expert loop reads only the experts a token hit.
The greedy continuation (phase greedy) is token-major over the prefill's caches and dequantizes only the experts each
token hit.

Sets (one directory each under --out, the ids of tools/ref/data/minimaxm3, the M3 sets' taps and shapes where the graphs
agree):
  ref_minimaxm31_a          41 ids     every tap below, 64 greedy tokens
  ref_minimaxm31_b          55 ids     l_out and the router's expert ids, 64 greedy tokens
  ref_minimaxm31_c          116 ids    the same
  ref_minimaxm31_d          1,536 ids  the last position's logits
  ref_minimaxm31_e          2,560 ids  the sparse attention's block scores and selection, under the vendor's rule
  ref_minimaxm31_e_perhead  the same ids under llama.cpp's rule
Every set holds result_output (the last position's logits row), topk.tsv and greedy.tsv. The taps are the M3 graph's node
names: attn_norm, Qcur and Kcur (projection, reshape, rotated: occurrences 0, 1, 2), Qcur_normed, Kcur_normed, msa_bs,
msa_bsf, msa_block_mask, msa_fattn, kqv_out, ffn_inp, ffn_norm, ffn_moe_probs, ffn_moe_topk (ordered by the selection
score), ffn_moe_weights_scaled, ffn_moe_out, ffn_shexp, ffn_out, l_out, result_norm. The last layer's nodes after
attention and result_norm hold the last position alone, as llama.cpp's graph does. The MSA taps are laid out as the M3
sets lay them out: [blocks, 4 heads, tokens] scores, [blocks, tokens, 4 heads] mask, blocks padded to a window of 256
positions, -FLT_MAX on a block the causal mask hides, +1e30 on a token's own block.

Selection rule. The vendor's indexer takes the max over the index heads and one top-16 shared by every attention head
(MiniMaxM3VLIndexer; the deployed kernel takes one `q2k` and expands it over the key heads). llama.cpp, and the card
kernel, select per index head: index head g picks the blocks of key head g's query group. The two agree while a query
sees at most 16 blocks (2,048 positions): every visible block is picked. Past that they differ, so set e carries the
vendor's rule and e_perhead the per-head one; `# msa_rule` in each manifest names it.

    dump_tf.py --folder DIR --out DIR [--phase prefill|greedy|all] [--ids DIR] [--greedy N] [--sets a,b,..]
               [--until L] [--resume] [--threads N]
    dump_tf.py --dequant-selftest --folder DIR
    dump_tf.py --kind --folder DIR                     prints `folder` (the vendor's) or `cut` (of its revision)
    dump_tf.py ... --fixture-line "$(refset-check --fixture-line FIRST_SHARD)"     a cut's `# fixture` line, required for a cut

--out must be new, or hold the state a --resume continues. Phase prefill ends with every set that has no greedy
continuation complete and the others waiting for phase greedy, which continues from the saved caches (and from its own
state with --resume). Exit 65 on an input this does not read, 64 on a usage error, 70 when memory runs out."""

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers

TRANSFORMERS = "5.12.1"
ARCH = "minimax-m3"
REPO = "MiniMaxAI/MiniMax-M3.1-Flash"
REV = "76faee1b4ac0f412e547781172e94548a6526a88"
# The index of the vendor's folder at REV: the identity of its tensor map, so a folder that is not that revision is refused.
INDEX_SHA256 = "55adcfd75b3433b98a9a045c26386776c179bddeed7eb86c40b73187d218abf1"
PFX = "language_model.model."
HEAD = "language_model.lm_head.weight"
EMBED = PFX + "embed_tokens.weight"
FINAL_NORM = PFX + "norm.weight"
FIXTURE_KEY = "bloomery.fixture."
TOPK_ROWS = 40
WINDOW = 256  # the positions llama.cpp's block axis is padded to
LOCAL_BIAS = 1e30
FLT_MAX = float(np.finfo(np.float32).max)
MEM_FLOOR = 6 << 30
MERGE_GAP = 4 << 20
READ_CHUNK = 256 << 20
ROPE_CALLS = None  # a list while a layer runs under a tap, else None

DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}
ITEM = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "U8": 1, "I32": 4, "I64": 8}

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# Every E4M3 byte as f32 (exact); 0x7F and 0xFF are NaN.
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
# The two values a packed byte holds, the low nibble first.
NIBBLES = torch.stack([E2M1[torch.arange(256) & 15], E2M1[torch.arange(256) >> 4]], dim=1).contiguous()

ALL_TAPS = frozenset(
    "attn_norm Qcur Kcur Qcur_normed Kcur_normed msa_bs msa_bsf msa_block_mask msa_fattn kqv_out ffn_inp ffn_norm "
    "ffn_moe_probs ffn_moe_topk ffn_moe_weights_scaled ffn_moe_out ffn_shexp ffn_out l_out result_norm".split()
)
MSA_TAPS = frozenset({"msa_bs", "msa_bsf", "msa_block_mask"})
# The nodes of the last layer that llama.cpp computes for the last position alone.
POST_ATTENTION = frozenset(
    "ffn_inp ffn_norm ffn_moe_probs ffn_moe_topk ffn_moe_weights_scaled ffn_moe_out ffn_shexp ffn_out l_out".split()
)


class Refusal(Exception):
    """An input this does not read: named, never a defined output."""


# --------------------------------------------------------------------------------------------------------------------
# The folder: safetensors headers and ranged reads
# --------------------------------------------------------------------------------------------------------------------


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_into(fd, buf, offset):
    """Fill the uint8 tensor `buf` from `fd` at `offset`: sequential chunked reads, a short file refused."""
    mv = memoryview(buf.numpy())
    done = 0
    while done < len(mv):
        n = os.preadv(fd, [mv[done:done + READ_CHUNK]], offset + done)
        if n <= 0:
            raise Refusal("short read at byte %d of %d" % (offset + done, offset + len(mv)))
        done += n


class Shard:
    """One safetensors file: its header, validated, and the file descriptor ranged reads go through."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        n = struct.unpack("<Q", os.pread(self.fd, 8, 0))[0]
        if n == 0 or n > 256 << 20 or 8 + n > size:
            raise Refusal("%s: header length %d in a file of %d bytes" % (path, n, size))
        header = json.loads(os.pread(self.fd, n, 8))
        self.meta = header.pop("__metadata__", {})
        self.base = 8 + n
        for name, e in header.items():
            dt, shape, (b, end) = e["dtype"], e["shape"], e["data_offsets"]
            if dt not in ITEM:
                raise Refusal("%s: %s has dtype %s, which this does not read" % (path, name, dt))
            if not 0 <= b <= end <= size - self.base or end - b != ITEM[dt] * math.prod(shape):
                raise Refusal("%s: %s %s %s at [%d, %d) does not fit the file" % (path, name, dt, shape, b, end))
        self.header = header


class Folder:
    """Tensors by checkpoint name over the folder's shards (index weight_map), read in place."""

    def __init__(self, path):
        self.path = path
        index = os.path.join(path, "model.safetensors.index.json")
        if not os.path.isfile(index):
            raise Refusal("%s: no model.safetensors.index.json" % path)
        self.index_sha256 = sha256_file(index)
        self.map = json.load(open(index))["weight_map"]
        self.shards = {}
        self.first_shard = min(set(self.map.values()))

    def shard(self, file):
        if file not in self.shards:
            self.shards[file] = Shard(os.path.join(self.path, file))
        return self.shards[file]

    def entry(self, name):
        if name not in self.map:
            raise Refusal("%s: not in the index of %s" % (name, self.path))
        sh = self.shard(self.map[name])
        if name not in sh.header:
            raise Refusal("%s: the index names %s, whose header has no such tensor" % (name, self.map[name]))
        return sh, sh.header[name]

    def names(self, prefix):
        return sorted(k for k in self.map if k.startswith(prefix))

    def meta(self):
        """The first shard's __metadata__."""
        return self.shard(self.first_shard).meta

    def spans(self, names):
        """The file ranges `names` lie in, merged where closer than MERGE_GAP: [(shard, begin, end, [(b, e, name)])]."""
        by_shard = {}
        for nm in names:
            sh, e = self.entry(nm)
            by_shard.setdefault(sh.path, (sh, []))[1].append((e["data_offsets"][0], e["data_offsets"][1], nm))
        out = []
        for sh, items in by_shard.values():
            groups = []
            for b, e, nm in sorted(items):
                if groups and b - groups[-1][1] <= MERGE_GAP:
                    groups[-1][1] = max(groups[-1][1], e)
                    groups[-1][2].append((b, e, nm))
                else:
                    groups.append([b, e, [(b, e, nm)]])
            out.extend((sh, g[0], g[1], g[2]) for g in groups)
        return out

    def read(self, names):
        """{name: tensor}. One-byte tensors are views into their range's buffer, wider ones are copies."""
        out = {}
        for sh, b, e, items in self.spans(names):
            buf = torch.empty(e - b, dtype=torch.uint8)
            read_into(sh.fd, buf, sh.base + b)
            for tb, te, nm in items:
                head = sh.header[nm]
                raw = buf[tb - b:te - b]
                if ITEM[head["dtype"]] != 1:
                    raw = raw.clone()
                out[nm] = raw.view(DTYPES[head["dtype"]]).reshape(head["shape"])
        return out

    def prefetch(self, names):
        """Ask the kernel to read the ranges of `names` ahead (Linux only; elsewhere nothing happens)."""
        if not hasattr(os, "posix_fadvise"):
            return
        for sh, b, e, _ in self.spans(names):
            os.posix_fadvise(sh.fd, sh.base + b, e - b, os.POSIX_FADV_WILLNEED)

    def rows(self, name, ids):
        """f32 rows `ids` of the 2-D BF16 tensor `name`: only those rows are read."""
        sh, e = self.entry(name)
        if e["dtype"] != "BF16" or len(e["shape"]) != 2:
            raise Refusal("%s: %s %s, want a BF16 matrix" % (name, e["dtype"], e["shape"]))
        n, width = e["shape"]
        out = torch.empty(len(ids), width, dtype=torch.float32)
        for i, r in enumerate(ids):
            if not 0 <= r < n:
                raise Refusal("%s: row %d outside 0..%d" % (name, r, n))
            row = torch.empty(2 * width, dtype=torch.uint8)
            read_into(sh.fd, row, sh.base + e["data_offsets"][0] + r * 2 * width)
            out[i] = row.view(torch.bfloat16).float()
        return out


# --------------------------------------------------------------------------------------------------------------------
# Dequantization
# --------------------------------------------------------------------------------------------------------------------


def scale_check(name, sc_u8, stats):
    """A UE4M3 scale byte a reader refuses (sign bit set, or 0x7F) is a corrupt file; zero scales are counted."""
    if int((sc_u8 >= 0x7F).sum()):
        raise Refusal("%s: UE4M3 scale bytes with the sign bit set or equal to 0x7F" % name)
    stats["zero_scales"] += int((sc_u8 == 0).sum())


def nvfp4(packed, sc_u8, g, out=None):
    """One NVFP4 matrix in f32 by compressed-tensors' rule, x_q x (scale / global_scale). `packed` [n, k/2] holds values
    2i (low nibble) and 2i + 1 (high) in byte i, `sc_u8` [n, k/16] the E4M3 scale bytes, `g` the 0-d F32 divisor."""
    n, half = packed.shape
    k = 2 * half
    if k % 16 or tuple(sc_u8.shape) != (n, k // 16):
        raise Refusal("NVFP4 matrix %s with scales %s" % (tuple(packed.shape), tuple(sc_u8.shape)))
    vals = F.embedding(packed.long(), NIBBLES).view(n, k // 16, 16)
    s = E4M3[sc_u8.long()] / g
    if out is None:
        out = torch.empty(n, k)
    torch.mul(vals, s[:, :, None], out=out.view(n, k // 16, 16))
    return out


def mxfp8(w_u8, s_u8, out=None):
    """One MXFP8 matrix in f32 by transformers' rule, e4m3 x 2^(scale_inv - 127), one scale per 32 values of a row."""
    n, k = w_u8.shape
    if k % 32 or tuple(s_u8.shape) != (n, k // 32):
        raise Refusal("MXFP8 matrix %s with scales %s" % (tuple(w_u8.shape), tuple(s_u8.shape)))
    sf = (s_u8.to(torch.float32) - 127.0).exp2()
    q = w_u8.view(torch.float8_e4m3fn).to(torch.float32).view(n, k // 32, 32)
    if out is None:
        out = torch.empty(n, k)
    torch.mul(q, sf[:, :, None], out=out.view(n, k // 32, 32))
    return out


def mxfp8_check(name, w, s):
    """An e4m3 NaN byte or the E8M0 NaN byte is a corrupt file."""
    if w.dtype != torch.float8_e4m3fn or s.dtype != torch.uint8:
        raise Refusal("%s: %s with %s scales, want F8_E4M3 with U8" % (name, w.dtype, s.dtype))
    if int(((w.view(torch.uint8) & 0x7F) == 0x7F).sum()) or int((s == 0xFF).sum()):
        raise Refusal("%s: NaN bytes in an MXFP8 matrix" % name)


def nvfp4_reference(packed, sc_u8, g):
    """compressed-tensors' dequantize of the same matrix in f32, and the library's version."""
    import compressed_tensors
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from compressed_tensors.quantization import preset_name_to_scheme
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    n, half = packed.shape
    scheme = preset_name_to_scheme("NVFP4", ["Linear"])
    x_q = unpack_fp4_from_uint8(packed, n, 2 * half, dtype=torch.float32)
    w = dequantize(
        x_q=x_q,
        scale=sc_u8.view(torch.float8_e4m3fn).to(torch.float32),
        args=scheme.weights,
        global_scale=g.reshape(1),
        dtype=torch.float32,
    )
    return w, compressed_tensors.__version__


def mxfp8_reference(w_u8, s_u8):
    """transformers' Fp8Dequantize of the same matrix in f32."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    return Fp8Dequantize(None)._dequantize_one(w_u8.view(torch.float8_e4m3fn), s_u8, output_dtype=torch.float32)


def same_bits(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def check_nvfp4(name, packed, sc_u8, g):
    ref, version = nvfp4_reference(packed, sc_u8, g)
    if not same_bits(nvfp4(packed, sc_u8, g), ref):
        raise Refusal("%s: the NVFP4 dequantization differs from compressed-tensors %s" % (name, version))
    return version


def check_mxfp8(name, w, s):
    if not same_bits(mxfp8(w.view(torch.uint8), s), mxfp8_reference(w.view(torch.uint8), s)):
        raise Refusal("%s: the MXFP8 dequantization differs from transformers' Fp8Dequantize" % name)


# --------------------------------------------------------------------------------------------------------------------
# The vendor modules, tapped
# --------------------------------------------------------------------------------------------------------------------

hf = None  # the vendor's modeling module, imported by load_vendor
ORIG_ROPE = None


def rope_logged(q, k, cos, sin, unsqueeze_dim=1):
    out = ORIG_ROPE(q, k, cos, sin, unsqueeze_dim)
    if ROPE_CALLS is not None:
        ROPE_CALLS.append((q.shape[1], out))
    return out


def load_vendor():
    """Import the vendor's modeling module and route its apply_rotary_pos_emb through the logger."""
    global hf, ORIG_ROPE
    if hf is not None:
        return hf
    if transformers.__version__ != TRANSFORMERS:
        raise Refusal("transformers %s, the oracle is %s" % (transformers.__version__, TRANSFORMERS))
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as module

    hf = module
    ORIG_ROPE = module.apply_rotary_pos_emb
    module.apply_rotary_pos_emb = rope_logged
    return module


def make_classes():
    """The tapped indexer and the lazy experts, subclasses of the vendor's."""
    m = load_vendor()

    class TappedIndexer(m.MiniMaxM3VLIndexer):
        """The vendor indexer. `forward` is the vendor's; when taps or the per-head rule ask, the per-head block scores are
        recomputed from the same modules and kept in `last`. Under the per-head rule one selection a index head
        replaces the pooled one, and `build_block_mask` expands it over each head's query group."""

        rule = "pooled"
        keep = False
        last = None
        tie_exchanges = 0

        def forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
            vendor = super().forward(hidden_states, position_embeddings, past_key_values, position_ids)
            self.last = None
            if not (self.keep or self.rule == "per-head"):
                return vendor
            if self.rule == "per-head" and (past_key_values is not None or self.num_heads != self.config.num_key_value_heads):
                raise Refusal("the per-head rule runs without a cache and with one index head a key head")
            bs, k_len = self._block_scores(hidden_states, position_embeddings, past_key_values, position_ids)
            q_block = position_ids // self.block_size
            local = torch.arange(self.local_blocks, device=bs.device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
            boost = bs.new_zeros(bs.shape[0], bs.shape[2], bs.shape[3])
            boost.scatter_(-1, local_idx, LOCAL_BIAS)
            topk = min(self.topk_blocks, bs.shape[-1])
            # the vendor's selection recomputed: the head-max scores with the local blocks forced
            pooled = bs.amax(dim=1).scatter(-1, local_idx, float("inf"))
            pv, pi = pooled.topk(topk, dim=-1)
            pi = pi.masked_fill(pv == float("-inf"), -1)
            self._same_selection(pi, pv, vendor, pooled)
            if self.rule == "pooled":
                chosen, result = pi[:, None].expand(-1, bs.shape[1], -1, -1), vendor
            else:
                heads = bs.scatter(-1, local_idx[:, None].expand(-1, bs.shape[1], -1, -1), float("inf"))
                hv, hi = heads.topk(topk, dim=-1)
                chosen = result = hi.masked_fill(hv == float("-inf"), -1)
            self.last = {"bs": bs, "boost": boost, "chosen": chosen, "k_len": k_len}
            return result

        def _block_scores(self, hidden_states, position_embeddings, past_key_values, position_ids):
            """The per-head block-max scores [B, heads, Sq, blocks], -inf on a block no key of the query's causal prefix
            lies in, and the key length."""
            batch, q_len, _ = hidden_states.shape
            idx_q = self.q_norm(self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)).transpose(1, 2)
            idx_k = self.k_norm(self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)).transpose(1, 2)
            cos, sin = position_embeddings
            idx_q, idx_k = ORIG_ROPE(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])
            if past_key_values is not None:
                idx_k = past_key_values.layers[self.layer_idx].idx_keys
            k_len = idx_k.shape[2]
            nblk = -(-k_len // self.block_size)
            pad = nblk * self.block_size - k_len
            scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
            k_positions = torch.arange(k_len, device=idx_q.device)
            scores = scores.masked_fill(k_positions[None, None, None, :] > position_ids[:, None, :, None], float("-inf"))
            if pad:
                scores = F.pad(scores, (0, pad), value=float("-inf"))
            return scores.view(batch, self.num_heads, q_len, nblk, self.block_size).amax(dim=-1), k_len

        def _same_selection(self, mine, mine_scores, vendor, pooled):
            """The recomputed pooled selection is the vendor's: the same blocks, or at a tie the same scores."""
            if mine.shape != vendor.shape:
                raise Refusal("the indexer's selection is %s, its recomputation %s" % (tuple(vendor.shape), tuple(mine.shape)))
            a, b = mine.sort(dim=-1).values, vendor.sort(dim=-1).values
            if torch.equal(a, b):
                return
            theirs = pooled.gather(-1, vendor.clamp(min=0)).masked_fill(vendor < 0, float("-inf"))
            if not torch.equal(mine_scores.sort(dim=-1).values, theirs.sort(dim=-1).values):
                raise Refusal("the indexer's selection differs from its recomputation beyond a tie")
            self.tie_exchanges += int((a != b).any(dim=-1).sum())

        def build_block_mask(self, block_indices, attention_mask, key_length, dtype, device, position_ids):
            if self.rule != "per-head":
                return super().build_block_mask(block_indices, attention_mask, key_length, dtype, device, position_ids)
            batch, heads, q_len, _ = block_indices.shape
            nblk = -(-key_length // self.block_size)
            safe = block_indices.masked_fill(block_indices < 0, nblk)
            bias = block_indices.new_full((batch, heads, q_len, nblk + 1), float("-inf"), dtype=dtype)
            bias.scatter_(-1, safe, 0.0)
            keep = (bias[..., :nblk] == 0.0).repeat_interleave(self.block_size, dim=-1)[..., :key_length]
            if attention_mask is not None:
                keep = keep & (attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0)
            else:
                k_positions = torch.arange(key_length, device=device)
                keep = keep & ~(k_positions[None, None, None, :] > position_ids[:, None, :, None])
            mask = torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, torch.finfo(dtype).min)
            rep = self.config.num_attention_heads // heads
            return mask[:, :, None].expand(batch, heads, rep, q_len, key_length).reshape(batch, heads * rep, q_len, key_length)

    class LazyExperts(m.MiniMaxM3VLExperts):
        """The vendor's expert loop over stacks the `store` fills with the experts a call hits."""

        store = None

        def forward(self, hidden_states, top_k_index, top_k_weights):
            self.store.fill(torch.unique(top_k_index).tolist())
            return super().forward(hidden_states, top_k_index, top_k_weights)

    return TappedIndexer, LazyExperts


# --------------------------------------------------------------------------------------------------------------------
# One layer's weights
# --------------------------------------------------------------------------------------------------------------------

MXFP8_ATTN = (
    ("self_attn.q_proj", "self_attn.q_proj.weight"),
    ("self_attn.k_proj", "self_attn.k_proj.weight"),
    ("self_attn.v_proj", "self_attn.v_proj.weight"),
    ("self_attn.o_proj", "self_attn.o_proj.weight"),
    ("self_attn.index_q_proj", "self_attn.indexer.q_proj.weight"),
    ("self_attn.index_k_proj", "self_attn.indexer.k_proj.weight"),
)
NORMS = (
    ("input_layernorm.weight", "input_layernorm.weight"),
    ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
    ("self_attn.q_norm.weight", "self_attn.q_norm.weight"),
    ("self_attn.k_norm.weight", "self_attn.k_norm.weight"),
    ("self_attn.index_q_norm.weight", "self_attn.indexer.q_norm.weight"),
    ("self_attn.index_k_norm.weight", "self_attn.indexer.k_norm.weight"),
)
EXPERT_LEAVES = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")


def layer_prefix(L):
    return "%slayers.%d." % (PFX, L)


def expert_names(P, e):
    return ["%sblock_sparse_moe.experts.%d.%s.%s" % (P, e, w, leaf) for w in ("w1", "w2", "w3") for leaf in EXPERT_LEAVES]


def mlp_matrices(cfg, P, L):
    """The checkpoint stems of a layer's FFN matrices (each with .weight and .weight_scale_inv): gate, up, down."""
    stem = P + ("block_sparse_moe.shared_experts." if cfg.mlp_layer_types[L] == "sparse" else "mlp.")
    return [stem + "%s_proj" % w for w in ("gate", "up", "down")]


def layer_names(cfg, L):
    """Every checkpoint tensor of layer L but the routed experts', in the layout of the layer's kind."""
    P = layer_prefix(L)
    stems = [P + s for s, _ in MXFP8_ATTN] + mlp_matrices(cfg, P, L)
    names = [s + "." + leaf for s in stems for leaf in ("weight", "weight_scale_inv")] + [P + n for n, _ in NORMS]
    if cfg.mlp_layer_types[L] == "sparse":
        names += [P + "block_sparse_moe.gate.weight", P + "block_sparse_moe.e_score_correction_bias"]
    return names


def all_names(cfg, L):
    """layer_names and every routed expert's planes."""
    P = layer_prefix(L)
    names = layer_names(cfg, L)
    if cfg.mlp_layer_types[L] == "sparse":
        names += [n for e in range(cfg.num_local_experts) for n in expert_names(P, e)]
    return names


class Stats:
    """What the run counted: one line in each manifest."""

    def __init__(self):
        self.d = {
            "mxfp8_matrices": 0,
            "nvfp4_matrices": 0,
            "dequant_values": 0,
            "zero_scales": 0,
            "input_global_scale": {},
            "checked_nvfp4": 0,
            "checked_mxfp8": 0,
            "tie_exchanges": 0,
            "ct_version": "-",
        }

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def line(self):
        igs = self.d["input_global_scale"]
        return (
            "mxfp8 matrices %d, nvfp4 matrices %d, zero UE4M3 scales %d, input_global_scale values %s, dequantization "
            "checked bit for bit against compressed-tensors %s (%d nvfp4 matrices) and Fp8Dequantize (%d mxfp8), indexer "
            "selection ties %d"
            % (
                self.d["mxfp8_matrices"],
                self.d["nvfp4_matrices"],
                self.d["zero_scales"],
                ", ".join("%.9g x%d" % (v, n) for v, n in sorted(igs.items())) or "none",
                self.d["ct_version"],
                self.d["checked_nvfp4"],
                self.d["checked_mxfp8"],
                self.d["tie_exchanges"],
            )
        )


def norm_weight(t):
    """A norm weight as stored, widened: the vendor module adds 1."""
    return t.float()


def cat_gate_up(gate, up):
    """The vendor's fused gate_up matrix: the gate rows, then the up rows."""
    return torch.cat([gate, up], dim=0)


def mx_matrix(raw, stats, stem, check=False):
    """The f32 matrix of the MXFP8 tensor pair `stem`.weight and `stem`.weight_scale_inv in `raw`."""
    w, s = raw[stem + ".weight"], raw[stem + ".weight_scale_inv"]
    mxfp8_check(stem, w, s)
    if check:
        check_mxfp8(stem, w, s)
        stats["checked_mxfp8"] += 1
    stats["mxfp8_matrices"] += 1
    stats["dequant_values"] += w.numel()
    return mxfp8(w.view(torch.uint8), s)


class ExpertStore:
    """The routed experts of one layer: planes read from the folder, dequantized into the stacks on demand."""

    def __init__(self, folder, cfg, L, gate_up, down, stats, used):
        self.folder, self.cfg, self.L, self.stats, self.used = folder, cfg, L, stats, used
        self.P = layer_prefix(L)
        self.gate_up, self.down = gate_up, down
        self.filled = set()
        self.raw = {}

    def preload(self):
        """Read every expert's planes in one pass (prefill: a layer's experts are one run of the shard)."""
        self.raw.update(self.folder.read([n for e in range(self.cfg.num_local_experts) for n in expert_names(self.P, e)]))

    def release(self):
        self.raw.clear()

    def planes(self, e):
        names = expert_names(self.P, e)
        if names[0] not in self.raw:
            self.raw.update(self.folder.read(names))
        return {n: self.raw[n] for n in names}

    def fill(self, ids):
        """Dequantize the experts `ids` that this layer's stacks do not hold yet."""
        for e in ids:
            if e not in self.filled:
                self._fill_one(e)
                self.filled.add(e)

    def _fill_one(self, e):
        cfg = self.cfg
        hid, inter = cfg.hidden_size, cfg.intermediate_size
        planes = self.planes(e)
        self.used.update(planes)
        for w, rows, cols, dst in (
            ("w1", inter, hid, self.gate_up[e, :inter]),
            ("w3", inter, hid, self.gate_up[e, inter:]),
            ("w2", hid, inter, self.down[e]),
        ):
            stem = "%sblock_sparse_moe.experts.%d.%s." % (self.P, e, w)
            packed, sc = planes[stem + "weight_packed"], planes[stem + "weight_scale"]
            g, igs = planes[stem + "weight_global_scale"], planes[stem + "input_global_scale"]
            if packed.dtype != torch.uint8 or tuple(packed.shape) != (rows, cols // 2):
                raise Refusal("%sweight_packed: %s %s, want U8 %s" % (stem, packed.dtype, tuple(packed.shape), (rows, cols // 2)))
            if sc.dtype != torch.float8_e4m3fn or tuple(sc.shape) != (rows, cols // 16):
                raise Refusal("%sweight_scale: %s %s, want F8_E4M3 %s" % (stem, sc.dtype, tuple(sc.shape), (rows, cols // 16)))
            for leaf, t in (("weight_global_scale", g), ("input_global_scale", igs)):
                if t.dtype != torch.float32 or t.numel() != 1 or not math.isfinite(float(t)) or float(t) <= 0:
                    raise Refusal("%s%s: %s %s = %s, want one finite positive F32" % (stem, leaf, t.dtype, tuple(t.shape), t.tolist()))
            v = float(igs)
            self.stats["input_global_scale"][v] = self.stats["input_global_scale"].get(v, 0) + 1
            sc_u8 = sc.view(torch.uint8)
            scale_check(stem + "weight_scale", sc_u8, self.stats)
            nvfp4(packed, sc_u8, g.reshape(()), out=dst)
            self.stats["nvfp4_matrices"] += 1
            self.stats["dequant_values"] += rows * cols

    def check_one(self, e):
        """One expert matrix against compressed-tensors, bit for bit."""
        stem = "%sblock_sparse_moe.experts.%d.w1." % (self.P, e)
        planes = self.planes(e)
        self.stats["ct_version"] = check_nvfp4(
            stem + "weight_packed",
            planes[stem + "weight_packed"],
            planes[stem + "weight_scale"].view(torch.uint8),
            planes[stem + "weight_global_scale"].reshape(()),
        )
        self.stats["checked_nvfp4"] += 1


def stack_buffers(cfg):
    """The expert stacks every sparse layer's weights are dequantized into: allocated once, filled a layer at a time."""
    e, hid, inter = cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size
    return torch.empty(e, 2 * inter, hid), torch.empty(e, hid, inter)


def build_layer(run, L, preload, check):
    """The vendor decoder layer L on f32 weights, its ExpertStore (None on a dense layer) and the tensors it read."""
    cfg, folder, stats = run.cfg, run.folder, run.stats
    TappedIndexer, LazyExperts = run.classes
    P = layer_prefix(L)
    sparse = cfg.mlp_layer_types[L] == "sparse"
    raw = folder.read(layer_names(cfg, L))
    used = set(raw)
    with torch.device("meta"):
        layer = hf.MiniMaxM3VLDecoderLayer(cfg, L)
        if sparse:
            layer.mlp.experts = LazyExperts(cfg)
    layer.self_attn.indexer.__class__ = TappedIndexer
    sd = {}
    for stem, key in MXFP8_ATTN:
        sd[key] = mx_matrix(raw, stats, P + stem, check=check and stem == "self_attn.q_proj")
    for name, key in NORMS:
        t = raw[P + name]
        if t.dtype != torch.bfloat16:
            raise Refusal("%s: %s, want BF16" % (P + name, t.dtype))
        sd[key] = norm_weight(t)
    gate, up, down = (mx_matrix(raw, stats, s) for s in mlp_matrices(cfg, P, L))
    store = None
    if sparse:
        moe = P + "block_sparse_moe."
        for name, key in ((moe + "gate.weight", "mlp.gate.weight"), (moe + "e_score_correction_bias", "mlp.gate.e_score_correction_bias")):
            if raw[name].dtype != torch.float32:
                raise Refusal("%s: %s, want F32" % (name, raw[name].dtype))
            sd[key] = raw[name]
        sd["mlp.shared_experts.gate_up_proj.weight"] = cat_gate_up(gate, up)
        sd["mlp.shared_experts.down_proj.weight"] = down
        gate_up, down_stack = run.stacks()
        sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = gate_up, down_stack
        store = ExpertStore(folder, cfg, L, gate_up, down_stack, stats, used)
        layer.mlp.experts.store = store
    else:
        sd["mlp.gate_up_proj.weight"] = cat_gate_up(gate, up)
        sd["mlp.down_proj.weight"] = down
    for key, v in layer.state_dict().items():
        if key not in sd or tuple(sd[key].shape) != tuple(v.shape):
            raise Refusal("layer %d %s: %s, the module wants %s" % (L, key, tuple(sd[key].shape) if key in sd else None, tuple(v.shape)))
    layer.load_state_dict(sd, strict=True, assign=True)
    layer.eval()
    if store is not None and preload:
        store.preload()
    return layer, store, used


def leftover(folder, L, used):
    """The layer's tensors no one read: a corrupt map or a name this does not know."""
    rest = [n for n in folder.names(layer_prefix(L)) if n not in used]
    if rest:
        raise Refusal("layer %d: tensors not read: %s" % (L, rest[:6]))


# --------------------------------------------------------------------------------------------------------------------
# Sets, streams and the writer
# --------------------------------------------------------------------------------------------------------------------


class SetSpec:
    def __init__(self, name, ids, taps, greedy, rule="pooled"):
        self.name, self.ids_file, self.taps, self.greedy, self.rule = name, ids, frozenset(taps), greedy, rule

    @property
    def dirname(self):
        return "ref_minimaxm31_" + self.name

    def key(self):
        return (self.name, self.ids_file, tuple(sorted(self.taps)), self.greedy, self.rule)


def default_sets(greedy):
    return [
        SetSpec("a", "a", ALL_TAPS, greedy),
        SetSpec("b", "b", {"l_out", "ffn_moe_topk"}, greedy),
        SetSpec("c", "c", {"l_out", "ffn_moe_topk"}, greedy),
        SetSpec("d", "d", set(), 0),
        SetSpec("e", "e", MSA_TAPS, 0),
        SetSpec("e_perhead", "e", MSA_TAPS, 0, rule="per-head"),
    ]


def stem_of(name):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


class SetWriter:
    """A set's directory: raw f32 files per node, the manifest rows, then the manifest, topk.tsv and greedy.tsv."""

    def __init__(self, out, spec):
        self.dir = os.path.join(out, spec.dirname)
        os.makedirs(self.dir, exist_ok=True)
        self.spec = spec
        self.rows = []
        self.seen = {}
        self.written = 0

    def tensor(self, name, arr, ty="f32", op="hf", ints=False):
        """`arr` laid out as the node's ne reversed (ne0 the last axis). With `ints` the values are integers: the i32 twins
        of both layouts and their `int` rows go with the f32 files."""
        a = np.ascontiguousarray(arr)
        occ = self.seen.get(name, 0)
        self.seen[name] = occ + 1
        ne = list(reversed(a.shape)) + [1] * (4 - a.ndim)
        stem = "%s.%d" % (stem_of(name), occ)
        f = a.astype(np.float32)
        for fn in [stem + ".f32"] + ([stem + ".logical.f32"] if ints else []):
            f.astype("<f4").tofile(os.path.join(self.dir, fn))
        self.rows.append(
            "tensor\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%.17g\t%s\t1\t%d\t-\t-"
            % (name, occ, ty, ne[0], ne[1], ne[2], ne[3], f.size * 4, float(np.sum(f, dtype=np.float64)), op, int(ints))
        )
        self.written += 1
        if ints:
            b = a.astype(np.int32)
            wide = b.astype(np.int64)
            for layout, at in (("flat", ""), ("logical", ".logical")):
                fn = "%s%s.i32" % (stem, at)
                b.astype("<i4").tofile(os.path.join(self.dir, fn))
                self.rows.append(
                    "int\t%s\t%d\ttensor\t%s\ti32\t%s\t%d\t%d\t%d\t%d\t%s"
                    % (name, occ, ty, layout, b.size, b.size * 4, int(wide.sum()), int(np.abs(wide).max()), fn)
                )


def top_of(row, k):
    """The k largest logits of a row with their f64 log-softmax: (id, logit, logprob), a tie to the lower id."""
    r = np.asarray(row, dtype=np.float32)
    mx = float(r.max())
    lse = mx + math.log(float(np.exp(r.astype(np.float64) - mx).sum()))
    idx = np.lexsort((np.arange(r.size), -r))[:k]
    return [(int(i), float(r[i]), float(r[i]) - lse) for i in idx]


def step_of(row):
    """A greedy step: the argmax, its f64 logprob and the f32 gap to the runner-up."""
    top = top_of(row, 2)
    return top[0][0], top[0][2], float(np.float32(top[0][1]) - np.float32(top[1][1]))


def finite_row(who, row):
    if not np.isfinite(row).all():
        raise Refusal("%s: a logit is not finite" % who)


class Stream:
    """One prompt through the layers."""

    def __init__(self, spec, ids):
        self.spec, self.ids = spec, ids
        self.n = len(ids)
        self.h = self.cache = self.mask = self.pe = self.pos = None
        self.steps = []


# --------------------------------------------------------------------------------------------------------------------
# What a layer call shows
# --------------------------------------------------------------------------------------------------------------------


class Obs:
    """What the hooks saw during one layer call of one stream."""

    def __init__(self, layer):
        self.d = {}
        self.handles = []
        at, post = layer.self_attn, layer.post_attention_layernorm

        def keep(key, pre=False):
            def hook(module, args, output=None):
                self.d[key] = args[0] if pre else output
            return hook

        self.handles += [
            layer.input_layernorm.register_forward_hook(keep("attn_norm")),
            at.q_proj.register_forward_hook(keep("q_proj")),
            at.k_proj.register_forward_hook(keep("k_proj")),
            at.q_norm.register_forward_hook(keep("q_norm")),
            at.k_norm.register_forward_hook(keep("k_norm")),
            at.o_proj.register_forward_pre_hook(keep("kqv_out", pre=True)),
            post.register_forward_pre_hook(keep("ffn_inp", pre=True)),
            post.register_forward_hook(keep("ffn_norm")),
            layer.mlp.register_forward_hook(keep("ffn_out")),
        ]
        if hasattr(layer.mlp, "gate"):
            self.handles += [
                layer.mlp.gate.register_forward_hook(keep("router")),
                layer.mlp.experts.register_forward_hook(keep("experts")),
                layer.mlp.shared_experts.register_forward_hook(keep("ffn_shexp")),
            ]

    def close(self):
        for h in self.handles:
            h.remove()


def pad_blocks(x, window, value):
    """Pad the last (blocks) axis of `x` to `window` blocks with `value`."""
    short = window - x.shape[-1]
    return F.pad(x, (0, short), value=value) if short else x


def emit_layer(w, cfg, L, obs, rope, indexer):
    """Write the taps of w's set for layer L from what the hooks saw."""
    want, d = w.spec.taps, obs.d
    T = d["attn_norm"].shape[1]
    last = L == cfg.num_hidden_layers - 1
    row = slice(T - 1, T) if last else slice(0, T)
    nq, nk, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    if "attn_norm" in want:
        w.tensor("attn_norm-%d" % L, d["attn_norm"][0].float().numpy())
    main = [out for heads, out in rope if heads == nq]
    if not main:
        raise Refusal("layer %d: the attention's rotary call was not seen" % L)
    for tag, proj, norm, heads, rot in (
        ("Qcur", "q_proj", "q_norm", nq, main[0][0]),
        ("Kcur", "k_proj", "k_norm", nk, main[0][1]),
    ):
        if tag in want:
            flat = d[proj][0].float().numpy()
            w.tensor("%s-%d" % (tag, L), flat, op="MUL_MAT")
            w.tensor("%s-%d" % (tag, L), flat.reshape(T, heads, hd), op="RESHAPE")
        if tag + "_normed" in want:
            w.tensor("%s_normed-%d" % (tag, L), d[norm][0].float().numpy())
        if tag in want:
            w.tensor("%s-%d" % (tag, L), rot[0].transpose(0, 1).float().numpy(), op="ROPE")
    if want & MSA_TAPS:
        emit_msa(w, want, L, indexer)
    if "msa_fattn" in want:
        attn = d["kqv_out"][0].float().reshape(T, nk, nq // nk, hd)
        w.tensor("msa_fattn-%d" % L, attn.permute(1, 0, 2, 3).numpy())
    if "kqv_out" in want:
        w.tensor("kqv_out-%d" % L, d["kqv_out"][0].float().numpy())
    for name in ("ffn_inp", "ffn_norm"):
        if name in want:
            w.tensor("%s-%d" % (name, L), d[name][0, row].float().numpy())
    if "router" in d:
        emit_router(w, want, L, d, cfg, row)
        if "ffn_moe_out" in want:
            moe = d["experts"].reshape(T, -1)[row].float() * np.float32(cfg.routed_scaling_factor)
            w.tensor("ffn_moe_out-%d" % L, moe.numpy())
        if "ffn_shexp" in want:
            w.tensor("ffn_shexp-%d" % L, d["ffn_shexp"].reshape(T, -1)[row].float().numpy())
    if "ffn_out" in want:
        w.tensor("ffn_out-%d" % L, d["ffn_out"][0, row].float().numpy())
    if "l_out" in want:
        w.tensor("l_out-%d" % L, d["l_out"][0, row].float().numpy())


def emit_router(w, want, L, d, cfg, row):
    """ffn_moe_probs, ffn_moe_topk (ordered by the selection score, a tie to the lower id) and the scaled weights."""
    logits, weights, index = d["router"]
    probs = torch.sigmoid(logits.float())
    if "ffn_moe_probs" in want:
        w.tensor("ffn_moe_probs-%d" % L, probs[row].numpy())
    if not want & {"ffn_moe_topk", "ffn_moe_weights_scaled"}:
        return
    sel = probs + d["bias"]
    order = sel.sort(dim=-1, descending=True, stable=True).indices[:, : cfg.num_experts_per_tok]
    if not torch.equal(order.sort(dim=-1).values, index.sort(dim=-1).values):
        # the router's own top-k differs from the sorted one only where two scores tie
        if not torch.equal(sel.gather(-1, order).sort(dim=-1).values, sel.gather(-1, index).sort(dim=-1).values):
            raise Refusal("layer %d: the router's selection differs from its recomputation beyond a tie" % L)
        order = index.gather(-1, sel.gather(-1, index).sort(dim=-1, descending=True, stable=True).indices)
    if "ffn_moe_topk" in want:
        w.tensor("ffn_moe_topk-%d" % L, order[row].numpy(), ty="i32", ints=True)
    if "ffn_moe_weights_scaled" in want:
        by_expert = torch.zeros_like(probs).scatter_(-1, index, weights.float())
        scaled = by_expert.gather(-1, order) * np.float32(cfg.routed_scaling_factor)
        w.tensor("ffn_moe_weights_scaled-%d" % L, scaled[row].numpy()[:, :, None])


def emit_msa(w, want, L, indexer):
    """msa_bs, msa_bsf and msa_block_mask in the M3 sets' layout."""
    info = indexer.last
    if info is None:
        raise Refusal("layer %d: the indexer kept no block scores" % L)
    bs, boost, chosen = info["bs"][0], info["boost"][0], info["chosen"][0]
    heads, T, _ = bs.shape
    window = -(-info["k_len"] // WINDOW) * (WINDOW // indexer.block_size)
    scores = pad_blocks(bs.masked_fill(bs == float("-inf"), -FLT_MAX), window, -FLT_MAX)
    if "msa_bs" in want:
        w.tensor("msa_bs-%d" % L, scores.permute(1, 0, 2).numpy())
    if "msa_bsf" in want:
        w.tensor("msa_bsf-%d" % L, (scores + pad_blocks(boost, window, 0.0)[None]).permute(1, 0, 2).numpy())
    if "msa_block_mask" in want:
        mask = torch.full((heads, T, window + 1), float("-inf"))
        mask.scatter_(-1, chosen.masked_fill(chosen < 0, window), 0.0)
        w.tensor("msa_block_mask-%d" % L, mask[..., :window].numpy(), ty="f16")


# --------------------------------------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------------------------------------


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            out[k] = int(v.split()[0]) * 1024
    return out


def watch_memory(stop):
    """Abort the run (exit 70) when MemAvailable falls under the floor: a swapping oracle takes hours."""
    if not os.path.exists("/proc/meminfo"):
        return
    while not stop.is_set():
        avail = meminfo()["MemAvailable"]
        if avail < MEM_FLOOR:
            print("ABORT: MemAvailable %d under the floor %d" % (avail, MEM_FLOOR), flush=True)
            os._exit(70)
        stop.wait(2)


def mem_gib():
    return meminfo()["MemAvailable"] / 2**30 if os.path.exists("/proc/meminfo") else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------------------------


def text_config(folder_path):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    cfg_json = json.load(open(os.path.join(folder_path, "config.json")))
    cfg = MiniMaxM3VLTextConfig(**cfg_json["text_config"])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    return cfg, cfg_json


def folder_kind(folder):
    """"folder" for the vendor's own folder (the revision's index by its sha256), "cut" for a cut of that revision (the first
    shard's bloomery.fixture.* keys, naming it); a folder that is neither is refused."""
    keys = {k: v for k, v in folder.meta().items() if k.startswith(FIXTURE_KEY)}
    if not keys:
        if folder.index_sha256 != INDEX_SHA256:
            raise Refusal(
                "%s: index sha256 %s, revision %s of %s has %s" % (folder.path, folder.index_sha256, REV[:8], REPO, INDEX_SHA256)
            )
        return "folder"
    if keys.get(FIXTURE_KEY + "rev") != REV:
        raise Refusal("%s: a cut of revision %r, the oracle is %s" % (folder.path, keys.get(FIXTURE_KEY + "rev"), REV))
    return "cut"


def fixture_line(folder, given=None):
    """The `# fixture` text of a cut folder, or None for the vendor's own folder. The text is refset's, the one owner of the
    line (`refset-check --fixture-line <first shard>`, passed as --fixture-line): it is held here only to the shard it names,
    every bloomery.fixture.* key of the shard but the card budget, each once with the shard's value, so a line of another
    cut, of part of one or of none is refused; a folder that is neither the vendor's nor a cut of its revision is too."""
    if folder_kind(folder) == "folder":
        if given is not None:
            raise Refusal("%s: --fixture-line for a folder whose first shard has no %s* keys: the vendor's own folder has no fixture line" % (folder.path, FIXTURE_KEY))
        return None
    keys = {k: v for k, v in folder.meta().items() if k.startswith(FIXTURE_KEY) and k != FIXTURE_KEY + "card_budget"}
    if given is None:
        raise Refusal(
            "%s: a cut of the revision, whose `# fixture` line is refset's: pass --fixture-line \"$(refset-check --fixture-line <first shard>)\"" % folder.path
        )
    prefix = "# fixture\t"
    if not given.startswith(prefix) or "\n" in given:
        raise Refusal("--fixture-line %r is not one '# fixture<TAB>key=value ...' line" % given)
    held = {}
    for token in given[len(prefix):].split(" "):
        k, sep, v = token.partition("=")
        if not sep or not k.startswith(FIXTURE_KEY) or k in held:
            raise Refusal("--fixture-line: %r is not one %s* key=value, once" % (token, FIXTURE_KEY))
        held[k] = v
    for k in sorted(set(keys) | set(held)):
        if held.get(k) != keys.get(k):
            raise Refusal(
                "--fixture-line states %s=%s, refset's line of the first shard of %s has %s" % (k, held.get(k, "nothing"), folder.path, keys.get(k, "nothing"))
            )
    return given[len(prefix):]


def dequant_selftest(folder, cfg, stats):
    """The dequantization against compressed-tensors and Fp8Dequantize on tensors of the folder, bit for bit: every
    MXFP8 matrix of the first dense layer and of the first sparse layer, and three matrices of four of its experts."""
    dense = next(i for i, t in enumerate(cfg.mlp_layer_types) if t == "dense")
    sparse = next(i for i, t in enumerate(cfg.mlp_layer_types) if t == "sparse")
    for L in sorted({dense, sparse}):
        raw = folder.read(layer_names(cfg, L))
        P = layer_prefix(L)
        for stem in [P + s for s, _ in MXFP8_ATTN] + mlp_matrices(cfg, P, L):
            mx_matrix(raw, stats, stem, check=True)
    store = ExpertStore(folder, cfg, sparse, None, None, stats, set())
    n = cfg.num_local_experts
    for e in sorted({0, 1, n // 2, n - 1}):
        planes = store.planes(e)
        for w in ("w1", "w2", "w3"):
            stem = "%sblock_sparse_moe.experts.%d.%s." % (store.P, e, w)
            sc = planes[stem + "weight_scale"].view(torch.uint8)
            scale_check(stem + "weight_scale", sc, stats)
            stats["ct_version"] = check_nvfp4(
                stem + "weight_packed", planes[stem + "weight_packed"], sc, planes[stem + "weight_global_scale"].reshape(())
            )
            stats["checked_nvfp4"] += 1
    return stats.line()


def save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def cache_tensors(cache):
    """A DynamicCache's keys, values and index keys a layer, copies; None for a layer nothing was cached for."""
    out = []
    for layer in cache.layers:
        if getattr(layer, "keys", None) is None:
            out.append(None)
        else:
            out.append((layer.keys.clone(), layer.values.clone(), layer.idx_keys.clone()))
    return out


def restored_cache(cfg, tensors):
    """A new DynamicCache holding `tensors` (cache_tensors' output)."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=cfg)
    for L, t in enumerate(tensors):
        if t is not None:
            cache.update(t[0], t[1], L)
            cache.layers[L].update_index(t[2])
    return cache


class Run:
    """One invocation: the folder, the sets and their streams, the writers, and the state on disk."""

    def __init__(self, a):
        self.a = a
        self.folder_arg = a.folder.rstrip("/")
        self.folder = Folder(self.folder_arg)
        self.cfg, self.cfg_json = text_config(self.folder_arg)
        if a.dequant_selftest:
            folder_kind(self.folder)
            self.fixture = None
        else:
            self.fixture = fixture_line(self.folder, a.fixture_line)
        self.classes = make_classes()
        self.stats = Stats()
        self._stacks = None
        self.src = os.path.abspath(__file__)
        with open(self.src, "rb") as f:
            self.md5 = hashlib.md5(f.read()).hexdigest()
        # The public torch version: the local tag (+cpu) is the wheel's index, the numbers are the release's.
        torch_release = str(torch.__version__).split("+")[0]
        self.build = "transformers-%s+torch-%s+dump_tf-%s" % (transformers.__version__, torch_release, self.md5[:12])
        self.modeling_sha256 = sha256_file(hf.__file__)
        wanted = None if a.sets is None else a.sets.split(",")
        sets = default_sets(a.greedy)
        if wanted is not None:
            unknown = [n for n in wanted if n not in {s.name for s in sets}]
            if unknown:
                raise Refusal("--sets: no set named %s (the sets are %s)" % (unknown, [s.name for s in sets]))
            sets = [s for s in sets if s.name in wanted]
        if not sets:
            raise Refusal("--sets: no sets")
        self.streams = []
        for spec in sets:
            path = os.path.join(a.ids, spec.ids_file + ".ids")
            try:
                ids = [int(x) for x in open(path).read().split()]
            except (OSError, ValueError) as e:
                raise Refusal("%s: %s" % (path, e))
            if not ids or any(t < 0 or t >= self.cfg.vocab_size for t in ids):
                raise Refusal("%s: no ids, or an id outside the vocabulary of %d" % (path, self.cfg.vocab_size))
            self.streams.append(Stream(spec, ids))
        self.state_dir = os.path.join(a.out, "_state") if a.out else None
        self.writers = {}
        self.identity = (
            self.md5,
            self.folder_arg,
            self.folder.index_sha256,
            self.fixture,
            self.cfg_json,
            self.modeling_sha256,
            str(torch.__version__),
            tuple((s.spec.key(), tuple(s.ids)) for s in self.streams),
        )

    def stacks(self):
        if self._stacks is None:
            self._stacks = stack_buffers(self.cfg)
        return self._stacks

    def path(self, name):
        return os.path.join(self.state_dir, name)

    def start(self, continues):
        """Create the set directories; a new run needs an --out that holds nothing."""
        os.makedirs(self.a.out, exist_ok=True)
        if not continues and os.listdir(self.a.out):
            raise Refusal("--out %s is not empty: remove it, or --resume the run it holds" % self.a.out)
        os.makedirs(self.state_dir, exist_ok=True)
        for s in self.streams:
            self.writers[s.spec.name] = SetWriter(self.a.out, s.spec)

    def prepare(self):
        """Embeddings, masks and rotary tables of every stream, as the vendor's TextModel.forward builds them."""
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        self.rotary = hf.MiniMaxM3VLRotaryEmbedding(config=self.cfg)
        for s in self.streams:
            s.h = self.folder.rows(EMBED, s.ids).unsqueeze(0)
            s.pos = torch.arange(s.n).unsqueeze(0)
            if s.spec.greedy:
                s.cache = DynamicCache(config=self.cfg)
                if type(s.cache.layers[0]).__name__ != "MiniMaxM3VLSparseCacheLayer":
                    raise Refusal("the cache holds %s layers, the sparse layers need MiniMaxM3VLSparseCacheLayer" % type(s.cache.layers[0]).__name__)
            s.mask = create_causal_mask(config=self.cfg, inputs_embeds=s.h, attention_mask=None, past_key_values=s.cache, position_ids=s.pos)
            s.pe = self.rotary(s.h, position_ids=s.pos)

    def final_head(self):
        """The final norm and the f32 head."""
        cfg = self.cfg
        t = self.folder.read([FINAL_NORM, HEAD])
        if t[FINAL_NORM].dtype != torch.bfloat16 or t[HEAD].dtype != torch.bfloat16:
            raise Refusal("the final norm and the head are BF16")
        if tuple(t[HEAD].shape) != (cfg.vocab_size, cfg.hidden_size):
            raise Refusal("%s: %s, want %s" % (HEAD, tuple(t[HEAD].shape), (cfg.vocab_size, cfg.hidden_size)))
        norm = hf.MiniMaxM3VLRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.weight.data = norm_weight(t[FINAL_NORM])
        return norm, t[HEAD].float()

    def manifest_header(self, s):
        spec, cfg = s.spec, self.cfg
        lines = [
            "# MiniMax-M3.1-Flash oracle: transformers MiniMaxM3VL f32 eager forward of the safetensors folder, raw f32, little-endian",
            "# model\t%s" % os.path.join(self.folder_arg, self.folder.first_shard),
        ]
        if self.fixture is not None:
            lines.append("# fixture\t%s" % self.fixture)
        lines += [
            "# build\t%s" % self.build,
            "# arch\t%s" % ARCH,
            "# model_file\t%s@%s" % (REPO, REV),
            "# tokens\t%s" % ",".join(map(str, s.ids)),
            "# topk\ttopk.tsv\t%d" % TOPK_ROWS,
            "# greedy\tgreedy.tsv\t%d" % spec.greedy,
            "# transformers\t%s modeling_minimax_m3_vl sha256 %s" % (transformers.__version__, self.modeling_sha256),
            "# torch\t%s" % torch.__version__,
            "# dequant\tNVFP4: e2m1 x (e4m3 scale / weight_global_scale); MXFP8: e4m3 x 2^(scale_inv - 127); norms as stored "
            "(the module adds 1); input_global_scale validated, not applied",
            "# msa_rule\t%s" % spec.rule,
            "# msa_layout\tblocks padded to a window of %d positions; -FLT_MAX on a block the causal mask hides; msa_bsf adds "
            "%g on a token's own block; msa_block_mask is 0 on the blocks picked and -inf on the rest" % (WINDOW, LOCAL_BIAS),
            "# positions\tevery position; the last layer's nodes after attention, result_norm and result_output at the last "
            "position only",
            "# routes\tffn_moe_topk ordered by the selection score (sigmoid probability plus bias), a tie to the lower id; "
            "ffn_moe_weights_scaled in the same order, the vendor's normalized weights times routed_scaling_factor",
            "# layers\t%d" % cfg.num_hidden_layers,
            "# checks\t%s" % self.stats.line(),
            "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1",
        ]
        return lines

    def finalize(self, s):
        """Write the set's greedy.tsv and manifest, the manifest by rename so a dead run leaves none."""
        w = self.writers[s.spec.name]
        if len(s.steps) != s.spec.greedy:
            raise Refusal("%s: %d greedy steps, the set states %d" % (s.spec.name, len(s.steps), s.spec.greedy))
        with open(os.path.join(w.dir, "greedy.tsv"), "w") as f:
            f.write("# step\tid\tmargin\tlogprob\n")
            for k, (i, lp, gap) in enumerate(s.steps):
                f.write("%d\t%d\t%.9g\t%.17g\n" % (k, i, gap, lp))
        lines = self.manifest_header(s)
        if any(r.startswith("int\t") for r in w.rows):
            lines.append("# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile")
        lines += w.rows
        lines.append("# complete\t%d\t0" % w.written)
        path = os.path.join(w.dir, "MANIFEST.tsv")
        with open(path + ".partial", "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(path + ".partial", path)

    # ----------------------------------------------------------------------------------------------------------------

    def run_layer(self, L, layer, taps):
        """Layer L over every stream: the vendor forward, then the taps of each set."""
        global ROPE_CALLS
        cfg = self.cfg
        obs = Obs(layer)
        indexer = layer.self_attn.indexer
        times = []
        try:
            for s in self.streams:
                obs.d.clear()
                indexer.rule, indexer.keep, indexer.last = s.spec.rule, bool(s.spec.taps & MSA_TAPS) and taps, None
                if hasattr(layer.mlp, "gate"):
                    obs.d["bias"] = layer.mlp.gate.e_score_correction_bias
                t0 = time.time()
                ROPE_CALLS = []
                out = layer(s.h, position_embeddings=s.pe, attention_mask=s.mask, position_ids=s.pos, past_key_values=s.cache)
                rope, ROPE_CALLS = ROPE_CALLS, None
                obs.d["l_out"] = out
                if taps and s.spec.taps:
                    emit_layer(self.writers[s.spec.name], cfg, L, obs, rope, indexer)
                s.h = out
                times.append("%s %.1fs" % (s.spec.name, time.time() - t0))
        finally:
            ROPE_CALLS = None
            obs.close()
        self.stats["tie_exchanges"] += indexer.tie_exchanges
        indexer.tie_exchanges = 0
        return times

    def state(self, next_layer):
        return {
            "identity": self.identity,
            "next": next_layer,
            "h": {s.spec.name: s.h for s in self.streams},
            "caches": {s.spec.name: cache_tensors(s.cache) for s in self.streams if s.cache is not None},
            "rows": {n: (w.rows, w.seen, w.written) for n, w in self.writers.items()},
            "stats": self.stats.d,
            "steps": {s.spec.name: s.steps for s in self.streams},
        }

    def restore(self, st):
        if st.get("identity") != self.identity:
            raise Refusal("%s: the writer, folder, cut, model code, torch or sets changed; start again without --resume" % self.state_dir)
        for s in self.streams:
            s.h = st["h"][s.spec.name]
            s.steps = st["steps"][s.spec.name]
            if s.cache is not None:
                s.cache = restored_cache(self.cfg, st["caches"][s.spec.name])
        for n, (rows, seen, written) in st["rows"].items():
            self.writers[n].rows, self.writers[n].seen, self.writers[n].written = rows, seen, written
        self.stats.d = st["stats"]
        return st["next"]

    def prefill(self, until, resume):
        cfg, folder = self.cfg, self.folder
        n_layers = cfg.num_hidden_layers
        start = 0
        if resume:
            path = self.path("prefill.pt")
            if not os.path.isfile(path):
                raise Refusal("%s: no saved state for --resume" % path)
            start = self.restore(torch.load(path, weights_only=False))
        if not 0 < until <= n_layers or until < start:
            raise Refusal("--until %d: not in %d..%d" % (until, max(start, 1), n_layers))
        t_all = time.time()
        for L in range(start, until):
            t0 = time.time()
            values0 = self.stats["dequant_values"]
            layer, store, used = build_layer(self, L, preload=True, check=True)
            if store is not None:
                store.check_one(L % cfg.num_local_experts)
                store.fill(range(cfg.num_local_experts))
                store.release()
            if L + 1 < n_layers:
                folder.prefetch(all_names(cfg, L + 1))
            t_load = time.time() - t0
            values = (self.stats["dequant_values"] - values0) / 1e9
            times = self.run_layer(L, layer, taps=True)
            leftover(folder, L, used)
            save(self.state(L + 1), self.path("prefill.pt"))
            print(
                "layer %d %s read+dequant %.1fs (%.2f G values, %.2f G values/s of it), %s, MemAvailable %.1f GiB, %.1fs"
                % (L, cfg.mlp_layer_types[L], t_load, values, values / t_load, " ".join(times), mem_gib(), time.time() - t0),
                flush=True,
            )
            del layer, store
        if until < n_layers:
            print("PAUSED at layer %d of %d after %.1fs (state in %s)" % (until, n_layers, time.time() - t_all, self.state_dir), flush=True)
            return False
        self.finish_prefill()
        return True

    def finish_prefill(self):
        """The final norm and the head over each prompt's last row; topk.tsv; the sets with no continuation complete."""
        cfg = self.cfg
        norm, head = self.final_head()
        x = norm(torch.cat([s.h[:, -1] for s in self.streams], dim=0))
        logits = F.linear(x, head).numpy()
        for i, s in enumerate(self.streams):
            row = logits[i]
            finite_row(s.spec.name, row)
            w = self.writers[s.spec.name]
            if "result_norm" in s.spec.taps:
                w.tensor("result_norm", x[i:i + 1].numpy())
            w.tensor("result_output", row[None, :])
            with open(os.path.join(w.dir, "topk.tsv"), "w") as f:
                for r, (tok, logit, lp) in enumerate(top_of(row, TOPK_ROWS)):
                    f.write("%d\t%d\t%.9g\t%.17g\n" % (r, tok, logit, lp))
            s.steps = [step_of(row)] if s.spec.greedy else []
            if not s.spec.greedy:
                self.finalize(s)
        save(self.state(cfg.num_hidden_layers), self.path("prefill.pt"))
        print("prefill complete: %s" % " ".join("%s argmax %d" % (s.spec.name, top_of(logits[i], 1)[0][0]) for i, s in enumerate(self.streams)), flush=True)

    def greedy(self, resume):
        """The greedy continuation of the sets that state one, token-major over the prefill's caches."""
        from transformers.masking_utils import create_causal_mask

        cfg = self.cfg
        n_layers = cfg.num_hidden_layers
        todo = [s for s in self.streams if s.spec.greedy]
        if not todo:
            return
        path = self.path("prefill.pt")
        if not os.path.isfile(path):
            raise Refusal("%s: phase greedy continues a finished prefill" % path)
        st = torch.load(path, weights_only=False)
        if st["next"] != n_layers:
            raise Refusal("%s: the prefill stopped at layer %d of %d" % (path, st["next"], n_layers))
        self.restore(st)
        gpath = self.path("greedy.pt")
        if resume:
            if not os.path.isfile(gpath):
                raise Refusal("%s: no saved state for --resume" % gpath)
            g = torch.load(gpath, weights_only=False)
            if g.get("identity") != self.identity:
                raise Refusal("%s: the writer, folder, cut, model code, torch or sets changed; start again without --resume" % gpath)
            for s in todo:
                s.steps = g["steps"][s.spec.name]
                s.cache = restored_cache(cfg, g["caches"][s.spec.name])
        norm, head = self.final_head()
        layers = {}
        n_steps = max(s.spec.greedy for s in todo)
        t_all = time.time()
        while min(len(s.steps) for s in todo) < n_steps:
            t0 = time.time()
            values0 = self.stats["dequant_values"]
            live = [s for s in todo if len(s.steps) < s.spec.greedy]
            for s in live:
                s.h = self.folder.rows(EMBED, [s.steps[-1][0]]).unsqueeze(0)
                s.pos = torch.tensor([[s.n + len(s.steps) - 1]])
                s.pe = self.rotary(s.h, position_ids=s.pos)
                s.mask = create_causal_mask(config=cfg, inputs_embeds=s.h, attention_mask=None, past_key_values=s.cache, position_ids=s.pos)
            for L in range(n_layers):
                if L not in layers:
                    layer, store, _ = build_layer(self, L, preload=False, check=False)
                    layers[L] = (layer, store)
                layer, store = layers[L]
                if store is not None:
                    store.filled.clear()
                for s in live:
                    layer.self_attn.indexer.rule, layer.self_attn.indexer.keep = "pooled", False
                    s.h = layer(s.h, position_embeddings=s.pe, attention_mask=s.mask, position_ids=s.pos, past_key_values=s.cache)
            x = norm(torch.cat([s.h[:, -1] for s in live], dim=0))
            rows = F.linear(x, head).numpy()
            for i, s in enumerate(live):
                finite_row(s.spec.name, rows[i])
                s.steps.append(step_of(rows[i]))
            save(
                {
                    "identity": self.identity,
                    "steps": {s.spec.name: s.steps for s in todo},
                    "caches": {s.spec.name: cache_tensors(s.cache) for s in todo},
                },
                gpath,
            )
            print(
                "greedy step %d %s dequant %.1f G values, MemAvailable %.1f GiB, %.1fs"
                % (
                    len(live[0].steps) - 1,
                    " ".join("%s %d" % (s.spec.name, s.steps[-1][0]) for s in live),
                    (self.stats["dequant_values"] - values0) / 1e9,
                    mem_gib(),
                    time.time() - t0,
                ),
                flush=True,
            )
        for s in todo:
            self.finalize(s)
        print("greedy complete in %.1fs this sitting" % (time.time() - t_all), flush=True)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("dump_tf: %s" % message, file=sys.stderr)
        sys.exit(64)


def parse(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = Parser(description="the MiniMax-M3.1-Flash oracle sets", add_help=True)
    ap.add_argument("--folder", required=True, help="the checkpoint folder, as the sets' # model line spells its first shard")
    ap.add_argument("--out", help="where the set directories go (new, or holding the state --resume continues)")
    ap.add_argument("--phase", choices=("prefill", "greedy", "all"), default="all")
    ap.add_argument("--ids", default=os.path.join(here, "..", "data", "minimaxm3"), help="the directory of a..e.ids")
    ap.add_argument("--greedy", type=int, default=64, help="greedy tokens of the sets that state a continuation")
    ap.add_argument("--sets", help="comma-separated set names (default: all)")
    ap.add_argument("--until", type=int, help="stop after this many layers with the state saved")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--kind", action="store_true", help="only print `folder` (the vendor's) or `cut` (of its revision) for --folder")
    ap.add_argument("--fixture-line", help="a cut's `# fixture` line as `refset-check --fixture-line <first shard>` prints it (required for a cut, refused for the vendor's folder)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--dequant-selftest", action="store_true", help="only check the dequantization against the references")
    a = ap.parse_args(argv)
    if not (a.dequant_selftest or a.kind) and not a.out:
        ap.error("--out is required")
    if a.greedy < 1:
        ap.error("--greedy wants at least 1")
    if a.resume and a.phase == "all":
        ap.error("--resume continues one phase: name it with --phase prefill or --phase greedy")
    return a


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.kind:
        try:
            print(folder_kind(Folder(a.folder)))
        except Refusal as e:
            print("dump_tf: %s" % e, file=sys.stderr)
            return 65
        return 0
    stop = threading.Event()
    threading.Thread(target=watch_memory, args=(stop,), daemon=True).start()
    try:
        load_vendor()
        run = Run(a)
        print("build %s modeling sha256 %s folder %s%s" % (run.build, run.modeling_sha256, run.folder_arg, " (a cut: %s)" % run.fixture if run.fixture else ""), flush=True)
        if a.dequant_selftest:
            print("dequant self-test PASS: %s" % dequant_selftest(run.folder, run.cfg, run.stats), flush=True)
            return 0
        print("dequant self-test PASS: %s" % dequant_selftest(run.folder, run.cfg, run.stats), flush=True)
        run.start(a.resume or a.phase == "greedy")
        run.prepare()
        if a.phase in ("prefill", "all"):
            done = run.prefill(a.until or run.cfg.num_hidden_layers, a.resume and a.phase == "prefill")
            if not done:
                return 0
        if a.phase in ("greedy", "all"):
            run.greedy(a.resume and a.phase == "greedy")
        return 0
    except Refusal as e:
        print("dump_tf: %s" % e, file=sys.stderr)
        return 65
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
