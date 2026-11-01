"""The Qwen3.8-Flash-Next NVFP4 oracle (refset family hf-qwen4exp-nvfp4, crates/refset/src/arch/qwen4exp/nvfp4.rs): the
forward of nvidia/Qwen3.8-Flash-Next-NVFP4 in transformers' qwen4_exp at TRANSFORMERS_REV, f32 on the CPU, with the
weights the split `prepare` writes from that folder holds.

Every weight is dequantized by its own rule (quantization_config.quantized_layers of config.json picks the rule per
module), then, for the dense arm `q8`, rounded to the type `prepare` writes it in under BLOOMERY_PREPARE_Q8=1:
  NVFP4 (routed experts)     e2m1 x e4m3(weight_scale, one per 16 along K) x weight_scale_2
  FP8 (the PLE table)        e4m3 x weight_scale (one BF16 value for the table), only the rows the ids hash to
  BF16 projections           arm q8: Q8_0 by ggml's reference rule (d = amax / 127 stored as f16, q = roundf(x / d)),
                             the tensors prepare rewrites to Q8_0 (Q8_SUFFIXES, Q8_TOP); arm bf16: widened, as
                             prepare's default split stores them
  other BF16 / F32 / I64     widened to f32 (I64 buffers kept, and checked equal to the values the module derives)
input_scale is ignored: activations stay f32 at every linear.

One decoder layer is resident at a time: built on the meta device, loaded with the dequantized f32 weights of the q8
arm, run on the q8 sets' hidden streams, loaded again with the bf16 arm's dense weights (the experts and the PLE rows
are the same tensors), run on the bf16 sets', then dropped. The forward loop repeats Qwen4ExpTextModel.forward
(embedding repeated to hc_count streams, 3D mrope position ids with text positions first, create_causal_mask with
allow_is_causal_skip=False, create_recurrent_attention_mask, ple_input_ids) and calls transformers' own
Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding and Qwen4ExpTextGatedResidual (the final mixer). The PLE table
(320,001,536 rows of 160) is never materialized: the layer's nn.Embedding is replaced by a row reader that dequantizes
the rows it is asked for.

Output: one set directory per arm in the ik dump layout (MANIFEST.tsv with `# kind`/`# int` columns, raw f32 files
named <stem>.0.f32, `# complete` trailer), the tap names ik's qwen4exp dumps use. An integer tap (the route ids)
is written as ik's dumper writes one: its f32 file and a `.logical` copy, and the exact i32 twin of each
(<stem>.0.i32, <stem>.0.logical.i32), so the gates read its ids where they read ik's (the logical twin):
  hf_qwen4exp_nvfp4           the 5-token batch of ref_qwen4exp, q8 arm, every position
  hf_qwen4exp_nvfp4_d3k       the 3,001 ids of ref_qwen4exp_d3k, q8 arm, a decode-step set (`# decode_pos 3000`),
                              every position; result_output at the last position
  hf_qwen4exp_nvfp4_bf16      the batch, bf16 arm, every position
  hf_qwen4exp_nvfp4_d3k_bf16  the 3,001 ids, bf16 arm; l_out and result_output at the last position, routes at every
                              position
`# model` names the folder as a node's model store mounts it (MODEL_ID) and `# build` the md5 of this file, which the
family pins.

DIR may also be a cut of that folder (`fixture cut`, whose first shard's `bloomery.fixture.*` keys name REPO at
MODEL_REV): the fixture tier's model, read the same way. Its sets' `# model` names its first shard as DIR names it,
and `# fixture` is the line --fixture-line gives (`refset-check --fixture-line` of that shard), the identity of refset
family fx-hf-qwen4exp-nvfp4, whose sets are these under the `fx_` prefix of their directory.

--until L stops after layer L - 1 with the run's state saved, and --resume continues from it, so a long run can be cut
into sittings.

    qwen4exp-nvfp4-oracle.py --model DIR --out DIR --prose IDS [--fixture-line LINE] [--until L] [--resume]

DIR is the checkpoint folder at MODEL_REV or a cut of it; IDS holds the 3,001 ids, whitespace separated. Needs torch,
numpy, safetensors and transformers at TRANSFORMERS_REV; reads about 13 GiB of the folder a layer and holds one layer.
"""

import argparse
import glob
import hashlib
import json
import os
import threading
import time

import numpy as np
import torch

import transformers
from safetensors import safe_open
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

TRANSFORMERS_REV = "4c414256ddce7b3c896cb624d55d1299ad963d20"
REPO = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_REV = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
# The checkpoint folder as a node's container sees the model store.
MODEL_ID = "/models/nvidia__Qwen3.8-Flash-Next-NVFP4/" + MODEL_REV
REF_TOKENS = [760, 6511, 314, 9338, 369]  # tools/ref/models/qwen4exp.sh REF_TOKENS
PFX = "model.language_model."
MEM_FLOOR = 4 << 30  # MemAvailable floor (bytes)
SWAP_DROP = 2 << 30  # abort when SwapFree falls this far below its start
EXPERT_CHUNK = 64  # experts dequantized in one batch of torch ops

# The BF16 matrices prepare writes as Q8_0 (model-hf qwen35moe map): within a decoder layer, and outside the layers.
Q8_SUFFIXES = (
    "attn_hyper_connection.input_mix_weight_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "ple.key_proj.weight",
    "ple.value_proj.weight",
)
Q8_TOP = (
    PFX + "embed_tokens.weight",
    PFX + "hyper_connection_mixer.input_mix_weight_down.weight",
    PFX + "hyper_connection_mixer.input_mix_weight_up.weight",
    "lm_head.weight",
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# Every E4M3 byte as f32 (exact); 0x7F and 0xFF are NaN.
E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        out[k] = int(v.split()[0]) * 1024
    return out


def watch_memory(log_path, stop):
    start_swap = meminfo()["SwapFree"]
    low_avail = None
    with open(log_path, "a") as log:
        while not stop.is_set():
            m = meminfo()
            low_avail = m["MemAvailable"] if low_avail is None else min(low_avail, m["MemAvailable"])
            log.write("%d MemAvailable %d SwapFree %d low %d\n"
                      % (time.time(), m["MemAvailable"], m["SwapFree"], low_avail))
            log.flush()
            if m["MemAvailable"] < MEM_FLOOR or m["SwapFree"] < start_swap - SWAP_DROP:
                log.write("ABORT memory floor\n")
                log.flush()
                print("ABORT: MemAvailable %d, SwapFree %d (start %d)" % (m["MemAvailable"], m["SwapFree"], start_swap),
                      flush=True)
                os._exit(70)
            stop.wait(2)


def roundf(v):
    """C roundf: nearest, halves away from zero (torch.round takes halves to even)."""
    r = torch.round(v)
    t = torch.trunc(v)
    return torch.where((v - t).abs() == 0.5, t + torch.sign(v), r)


def q8_0_blocks(w):
    """ggml's quantize_row_q8_0_ref over the last axis of f32 `w`: (d as f16, q as int8), one d a 32-value block."""
    if w.shape[-1] % 32:
        raise SystemExit("Q8_0 of a row of %d values" % w.shape[-1])
    x = w.reshape(-1, 32)
    d = x.abs().amax(dim=1, keepdim=True) / 127.0
    inv = torch.where(d != 0, 1.0 / torch.where(d != 0, d, torch.ones_like(d)), torch.zeros_like(d))
    q = roundf(x * inv)
    return d.to(torch.float16), q.to(torch.int8)


def q8_0(w):
    """`w` as the Q8_0 values a reader dequantizes: q x f16(d)."""
    out = torch.empty_like(w)
    flat, dst = w.reshape(-1, w.shape[-1]), out.reshape(-1, w.shape[-1])
    for r in range(0, flat.shape[0], 16384):
        d, q = q8_0_blocks(flat[r:r + 16384])
        dst[r:r + 16384] = (q.float() * d.float()).reshape(-1, w.shape[-1])
    return out


def q8_name(name):
    if name in Q8_TOP:
        return True
    rest = name[len(PFX):] if name.startswith(PFX + "layers.") else None
    return rest is not None and rest.split(".", 2)[2] in Q8_SUFFIXES


def cut_shard(folder, line):
    """The first shard of `folder`, as `folder` names it, when the folder is a cut of REPO at MODEL_REV and `line` its
    `# fixture` line; None for the checkpoint folder itself, which must be at MODEL_REV and take no line."""
    first = glob.glob(os.path.join(folder, "*-00001-of-*.safetensors"))
    if len(first) != 1:
        raise SystemExit("%s: %d first shards, want one" % (folder, len(first)))
    meta = safe_open(first[0], framework="pt").metadata() or {}
    if "bloomery.fixture.version" not in meta:
        if not os.path.realpath(folder).endswith(MODEL_REV):
            raise SystemExit("model folder is not rev %s" % MODEL_REV)
        if line is not None:
            raise SystemExit("--fixture-line names a cut's line, and %s is the checkpoint folder" % folder)
        return None
    source = (meta.get("bloomery.fixture.source"), meta.get("bloomery.fixture.rev"))
    if source != (REPO, MODEL_REV):
        raise SystemExit("%s is a cut of %s@%s, not of %s@%s" % (first[0], source[0], source[1], REPO, MODEL_REV))
    if line is None or not line.startswith("# fixture\t") or "\n" in line:
        raise SystemExit("%s is a cut: --fixture-line gives its one `# fixture` line" % folder)
    return first[0]


class Checkpoint:
    """Tensor access by checkpoint name over the folder's safetensors files (index weight_map)."""

    def __init__(self, folder):
        self.folder = folder
        self.map = json.load(open(folder + "/model.safetensors.index.json"))["weight_map"]
        self.handles = {}
        self.used = set()

    def handle(self, name):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.folder, f), framework="pt")
        return self.handles[f]

    def get(self, name):
        self.used.add(name)
        return self.handle(name).get_tensor(name)

    def row(self, name, r):
        return self.handle(name).get_slice(name)[r:r + 1]

    def names(self, prefix):
        return [k for k in self.map if k.startswith(prefix)]


def nvfp4_stack(w, sc_u8, s2):
    """ModelOpt NVFP4 of `c` matrices in torch f32 (`w` [c, rows, half], `sc_u8` [c, rows, half / 8], `s2` [c]): byte i
    of a row holds values 2i (low nibble) and 2i+1 (high)."""
    c, rows, half = w.shape
    sc = E4M3[sc_u8.long()]
    q = torch.stack([E2M1[(w & 0xF).long()], E2M1[(w >> 4).long()]], dim=-1).view(c, rows, half // 8, 16)
    return (q * sc[..., None]).view(c, rows, 2 * half) * s2.view(c, 1, 1)


def nvfp4_f64(w, sc_u8, s2):
    """The same rule by a second path: numpy f64 of each factor, then f32."""
    lo, hi = w.numpy() & 0xF, w.numpy() >> 4
    e2m1 = E2M1.double().numpy()
    q = np.stack([e2m1[lo], e2m1[hi]], axis=-1).reshape(w.shape[0], -1)
    sc = np.repeat(E4M3.double().numpy()[sc_u8.numpy()], 16, axis=1)
    return torch.from_numpy((q * sc * float(s2)).astype(np.float32))


def check_bits(name, x, w, sc_u8, s2):
    """The torch f32 path against the f64 path: bit-equal or stop."""
    if not torch.equal(nvfp4_f64(w, sc_u8, s2).view(torch.int32), x.view(torch.int32)):
        raise SystemExit("%s: torch dequant differs from the f64 rule" % name)


def scale_bytes(sc_u8, stats):
    """Count the scale bytes a UE4M3 reader refuses (bit 7 set, 0x7F) and the zero ones."""
    stats["bit7"] += int((sc_u8 & 0x80).bool().sum())
    stats["0x7F"] += int((sc_u8 == 0x7F).sum())
    stats["zero"] += int((sc_u8 == 0).sum())


class PleRows(torch.nn.Module):
    """Stands in for the PLE nn.Embedding: dequantizes only the rows asked for (E4M3 rows of 160 values, one BF16
    weight_scale for the table), shard s holding rows [s * rows_per_shard, (s + 1) * rows_per_shard)."""

    def __init__(self, ck, prefix, shards, rows_per_shard, dim):
        super().__init__()
        self.register_buffer("weight", torch.empty(0), persistent=False)  # the module reads weight.device only
        self.ck, self.prefix, self.rows_per_shard, self.dim = ck, prefix, rows_per_shard, dim
        s = ck.get(prefix + ".weight_scale")
        if s.dtype != torch.bfloat16 or s.numel() != 1:
            raise SystemExit("%s.weight_scale: %s %s, want one BF16" % (prefix, s.dtype, tuple(s.shape)))
        self.scale = float(s.float().reshape(()))
        for i in range(shards):
            name = "%s.shard_%d.weight" % (prefix, i)
            t = ck.handle(name).get_slice(name)
            if tuple(t.get_shape()) != (rows_per_shard, dim) or t.get_dtype() != "F8_E4M3":
                raise SystemExit("%s: %s %s, want F8_E4M3 %s"
                                 % (name, t.get_dtype(), t.get_shape(), (rows_per_shard, dim)))
            ck.used.add(name)
        self.cache = {}
        self.stats = {"rows": 0, "nan": 0, "zero": 0}

    def load(self, r):
        s, local = divmod(r, self.rows_per_shard)
        b = self.ck.row("%s.shard_%d.weight" % (self.prefix, s), local).view(torch.uint8)
        nan = int(((b & 0x7F) == 0x7F).sum())
        if nan:
            raise SystemExit("%s row %d: %d NaN bytes" % (self.prefix, r, nan))
        x = E4M3[b.long()] * self.scale
        ref = torch.from_numpy((E4M3.double().numpy()[b.numpy()] * self.scale).astype(np.float32))
        if not torch.equal(ref.view(torch.int32), x.view(torch.int32)):
            raise SystemExit("%s row %d: torch dequant differs from the f64 rule" % (self.prefix, r))
        self.stats["rows"] += 1
        self.stats["zero"] += int((b == 0).sum())
        return x[0]

    def forward(self, ids):
        for r in torch.unique(ids).tolist():
            if r not in self.cache:
                self.cache[r] = self.load(r)
        flat = ids.reshape(-1).tolist()
        return torch.stack([self.cache[r] for r in flat]).reshape(*ids.shape, self.dim)


class Rules:
    def __init__(self, folder, cfg_json):
        q = cfg_json["quantization_config"]
        if q.get("quant_algo") != "MIXED_PRECISION":
            raise SystemExit("quantization_config: not MIXED_PRECISION")
        self.layers = q["quantized_layers"]
        self.ck = Checkpoint(folder)
        self.counts = {"NVFP4": 0, "Q8_0": 0, "plain": 0, "I64": 0}
        self.scale_bytes = {"bit7": 0, "0x7F": 0, "zero": 0, "matrices": 0}
        self.checked_nvfp4 = 0

    def tensor(self, name):
        """(q8 arm, bf16 arm) f32 values of the checkpoint tensor `name`; one tensor for both unless Q8_0 rounds it."""
        module = name.rsplit(".", 1)[0]
        if name.endswith(".weight") and module in self.layers:
            raise SystemExit("%s: quantized_layers rule %s where a dense tensor is read" % (name, self.layers[module]))
        t = self.ck.get(name)
        if t.dtype == torch.int64:
            self.counts["I64"] += 1
            return t, t
        if t.dtype not in (torch.bfloat16, torch.float32):
            raise SystemExit("%s: %s with no quantized_layers rule" % (name, t.dtype))
        wide = t.float()
        if not q8_name(name):
            self.counts["plain"] += 1
            return wide, wide
        if t.dtype != torch.bfloat16 or t.dim() != 2:
            raise SystemExit("%s: %s %s, want a BF16 matrix for Q8_0" % (name, t.dtype, tuple(t.shape)))
        self.counts["Q8_0"] += 1
        return q8_0(wide), wide

    def experts(self, layer, cfg):
        """gate_up_proj [E, 2I, H] and down_proj [E, H, I] in f32, transformers' stacked layout (gate rows first)."""
        module = "%slayers.%d.mlp.experts" % (PFX, layer)
        r = self.layers.get(module)
        if r != {"quant_algo": "NVFP4", "group_size": 16}:
            raise SystemExit("%s: rule %s, want NVFP4 group 16 with plain scales" % (module, r))
        n, hid, inter = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        gate_up = torch.empty(n, 2 * inter, hid)
        down = torch.empty(n, hid, inter)
        parts = (("gate_proj", gate_up[:, :inter]), ("up_proj", gate_up[:, inter:]), ("down_proj", down))
        for p, dst in parts:
            for e0 in range(0, n, EXPERT_CHUNK):
                ws, scs, s2s = [], [], []
                for e in range(e0, min(e0 + EXPERT_CHUNK, n)):
                    name = "%s.%d.%s.weight" % (module, e, p)
                    w = self.ck.get(name)
                    sc = self.ck.get(name + "_scale")
                    s2 = self.ck.get(name + "_scale_2")
                    rows, half = w.shape
                    if w.dtype != torch.uint8 or sc.dtype != torch.float8_e4m3fn or sc.shape != (rows, half // 8) \
                            or s2.dtype != torch.float32 or s2.numel() != 1:
                        raise SystemExit("%s: not an NVFP4 triple" % name)
                    ws.append(w)
                    scs.append(sc.view(torch.uint8))
                    s2s.append(float(s2.reshape(())))
                w, sc = torch.stack(ws), torch.stack(scs)
                scale_bytes(sc, self.scale_bytes)
                self.scale_bytes["matrices"] += len(ws)
                x = nvfp4_stack(w, sc, torch.tensor(s2s, dtype=torch.float32))
                self.counts["NVFP4"] += len(ws)
                if e0 == 0 and p == "gate_proj":
                    check_bits("%s.0.%s.weight" % (module, p), x[0], ws[0], scs[0], s2s[0])
                    self.checked_nvfp4 += 1
                dst[e0:e0 + len(ws)] = x
        if self.scale_bytes["bit7"] or self.scale_bytes["0x7F"]:
            raise SystemExit("layer %d: UE4M3 scale bytes a reader refuses: %s" % (layer, self.scale_bytes))
        return gate_up, down


def ple_constants(cfg, ple_layer_index):
    """layer_multipliers, head vocab sizes and offsets as Qwen4ExpTextNGramEmbedding.__init__ derives them."""
    heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    sizes, offsets, total = [], [], 0
    for h in range(heads):
        size = hf._find_nth_prime_after(cfg.ngram_vocab_size_base - 1, ple_layer_index * heads + h + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    mult = hf._build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_layer_index, cfg.seed)
    return {"layer_multipliers": mult, "ngram_heads_vocab_sizes": torch.tensor(sizes),
            "ngram_heads_offsets": torch.tensor(offsets)}


def layer_state(rules, cfg, i):
    """Decoder layer i loaded with the q8 arm's f32 weights, and the bf16 arm's dense weights that differ."""
    with torch.device("meta"):
        layer = hf.Qwen4ExpTextDecoderLayer(cfg, i)
    p = "%slayers.%d." % (PFX, i)
    ple_rows = None
    if layer.ple is not None:
        emb = layer.ple.ple_embedding
        r = rules.layers.get(p + "ple.ple_embedding.ngram_embedding")
        if r != {"quant_algo": "FP8"}:
            raise SystemExit("layer %d PLE rule %s, want FP8 per tensor" % (i, r))
        shards = cfg.split_ngram_parts
        rows, dim = emb.ngram_embedding.num_embeddings, emb.ngram_embedding.embedding_dim
        if rows % shards:
            raise SystemExit("PLE rows %d not a multiple of %d shards" % (rows, shards))
        ple_rows = PleRows(rules.ck, p + "ple.ple_embedding.ngram_embedding", shards, rows // shards, dim)
        emb.ngram_embedding = ple_rows
        want = ple_constants(cfg, emb.ple_layer_index)
    sd, bf16 = {}, {}
    for key in layer.state_dict():
        if key == "mlp.experts.gate_up_proj":
            sd[key], sd["mlp.experts.down_proj"] = rules.experts(i, cfg)
        elif key == "mlp.experts.down_proj":
            continue
        else:
            sd[key], wide = rules.tensor(p + key)
            if wide is not sd[key]:
                bf16[key] = wide
            leaf = key.rsplit(".", 1)[-1]
            if ple_rows is not None and key.startswith("ple.ple_embedding.") and leaf in want:
                if not torch.equal(sd[key], want[leaf]):
                    raise SystemExit("layer %d %s: checkpoint %s, module derives %s"
                                     % (i, key, sd[key].tolist(), want[leaf].tolist()))
    for key, v in layer.state_dict().items():
        if tuple(sd[key].shape) != tuple(v.shape):
            raise SystemExit("layer %d %s: %s, module wants %s" % (i, key, tuple(sd[key].shape), tuple(v.shape)))
    layer.load_state_dict(sd, strict=True, assign=True)
    layer.eval()
    left = [n for n in rules.ck.names(p) if n not in rules.ck.used and not n.endswith(".input_scale")]
    if left:
        raise SystemExit("layer %d: checkpoint tensors not used: %s" % (i, left[:8]))
    return layer, ple_rows, bf16, {k: sd[k] for k in bf16}


class Arm:
    def __init__(self, name, ids, dense, every_position):
        self.name, self.ids, self.dense, self.every_position = name, ids, dense, every_position
        self.rows = []
        self.h = None


def positions(arm):
    """The positions a set's taps hold, as its `# positions` line states them."""
    if len(arm.ids) <= 8:
        return "every position"
    if arm.every_position:
        return "every position; result_output at the last position only"
    return "l_out and result_output at the last position only; routes at every position"


class SetWriter:
    """ik dump layout: <stem>.0.f32 per tap, MANIFEST.tsv rows, `# complete` trailer."""

    def __init__(self, out, arm):
        self.dir = os.path.join(out, arm.name)
        os.makedirs(self.dir, exist_ok=True)
        manifest = os.path.join(self.dir, "MANIFEST.tsv")
        if os.path.exists(manifest):
            os.replace(manifest, manifest + ".incomplete")
        self.arm = arm

    def tensor(self, name, t, ty="f32", int_twin=False):
        """t laid out [.., ne1, ne0] (C order): ne0 is the last axis. With int_twin, the i32 tensor's logical copy
        and both exact twins as well, named by tools/bloomery/manifest.py's dump_file_name rule."""
        a = t.detach().float().contiguous().numpy()
        ne = list(reversed(a.shape)) + [1] * (4 - a.ndim)
        stem = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        layouts = [("flat", ".0")] + ([("logical", ".0.logical")] if int_twin else [])
        for _, at in layouts:
            a.astype("<f4").tofile(os.path.join(self.dir, stem + at + ".f32"))
        self.arm.rows.append("tensor\t%s\t0\t%s\t%d\t%d\t%d\t%d\t%d\t%.17g\thf\t1\t%d\t-\t-"
                             % (name, ty, ne[0], ne[1], ne[2], ne[3], a.size * 4, float(np.sum(a, dtype=np.float64)),
                                int(int_twin)))
        if int_twin:
            b = t.detach().to(torch.int32).contiguous().numpy()
            wide = b.astype(np.int64)
            for layout, at in layouts:
                b.astype("<i4").tofile(os.path.join(self.dir, stem + at + ".i32"))
                self.arm.rows.append("int\t%s\t0\ttensor\t%s\ti32\t%s\t%d\t%d\t%d\t%d\t%s"
                                     % (name, ty, layout, b.size, b.size * 4, int(wide.sum()), int(np.abs(wide).max()),
                                        stem + at + ".i32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prose", required=True)
    ap.add_argument("--fixture-line", help="a cut's `# fixture` line (refset-check --fixture-line of its first shard)")
    ap.add_argument("--until", type=int, default=None, help="stop after this many layers with the state saved")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.abspath(__file__), "rb") as f:
        build = hashlib.md5(f.read()).hexdigest()
    print("build %s transformers %s (%s) modeling_qwen4_exp %s torch %s" % (
        build, TRANSFORMERS_REV, transformers.__version__, sha256(hf.__file__), torch.__version__), flush=True)
    shard = cut_shard(a.model, a.fixture_line)
    model = os.path.realpath(a.model)
    cfg_json = json.load(open(model + "/config.json"))
    cfg = Qwen4ExpTextConfig(**cfg_json["text_config"])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    n_layers = cfg.num_hidden_layers
    until = n_layers if a.until is None else a.until
    if not 0 < until <= n_layers:
        raise SystemExit("--until %d: not in 1..%d" % (until, n_layers))

    prose_bytes = open(a.prose, "rb").read()
    try:
        prose = [int(x) for x in prose_bytes.split()]
    except ValueError:
        raise SystemExit("--prose: expected integer token ids")
    if len(prose) != 3001:
        raise SystemExit("--prose: expected 3,001 ids for the D3K reference")
    if any(t < 0 or t >= cfg.vocab_size for t in prose):
        raise SystemExit("--prose: token id outside the model vocabulary")
    arms = [Arm("hf_qwen4exp_nvfp4", REF_TOKENS, "q8", True),
            Arm("hf_qwen4exp_nvfp4_d3k", prose, "q8", True),
            Arm("hf_qwen4exp_nvfp4_bf16", REF_TOKENS, "bf16", True),
            Arm("hf_qwen4exp_nvfp4_d3k_bf16", prose, "bf16", False)]
    ckpt_path = os.path.join(a.out, "resume.pt")
    identity = (
        build, model, a.fixture_line, cfg_json, sha256(hf.__file__), str(torch.__version__),
        tuple((arm.name, tuple(arm.ids), arm.dense, arm.every_position) for arm in arms),
    )
    ck = None
    if a.resume:
        if not os.path.isfile(ckpt_path):
            raise SystemExit("resume.pt: no saved state for --resume")
        ck = torch.load(ckpt_path)
        if ck.get("identity") != identity:
            raise SystemExit("resume.pt: writer, model, cut or tokens changed; restart without --resume")
    writers = {arm.name: SetWriter(a.out, arm) for arm in arms}

    stop = threading.Event()
    threading.Thread(target=watch_memory, args=(os.path.join(a.out, "mem.log"), stop), daemon=True).start()
    rules = Rules(model, cfg_json)

    # Per-arm inputs Qwen4ExpTextModel.forward builds before its layer loop (no cache, no attention mask given).
    rotary = hf.Qwen4ExpTextRotaryEmbedding(config=cfg)
    emb = rules.ck.get(PFX + "embed_tokens.weight")
    if emb.dtype != torch.bfloat16:
        raise SystemExit("embed_tokens: %s, want BF16" % emb.dtype)
    inputs = {}
    for arm in arms:
        ids = torch.tensor([arm.ids])
        x = emb[ids].float()
        if arm.dense == "q8":
            x = q8_0(x)
        pos = torch.arange(len(arm.ids)).view(1, 1, -1).expand(4, 1, -1)
        text_pos, pos = pos[0], pos[1:]
        mask_kwargs = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": None,
                       "position_ids": text_pos, "allow_is_causal_skip": False}
        causal = create_causal_mask(**mask_kwargs)
        conv_mask = create_recurrent_attention_mask(**mask_kwargs)
        ple_ids = ids
        if conv_mask is not None:
            ple_ids = torch.where(conv_mask.bool(), ids, cfg.eos_token_id)
        inputs[arm.name] = (causal, conv_mask, ple_ids, rotary(x, pos))
        arm.h = x.repeat(1, 1, cfg.hc_count)
    del emb

    t_all = time.time()
    start = 0
    ple_stats = None
    if ck is not None:
        start = ck["next"]
        for arm in arms:
            arm.h, arm.rows = ck["arms"][arm.name]
        rules.counts, rules.scale_bytes, rules.checked_nvfp4 = ck["counts"], ck["scale_bytes"], ck["checked"]
        ple_stats = ck["ple"]
        print("resume at layer", start, flush=True)
    else:
        for arm in arms:
            if arm.every_position:
                writers[arm.name].tensor("hc_residual", arm.h[0].view(len(arm.ids), cfg.hc_count, cfg.hidden_size))
    if start > until or start == until < n_layers:
        raise SystemExit("resume at layer %d, nothing to run before --until %d" % (start, until))

    for i in range(start, until):
        t0 = time.time()
        rules.ck.used.clear()
        layer, ple_rows, bf16, q8 = layer_state(rules, cfg, i)
        t_load = time.time() - t0
        routes = {}
        hook = layer.mlp.gate.register_forward_hook(lambda m, inp, out: routes.update(out=out))
        dense = "q8"
        for arm in arms:
            if arm.dense != dense:
                layer.load_state_dict(bf16 if arm.dense == "bf16" else q8, strict=False, assign=True)
                dense = arm.dense
            causal, conv_mask, ple_ids, pe = inputs[arm.name]
            routes.clear()
            arm.h = layer(arm.h, position_embeddings=pe, attention_mask=causal, conv_mask=conv_mask,
                          past_key_values=None, ple_input_ids=ple_ids)
            w = writers[arm.name]
            l_out = arm.h[0].view(len(arm.ids), cfg.hc_count, cfg.hidden_size)
            w.tensor("l_out-%d" % i, l_out if arm.every_position else l_out[-1:])
            logits, weights, idx = routes["out"]
            w.tensor("ffn_moe_logits-%d" % i, logits)
            w.tensor("ffn_moe_probs-%d" % i, torch.softmax(logits, dim=-1, dtype=torch.float))
            w.tensor("ffn_moe_topk-%d" % i, idx, ty="i32", int_twin=True)
            w.tensor("ffn_moe_weights_norm-%d" % i, weights)
        hook.remove()
        if ple_rows is not None:
            ple_stats = dict(ple_rows.stats)
            print("PLE layer %d rows read %d, zero bytes %d, no NaN byte, torch == f64 rule on every row" % (
                i, ple_stats["rows"], ple_stats["zero"]), flush=True)
        del layer, ple_rows, bf16, q8
        torch.save({"identity": identity, "next": i + 1, "arms": {arm.name: (arm.h, arm.rows) for arm in arms},
                    "counts": rules.counts, "scale_bytes": rules.scale_bytes, "checked": rules.checked_nvfp4,
                    "ple": ple_stats}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        m = meminfo()
        print("layer %d %s load %.1fs total %.1fs MemAvailable %.1f GiB SwapFree %.1f GiB" % (
            i, cfg.layer_types[i], t_load, time.time() - t0, m["MemAvailable"] / 2**30, m["SwapFree"] / 2**30),
              flush=True)
    if until < n_layers:
        stop.set()
        print("PAUSED at layer %d of %d after %.1f s (state in %s)" % (until, n_layers, time.time() - t_all, ckpt_path),
              flush=True)
        return

    rules.ck.used.clear()
    mixers, heads = {}, {}
    mix_q8, mix_bf16 = {}, {}
    for k in hf.Qwen4ExpTextGatedResidual(cfg, use_combine=False).state_dict():
        mix_q8[k], mix_bf16[k] = rules.tensor(PFX + "hyper_connection_mixer." + k)
    head_q8, head_bf16 = rules.tensor("lm_head.weight")
    for dense, msd, hw in (("q8", mix_q8, head_q8), ("bf16", mix_bf16, head_bf16)):
        mixers[dense] = hf.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        mixers[dense].load_state_dict(msd, strict=True)
        heads[dense] = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        heads[dense].weight.data = hw
    rules.ck.used.add(PFX + "embed_tokens.weight")
    top = [n for n in rules.ck.names(PFX) if not n.startswith(PFX + "layers.") and n not in rules.ck.used]
    if top:
        raise SystemExit("top-level checkpoint tensors not used: %s" % top[:8])
    for arm in arms:
        x = mixers[arm.dense](arm.h)[0]
        head = heads[arm.dense]
        w = writers[arm.name]
        if len(arm.ids) <= 8:
            w.tensor("result_embd", x)
            w.tensor("result_output", head(x))
            continue
        if arm.every_position:
            w.tensor("result_embd", x)
        w.tensor("result_output", head(x[-1:]))
        # next-id metrics (logged, not written): ids[t+1] against the logits at t, t < len - 1
        target = torch.tensor(arm.ids[1:])
        nll, top1, n = 0.0, 0, len(target)
        for c in range(0, n, 256):
            lo = head(x[c:min(c + 256, n)])
            nll += float(torch.nn.functional.cross_entropy(lo.double(), target[c:c + 256], reduction="sum"))
            top1 += int((lo.argmax(-1) == target[c:c + 256]).sum())
        print("%s next-id over %d positions: mean NLL %.4f (ppl %.3f), top1 %d (%.4f)" % (
            arm.name, n, nll / n, float(np.exp(nll / n)), top1, top1 / n), flush=True)

    for arm in arms:
        w = writers[arm.name]
        with open(os.path.join(w.dir, "MANIFEST.tsv"), "w") as f:
            f.write("# hf_dump: transformers qwen4_exp f32 forward of an NVIDIA ModelOpt NVFP4 folder, raw f32, "
                    "little-endian\n")
            f.write("# model\t%s\n" % (shard or MODEL_ID))
            if shard:
                f.write("%s\n" % a.fixture_line)
            f.write("# model_dir\t%s\n" % model)
            f.write("# build\t%s\n" % build)
            f.write("# transformers\t%s %s\n" % (TRANSFORMERS_REV, transformers.__version__))
            f.write("# torch\t%s\n" % torch.__version__)
            f.write("# arch\tqwen4exp\n")
            f.write("# model_file\t%s@%s\n" % (REPO, MODEL_REV))
            f.write("# tokens\t%s\n" % ",".join(map(str, arm.ids)))
            if arm.ids is not REF_TOKENS:
                f.write("# tokens_file\t%s\n" % os.path.realpath(a.prose))
                f.write("# tokens_file_sha256\t%s\n" % hashlib.sha256(prose_bytes).hexdigest())
                f.write("# tokens_count\t%d\n" % len(arm.ids))
                f.write("# prefill\t%d\n" % (len(arm.ids) - 1))
                f.write("# decode_pos\t%d\n" % (len(arm.ids) - 1))
            f.write("# dequant\tNVFP4: e2m1 x e4m3 x weight_scale_2; PLE FP8 rows: e4m3 x weight_scale; "
                    "input_scale ignored\n")
            f.write("# dense\t%s\n" % ("Q8_0 (ggml reference rule) of the BF16 matrices prepare writes as Q8_0, the other "
                                       "BF16 and F32 widened" if arm.dense == "q8" else "BF16 and F32 widened"))
            f.write("# activations\tf32 at every linear\n")
            f.write("# positions\t%s\n" % positions(arm))
            f.write("# routes\tffn_moe_topk ordered by ffn_moe_probs descending (torch.topk); "
                    "ffn_moe_weights_norm in the same order\n")
            f.write("# layers\t%d of %d (MTP layer not run)\n" % (n_layers, cfg.num_hidden_layers))
            f.write("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n")
            f.write("# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\n")
            f.write("# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n")
            for r in arm.rows:
                f.write(r + "\n")
            f.write("# complete\t%d\t0\n" % sum(1 for r in arm.rows if r.startswith("tensor\t")))
    stop.set()
    print("counts", json.dumps(rules.counts), "nvfp4 scales", json.dumps(rules.scale_bytes),
          "nvfp4 torch == f64 checked", rules.checked_nvfp4, "ple", json.dumps(ple_stats), flush=True)
    print("DONE %d layers in %.1f s this sitting" % (n_layers - start, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
